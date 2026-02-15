import argparse
import os
import sys
import json
import time
from pathlib import Path
import torch
import cv2
import numpy as np
from loguru import logger
from collections import defaultdict

# Anomalib TorchInferencer 사용
try:
    from anomalib.deploy import TorchInferencer
    INFERENCER_AVAILABLE = True
except ImportError:
    INFERENCER_AVAILABLE = False

def find_validation_root(base_path):
    """사용자님이 지정하신 'datasets/256x256 fit/validation' 경로를 정밀 탐색합니다."""
    base = Path(base_path).resolve()
    logger.info(f"[*] 검증 데이터 탐색 시작: {base}")
    
    # 1순위: 'datasets/256x256 fit/validation' 정밀 탐색
    for p in base.rglob("*/validation"):
        if "256x256 fit" in str(p):
            logger.success(f"OK: 검증 데이터셋 발견: {p}")
            return p
            
    # 2순위: 'validation' 폴더 탐색
    for p in base.rglob("validation"):
        if p.is_dir():
            logger.warning(f"WARN: 'validation' 폴더 발견: {p}")
            return p
            
    logger.error("ERR: 'validation' 폴더를 찾을 수 없습니다.")
    return None

def run_evaluation(data_path, model_path, output_dir):
    logger.info("==================================================")
    logger.info("STAGE 2: INFERENCE & PERFORMANCE EVALUATION")
    logger.info("==================================================")

    if not INFERENCER_AVAILABLE:
        logger.error("[ERR] 'TorchInferencer'를 로드할 수 없습니다.")
        return

    # 1. 모델 수동 조립 (Architecture Reconstruction)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[*] 사용 장치: {device}")
    
    try:
        from anomalib.models import Fastflow
        
        # [RECONSTRUCTION] 설계도(뼈대) 먼저 세우기: resnet18 기반의 Fastflow
        logger.info("[*] 모델 설계도(Fastflow-ResNet18) 기반 뼈대 생성 중...")
        model = Fastflow(backbone="resnet18")
        
        # 가중치 파일 로드
        if not os.path.exists(model_path):
            logger.error(f"[ERR] 모델 파일을 찾을 수 없습니다: {model_path}")
            return
            
        ckpt = torch.load(model_path, map_location="cpu")
        
        # 가중치 정제 (state_dict 추출)
        # ckpt가 lightning 형식({"state_dict": ...})이거나 raw state_dict일 경우 대응
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        
        # 간혹 'model' 키로 한 번 더 감싸져 있는 경우 대응
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
            
        # 뼈대에 지능(가중치) 주입
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval() # 명시적으로 eval 모드 전환
        logger.success("[OK] 모델 가중치 정밀 조립 완료!")

        # 조립된 '객체(nn.Module)'를 TorchInferencer가 기대하는 형식으로 임시 저장
        # TorchInferencer는 내부적으로 torch.load(path)['model']을 사용하거나 아예 객체를 기대함
        temp_model_path = "/tmp/reconstructed_model.pt"
        os.makedirs("/tmp", exist_ok=True)
        torch.save({"model": model}, temp_model_path)
        
        # 최종적으로 조립된 모델의 경로로 업데이트
        logger.info(f"[SAVED] 조립된 모델 임시 저장: {temp_model_path}")
        inferencer = TorchInferencer(path=temp_model_path, device=device)
        logger.success("[OK] 최종 TorchInferencer 로드 성공")
    except Exception as e:
        logger.error(f"[ERR] 모델 조립 및 로드 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return

    # 2. 경로 설정
    validation_root = find_validation_root(data_path)
    if not validation_root: return
    
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # 3. 평가 데이터 초기화 (Confusion Matrix용)
    results_summary = []
    matrix = defaultdict(int) 

    # 4. 카테고리 순회
    categories = [d for d in validation_root.iterdir() if d.is_dir()]
    logger.info(f"[*] 카테고리 목록: {[c.name for c in categories]}")

    for cat_dir in categories:
        cat_name = cat_dir.name
        is_actual_anomaly = 0 if cat_name.lower() == "good" else 1
        
        cat_output = output_base / "heatmaps" / cat_name
        cat_output.mkdir(parents=True, exist_ok=True)
        
        img_files = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpeg"))
        logger.info(f"[*] {cat_name} 처리 중... ({len(img_files)}장)")

        for img_path in img_files:
            try:
                # 추론 수행
                prediction = inferencer.predict(image=str(img_path))
                
                # 시각화 저장 (Heatmap)
                heatmap = prediction.heatmap
                cv2.imwrite(str(cat_output / f"heatmap_{img_path.name}"), heatmap)
                
                # 분류 결과 추출
                pred_label = int(prediction.pred_label) if hasattr(prediction, 'pred_label') else (1 if prediction.pred_score > 0.5 else 0)
                pred_score = float(prediction.pred_score)

                # 메트릭 업데이트
                if is_actual_anomaly == 0: 
                    if pred_label == 0: matrix["TN"] += 1
                    else: matrix["FP"] += 1
                else: 
                    if pred_label == 1: matrix["TP"] += 1
                    else: matrix["FN"] += 1
                
                results_summary.append({
                    "image": img_path.name,
                    "actual": "Anomaly" if is_actual_anomaly else "Normal",
                    "predicted": "Anomaly" if pred_label else "Normal",
                    "score": pred_score
                })

            except Exception as e:
                logger.warning(f"[WARN] 처리 실패 ({img_path.name}): {e}")

    # 5. 최종 리포트 생성
    total = sum(matrix.values())
    accuracy = (matrix["TP"] + matrix["TN"]) / total if total > 0 else 0
    
    logger.info("--------------------------------------------------")
    logger.info("STAGE 2 EVALUATION REPORT")
    logger.info(f"[*] Accuracy: {accuracy:.4f}")
    logger.info(f"[*] Confusion Matrix: {dict(matrix)}")
    logger.info("--------------------------------------------------")

    # 결과 파일 저장
    report = {
        "metrics": dict(matrix),
        "overall_accuracy": accuracy,
        "details": results_summary
    }
    with open(output_base / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    logger.success(f"[FINISH] Stage 2 완료. 히트맵 및 리포트 저장됨: {output_dir}")

if __name__ == "__main__":
    # 디버깅: 에저에서 들어오는 원본 인자 확인
    logger.info(f"[*] Raw Arguments: {sys.argv}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input validation folders")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    
    try:
        args = parser.parse_args()
        logger.info(f"[OK] Parsed Arguments: data={args.data_path}, model={args.model_path}, out={args.output_dir}")
        
        sys.stdout.reconfigure(line_buffering=True)
        run_evaluation(args.data_path, args.model_path, args.output_dir)
    except Exception as e:
        logger.error(f"[FATAL] Argument issue: {e}")
        sys.exit(1)