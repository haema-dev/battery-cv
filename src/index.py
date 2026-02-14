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
    logger.info(f"🔎 검증 데이터 탐색 시작: {base}")
    
    # 1순위: 'datasets/256x256 fit/validation' 정밀 탐색
    for p in base.rglob("*/validation"):
        if "256x256 fit" in str(p):
            logger.success(f"✅ 검증 데이터셋 발견: {p}")
            return p
            
    # 2순위: 'validation' 폴더 탐색
    for p in base.rglob("validation"):
        if p.is_dir():
            logger.warning(f"⚠️ 'validation' 폴더 발견: {p}")
            return p
            
    logger.error("❌ 'validation' 폴더를 찾을 수 없습니다.")
    return None

def run_evaluation(data_path, model_path, output_dir):
    logger.info("==================================================")
    logger.info("🚀 STAGE 2: INFERENCE & PERFORMANCE EVALUATION")
    logger.info("==================================================")

    if not INFERENCER_AVAILABLE:
        logger.error("❌ 'TorchInferencer'를 로드할 수 없습니다.")
        return

    # 1. Inferencer 초기화
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ 사용 장치: {device}")
    
    try:
        # 모델 경로 존재 확인
        if not os.path.exists(model_path):
            logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return
            
        # [SURGEON PATCH] 'model' key missing 대응
        # TorchInferencer는 내부적으로 ckpt['model']을 찾으려 하지만,
        # 간혹 ckpt 자체가 state_dict이거나 다른 구조일 경우 에러가 발생합니다.
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" not in ckpt:
            logger.warning("⚠️ 'model' key가 없습니다. 자동 구조 복원 시도...")
            # 만약 ckpt 자체가 state_dict인 경우 'model' key로 감싸서 임시 파일 생성
            temp_model_path = "/tmp/fixed_model.pt"
            os.makedirs("/tmp", exist_ok=True)
            torch.save({"model": ckpt}, temp_model_path)
            model_path = temp_model_path
            logger.success(f"✅ 구조 복원 완료: {model_path}")

        inferencer = TorchInferencer(path=model_path, device=device)
        logger.success("✅ 모델 로드 성공")
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
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
    logger.info(f"📂 카테고리 목록: {[c.name for c in categories]}")

    for cat_dir in categories:
        cat_name = cat_dir.name
        is_actual_anomaly = 0 if cat_name.lower() == "good" else 1
        
        cat_output = output_base / "heatmaps" / cat_name
        cat_output.mkdir(parents=True, exist_ok=True)
        
        img_files = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpeg"))
        logger.info(f"🔍 {cat_name} 처리 중... ({len(img_files)}장)")

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
                logger.warning(f"⚠️ 처리 실패 ({img_path.name}): {e}")

    # 5. 최종 리포트 생성
    total = sum(matrix.values())
    accuracy = (matrix["TP"] + matrix["TN"]) / total if total > 0 else 0
    
    logger.info("--------------------------------------------------")
    logger.info("📊 STAGE 2 EVALUATION REPORT")
    logger.info(f"✅ Accuracy: {accuracy:.4f}")
    logger.info(f"📝 Confusion Matrix: {dict(matrix)}")
    logger.info("--------------------------------------------------")

    # 결과 파일 저장
    report = {
        "metrics": dict(matrix),
        "overall_accuracy": accuracy,
        "details": results_summary
    }
    with open(output_base / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    logger.success(f"🎉 Stage 2 완료. 히트맵 및 리포트 저장됨: {output_dir}")

if __name__ == "__main__":
    # 디버깅: 에저에서 들어오는 원본 인자 확인
    logger.info(f"📋 Raw Arguments: {sys.argv}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input validation folders")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    
    try:
        args = parser.parse_args()
        logger.info(f"✅ Parsed Arguments: data={args.data_path}, model={args.model_path}, out={args.output_dir}")
        
        sys.stdout.reconfigure(line_buffering=True)
        run_evaluation(args.data_path, args.model_path, args.output_dir)
    except Exception as e:
        logger.error(f"❌ FATAL: Argument issue: {e}")
        sys.exit(1)