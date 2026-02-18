# -*- coding: utf-8 -*-
# Version trigger for Azure ML - v6 (Strict Compliance)
import os
import sys
import torch
import argparse
import mlflow
import json
import time
import cv2
import random
import numpy as np
from loguru import logger
from anomalib.models import Fastflow
from torch import optim
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.loggers import AnomalibMLFlowLogger
from pathlib import Path
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
from lightning.pytorch.callbacks import EarlyStopping
import lightning

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_matched_weights(model_path, model):
    """
    [Definitive Fix] 가중치를 추출하고 매칭 전략에 따라 모델에 직접 주입합니다.
    - 엔진의 ckpt_path 피드백 루프를 우회하여 확실한 주입을 보장합니다.
    """
    logger.info(f"[*] 가중치 수동 주입 시작: {model_path}")
    raw_ckpt = torch.load(model_path, map_location="cpu")
    
    if isinstance(raw_ckpt, dict):
        state_dict = raw_ckpt.get("state_dict", raw_ckpt.get("model", raw_ckpt))
    else:
        state_dict = raw_ckpt

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    
    strategies = [
        ("As-is", lambda d: d),
        ("Add 'model.'", lambda d: {f"model.{k}": v for k, v in d.items()}),
        ("Remove 'model.'", lambda d: {k[6:] if k.startswith("model.") else k: v for k, v in d.items()})
    ]
    
    best_matching_dict = state_dict
    max_matches = 0
    best_strategy = "None"
    
    for name, func in strategies:
        try:
            test_dict = func(state_dict)
            matches = len(model_keys.intersection(test_dict.keys()))
            if matches > max_matches:
                max_matches = matches
                best_strategy = name
                best_matching_dict = test_dict
        except Exception: continue

    logger.info(f"[*] 매칭 전략: {best_strategy} (매칭률: {(max_matches/len(model_keys))*100:.1f}%)")
    
    # 모델에 존재하는 키만 필터링
    final_state_dict = {k: v for k, v in best_matching_dict.items() if k in model_keys}
    
    # 직접 주입 (Strict=False로 유연하게 대응하되, 매칭률 로그로 검증)
    model.load_state_dict(final_state_dict, strict=False)
    
    # 주입 상태 진단 (가중치가 모두 0은 아닌지 확인)
    first_key = list(final_state_dict.keys())[0] if final_state_dict else None
    if first_key:
        weight_mean = final_state_dict[first_key].abs().mean().item()
        logger.info(f"[*] 가중치 주입 샘플 검증 ({first_key}): Mean Abs = {weight_mean:.6f}")
    
    return True

def main():
    # ================== 1. Input/Output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model checkpoint")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=50) # 진단용 표준 epoch 설정
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="evaluation", choices=["training", "evaluation", "prediction"])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for mass inference")

    args = parser.parse_args()
    set_seed(args.seed)
    
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 MODE: {args.mode.upper()} | BACKBONE: {args.backbone}")

    try:
        # ================== 2. Anomalib 데이터 및 모델 구성 ==================== #
        dataset_root = Path(args.data_path)
        
        transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((256, 256)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Prediction 모드에서는 정답 라벨 없이 모든 이미지를 훑습니다.
        if args.mode == "prediction":
            from anomalib.data import PredictDataset
            # validation 폴더의 모든 이미지를 대상으로 전수 검사 실시
            predict_dir = dataset_root / "validation"
            datamodule = PredictDataset(path=predict_dir, transform=transform)
            loader = torch.utils.data.DataLoader(datamodule, batch_size=args.batch_size, shuffle=False)
            logger.info(f"📁 Prediction 대상 경로: {predict_dir}")
        else:
            val_path = dataset_root / "validation"
            abnormal_dirs = [f"validation/{d.name}" for d in val_path.iterdir() if d.is_dir() and d.name != "good"] if val_path.exists() else []
            datamodule = Folder(
                name="battery", root=str(dataset_root),
                normal_dir="train/good", normal_test_dir="validation/good",
                abnormal_dir=abnormal_dirs if abnormal_dirs else None,
                train_batch_size=args.batch_size, eval_batch_size=args.batch_size,
                train_transform=transform, eval_transform=transform,
                task="classification", seed=args.seed
            )

        model = Fastflow(backbone=args.backbone, flow_steps=8)
        model.setup()

        if args.model_path and os.path.exists(args.model_path):
            load_matched_weights(args.model_path, model)

        # [Critical Fix] Stage 2에서 검증된 최적 임계값(-0.2604) 강제 적용
        # 모델 로드 후 임계값이 초기화되는 것을 막기 위해 명시적으로 주입합니다.
        SAVED_THRESHOLD = -0.2604
        if hasattr(model, "image_threshold"):
            if hasattr(model.image_threshold, "value"):
                model.image_threshold.value = torch.tensor(SAVED_THRESHOLD)
            else:
                model.image_threshold = torch.tensor(SAVED_THRESHOLD)
            logger.info(f"[*] 임계값 복구 완료: {SAVED_THRESHOLD}")

        # ================== 3. 엔진 설정 및 실행 ==================== #
        early_stop = EarlyStopping(monitor="image_AUROC", patience=10, mode="max", verbose=True)
        mlflow_logger = AnomalibMLFlowLogger(experiment_name="Battery_S2_Diagnostics", save_dir=str(OUTPUT_DIR))
        
        engine = Engine(
            max_epochs=args.epochs,
            devices=1,
            accelerator="auto",
            logger=mlflow_logger,
            callbacks=[early_stop] if args.mode == "training" else [],
            default_root_dir=str(OUTPUT_DIR)
        )

        if args.mode == "training":
            logger.info("🔥 [ST5] Training 모드 시작")
            engine.fit(model=model, datamodule=datamodule)
        elif args.mode == "evaluation":
            logger.info("🔍 [ST5] Evaluation 모드 시작")
            engine.test(model=model, datamodule=datamodule, ckpt_path=None)
        elif args.mode == "prediction":
            logger.info("📡 [ST5] 전수검사 (Prediction) 모드 및 Heatmap 생성 시작")
            from anomalib.utils.visualization import ImageVisualizer
            # Anomalib 1.1.3 시각화 도구 준비
            visualizer = ImageVisualizer(mode="full", task="classification")
            
            predictions = engine.predict(model=model, dataloaders=loader)
            
            # 결과 수집 및 CSV 저장 (Stage 6 리포팅용)
            import pandas as pd
            records = []
            
            # 히트맵 저장 폴더 생성
            vis_dir = OUTPUT_DIR / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            for batch in predictions:
                # Anomalib 1.1.3 Predict 결과 구조에 맞춰 데이터 추출
                paths = batch["image_path"]
                images = batch["image"]
                anomaly_maps = batch["anomaly_maps"]
                scores = batch["pred_scores"].cpu().numpy()
                labels = batch["pred_labels"].cpu().numpy()
                
                for i in range(len(paths)):
                    path = paths[i]
                    score = float(scores[i])
                    label = bool(labels[i])
                    
                    # 히트맵 이미지 생성 (RGB numpy array 반환)
                    res_image = visualizer.visualize(
                        image=images[i],
                        anomaly_map=anomaly_maps[i],
                        score=score,
                        label=label
                    )
                    
                    # 파일 저장 로직 (BGR 변환 후 OpenCV 사용)
                    file_name = Path(path).name
                    save_path = vis_dir / f"vis_{file_name}"
                    cv2.imwrite(str(save_path), cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR))
                    
                    records.append({
                        "file_path": path,
                        "file_name": file_name,
                        "parent_dir": Path(path).parent.name,
                        "anomaly_score": score,
                        "is_defect": label,
                        "vis_path": str(save_path)
                    })
            
            df = pd.DataFrame(records)
            csv_path = OUTPUT_DIR / "results.csv"
            df.to_csv(csv_path, index=False)
            logger.success(f"📊 전수검사 및 히트맵 저장 완료: {vis_dir} ({len(df)} images)")
            logger.success(f"📊 CSV 완료: {csv_path}")
        
        # 최종 가중치 저장 및 결과 보고
        torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
        logger.success(f"[FINISH] Output saved at: {OUTPUT_DIR}")
        logger.success(f"[FINISH] 작업 완료. Stage 5 전수검사 모듈 준비 완료.")

    except Exception as e:
        logger.error(f"[FATAL] 오류 발생: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
