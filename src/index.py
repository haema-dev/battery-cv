# -*- coding: utf-8 -*-
import os
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
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.loggers import AnomalibMLFlowLogger
from pathlib import Path
from torchvision.transforms.v2 import Resize
from lightning.pytorch.callbacks import EarlyStopping

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # ================== 1. Input/Output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backbone", type=str, default="resnet18", help="Feature extractor backbone")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")

    args = parser.parse_args()
    set_seed(args.seed)
    base_path = Path(args.data_path)
    
    logger.info("==================================================")
    logger.info(" S1_FastFlow_Training: [Production Ready Mode]")
    logger.info(f" 마운트 루트: {base_path}")
    logger.info(f" 설정: Backbone={args.backbone}, Epochs={args.epochs}, Seed={args.seed}")
    logger.info("==================================================")

    # [디버깅] 실제 마운트된 파일 구조를 2단계까지 출력 (ls -R 스타일)
    try:
        logger.info(" [Debug] 마운트된 디렉토리 구조 탐색 중...")
        for root, dirs, files in os.walk(base_path):
            level = len(Path(root).relative_to(base_path).parts)
            if level <= 2: # 너무 길어지지 않게 2단계까지만
                indent = "  " * level
                logger.info(f"{indent} {Path(root).name}/ ({len(files)} files)")
            if level > 2: continue # 더 깊은 곳은 생략
    except Exception as e:
        logger.warning(f" 구조 출력 중 오류 (무시 가능): {e}")

    # [Fail-Fast] 필수 폴더 존재 여부 체크
    train_path = base_path / "train/good"
    val_path = base_path / "validation"
    
    check_targets = {
        "학습용 정상 데이터 (train/good)": train_path,
        "검증용 데이터 (validation)": val_path
    }
    
    missing_critical = False
    for label, path in check_targets.items():
        if path.exists():
            logger.info(f" {label} 확인 완료: {path}")
        else:
            logger.error(f" {label}을(를) 찾을 수 없음: {path}")
            if label == "학습용 정상 데이터 (train/good)":
                missing_critical = True

    if missing_critical:
        raise FileNotFoundError(f" 필수 학습 경로가 없습니다. 위 로그를 보고 데이터 구조를 확인하세요.")

    dataset_root = base_path

    # ================== 2. MLflow & Output 설정 ==================== #
    mlflow.start_run()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f" 사용 장치: {device}")

    try:
        # ================== 3. Anomalib 데이터 구성 ==================== #
        logger.info(f" 데이터셋 로딩 중: {dataset_root}")
        
        # [Dynamic Detection] 'good'을 제외한 모든 폴더를 불량(abnormal) 카테고리로 수집합니다.
        val_root = base_path / "validation"
        abnormal_dirs = []
        if val_root.exists():
            abnormal_dirs = [f"validation/{d.name}" for d in val_root.iterdir() if d.is_dir() and d.name != "good"]
        
        logger.info(f" 검증용 불량 카테고리 자동 감지: {abnormal_dirs}")

        datamodule = Folder(
            name="battery",
            root=str(dataset_root),
            normal_dir="train/good",
            normal_test_dir="validation/good",
            abnormal_dir=abnormal_dirs if abnormal_dirs else None,
            train_batch_size=32,
            eval_batch_size=8,
            num_workers=4,
            augmentations=Resize((256, 256)),
            seed=args.seed,
            task="classification"
        )

        # ================== 3. 모델 및 콜백 설정 ==================== #
        logger.info(f"🏗️ 모델 생성 중: FastFlow (Backbone: {args.backbone})")
        # evaluator=False prevents internal metric initialization that might expect gt_mask
        model = Fastflow(
            backbone=args.backbone, 
            flow_steps=8, 
            evaluator=False,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Early Stopping 설정: 성능 향상이 없으면 조기 종료하여 자원 절약
        early_stop = EarlyStopping(
            monitor="image_F1Score", 
            patience=5, 
            mode="max",
            verbose=True
        )

        mlflow_logger = AnomalibMLFlowLogger(experiment_name="Battery_Anomaly", save_dir=str(OUTPUT_DIR))
        
        engine = Engine(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            default_root_dir=str(OUTPUT_DIR),
            logger=mlflow_logger,
            callbacks=[early_stop],
            image_metrics=["AUROC", "F1Score"],
            pixel_metrics=None,
            task="classification"
        )

        # ================== 4. 학습 및 저장 ==================== #
        logger.info(f" Training started (Seed: {args.seed})...")
        engine.fit(model=model, datamodule=datamodule)
        
        # [Threshold Finalization] 테스트를 수행하여 최적의 임계값(Threshold)을 확정하고 로그에 기록합니다.
        logger.info(" Finalizing threshold and calculating metrics...")
        engine.test(model=model, datamodule=datamodule)
        
        # 최적 임계값 로깅
        if hasattr(model, "image_threshold"):
            logger.info(f" Calculated Image Threshold: {model.image_threshold.value.item():.4f}")
        if hasattr(model, "pixel_threshold"):
            logger.info(f" Calculated Pixel Threshold: {model.pixel_threshold.value.item():.4f}")

        ckpt_path = OUTPUT_DIR / "model.ckpt"
        engine.trainer.save_checkpoint(ckpt_path)
        
        model_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_path)
        logger.success(f" Model saved (including thresholds in .ckpt)")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mlflow.log_param("gpu_name", gpu_name)

        # ================== 5. 결과 기록 ==================== #
        info = {
            "backbone": args.backbone,
            "seed": args.seed,
            "epochs": args.epochs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(OUTPUT_DIR / "info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        mlflow.log_params(info)
        mlflow.log_artifact(str(OUTPUT_DIR))
        logger.success(" 모든 프로세스가 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f" 학습 중 오류 발생: {e}")
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()

# success plz
