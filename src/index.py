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
    parser.add_argument("--epochs", type=int, default=10) # 진단용이므로 기본 epoch 단축
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="evaluation", choices=["training", "evaluation"])

    args = parser.parse_args()
    set_seed(args.seed)
    
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 MODE: {args.mode.upper()} | BACKBONE: {args.backbone}")

    try:
        # ================== 2. Anomalib 데이터 및 모델 구성 ==================== #
        dataset_root = Path(args.data_path)
        val_path = dataset_root / "validation"
        abnormal_dirs = [f"validation/{d.name}" for d in val_path.iterdir() if d.is_dir() and d.name != "good"] if val_path.exists() else []

        transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((256, 256)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        datamodule = Folder(
            name="battery", root=str(dataset_root),
            normal_dir="train/good", normal_test_dir="validation/good",
            abnormal_dir=abnormal_dirs if abnormal_dirs else None,
            train_batch_size=32, eval_batch_size=8,
            train_transform=transform, eval_transform=transform,
            task="classification", seed=args.seed
        )

        model = Fastflow(backbone=args.backbone, flow_steps=8)
        
        # [Manual Injection] 모델 수동 초기화 및 가중치 강제 주입
        datamodule.setup(stage="test")
        model.setup()

        if args.model_path and os.path.exists(args.model_path):
            load_matched_weights(args.model_path, model)

        # ================== 3. 엔진 설정 및 실행 ==================== #
        mlflow_logger = AnomalibMLFlowLogger(experiment_name="Battery_S2_Diagnostics", save_dir=str(OUTPUT_DIR))
        engine = Engine(
            max_epochs=args.epochs, devices=1, accelerator="auto",
            logger=mlflow_logger, default_root_dir=str(OUTPUT_DIR)
        )

        if args.mode == "training":
            logger.info("🔥 [DIAGNOSIS] Training 모드 시작 (가중치 기반 Fine-tuning)")
            engine.fit(model=model, datamodule=datamodule)
        else:
            logger.info("🔍 [DIAGNOSIS] Evaluation 모드 시작 (수동 주입된 가중치 기반)")
            # ckpt_path=None으로 설정하여 프레임워크의 자동 로드를 방지하고 주입된 가중치를 그대로 사용
            engine.test(model=model, datamodule=datamodule, ckpt_path=None)
        
        # 임계값 및 결과 확인
        if hasattr(model, "image_threshold"):
            thresh = model.image_threshold.value.item() if hasattr(model.image_threshold, "value") else model.image_threshold
            logger.success(f"[*] Calculated Image Threshold: {thresh:.4f}")

        # 최종 가중치 저장
        torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
        logger.success(f"[FINISH] Output saved at: {OUTPUT_DIR}")

        # 최종 가중치 저장
        model_pt_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_pt_path)
        logger.success(f"[FINISH] 작업 완료. 결과 저장 위치: {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"[FATAL] 오류 발생: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()