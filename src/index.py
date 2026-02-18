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
from torchvision.transforms.v2 import Compose, Normalize, Resize
from lightning.pytorch.callbacks import EarlyStopping
import lightning

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def convert_to_lightning_checkpoint(model_path, model, output_dir, transform=None):
    """
    [Rigorous Final Fix] raw state_dict를 Anomalib/Lightning 정식 체크포인트로 변환합니다.
    - BestFit Matcher: 모델의 실제 요구사항에 맞춰 'model.' 접두어를 지능적으로 가공합니다.
    - Strict Filtering: 모델에 없는 키(예: 구버전 post_processor 등)를 제거하여 엔진의 strict 로드를 통과시킵니다.
    """
    logger.info(f"[*] 가중치 규격 변환 및 래핑 시작: {model_path}")
    raw_ckpt = torch.load(model_path, map_location="cpu")
    
    # 1. 여러 포맷(Anomalib/Lightning/Raw)에서 state_dict 추출
    if isinstance(raw_ckpt, dict):
        state_dict = raw_ckpt.get("state_dict", raw_ckpt.get("model", raw_ckpt))
    else:
        state_dict = raw_ckpt

    # 2. BestFit Matcher: 모델의 실제 요구사항과 체크포인트 키 대조
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    
    # [Robustness] LightningModule이 아직 setup되지 않아 state_dict가 비어있는 경우 내부 모델 확인
    if not model_keys and hasattr(model, "model"):
        logger.info("[*] LightningModule 키가 비어있음. 내부 모델 구조를 분석합니다.")
        inner_keys = set(model.model.state_dict().keys())
        # Anomalib LightningModule은 보통 내부 모델 키에 'model.'을 붙여 관리합니다.
        model_keys = {f"model.{k}" for k in inner_keys}

    strategies = [
        ("As-is", lambda d: d),
        ("Add 'model.'", lambda d: {f"model.{k}": v for k, v in d.items()}),
        ("Remove 'model.'", lambda d: {k[6:] if k.startswith("model.") else k: v for k, v in d.items()})
    ]
    
    best_matching_dict = state_dict
    max_matches = 0
    best_strategy = "None"
    
    num_model_keys = len(model_keys)
    logger.info(f"[*] 매칭 전략 탐색 시작 (모델 키 총 {num_model_keys}개)")
    
    if num_model_keys > 0:
        for name, func in strategies:
            try:
                test_dict = func(state_dict)
                matches = len(model_keys.intersection(test_dict.keys()))
                logger.info(f"    - 전략 '{name}': {matches}개 매칭")
                if matches > max_matches:
                    max_matches = matches
                    best_strategy = name
                    best_matching_dict = test_dict
            except Exception:
                continue
        
        match_rate = (max_matches / num_model_keys) * 100
        logger.info(f"[*] 최종 채택 전략: {best_strategy} (매칭률: {match_rate:.1f}%)")
    else:
        logger.warning("[!] 모델 키를 감지하지 못했습니다. 기본 전략(As-is)을 사용합니다.")
        best_strategy = "Default (As-is)"

    # 3. Strict Filtering: 모델에 존재하지 않는 불필요한 키 제거 (RuntimeError 방지)
    final_state_dict = {k: v for k, v in best_matching_dict.items() if k in model_keys}
    
    # 4. 정식 규격 래핑
    lightning_ckpt = {
        "state_dict": final_state_dict,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": getattr(lightning, "__version__", "2.1.0"),
        "transform": transform,
        "callbacks": {},
        "optimizer_states": [],
        "lr_schedulers": []
    }
    
    wrapped_path = Path(output_dir) / "wrapped_checkpoint.ckpt"
    torch.save(lightning_ckpt, wrapped_path)
    return str(wrapped_path)

def main():
    # ================== 1. Input/Output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model checkpoint")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backbone", type=str, default="resnet18", help="Feature extractor backbone")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--mode", type=str, default="evaluation", choices=["training", "evaluation"])

    args = parser.parse_args()
    set_seed(args.seed)
    
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("==================================================")
    logger.info(" STAGE 2: PM Selection - FastFlow Training/Eval")
    logger.info(f" 데이터 경로: {args.data_path}")
    if args.model_path:
        logger.info(f" 모델 로드 경로: {args.model_path}")
    logger.info(f" 설정: Backbone={args.backbone}, Mode={args.mode}")
    logger.info("==================================================")

    try:
        # ================== 2. Anomalib 데이터 구성 ==================== #
        dataset_root = Path(args.data_path)
        val_path = dataset_root / "validation"
        
        # 불량 카테고리 자동 감지
        abnormal_dirs = []
        if val_path.exists():
            abnormal_dirs = [f"validation/{d.name}" for d in val_path.iterdir() if d.is_dir() and d.name != "good"]
        logger.info(f"[*] 불량 카테고리 감지: {abnormal_dirs}")

        transform = Compose([
            Resize((256, 256)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        datamodule = Folder(
            name="battery",
            root=str(dataset_root),
            normal_dir="train/good",
            normal_test_dir="validation/good",
            abnormal_dir=abnormal_dirs if abnormal_dirs else None,
            train_batch_size=32,
            eval_batch_size=8,
            num_workers=4,
            train_transform=transform,
            eval_transform=transform,
            task="classification",
            seed=args.seed
        )

        # ================== 3. 모델 생성 및 수동 초기화 ==================== #
        logger.info(f"🏗️ 모델 생성 중: FastFlow ({args.backbone})")
        model = Fastflow(backbone=args.backbone, flow_steps=8)
        
        # [Strict Fix] 엔진 구동 전 모델 레이어를 명시적으로 생성 (Key 감지용)
        # Anomalib 1.1.3에서는 setup()을 호출해야 내부 레이어(feature_extractor 등)가 실체화됩니다.
        datamodule.setup(stage="test")
        model.setup()

        # ================== 4. 가중치 래핑 (Rigorous Fix) ==================== #
        # 엔진이 "직접" 로드하게 함으로써 프레임워크 초기화 시 발생하는 리셋 문제를 해결합니다.
        tmp_ckpt_path = None
        if args.model_path and os.path.exists(args.model_path):
            tmp_ckpt_path = convert_to_lightning_checkpoint(args.model_path, model, OUTPUT_DIR, transform=transform)
            logger.info(f"[*] 임시 체크포인트 준비 완료: {tmp_ckpt_path}")

        # ================== 5. 엔진 설정 및 실행 ==================== #
        early_stop = EarlyStopping(monitor="image_AUROC", patience=5, mode="max", verbose=True)
        mlflow_logger = AnomalibMLFlowLogger(experiment_name="Battery_S1_AnomalyDetection", save_dir=str(OUTPUT_DIR))

        engine = Engine(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            default_root_dir=str(OUTPUT_DIR),
            logger=mlflow_logger,
            callbacks=[early_stop],
            gradient_clip_val=1.0
        )

        if args.mode == "training":
            logger.info("[*] 학습 모드 시작")
            engine.fit(model=model, datamodule=datamodule)
        else:
            logger.info("[*] 평가 모드 시작 (가중치 주입)")
            engine.test(model=model, datamodule=datamodule, ckpt_path=tmp_ckpt_path)
        
        # 임계값 결과 확인
        if hasattr(model, "image_threshold"):
            thresh = model.image_threshold.value.item() if hasattr(model.image_threshold, "value") else model.image_threshold
            logger.success(f"[*] Calculated Image Threshold: {thresh:.4f}")

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