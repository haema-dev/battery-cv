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
    [Strict Fix] raw state_dict를 Lightning 정식 체크포인트 포맷으로 변환합니다.
    사용자님의 제안에 따라 필수 메타데이터(transform, version, epoch 등)를 포함하여 
    전용 래퍼 체크포인트를 생성합니다. 
    이는 프레임워크 초기화 시 가중치가 리셋되는 현상을 물리적으로 방지합니다.
    """
    logger.info(f"[*] 가중치 규격 변환 및 래핑 시작: {model_path}")
    raw_ckpt = torch.load(model_path, map_location="cpu")
    
    # 딕셔너리 구조에서 state_dict 추출 (사용자님의 제안 반영)
    state_dict = raw_ckpt.get("state_dict", raw_ckpt)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    # [Smart Matcher Logic] 모델 키 구조 분석 및 보정
    # LightningModule 내부의 실제 파라미터 이름과 체크포인트의 이름 불일치 해결
    model_keys = set(model.state_dict().keys())
    has_model_prefix = any(k.startswith("model.") for k in model_keys)
    ckpt_has_prefix = any(k.startswith("model.") for k in state_dict.keys())
    
    final_state_dict = {}
    if has_model_prefix and not ckpt_has_prefix:
        logger.info("[*] 규격 조정: 가중치 키에 'model.' 접두어를 추가합니다.")
        for k, v in state_dict.items():
            final_state_dict[f"model.{k}"] = v
    elif not has_model_prefix and ckpt_has_prefix:
        logger.info("[*] 규격 조정: 가중치 키에서 'model.' 접두어를 제거합니다.")
        for k, v in state_dict.items():
            final_state_dict[k.replace("model.", "")] = v
    else:
        final_state_dict = state_dict

    # [CRITICAL] Lightning 및 Anomalib 1.1.3 필수 메타데이터 포함
    # 주의: transform 객체의 직렬화(Pickling)가 실패할 경우 예외 처리가 필요할 수 있습니다.
    lightning_ckpt = {
        "state_dict": final_state_dict,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": getattr(lightning, "__version__", "2.1.0"),
        "transform": transform,  # Anomalib 1.1.3 필수 키
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

        # ================== 3. 모델 생성 ==================== #
        logger.info(f"🏗️ 모델 생성 중: FastFlow ({args.backbone})")
        model = Fastflow(backbone=args.backbone, flow_steps=8)
        
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