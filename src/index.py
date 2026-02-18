# -*- coding: utf-8 -*-
# Version trigger for Azure ML - v5
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


def convert_to_lightning_checkpoint(model_path, model, output_dir):
    """
    [Rigorous Fix] raw state_dict를 Lightning 정식 체크포인트 포맷으로 래핑합니다.
    사용자님의 제안에 따라 메타데이터를 추가하여 엔진의 공식 로드 경로를 활성화합니다.
    """
    logger.info(f"[*] 가중치 규격 변환 및 래핑 시작: {model_path}")
    raw_ckpt = torch.load(model_path, map_location="cpu")
    state_dict = raw_ckpt.get("state_dict", raw_ckpt)
    
    # [Smart Matcher Logic] 모델 키 구조 분석 및 보정
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

    # 가짜 체크포인트 생성 (Lightning 필수 메타데이터 포함)
    lightning_ckpt = {
        "state_dict": final_state_dict,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": getattr(lightning, "__version__", "2.1.0"),
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
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model checkpoint (Optional for Eval Mode)")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backbone", type=str, default="resnet18", help="Feature extractor backbone")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--mode", type=str, default="evaluation", choices=["training", "evaluation"], help="Execution mode")

    args = parser.parse_args()
    set_seed(args.seed)
    base_path = Path(args.data_path)
    
    logger.info("==================================================")
    logger.info(" STAGE 2: PM Selection - FastFlow Training/Eval")
    logger.info(f" 마운트 루트: {base_path}")
    if args.model_path:
        logger.info(f" 모델 로드 경로: {args.model_path}")
    logger.info(f" 설정: Backbone={args.backbone}, Epochs={args.epochs}")
    logger.info("==================================================")

    # 필수 폴더 존재 여부 체크
    val_path = base_path / "validation"
    dataset_root = base_path

    # ================== 2. MLflow & Output 설정 ==================== #
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f" 사용 장치: {device}")

    try:
        # ================== 3. Anomalib 데이터 구성 ==================== #
        logger.info(f" 데이터셋 로딩 중: {dataset_root}")
        
        abnormal_dirs = []
        if val_path.exists():
            abnormal_dirs = [f"validation/{d.name}" for d in val_path.iterdir() if d.is_dir() and d.name != "good"]
        
        logger.info(f" 검증용 불량 카테고리 자동 감지: {abnormal_dirs}")

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

        # ================== 4. 모델 생성 및 초기화 ==================== #
        logger.info(f"🏗️ 모델 생성 중: FastFlow (Backbone: {args.backbone})")
        
        model = Fastflow(
            backbone=args.backbone, 
            flow_steps=8
        )
        
        # [Rigorous Strategy] 가짜 체크포인트 생성 및 엔진 전달
        tmp_ckpt_path = None
        if args.model_path and os.path.exists(args.model_path):
            tmp_ckpt_path = convert_to_lightning_checkpoint(args.model_path, model, OUTPUT_DIR)
            logger.info(f"[*] 임시 체크포인트 준비 완료: {tmp_ckpt_path}")

        early_stop = EarlyStopping(
            monitor="image_AUROC", 
            patience=5, 
            mode="max",
            verbose=True
        )

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

        # ================== 5. 실행 (학습 또는 평가) ==================== #
        if args.mode == "training":
            logger.info(" [Mode: Training] 학습을 시작합니다.")
            engine.fit(model=model, datamodule=datamodule)
        if args.mode == "evaluation":
            logger.info(" [Mode: Evaluation] 학습을 생략하고 평가를 수행합니다.")
            logger.info(" Calculating final metrics and thresholds...")
            
            engine.test(
                model=model, 
                datamodule=datamodule, 
                ckpt_path=tmp_ckpt_path
            )
        
        if hasattr(model, "image_threshold"):
            try:
                thresh = model.image_threshold.value.item() if hasattr(model.image_threshold, "value") else model.image_threshold
                logger.info(f" Calculated Image Threshold: {thresh:.4f}")
            except Exception as e:
                logger.warning(f" Threshold 값을 읽어오는 데 실패했습니다: {e}")

        model_pt_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_pt_path)
        logger.success(f" [FINISH] 모든 결과가 {OUTPUT_DIR}에 저장되었습니다.")

    except Exception as e:
        logger.error(f" [FATAL] 실행 중 오류 발생: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    finally:
        pass

if __name__ == "__main__":
    main()