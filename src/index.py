# -*- coding: utf-8 -*-
# Version trigger for Azure ML - v4
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TunableFastflow(Fastflow):
    def __init__(self, *args, lr: float = 0.001, weight_decay: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
    # [수정] configure_evaluator 제거 (Engine이 자동 관리)

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
    # [수정] 수동 mlflow.start_run() 제거 (AnomalibMLFlowLogger가 자동 관리)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f" 사용 장치: {device}")

    try:
        # ================== 3. Anomalib 데이터 구성 ==================== #
        logger.info(f" 데이터셋 로딩 중: {dataset_root}")
        
        # [Dynamic Detection] 'good'을 제외한 모든 폴더를 불량 카테고리로 수집
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
        
        # [수정] evaluator 인자 제거 (Engine이 자동 관리)
        model = TunableFastflow(
            backbone=args.backbone, 
            flow_steps=8, 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # [Stage 2 Integration] 로드할 모델 파일이 있다면 가중치 주입
        if args.model_path and os.path.exists(args.model_path):
            logger.info(f"[*] 사전 학습된 가중치 로드: {args.model_path}")
            ckpt = torch.load(args.model_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            
            # [수수술적 로깅] 가중치 로드 현황 정밀 진단
            model_keys = set(model.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            intersect_keys = model_keys.intersection(loaded_keys)
            
            logger.info(f"[*] 가중치 분석: 모델 키({len(model_keys)}개), 체크포인트 키({len(loaded_keys)}개)")
            logger.info(f"[*] 매칭된 키 개수: {len(intersect_keys)}개")
            
            model.load_state_dict(state_dict, strict=False)
            logger.success(f"[OK] 가중치 주입 완료 (매칭율: {len(intersect_keys)/len(model_keys)*100:.1f}%)")

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
        if not args.model_path:
            logger.info(" [Mode: Training] 학습을 시작합니다.")
            engine.fit(model=model, datamodule=datamodule)
        else:
            logger.info(" [Mode: Evaluation] 학습을 생략하고 평가를 수행합니다.")

        # 최종 성능 측정 및 임계값 확정
        logger.info(" Calculating final metrics and thresholds...")
        engine.test(model=model, datamodule=datamodule)
        
        # [수정] 임계값 접근 안전성 확보
        if hasattr(model, "image_threshold"):
            try:
                # .value가 프로젝트 버전에 따라 달라질 수 있으므로 안전하게 접근
                thresh = model.image_threshold.value.item() if hasattr(model.image_threshold, "value") else model.image_threshold
                logger.info(f" Calculated Image Threshold: {thresh:.4f}")
            except Exception as e:
                logger.warning(f" Threshold 값을 읽어오는 데 실패했습니다: {e}")

        # 결과 저장
        model_pt_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_pt_path)
        logger.success(f" [FINISH] 모든 결과가 {OUTPUT_DIR}에 저장되었습니다.")

    except Exception as e:
        logger.error(f" [FATAL] 실행 중 오류 발생: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    finally:
        pass # [수정] 수동 mlflow.end_run() 제거

if __name__ == "__main__":
    main()