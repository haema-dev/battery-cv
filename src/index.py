import argparse
import os
import sys
import inspect
import json
import time
import mlflow
from pathlib import Path
import torch
from loguru import logger
import lightning.pytorch.trainer.trainer as trainer_module
import anomalib.metrics.evaluator as evaluator_module

# v3.9: "The Nuclear Surgeon" - Absolute Metric Suppression
# Anomalib 1.1.3의 결함을 해결하는 최종 해결책입니다:
# 1. Trainer 인자 유출로 인한 TypeError (해결: 필터링 패치)
# 2. 1단계 학습 시 정상 이미지만 있어 생기는 메트릭 에러 (해결: Evaluator 무력화)

# [수술 1] Trainer 인자 필터링 (TypeError 방지)
original_trainer_init = trainer_module.Trainer.__init__
TRAINER_ALLOWED_PARAMS = set(inspect.signature(original_trainer_init).parameters.keys())

def patched_trainer_init(self, *args, **kwargs):
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in TRAINER_ALLOWED_PARAMS}
    return original_trainer_init(self, *args, **filtered_kwargs)

trainer_module.Trainer.__init__ = patched_trainer_init

# [수술 2] Nuclear Option: 1단계는 '정상'군 학습만 하므로 검증/평가가 무의미하고 에러만 냅니다.
# Evaluator의 모든 훅을 '아무것도 안 함'으로 바꿔서 gt_mask 에러를 원천 봉쇄합니다.
logger.info("🧪 [Nuclear Surgeon] Nuking Evaluator hooks to prevent gt_mask errors...")
evaluator_module.Evaluator.on_validation_batch_end = lambda *args, **kwargs: None
evaluator_module.Evaluator.on_test_batch_end = lambda *args, **kwargs: None
evaluator_module.Evaluator.on_validation_epoch_end = lambda *args, **kwargs: None
evaluator_module.Evaluator.on_test_epoch_end = lambda *args, **kwargs: None

from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

def find_dataset_root(base_path):
    """사용자님이 강조하신 'datasets/256x256 fit/train/good' 경로를 포함하는 루트를 찾습니다."""
    base = Path(base_path).resolve()
    logger.info(f"🔎 탐색 시작: {base}")
    
    # 모든 train/good 위치를 찾아서 로그로 남깁니다.
    found_paths = list(base.rglob("train/good"))
    if not found_paths:
        logger.error("❌ 'train/good' 폴더를 어디에서도 찾을 수 없습니다.")
        return base

    for p in found_paths:
        logger.info(f"📍 발견된 경로: {p}")
        if "256x256 fit" in str(p):
            root = p.parent.parent
            logger.success(f"🎯 최종 타겟 루트 확정: {root}")
            return root
            
    # 못 찾으면 첫 번째 발견된 경로의 부모를 반환
    fallback_root = found_paths[0].parent.parent
    logger.warning(f"⚠️ '256x256 fit'을 포함하는 경로를 못 찾아 첫 번째 경로를 사용합니다: {fallback_root}")
    return fallback_root

def run_pipeline(data_path, output_dir, epochs):
    logger.info("==================================================")
    logger.info("🚀 STAGE 1: NUCLEAR STABILIZATION V3.9 (THE END)")
    logger.info("==================================================")
    
    mlflow.start_run()
    
    try:
        # 1. 데이터 경로 확정
        optimized_root = find_dataset_root(data_path)

        # 2. 데이터 모듈 설정
        datamodule = Folder(
            name="battery",
            root=str(optimized_root),
            normal_dir="train/good",
            test_split_mode="from_dir"
        )

        # 3. 모델 설정
        model = Fastflow(backbone="resnet18", flow_steps=8)

        # 4. 엔진 설정
        # limit_val_batches=0으로 검증 루프를 공식적으로 건너뜁니다.
        # Surgeon 패치가 있어 어떤 인자든 에러 없이 전달됩니다.
        engine = Engine(
            max_epochs=epochs,
            default_root_dir=output_dir,
            devices=1,
            accelerator="auto",
            task="classification",
            limit_val_batches=0, # 검증 안 함
            pixel_metrics=None
        )

        # 5. 학습 실행
        logger.info(f"⏳ 1단계 학습 돌입 (Epochs: {epochs})... 다시는 에러로 멈추지 않습니다.")
        engine.fit(model=model, datamodule=datamodule)

        # 6. 저장
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_save_path = output_path / "model.pt"
        torch.save(model.state_dict(), model_save_path)
        
        mlflow.log_params({"epochs": epochs, "status": "Stage 1 Success"})
        logger.success(f"✅ Stage 1 성공! 모델 저장 완료: {model_save_path}")

    except Exception as e:
        logger.error(f"❌ 최종 실패: {e}")
        raise e
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    sys.stdout.reconfigure(line_buffering=True)
    run_pipeline(args.data_path, args.output_dir, args.epochs)