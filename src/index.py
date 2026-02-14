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

# v3.8: "The Hybrid Grandmaster Surgeon"
# 1. 사용자님이 지정하신 경로(datasets/256x256 fit/train/good) 정밀 조준
# 2. Anomalib 1.1.3의 모든 타입 에러(TypeError)와 메트릭 에러(gt_mask) 박멸
# 3. MLflow 로깅 및 캡처된 구조 탐색 통합

# [수술 1] Master Surgeon Patch: Trainer 인자 유출 방어
original_trainer_init = trainer_module.Trainer.__init__
TRAINER_ALLOWED_PARAMS = set(inspect.signature(original_trainer_init).parameters.keys())

def patched_trainer_init(self, *args, **kwargs):
    # Trainer가 인식하지 못하는 모든 인자(task, pixel_metrics 등)를 걸러냅니다.
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in TRAINER_ALLOWED_PARAMS}
    return original_trainer_init(self, *args, **filtered_kwargs)

trainer_module.Trainer.__init__ = patched_trainer_init

from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

def find_dataset_root(base_path):
    """사용자님이 강조하신 'datasets/256x256 fit/train/good' 경로를 정밀 탐색합니다."""
    base = Path(base_path)
    logger.info(f"🔎 데이터 탐색 중: {base}")
    
    # 1순위: 지정하신 '256x256 fit' 경로를 rglob으로 찾기 (공백 포함 대응)
    for p in base.rglob("*/train/good"):
        path_str = str(p)
        if "256x256 fit" in path_str:
            root_candidate = p.parent.parent # '256x256 fit' 폴더
            logger.success(f"✅ 타겟 데이터셋 발견: {root_candidate}")
            return root_candidate
            
    # 2순위: 일반적인 train/good이라도 찾기
    for p in base.rglob("train/good"):
        logger.warning(f"⚠️ 정확한 구조는 아니지만 'train/good' 발견: {p.parent.parent}")
        return p.parent.parent
            
    logger.error("❌ 지정된 학습 데이터를 찾을 수 없습니다.")
    return base

def run_pipeline(data_path, output_dir, epochs):
    logger.info("==================================================")
    logger.info("🚀 STAGE 1: HYBRID GRANDMASTER V3.8 (FINAL)")
    logger.info("==================================================")
    
    mlflow.start_run()
    
    try:
        # 1. 데이터 루트 탐색
        optimized_root = find_dataset_root(data_path)
        logger.info(f"📂 최종 학습 루트: {optimized_root}")

        # 2. 데이터 모듈 설정 (정상 데이터만 학습용으로 사용)
        datamodule = Folder(
            name="battery",
            root=str(optimized_root),
            normal_dir="train/good",
            test_split_mode="from_dir"
        )

        # 3. 모델 설정
        model = Fastflow(backbone="resnet18", flow_steps=8)

        # 4. 엔진 설정
        # Surgeon 패치가 있어 TypeError 걱정 없이 필요한 인자 전달 가능
        engine = Engine(
            max_epochs=epochs,
            default_root_dir=output_dir,
            devices=1,
            accelerator="auto",
            task="classification",
            pixel_metrics=None
        )

        # [수술 2] 메트릭 강제 고정 (Hot-Swap) - gt_mask 에러 최종 방화벽
        if hasattr(engine, "task"): engine.task = "classification"
        if hasattr(engine, "pixel_metrics"): engine.pixel_metrics = None
        if hasattr(model, "task"): model.task = "classification"

        # 5. 실행
        logger.info(f"⏳ 학습 시작 (목표 에포크: {epochs})...")
        engine.fit(model=model, datamodule=datamodule)

        # 6. 저장
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_save_path = output_path / "model.pt"
        torch.save(model.state_dict(), model_save_path)
        
        # MLflow 로깅
        mlflow.log_params({"epochs": epochs, "model": "FastFlow", "data": "256x256 fit"})
        logger.success(f"✅ Stage 1 성공! 모델 저장 완료: {model_save_path}")

    except Exception as e:
        logger.error(f"❌ 치명적 에러 발생: {e}")
        import traceback
        traceback.print_exc()
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