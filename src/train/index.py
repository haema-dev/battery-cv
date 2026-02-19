# -*- coding: utf-8 -*-
"""
배터리 이상탐지 PatchCore 학습 (Azure ML Pipeline용)
- PatchCore + WideResNet50 backbone
- 단일 GPU / 멀티 GPU(DDP) 자동 지원
- image-level 메트릭 (AUROC, F1) + threshold 확정
- engine.export()로 TorchInferencer 호환 모델 생성
"""
import os
import torch
import argparse
import mlflow
import json
import time
import random
import numpy as np
from loguru import logger
from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.loggers import AnomalibMLFlowLogger
from pathlib import Path
from torchvision.transforms.v2 import Compose, Normalize, Resize
from anomalib.metrics import AUROC, F1Score, Evaluator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=1, help="PatchCore는 1 epoch이면 충분")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coreset_sampling_ratio", type=float, default=0.1)
    parser.add_argument("--num_neighbors", type=int, default=9)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=0, help="GPU count (0=auto)")

    args = parser.parse_args()
    set_seed(args.seed)
    base_path = Path(args.data_path)

    logger.info("=" * 60)
    logger.info("S1_PatchCore_Training (Azure ML Pipeline)")
    logger.info(f"  Data path: {base_path}")
    logger.info(f"  Backbone: {args.backbone}, Coreset: {args.coreset_sampling_ratio}, Neighbors: {args.num_neighbors}")
    logger.info("=" * 60)

    # --- 디버깅: 마운트 구조 출력 ---
    try:
        for root, dirs, files in os.walk(base_path):
            level = len(Path(root).relative_to(base_path).parts)
            if level <= 2:
                indent = "  " * level
                logger.info(f"{indent}{Path(root).name}/ ({len(files)} files)")
            if level > 2:
                continue
    except Exception as e:
        logger.warning(f"  구조 출력 중 오류 (무시): {e}")

    # --- 필수 폴더 체크 ---
    train_path = base_path / "train/good"
    val_path = base_path / "validation"

    if not train_path.exists():
        raise FileNotFoundError(f"필수 학습 경로가 없습니다: {train_path}")
    logger.info(f"  train/good 확인: {train_path}")

    if val_path.exists():
        logger.info(f"  validation 확인: {val_path}")
    else:
        logger.warning(f"  validation 없음 - test 없이 학습만 진행")

    # --- DDP: 멀티 GPU 자동 감지 ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = local_rank == 0
    num_gpus = torch.cuda.device_count()

    logger.info(f"  DDP: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, GPUs={num_gpus}")

    if world_size > 1 and num_gpus > 1:
        use_devices = num_gpus
        strategy = "ddp"
        accelerator = "gpu"
    elif num_gpus >= 1:
        use_devices = 1
        strategy = "auto"
        accelerator = "gpu"
    else:
        use_devices = 1
        strategy = "auto"
        accelerator = "cpu"

    # DDP일 때 num_workers 조정 (OOM 방지)
    effective_workers = min(args.num_workers, max(2, args.num_workers // max(use_devices, 1)))
    logger.info(f"  Using: devices={use_devices}, strategy={strategy}, accelerator={accelerator}")
    logger.info(f"  Workers: {args.num_workers} -> {effective_workers} (adjusted)")

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- MLflow (main process만) ---
    mlflow_active = False
    if is_main_process:
        try:
            mlflow.start_run()
            mlflow.log_params({
                "model": "patchcore",
                "backbone": args.backbone,
                "coreset_sampling_ratio": args.coreset_sampling_ratio,
                "num_neighbors": args.num_neighbors,
                "num_gpus": world_size,
                "strategy": strategy,
            })
            mlflow_active = True
            logger.info("  MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"  MLflow init failed (training continues): {e}")

    try:
        # --- 데이터 구성 ---
        abnormal_dirs = []
        if val_path.exists():
            abnormal_dirs = [f"validation/{d.name}" for d in val_path.iterdir() if d.is_dir() and d.name != "good"]
        logger.info(f"  검증용 불량 카테고리: {abnormal_dirs}")

        transform = Compose([
            Resize((512, 512)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        datamodule = Folder(
            name="battery",
            root=str(base_path),
            normal_dir="train/good",
            normal_test_dir="validation/good" if (base_path / "validation/good").exists() else None,
            abnormal_dir=abnormal_dirs if abnormal_dirs else None,
            train_batch_size=8,
            eval_batch_size=4,
            num_workers=effective_workers,
            augmentations=transform,
            seed=args.seed
        )

        # --- 모델 + Evaluator ---
        image_auroc_val = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_auroc_test = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1_test = F1Score(fields=["pred_label", "gt_label"], prefix="image_")

        evaluator = Evaluator(
            val_metrics=[image_auroc_val],
            test_metrics=[image_auroc_test, image_f1_test],
        )

        model = Patchcore(
            backbone=args.backbone,
            pre_trained=True,
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=args.coreset_sampling_ratio,
            num_neighbors=args.num_neighbors,
            evaluator=evaluator,
        )

        # --- MLflow Logger ---
        mlflow_logger = None
        if is_main_process:
            try:
                mlflow_logger = AnomalibMLFlowLogger(
                    experiment_name="Battery_Anomaly_PatchCore",
                    save_dir=str(OUTPUT_DIR)
                )
            except Exception:
                pass

        # --- Engine ---
        engine = Engine(
            max_epochs=args.epochs,
            accelerator=accelerator,
            devices=use_devices,
            strategy=strategy,
            default_root_dir=str(OUTPUT_DIR),
            logger=mlflow_logger,
        )

        # --- DDP Race Condition 방지: versioned_dir 생성 시 FileExistsError 회피 ---
        # Anomalib Engine이 `from anomalib.utils.path import create_versioned_dir`로
        # 직접 import하므로, 모듈 패치뿐 아니라 engine 모듈의 참조도 함께 패치해야 함
        import anomalib.utils.path as _aml_path
        import anomalib.engine.engine as _aml_engine

        _orig_create_versioned_dir = _aml_path.create_versioned_dir

        def _safe_create_versioned_dir(root_dir):
            """DDP-safe wrapper: mkdir에서 FileExistsError 발생 시 무시"""
            try:
                return _orig_create_versioned_dir(root_dir)
            except FileExistsError:
                root_dir = Path(root_dir)
                versions = sorted(
                    [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
                    key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0
                )
                if versions:
                    logger.info(f"  DDP: 다른 프로세스가 이미 생성한 디렉토리 사용: {versions[-1]}")
                    return versions[-1]
                fallback = root_dir / "v1"
                fallback.mkdir(parents=True, exist_ok=True)
                return fallback

        # 두 군데 모두 패치 (모듈 레벨 + engine의 직접 참조)
        _aml_path.create_versioned_dir = _safe_create_versioned_dir
        if hasattr(_aml_engine, 'create_versioned_dir'):
            _aml_engine.create_versioned_dir = _safe_create_versioned_dir

        # --- 학습 ---
        t0 = time.time()
        logger.info("Training started...")
        engine.fit(model=model, datamodule=datamodule)

        # 원복
        _aml_path.create_versioned_dir = _orig_create_versioned_dir
        if hasattr(_aml_engine, 'create_versioned_dir'):
            _aml_engine.create_versioned_dir = _orig_create_versioned_dir
        elapsed = time.time() - t0

        # --- Threshold 확정 + 메트릭 ---
        logger.info("Finalizing threshold and calculating metrics...")
        test_results = engine.test(model=model, datamodule=datamodule)
        logger.info(f"Test results: {test_results}")

        if hasattr(model, "image_threshold"):
            threshold = model.image_threshold.value.item()
            logger.info(f"  Image Threshold: {threshold:.4f}")

        # --- 모델 저장 ---
        ckpt_path = OUTPUT_DIR / "model.ckpt"
        engine.trainer.save_checkpoint(ckpt_path)
        logger.info(f"  Checkpoint saved: {ckpt_path}")

        # TorchInferencer 호환 모델 export
        logger.info("Exporting model for inference...")
        try:
            exported_model_path = engine.export(
                model=model,
                export_type="torch",
                export_root=str(OUTPUT_DIR)
            )
            logger.info(f"  Exported: {exported_model_path}")
        except Exception as e:
            logger.warning(f"  Export failed (checkpoint still available): {e}")

        # 백업용 state_dict
        torch.save(model.state_dict(), OUTPUT_DIR / "model_weights.pt")

        logger.info(f"Training complete in {elapsed:.0f}s")

        # --- 결과 기록 ---
        info = {
            "model": "patchcore",
            "backbone": args.backbone,
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": args.coreset_sampling_ratio,
            "num_neighbors": args.num_neighbors,
            "seed": args.seed,
            "epochs": args.epochs,
            "image_size": [512, 512],
            "num_gpus": world_size,
            "strategy": strategy,
            "training_time_sec": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        if hasattr(model, "image_threshold"):
            info["image_threshold"] = model.image_threshold.value.item()

        with open(OUTPUT_DIR / "training_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        if mlflow_active:
            mlflow.log_metrics({"training_time_sec": elapsed})
            mlflow.log_artifact(str(OUTPUT_DIR))
        logger.success("All outputs saved successfully.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if mlflow_active:
            mlflow.end_run()


if __name__ == "__main__":
    main()
