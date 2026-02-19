# -*- coding: utf-8 -*-
# anomalib 2.2.0 | Python 3.11 | Azure ML GPU T4
# FastFlow 최고 성능 구성: wide_resnet50_2 + flow_steps=16 + FP16 + augmentation
import os
import random
import argparse
import numpy as np
import torch
import cv2
import mlflow
from pathlib import Path
from loguru import logger

from anomalib.models import Fastflow
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.loggers import AnomalibMLFlowLogger

from torchvision.transforms.v2 import (
    Compose, Normalize, Resize, ToImage, ToDtype,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


# ────────────────────────────────────────────────
# 재현성 보장
# ────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────
# anomalib 2.x 배치 구조 파싱 (dict / object 모두 지원)
# ────────────────────────────────────────────────
def parse_batch(batch):
    """anomalib 1.x(dict) 및 2.x(ImageBatch object) 모두 호환"""
    if isinstance(batch, dict):
        paths       = batch.get("image_path", [])
        images      = batch.get("image")
        amaps       = batch.get("anomaly_map") or batch.get("anomaly_maps")
        scores_t    = batch.get("pred_score")  or batch.get("pred_scores")
        labels_t    = batch.get("pred_label")  or batch.get("pred_labels")
    else:
        paths    = getattr(batch, "image_path", [])
        images   = getattr(batch, "image", None)
        amaps    = getattr(batch, "anomaly_map", None)
        scores_t = getattr(batch, "pred_score", None)
        labels_t = getattr(batch, "pred_label", None)

    scores = scores_t.cpu().numpy() if scores_t is not None else np.array([])
    labels = labels_t.cpu().numpy() if labels_t is not None else np.array([])
    return paths, images, amaps, scores, labels


# ────────────────────────────────────────────────
# 히트맵 + 원본 합성 (OpenCV)
# ────────────────────────────────────────────────
def blend_heatmap(image_tensor, amap_tensor, alpha: float = 0.6):
    amap = amap_tensor.cpu().numpy() if hasattr(amap_tensor, "cpu") else amap_tensor
    amap = amap.squeeze()
    am_min, am_max = amap.min(), amap.max()
    amap_norm = ((amap - am_min) / (am_max - am_min + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)

    orig = image_tensor.cpu().numpy() if hasattr(image_tensor, "cpu") else image_tensor
    if orig.ndim == 3 and orig.shape[0] == 3:
        orig = orig.transpose(1, 2, 0)
    orig_vis = ((orig - orig.min()) / (orig.max() - orig.min() + 1e-8) * 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_vis, cv2.COLOR_RGB2BGR)

    if orig_bgr.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))

    return cv2.addWeighted(orig_bgr, alpha, heatmap, 1 - alpha, 0)


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Battery FastFlow – anomalib 2.2.0")

    # 경로
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--model_path",  type=str, default=None)
    parser.add_argument("--output_dir",  type=str, default="./outputs")

    # 모드
    parser.add_argument("--mode", type=str, default="training",
                        choices=["training", "evaluation", "prediction"])

    # 모델 하이퍼파라미터 (최고 성능 기본값)
    parser.add_argument("--backbone",     type=str,   default="wide_resnet50_2",
                        help="resnet18 | wide_resnet50_2 | efficientnet_b5 등")
    parser.add_argument("--flow_steps",   type=int,   default=16,
                        help="Normalizing Flow 변환 횟수. 클수록 표현력↑ (기본 16)")
    parser.add_argument("--hidden_ratio", type=float, default=1.0)
    parser.add_argument("--image_size",   type=int,   default=256)

    # 학습 설정
    parser.add_argument("--epochs",       type=int,   default=300,
                        help="FastFlow은 수렴에 300-500 epoch 필요 (기본 300)")
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="T4 16GB + FP16: wide_resnet50_2 기준 32 안전 (~5GB)")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="EarlyStopping patience")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--precision",    type=str,   default="16-mixed",
                        help="T4 GPU FP16 혼합 정밀도 – 속도 2x, 메모리↓")

    args = parser.parse_args()
    set_seed(args.seed)

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[START] MODE={args.mode.upper()} | "
        f"BACKBONE={args.backbone} | "
        f"FLOW_STEPS={args.flow_steps} | "
        f"IMG={args.image_size}x{args.image_size} | "
        f"PRECISION={args.precision}"
    )

    try:
        dataset_root = Path(args.data_path)
        img_sz = args.image_size

        # ── 전처리 변환 ──────────────────────────────
        # anomalib 2.x: train_augmentations / val_augmentations / test_augmentations
        eval_transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((img_sz, img_sz)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 학습 시 양품(Normal) 이미지에 다양한 augmentation 적용 → 과적합 방지
        train_transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((img_sz, img_sz)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ── 불량 카테고리 자동 탐색 ──────────────────
        val_path = dataset_root / "validation"
        abnormal_dirs = None
        if val_path.exists():
            subs = [d for d in val_path.iterdir() if d.is_dir() and d.name != "good"]
            if subs:
                abnormal_dirs = [f"validation/{d.name}" for d in subs]
                logger.info(f"불량 카테고리 발견: {[d.name for d in subs]}")

        # ── 데이터 모듈 ──────────────────────────────
        if args.mode == "prediction":
            # 전수검사: validation 폴더 전체를 라벨 없이 추론
            try:
                from anomalib.data import PredictDataset
                predict_dir = val_path
                pred_dataset = PredictDataset(path=predict_dir, transform=eval_transform)
                loader = torch.utils.data.DataLoader(
                    pred_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True,
                )
                logger.info(f"PredictDataset 경로: {predict_dir} ({len(pred_dataset)} 이미지)")
            except Exception as e:
                logger.warning(f"PredictDataset 초기화 실패: {e} – Folder fallback 사용")
                loader = None
        else:
            datamodule = Folder(
                name="battery",
                root=str(dataset_root),
                normal_dir="train/good",
                normal_test_dir="validation/good",
                abnormal_dir=abnormal_dirs,
                train_batch_size=args.batch_size,
                eval_batch_size=args.batch_size,
                # ★ anomalib 2.x API: train_transform → train_augmentations
                train_augmentations=train_transform,
                val_augmentations=eval_transform,
                test_augmentations=eval_transform,
                num_workers=4,
                seed=args.seed,
            )

        # ── FastFlow 모델 ────────────────────────────
        # ★ anomalib 2.x: model.setup() 불필요, engine이 자동 처리
        model = Fastflow(
            backbone=args.backbone,
            pre_trained=True,
            flow_steps=args.flow_steps,
            conv3x3_only=False,   # wide_resnet50_2: 1x1+3x3 모두 사용
            hidden_ratio=args.hidden_ratio,
        )
        logger.info(f"FastFlow 모델 생성 완료 (backbone={args.backbone}, flow_steps={args.flow_steps})")

        # ── 사전학습 가중치 로드 (선택) ───────────────
        # .ckpt → Lightning 체크포인트: engine에 전달
        # .pt/.pth → state_dict: 직접 로드 후 ckpt_path=None
        ckpt_path = None
        if args.model_path and os.path.exists(args.model_path):
            if args.model_path.endswith(".ckpt"):
                logger.info(f"Lightning 체크포인트 로드: {args.model_path}")
                ckpt_path = args.model_path
            else:
                logger.info(f"state_dict 수동 로드: {args.model_path}")
                raw = torch.load(args.model_path, map_location="cpu")
                state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info(f"가중치 로드 완료 (missing={len(missing)}, unexpected={len(unexpected)})")

        # ── 콜백 구성 ────────────────────────────────
        callbacks = []
        if args.mode == "training":
            callbacks.append(EarlyStopping(
                monitor="image_AUROC",
                patience=args.patience,
                mode="max",
                verbose=True,
            ))
            callbacks.append(ModelCheckpoint(
                dirpath=str(OUTPUT_DIR / "checkpoints"),
                filename="fastflow-{epoch:03d}-auroc{image_AUROC:.4f}",
                monitor="image_AUROC",
                mode="max",
                save_top_k=3,
                save_last=True,
            ))

        # ── MLFlow 로거 ──────────────────────────────
        mlflow_logger = AnomalibMLFlowLogger(
            experiment_name="Battery_FastFlow_v2",
            save_dir=str(OUTPUT_DIR),
        )

        # ── Engine ───────────────────────────────────
        engine = Engine(
            max_epochs=args.epochs,
            devices=1,
            accelerator="auto",
            precision=args.precision,   # ★ FP16 mixed precision: T4 속도 최대 2배
            logger=mlflow_logger,
            callbacks=callbacks,
            default_root_dir=str(OUTPUT_DIR),
        )

        # ══════════════════════════════════════════════
        # 모드별 실행
        # ══════════════════════════════════════════════

        if args.mode == "training":
            logger.info("Training 시작")
            engine.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

            logger.info("Training 완료 후 Test 평가 실행")
            engine.test(model=model, datamodule=datamodule)

        elif args.mode == "evaluation":
            logger.info("Evaluation 시작")
            engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        elif args.mode == "prediction":
            logger.info("전수검사(Prediction) 시작")

            if loader is not None:
                predictions = engine.predict(
                    model=model, dataloaders=loader, ckpt_path=ckpt_path
                )
            else:
                predictions = engine.predict(
                    model=model, datamodule=datamodule, ckpt_path=ckpt_path
                )

            import pandas as pd
            records = []
            vis_dir = OUTPUT_DIR / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

            for batch in (predictions or []):
                paths, images, amaps, scores, labels = parse_batch(batch)

                for i in range(len(paths)):
                    path  = paths[i]
                    score = float(scores[i]) if i < len(scores) else 0.0
                    label = bool(labels[i])  if i < len(labels) else False

                    if amaps is not None and images is not None:
                        blended = blend_heatmap(images[i], amaps[i])
                        file_name = Path(path).name
                        save_path = vis_dir / f"vis_{file_name}"
                        cv2.imwrite(str(save_path), blended)
                        vis_path_str = str(save_path)
                    else:
                        file_name    = Path(path).name
                        vis_path_str = ""

                    records.append({
                        "file_path":     path,
                        "file_name":     file_name,
                        "parent_dir":    Path(path).parent.name,
                        "anomaly_score": score,
                        "is_defect":     label,
                        "vis_path":      vis_path_str,
                    })

            if records:
                df = pd.DataFrame(records)
                csv_path = OUTPUT_DIR / "results.csv"
                df.to_csv(csv_path, index=False)
                total   = len(df)
                defects = int(df["is_defect"].sum())
                logger.success(
                    f"전수검사 완료: {total}장 | 불량 {defects}장 ({defects/total*100:.1f}%) | "
                    f"양품 {total-defects}장"
                )
                logger.success(f"CSV: {csv_path} | 히트맵: {vis_dir}")
            else:
                logger.warning("예측 결과가 없습니다. 데이터 경로와 모델을 확인하세요.")

        # ── 모델 가중치 저장 ────────────────────────
        model_save_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_save_path)
        logger.success(f"[FINISH] 모델 저장: {model_save_path}")
        logger.success(f"[FINISH] 전체 출력: {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"[FATAL] {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
