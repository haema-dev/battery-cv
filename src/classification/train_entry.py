# -*- coding: utf-8 -*-
"""
배터리 불량 분류 모델 학습 (Azure ML Compute Cluster용)
- EfficientNet-B0 backbone + BCEWithLogitsLoss (multi-label: Damaged/Pollution)
- 단일 GPU / 멀티 GPU(DDP) 자동 지원
- 전처리된 이미지 + CSV를 데이터 에셋으로 마운트받아 학습
"""
import argparse
import json
import os
import time

import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from loguru import logger
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# Dataset & DataModule
# ============================================================
class DefectDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, classes=("Damaged", "Pollution")):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["file_name"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="black")

        labels = []
        defect_str = str(row["defect_types"])
        for cls in self.classes:
            labels.append(1 if cls in defect_str else 0)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, labels


class DefectDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, img_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["file_name"])
        df_defects = df[df["has_defect"].fillna(False).astype(bool)].copy()
        df_defects.loc[:, "defect_types"] = df_defects["defect_types"].fillna("Unknown")

        if len(df_defects) == 0:
            raise ValueError(f"No defect samples found in {self.csv_path}")

        train_df = df_defects.sample(frac=0.8, random_state=42)
        val_df = df_defects.drop(train_df.index)

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

        # 클래스 분포 출력
        for split_name, split_df in [("Train", train_df), ("Val", val_df)]:
            for cls in ["Damaged", "Pollution"]:
                cnt = split_df["defect_types"].str.contains(cls).sum()
                logger.info(f"  {split_name} - {cls}: {cnt}")

        self.train_dataset = DefectDataset(train_df, self.img_dir, transform=self.train_transform)
        self.val_dataset = DefectDataset(val_df, self.img_dir, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
        )


# ============================================================
# Model
# ============================================================
class DefectClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-3, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_classes)
        self.val_precision = torchmetrics.Precision(task="multilabel", num_labels=num_classes)
        self.val_recall = torchmetrics.Recall(task="multilabel", num_labels=num_classes)

    def get_gradcam_target_layer(self):
        return self.backbone.features[-1]

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, prog_bar=True, sync_dist=True)
        self.log("val_precision", self.val_precision, sync_dist=True)
        self.log("val_recall", self.val_recall, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Battery Defect Classification Training")
    parser.add_argument("--data_path", type=str, required=True, help="Mounted data asset path")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_freeze", action="store_true", help="Don't freeze backbone")
    parser.add_argument("--devices", type=int, default=0, help="GPU count (0=auto)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Battery Defect Classification Training (Azure ML)")
    logger.info(f"  Data path: {args.data_path}")
    logger.info(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"  Freeze backbone: {not args.no_freeze}")
    logger.info("=" * 60)

    data_path = args.data_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- 데이터 구조 확인 ---
    # 데이터 에셋 구조:
    #   <data_path>/
    #     images/        <- 전처리된 이미지
    #     labels.csv     <- 메타데이터 CSV
    img_dir = os.path.join(data_path, "images")
    csv_path = os.path.join(data_path, "labels.csv")

    if not os.path.exists(csv_path):
        # 폴더 루트에 csv가 있을 수도 있음
        csvs = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        if csvs:
            csv_path = os.path.join(data_path, csvs[0])
            logger.info(f"  CSV auto-detected: {csv_path}")
        else:
            raise FileNotFoundError(f"No CSV found in {data_path}")

    if not os.path.isdir(img_dir):
        # processed_images/ 폴더도 확인
        alt_img_dir = os.path.join(data_path, "processed_images")
        if os.path.isdir(alt_img_dir):
            img_dir = alt_img_dir
            logger.info(f"  Images found in processed_images/")
        else:
            # 이미지가 루트에 바로 있을 수도 있음
            pngs = [f for f in os.listdir(data_path) if f.endswith(".png")]
            if pngs:
                img_dir = data_path
                logger.info(f"  Images found in root: {len(pngs)} files")
            else:
                raise FileNotFoundError(f"No images directory found in {data_path}")

    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Image dir: {img_dir}")

    # --- DDP rank 확인 (환경변수 정리 전) ---
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # --- Azure ML 분산 환경변수 정리 (Lightning이 DDP를 직접 관리) ---
    for env_key in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        if env_key in os.environ:
            logger.info(f"  Removing Azure ML env: {env_key}={os.environ[env_key]}")
            del os.environ[env_key]

    # --- GPU 감지 ---
    num_gpus = torch.cuda.device_count()
    logger.info(f"  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    # GPU 설정: Lightning이 DDP를 직접 관리 (process_count_per_instance=1)
    if num_gpus > 1:
        use_devices = min(args.devices, num_gpus) if args.devices > 0 else num_gpus
        strategy = "ddp"
        accelerator = "gpu"
        logger.info(f"  Multi-GPU: {use_devices} devices, strategy=ddp")
    elif num_gpus == 1:
        use_devices = 1
        strategy = "auto"
        accelerator = "gpu"
    else:
        use_devices = 1
        strategy = "auto"
        accelerator = "cpu"

    logger.info(f"  Using: {use_devices} device(s), strategy={strategy}, accelerator={accelerator}")

    effective_batch = args.batch_size * use_devices
    logger.info(f"  Effective batch size: {args.batch_size} x {use_devices} = {effective_batch}")

    # --- MLflow (main process만, 인증 실패 시 skip) ---
    mlflow_active = False
    if is_main_process:
        try:
            mlflow.start_run()
            mlflow.log_params({
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "effective_batch_size": effective_batch,
                "lr": args.lr,
                "freeze_backbone": not args.no_freeze,
                "num_gpus": use_devices,
                "strategy": strategy,
            })
            mlflow_active = True
            logger.info("  MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"  MLflow init failed (training will continue without tracking): {e}")

    try:
        # --- DataModule ---
        dm = DefectDataModule(
            csv_path=csv_path,
            img_dir=img_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # --- Model ---
        model = DefectClassifier(
            num_classes=2,
            lr=args.lr,
            freeze_backbone=not args.no_freeze,
        )

        # --- Callbacks ---
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            monitor="val_f1",
            dirpath=ckpt_dir,
            filename="defect-{epoch:02d}-{val_f1:.3f}",
            save_top_k=3,
            mode="max",
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=7,
            mode="min",
            verbose=True,
        )

        # --- Trainer ---
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=accelerator,
            devices=use_devices,
            strategy=strategy,
            callbacks=[checkpoint_cb, early_stop_cb],
            default_root_dir=output_dir,
            precision="16-mixed" if num_gpus > 0 else "32-true",
            gradient_clip_val=1.0,
        )

        # --- Train ---
        t0 = time.time()
        trainer.fit(model, datamodule=dm)
        elapsed = time.time() - t0

        best_path = checkpoint_cb.best_model_path
        best_score = checkpoint_cb.best_model_score

        logger.info(f"Training complete in {elapsed:.0f}s")
        logger.info(f"Best checkpoint: {best_path}")
        logger.info(f"Best val_f1: {best_score:.4f}")

        # --- Save final outputs ---
        if mlflow_active:
            mlflow.log_metrics({
                "best_val_f1": best_score.item() if best_score else 0.0,
                "training_time_sec": elapsed,
            })

        info = {
            "best_checkpoint": best_path,
            "best_val_f1": best_score.item() if best_score else 0.0,
            "training_time_sec": elapsed,
            "num_gpus": use_devices,
            "strategy": strategy,
            "epochs_completed": trainer.current_epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        if mlflow_active:
            mlflow.log_artifacts(output_dir)
        logger.success("All outputs saved successfully.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if mlflow_active:
            mlflow.end_run()


if __name__ == "__main__":
    main()
