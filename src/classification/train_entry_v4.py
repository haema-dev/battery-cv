# -*- coding: utf-8 -*-
"""
배터리 불량 분류 모델 학습 V4 — 개선 버전
- EfficientNet-B0/B3 선택 가능
- 강화된 augmentation (RandomRotation, GaussianBlur, RandomErasing)
- Label smoothing + CosineAnnealing + Mixup
- 단일 GPU / 멀티 GPU(DDP) 자동 지원
"""
import argparse
import json
import os
import time
import random

import mlflow
import numpy as np
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
CLASS_NAMES = ["Damaged", "Pollution", "Damaged+Pollution"]


class DefectDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["file_name"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="black")

        defect_str = str(row["defect_types"])
        has_damaged = "Damaged" in defect_str
        has_pollution = "Pollution" in defect_str
        if has_damaged and has_pollution:
            label = 2
        elif has_pollution:
            label = 1
        else:
            label = 0
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return image, label


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: 배치 내 이미지 쌍을 선형 보간"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class DefectDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, img_dir, batch_size=32, num_workers=4, img_size=224):
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
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

        for split_name, split_df in [("Train", train_df), ("Val", val_df)]:
            dt = split_df["defect_types"].fillna("")
            has_d = dt.str.contains("Damaged")
            has_p = dt.str.contains("Pollution")
            cnt_d = (has_d & ~has_p).sum()
            cnt_p = (~has_d & has_p).sum()
            cnt_b = (has_d & has_p).sum()
            logger.info(f"  {split_name} - Damaged: {cnt_d}, Pollution: {cnt_p}, Both: {cnt_b}")

        # 클래스 가중치 계산
        dt = train_df["defect_types"].fillna("")
        has_d = dt.str.contains("Damaged")
        has_p = dt.str.contains("Pollution")
        counts = [(has_d & ~has_p).sum(), (~has_d & has_p).sum(), (has_d & has_p).sum()]
        max_count = max(counts)
        self.class_weights = torch.tensor(
            [max_count / c if c > 0 else 1.0 for c in counts], dtype=torch.float32
        )
        logger.info(f"  Class weights: {self.class_weights.tolist()}")

        self.train_dataset = DefectDataset(train_df, self.img_dir, transform=self.train_transform)
        self.val_dataset = DefectDataset(val_df, self.img_dir, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
        )


# ============================================================
# Model
# ============================================================
def build_backbone(name, num_classes):
    """백본 모델 생성"""
    if name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    elif name == "efficientnet_b3":
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    elif name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return backbone


class DefectClassifier(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-3, freeze_backbone=True,
                 class_weights=None, backbone_name="efficientnet_b0",
                 label_smoothing=0.0, use_mixup=False, mixup_alpha=0.2,
                 scheduler_type="cosine", warmup_epochs=3):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.backbone = build_backbone(backbone_name, num_classes)

        if freeze_backbone:
            feature_module = getattr(self.backbone, 'features', None)
            if feature_module is None:
                # ResNet: freeze all except fc
                for name, param in self.backbone.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False
            else:
                for param in feature_module.parameters():
                    param.requires_grad = False
                feature_module.eval()

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing,
        )
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        # per-class F1
        self.val_f1_per_class = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average=None,
        )

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        if self.hparams.freeze_backbone:
            feature_module = getattr(self.backbone, 'features', None)
            if feature_module:
                feature_module.eval()

        x, y = batch

        if self.use_mixup and self.training:
            x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            logits = self(x)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)

        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.val_f1_per_class(logits, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, prog_bar=True, sync_dist=True)
        self.log("val_precision", self.val_precision, sync_dist=True)
        self.log("val_recall", self.val_recall, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        f1_per = self.val_f1_per_class.compute()
        for i, name in enumerate(CLASS_NAMES[:len(f1_per)]):
            self.log(f"val_f1_{name}", f1_per[i], sync_dist=True)
        self.val_f1_per_class.reset()

    def configure_optimizers(self):
        # Discriminative LR: backbone lower, head higher
        if not self.hparams.freeze_backbone:
            backbone_params = []
            head_params = []
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if 'classifier' in name or 'fc' in name:
                        head_params.append(param)
                    else:
                        backbone_params.append(param)
            param_groups = [
                {"params": backbone_params, "lr": self.hparams.lr * 0.1},
                {"params": head_params, "lr": self.hparams.lr},
            ]
        else:
            param_groups = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(param_groups, lr=self.hparams.lr, weight_decay=1e-4)

        if self.hparams.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        else:
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
    parser = argparse.ArgumentParser(description="Battery Defect Classification V4")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_freeze", action="store_true")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3",
                        choices=["efficientnet_b0", "efficientnet_b3", "resnet50"])
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--devices", type=int, default=0)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Battery Defect Classification V4")
    logger.info(f"  Backbone: {args.backbone}")
    logger.info(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"  Label smoothing: {args.label_smoothing}, Mixup: {args.use_mixup}")
    logger.info(f"  Scheduler: {args.scheduler}, Img size: {args.img_size}")
    logger.info(f"  Freeze backbone: {not args.no_freeze}")
    logger.info("=" * 60)

    data_path = args.data_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(data_path, "images")
    csv_path = os.path.join(data_path, "labels.csv")

    if not os.path.exists(csv_path):
        csvs = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        if csvs:
            csv_path = os.path.join(data_path, csvs[0])
        else:
            raise FileNotFoundError(f"No CSV found in {data_path}")

    if not os.path.isdir(img_dir):
        alt_img_dir = os.path.join(data_path, "processed_images")
        if os.path.isdir(alt_img_dir):
            img_dir = alt_img_dir
        else:
            pngs = [f for f in os.listdir(data_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if pngs:
                img_dir = data_path
            else:
                raise FileNotFoundError(f"No images directory found in {data_path}")

    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Image dir: {img_dir}")

    # DDP setup
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

    effective_batch = args.batch_size * world_size
    logger.info(f"  Using: devices={use_devices}, strategy={strategy}")
    logger.info(f"  Effective batch size: {args.batch_size} x {world_size} = {effective_batch}")

    # MLflow
    mlflow_active = False
    if is_main_process:
        try:
            mlflow.start_run()
            mlflow.log_params({
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "backbone": args.backbone,
                "freeze_backbone": not args.no_freeze,
                "label_smoothing": args.label_smoothing,
                "use_mixup": args.use_mixup,
                "scheduler": args.scheduler,
                "img_size": args.img_size,
                "num_gpus": world_size,
            })
            mlflow_active = True
        except Exception as e:
            logger.warning(f"  MLflow init failed: {e}")

    try:
        dm = DefectDataModule(
            csv_path=csv_path, img_dir=img_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
            img_size=args.img_size,
        )
        dm.setup()

        model = DefectClassifier(
            num_classes=3,
            lr=args.lr,
            freeze_backbone=not args.no_freeze,
            class_weights=dm.class_weights,
            backbone_name=args.backbone,
            label_smoothing=args.label_smoothing,
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha,
            scheduler_type=args.scheduler,
        )

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            monitor="val_f1", dirpath=ckpt_dir,
            filename="defect-{epoch:02d}-{val_f1:.3f}",
            save_top_k=3, mode="max",
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss", patience=10, mode="min", verbose=True,
        )

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

        t0 = time.time()
        trainer.fit(model, datamodule=dm)
        elapsed = time.time() - t0

        best_path = checkpoint_cb.best_model_path
        best_score = checkpoint_cb.best_model_score

        logger.info(f"Training complete in {elapsed:.0f}s")
        logger.info(f"Best checkpoint: {best_path}")
        logger.info(f"Best val_f1: {best_score:.4f}")

        if mlflow_active:
            mlflow.log_metrics({
                "best_val_f1": best_score.item() if best_score else 0.0,
                "training_time_sec": elapsed,
            })

        info = {
            "best_checkpoint": best_path,
            "best_val_f1": best_score.item() if best_score else 0.0,
            "training_time_sec": elapsed,
            "num_gpus": world_size,
            "strategy": strategy,
            "backbone": args.backbone,
            "label_smoothing": args.label_smoothing,
            "use_mixup": args.use_mixup,
            "scheduler": args.scheduler,
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
