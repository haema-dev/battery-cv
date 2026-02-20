# -*- coding: utf-8 -*-
"""
Battery Defect Classification V6 - Anti-overfitting improvements
Changes from V5:
- CutMix + MixUp augmentation (torchvision)
- Label Smoothing (0.1) combined with class weights
- Higher dropout (0.5)
- Image size 256 (match crop size, no resize needed)
- Cosine Annealing with linear warmup (5 epochs)
- Gradient accumulation (effective batch 64)
- StochasticDepth already in EfficientNet-B0
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
import torch.nn.functional as F
import torchmetrics
from loguru import logger
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# Config
# ============================================================
CLASS_NAMES = ["Damaged", "Pollution", "Damaged+Pollution", "Normal"]
NUM_CLASSES = 4


# ============================================================
# Focal Loss with Label Smoothing
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


# ============================================================
# CutMix / MixUp
# ============================================================
def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, y[index], lam


def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# ============================================================
# Dataset & DataModule
# ============================================================
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
            image = Image.new("RGB", (256, 256), color="black")

        label_str = str(row.get("label", row.get("defect_types", "")))

        if label_str == "Normal":
            label = 3
        elif "Damaged" in label_str and "Pollution" in label_str:
            label = 2
        elif "Pollution" in label_str:
            label = 1
        else:
            label = 0

        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return image, label


class DefectDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, img_dir, batch_size=32, num_workers=4, img_size=256):
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
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

        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")

        # Stratified split
        train_dfs = []
        val_dfs = []
        for label_val in df['label'].unique():
            subset = df[df['label'] == label_val]
            train_part = subset.sample(frac=0.8, random_state=42)
            val_part = subset.drop(train_part.index)
            train_dfs.append(train_part)
            val_dfs.append(val_part)

        train_df = pd.concat(train_dfs)
        val_df = pd.concat(val_dfs)

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

        for split_name, split_df in [("Train", train_df), ("Val", val_df)]:
            counts = split_df['label'].value_counts()
            for cls in CLASS_NAMES:
                logger.info(f"  {split_name} - {cls}: {counts.get(cls, 0)}")

        # Class weights
        label_map = {"Damaged": 0, "Pollution": 1, "Damaged+Pollution": 2, "Normal": 3}
        train_labels = train_df['label'].map(label_map)
        counts = train_labels.value_counts().sort_index()
        max_count = counts.max()
        self.class_weights = torch.tensor(
            [max_count / counts.get(i, 1) for i in range(NUM_CLASSES)],
            dtype=torch.float32
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
class DefectClassifier(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=5e-4,
                 class_weights=None, backbone_name="efficientnet_b0",
                 focal_gamma=2.0, label_smoothing=0.1,
                 scheduler_type="cosine", warmup_epochs=5,
                 use_cutmix=True, use_mixup=True,
                 cutmix_alpha=1.0, mixup_alpha=0.2,
                 cutmix_prob=0.3, mixup_prob=0.3):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.warmup_epochs = warmup_epochs
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob

        # Build backbone
        if backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Focal Loss with Label Smoothing
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )
        self.criterion = FocalLoss(
            alpha=self.class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1_per_class = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average=None,
        )
        self.val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        rand_val = random.random()

        if self.training and self.use_cutmix and rand_val < self.cutmix_prob:
            x, y_a, y_b, lam = cutmix_data(x, y, self.cutmix_alpha)
            logits = self(x)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
        elif self.training and self.use_mixup and rand_val < (self.cutmix_prob + self.mixup_prob):
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
        self.val_cm(logits, y)
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
            logger.info(f"  val_f1_{name}: {f1_per[i]:.4f}")
        self.val_f1_per_class.reset()

        # Log confusion matrix
        cm = self.val_cm.compute()
        logger.info(f"  Confusion Matrix:\n{cm}")
        self.val_cm.reset()

    def configure_optimizers(self):
        # Discriminative LR: backbone features lower, classifier head higher
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
        optimizer = torch.optim.AdamW(param_groups, lr=self.hparams.lr, weight_decay=5e-4)

        if self.hparams.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Battery Defect Classification V6 (Anti-overfitting)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--cutmix_prob", type=float, default=0.3)
    parser.add_argument("--mixup_prob", type=float, default=0.3)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--devices", type=int, default=0)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Battery Defect Classification V6 (Anti-overfitting)")
    logger.info(f"  Backbone: {args.backbone}")
    logger.info(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"  Focal Loss gamma: {args.focal_gamma}, Label smoothing: {args.label_smoothing}")
    logger.info(f"  Scheduler: {args.scheduler}, Img size: {args.img_size}")
    logger.info(f"  CutMix prob: {args.cutmix_prob}, MixUp prob: {args.mixup_prob}")
    logger.info(f"  Accumulate: {args.accumulate}")
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
        pngs = [f for f in os.listdir(data_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if pngs:
            img_dir = data_path
        else:
            raise FileNotFoundError(f"No images directory found in {data_path}")

    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Image dir: {img_dir}")

    # GPU/DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = local_rank == 0
    num_gpus = torch.cuda.device_count()

    if world_size > 1 and num_gpus > 1:
        use_devices, strategy, accelerator = num_gpus, "ddp", "gpu"
    elif num_gpus >= 1:
        use_devices, strategy, accelerator = 1, "auto", "gpu"
    else:
        use_devices, strategy, accelerator = 1, "auto", "cpu"

    logger.info(f"  DDP: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, GPUs={num_gpus}")

    # MLflow
    mlflow_active = False
    if is_main_process:
        try:
            mlflow.start_run()
            mlflow.log_params({
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "backbone": args.backbone,
                "focal_gamma": args.focal_gamma, "label_smoothing": args.label_smoothing,
                "scheduler": args.scheduler, "img_size": args.img_size,
                "cutmix_prob": args.cutmix_prob, "mixup_prob": args.mixup_prob,
                "accumulate": args.accumulate, "num_classes": NUM_CLASSES,
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
            num_classes=NUM_CLASSES, lr=args.lr,
            class_weights=dm.class_weights,
            backbone_name=args.backbone,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            scheduler_type=args.scheduler,
            use_cutmix=True, use_mixup=True,
            cutmix_prob=args.cutmix_prob,
            mixup_prob=args.mixup_prob,
        )

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            monitor="val_f1", dirpath=ckpt_dir,
            filename="defect-{epoch:02d}-{val_f1:.3f}",
            save_top_k=3, mode="max",
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss", patience=15, mode="min", verbose=True,
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
            accumulate_grad_batches=args.accumulate,
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
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "backbone": args.backbone,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "scheduler": args.scheduler,
            "cutmix_prob": args.cutmix_prob,
            "mixup_prob": args.mixup_prob,
            "accumulate": args.accumulate,
            "img_size": args.img_size,
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
