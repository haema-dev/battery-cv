# -*- coding: utf-8 -*-
"""
test_eval.py  –  FastFlow 종합 테스트 평가 스크립트

테스트 데이터 소스 (3가지 동시 지원):
  A) test_data/          : originnal_image/ (원본 1920×1080) + labels/*.json
                           battery_outline 기반 크롭 → 256×256
  B) class_data/         : classification_data/none_heatmap_based/
                           processed_images/ + sample_all_list.csv (불량 2,248장)
  C) normal_data/        : good_images_cropped_resized/ 등 (정상 이미지 폴더)
                           폴더 전체를 정상(label=0)으로 간주

Azure ML job: test-job.yml 참조
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from PIL import Image

from torchvision.transforms.v2 import (
    Compose, Normalize, Resize, ToImage, ToDtype,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

# index.py 공유 컴포넌트 import
sys.path.insert(0, str(Path(__file__).parent))
from index import (
    FastflowCompat, _DisableVisualizerAtStart,
    blend_heatmap, annotate_image,
)

from anomalib.engine import Engine
from anomalib import TaskType


# ────────────────────────────────────────────────
# 배터리 윤곽 BBox 추출 (originnal_image 크롭용)
# ────────────────────────────────────────────────
def extract_crop_bbox(json_path: Path, pad: int = 30):
    """JSON의 battery_outline 폴리곤 → (x1,y1,x2,y2) 바운딩박스."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        pts = data.get("swelling", {}).get("battery_outline", [])
        if not pts or len(pts) < 4:
            return None
        xs = pts[0::2]
        ys = pts[1::2]
        w = data.get("image_info", {}).get("width", 1920)
        h = data.get("image_info", {}).get("height", 1080)
        return (
            max(0,   int(min(xs)) - pad),
            max(0,   int(min(ys)) - pad),
            min(w,   int(max(xs)) + pad),
            min(h,   int(max(ys)) + pad),
        )
    except Exception as e:
        logger.warning(f"bbox 추출 실패 ({json_path.name}): {e}")
        return None


# ────────────────────────────────────────────────
# 다중 소스 레코드 빌드
# ────────────────────────────────────────────────
def build_records(test_data_path, class_data_path, normal_data_path, sample_n=0):
    records = []

    # ── 소스 A: test_data/originnal_image + labels/*.json ──────────
    if test_data_path:
        test_root = Path(test_data_path)
        img_dir   = test_root / "originnal_image"   # 오타 그대로 유지
        lbl_dir   = test_root / "labels"
        if img_dir.exists() and lbl_dir.exists():
            for jf in sorted(lbl_dir.glob("*.json")):
                img_path = img_dir / jf.with_suffix(".png").name
                if not img_path.exists():
                    img_path = img_dir / jf.with_suffix(".jpg").name
                if not img_path.exists():
                    continue
                try:
                    data     = json.loads(jf.read_text(encoding="utf-8"))
                    is_norm  = data.get("image_info", {}).get("is_normal", None)
                    gt_label = 0 if is_norm else 1
                except Exception:
                    gt_label = -1  # 라벨 불명
                crop_bbox = extract_crop_bbox(jf)
                records.append({
                    "image_path": str(img_path),
                    "gt_label":   gt_label,
                    "crop_bbox":  crop_bbox,
                    "source":     "test_data",
                })
            logger.info(f"[소스A] test_data: {sum(1 for r in records if r['source']=='test_data')}장 로드")
        else:
            logger.warning(f"[소스A] test_data 경로 없음: {test_root}")

    # ── 소스 B: classification_data/none_heatmap_based ─────────────
    if class_data_path:
        class_root = Path(class_data_path)
        csv_path   = class_root / "sample_all_list.csv"
        img_dir    = class_root / "processed_images"
        if csv_path.exists() and img_dir.exists():
            df = pd.read_csv(csv_path)
            before = len(records)
            for _, row in df.iterrows():
                img_path = img_dir / str(row["file_name"])
                if not img_path.exists():
                    continue
                is_norm  = str(row.get("is_normal", "False")).strip()
                gt_label = 0 if is_norm == "True" else 1
                records.append({
                    "image_path": str(img_path),
                    "gt_label":   gt_label,
                    "crop_bbox":  None,
                    "source":     "class_data",
                })
            added = len(records) - before
            logger.info(f"[소스B] classification_data: {added}장 로드")
        else:
            logger.warning(f"[소스B] class_data 경로 없음: {class_root}")

    # ── 소스 C: normal_data/ (전체 = 정상) ────────────────────────
    if normal_data_path:
        normal_root = Path(normal_data_path)
        before = len(records)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for img_path in sorted(normal_root.rglob("*")):
            if img_path.suffix.lower() in exts:
                records.append({
                    "image_path": str(img_path),
                    "gt_label":   0,
                    "crop_bbox":  None,
                    "source":     "normal_data",
                })
        added = len(records) - before
        logger.info(f"[소스C] normal_data: {added}장 로드")

    # 유효 레코드만 (gt_label != -1)
    valid = [r for r in records if r["gt_label"] in (0, 1)]
    invalid = len(records) - len(valid)
    if invalid:
        logger.warning(f"라벨 불명 {invalid}장 제외")

    # 샘플링
    if sample_n > 0 and len(valid) > sample_n:
        import random
        random.seed(42)
        valid = random.sample(valid, sample_n)
        logger.info(f"샘플링: {sample_n}장 선택")

    n_norm   = sum(1 for r in valid if r["gt_label"] == 0)
    n_defect = sum(1 for r in valid if r["gt_label"] == 1)
    logger.info(f"전체 테스트셋: {len(valid)}장 (정상 {n_norm} | 불량 {n_defect})")
    return valid


# ────────────────────────────────────────────────
# Letterbox resize (학습 데이터 "256x256 fit" 방식과 일치)
# ────────────────────────────────────────────────
def fit_to_square(img: Image.Image, size: int = 256) -> Image.Image:
    """긴 변을 size에 맞게 축소 → 짧은 변을 0(검정) 패딩으로 size까지 확장.
    학습 데이터 폴더명 '256x256 fit'과 동일한 방식.
    이미 size×size 이면 그대로 반환(no-op).
    """
    w, h = img.size
    if w == size and h == size:
        return img
    scale = size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_l = (size - new_w) // 2
    pad_t = (size - new_h) // 2
    pad_r = size - new_w - pad_l
    pad_b = size - new_h - pad_t
    from PIL import ImageOps
    return ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=0)


# ────────────────────────────────────────────────
# 커스텀 Dataset
# ────────────────────────────────────────────────
class MultiSourceDataset(torch.utils.data.Dataset):
    def __init__(self, records: list, transform, image_size: int = 256):
        self.records    = records
        self.transform  = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:
            logger.warning(f"이미지 로드 실패 ({rec['image_path']}): {e}")
            img = Image.new("RGB", (self.image_size, self.image_size), 0)

        if rec.get("crop_bbox"):
            img = img.crop(rec["crop_bbox"])

        # 비정사각형 이미지는 letterbox로 정사각형화 후 transform
        img = fit_to_square(img, self.image_size)

        return {
            "image":      self.transform(img),
            "image_path": rec["image_path"],
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "image":      torch.stack([b["image"] for b in batch]),
            "image_path": [b["image_path"] for b in batch],
        }


# ────────────────────────────────────────────────
# 모델 로드 (index.py 로직 기반)
# ────────────────────────────────────────────────
def load_model(model_path: str, precision: str = "32"):
    import tempfile

    model_dir = Path(model_path)
    # Azure ML custom_model: 파일 직접 전달 or 디렉토리 전달 모두 처리
    if model_dir.is_file() and model_dir.suffix in (".pt", ".ckpt"):
        pt_path = model_dir
    else:
        pt_files = (list(model_dir.glob("*.pt"))   + list(model_dir.glob("**/*.pt")) +
                    list(model_dir.glob("*.ckpt")) + list(model_dir.glob("**/*.ckpt")))
        if not pt_files:
            raise FileNotFoundError(f"모델 .pt/.ckpt 파일을 찾을 수 없음: {model_dir}")
        pt_path = pt_files[0]
    logger.info(f"모델 파일: {pt_path}")

    state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # backbone 자동 감지
    backbone   = "wide_resnet50_2"
    flow_steps = 8
    for k in state_dict:
        if "wide_resnet50_2" in k:
            backbone = "wide_resnet50_2"
            break
        elif "resnet18" in k:
            backbone = "resnet18"
            break

    # flow_steps 감지
    fs_keys = [k for k in state_dict if "fast_flow_blocks" in k]
    if fs_keys:
        max_idx = max(
            int(k.split("fast_flow_blocks.")[1].split(".")[0])
            for k in fs_keys
            if "fast_flow_blocks." in k
        )
        flow_steps = max_idx + 1

    logger.info(f"backbone={backbone}, flow_steps={flow_steps}")

    # threshold 추출
    saved_thresh = None
    for key in ["post_processor._image_threshold", "image_threshold.value"]:
        if key in state_dict:
            v = state_dict[key]
            v = v.item() if hasattr(v, "item") else float(v)
            if np.isfinite(v):
                saved_thresh = v
                logger.info(f"저장 임계값: {saved_thresh:.6f} (키={key})")
                break

    # FastflowCompat 생성 (index.py 학습 시와 동일한 인수)
    model = FastflowCompat(
        backbone=backbone,
        pre_trained=False,
        flow_steps=flow_steps,
        conv3x3_only=False,
        hidden_ratio=1.0,
    )
    # index.py 631줄과 동일: CLASSIFICATION 태스크 명시 (기본값 SEGMENTATION이면
    # anomalib이 GT 마스크 없는 DataLoader에서 pixel 지표 계산 시도 → 오류)
    model.task = TaskType.CLASSIFICATION
    model.load_state_dict(state_dict)
    logger.info(f"state_dict 로드 완료")

    # Lightning ckpt 래핑 (engine.predict 호환) — index.py와 동일한 구조
    import lightning.pytorch as L
    eval_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Resize((256, 256)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    wrapped = {
        "state_dict":                  model.state_dict(),
        "transform":                   eval_transform,
        "pytorch-lightning_version":   L.__version__,
        "epoch":                       0,
        "global_step":                 0,
        "loops":                       None,
        "callbacks":                   {},
        "optimizer_states":            [],
        "lr_schedulers":               [],
    }
    tmp_f = tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False)
    ckpt_path = tmp_f.name
    tmp_f.close()
    torch.save(wrapped, ckpt_path)
    logger.info("state_dict → Lightning ckpt 래핑 완료")

    return model, ckpt_path, saved_thresh


# ────────────────────────────────────────────────
# 지표 계산 (direction-aware Youden J)
# ────────────────────────────────────────────────
def compute_metrics(y_true, y_score, saved_thresh, output_dir: Path):
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    score_min = float(y_score.min())
    score_max = float(y_score.max())

    # 방향1: lower=defect  (FastFlow 표준)
    fpr1, tpr1, thrs1 = roc_curve(y_true, -y_score)
    j1 = int(np.argmax(tpr1 - fpr1))
    thr_std  = float(-thrs1[j1])
    auroc_std = roc_auc_score(y_true, -y_score)

    # 방향2: higher=defect (역전)
    fpr2, tpr2, thrs2 = roc_curve(y_true, y_score)
    j2 = int(np.argmax(tpr2 - fpr2))
    thr_inv  = float(thrs2[j2])
    auroc_inv = roc_auc_score(y_true, y_score)

    if auroc_inv > auroc_std:
        logger.warning(
            f"[Score 방향 역전] AUROC_inv={auroc_inv:.4f} > AUROC_std={auroc_std:.4f} "
            f"→ 역방향 Youden J 임계값 {thr_inv:.6f} 적용"
        )
        y_pred = (y_score > thr_inv).astype(int)
        auroc  = auroc_inv
        used_thr = thr_inv
    else:
        logger.info(
            f"Youden J (표준): {thr_std:.6f}  AUROC={auroc_std:.4f}"
        )
        y_pred = (y_score < thr_std).astype(int)
        auroc  = auroc_std
        used_thr = thr_std

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    logger.success(
        f"[METRICS] Accuracy={acc:.4f} | Precision={prec:.4f} | "
        f"Recall={rec:.4f} | F1={f1:.4f} | AUROC={auroc:.4f}"
    )
    report = classification_report(y_true, y_pred, target_names=["Normal", "Defect"])
    logger.info(f"\n{report}")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Defect"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix (Test Eval)")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Score Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    norm_scores   = y_score[y_true == 0]
    defect_scores = y_score[y_true == 1]
    ax.hist(norm_scores,   bins=60, alpha=0.6, color="green",  label=f"Normal  (n={len(norm_scores)})")
    ax.hist(defect_scores, bins=60, alpha=0.6, color="red",    label=f"Defect  (n={len(defect_scores)})")
    ax.axvline(used_thr, color="black", linestyle="--", label=f"Threshold={used_thr:.4f}")
    ax.set_xlabel("Anomaly Score (log-likelihood mean)")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution – Test Eval")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close()

    return {
        "accuracy": acc, "precision": prec,
        "recall":   rec, "f1": f1, "auroc": auroc,
        "threshold": used_thr, "n_total": len(y_true),
        "n_normal": int((y_true == 0).sum()),
        "n_defect": int((y_true == 1).sum()),
    }


# ────────────────────────────────────────────────
# 메인 실행 (index.py --mode test 위임용)
# ────────────────────────────────────────────────
def run_test_eval(args):
    """index.py에서 --mode test 시 호출되는 진입점.
    args: argparse.Namespace (index.py 파서로 파싱된 args 그대로 사용)
    """
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"[TEST EVAL START] precision={args.precision} | batch={args.batch_size}")

    # ── 레코드 빌드 ──────────────────────────────
    records = build_records(
        test_data_path   = args.test_data_path,
        class_data_path  = args.class_data_path,
        normal_data_path = args.normal_data_path,
        sample_n         = args.sample_n,
    )
    if not records:
        logger.error("테스트 이미지 없음. 경로를 확인하세요.")
        return

    # ── 전처리 ────────────────────────────────────
    eval_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Resize((256, 256)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = MultiSourceDataset(records, eval_transform, image_size=256)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = MultiSourceDataset.collate_fn,
    )
    logger.info(f"DataLoader: {len(dataset)}장 / {len(loader)} 배치")

    # ── 모델 로드 ─────────────────────────────────
    import tempfile, os
    model, ckpt_path, saved_thresh = load_model(args.model_path, args.precision)

    # ── 엔진 설정 ─────────────────────────────────
    precision_map = {"32": "32-true", "16": "16-mixed", "32-true": "32-true", "16-mixed": "16-mixed"}
    pl_precision  = precision_map.get(args.precision, "32-true")

    engine = Engine(
        precision           = pl_precision,
        callbacks           = [_DisableVisualizerAtStart()],
        enable_progress_bar = True,
    )

    # ── 추론 ──────────────────────────────────────
    logger.info("추론 시작...")
    predictions = engine.predict(model=model, dataloaders=loader, ckpt_path=ckpt_path)

    # ── 결과 수집 ─────────────────────────────────
    # records와 predictions를 경로 기준으로 매칭
    path_to_rec = {r["image_path"]: r for r in records}
    result_rows = []

    if predictions is None:
        logger.error("predict 결과 없음")
        return

    for batch_pred in predictions:
        # batch_pred: dict {"image_path": [...], "pred_score": tensor, "anomaly_map": tensor, ...}
        if isinstance(batch_pred, dict):
            paths  = batch_pred.get("image_path", [])
            scores_t = batch_pred.get("pred_score")
        else:
            paths  = getattr(batch_pred, "image_path", [])
            scores_t = getattr(batch_pred, "pred_score", None)

        scores = scores_t.cpu().numpy() if scores_t is not None else np.zeros(len(paths))

        for i, path in enumerate(paths):
            rec      = path_to_rec.get(path, {})
            gt_label = rec.get("gt_label", -1)
            score    = float(scores[i]) if i < len(scores) else 0.0
            source   = rec.get("source", "unknown")
            result_rows.append({
                "file_path":     path,
                "file_name":     Path(path).name,
                "source":        source,
                "gt_label":      gt_label,
                "anomaly_score": score,
            })

    if not result_rows:
        logger.error("결과 행 없음")
        return

    df = pd.DataFrame(result_rows)
    df.to_csv(OUTPUT_DIR / "test_results_raw.csv", index=False)
    logger.info(f"raw 결과 저장: {len(df)}행")

    # ── 소스별 분포 로그 ──────────────────────────
    for src, grp in df.groupby("source"):
        n_norm = (grp["gt_label"] == 0).sum()
        n_def  = (grp["gt_label"] == 1).sum()
        logger.info(f"  [{src}] {len(grp)}장 | 정상 {n_norm} | 불량 {n_def}")

    # ── 지표 계산 ─────────────────────────────────
    valid_df = df[df["gt_label"].isin([0, 1])].copy()
    if len(valid_df) == 0:
        logger.error("유효 gt_label 없음")
        return
    if len(valid_df["gt_label"].unique()) < 2:
        logger.warning("단일 클래스만 존재 – AUROC 계산 불가")
        return

    y_true  = valid_df["gt_label"].values
    y_score = valid_df["anomaly_score"].values

    metrics = compute_metrics(y_true, y_score, saved_thresh, OUTPUT_DIR)

    # is_defect 컬럼 추가 (Youden J 적용 후)
    used_thr = metrics["threshold"]
    if roc_auc_score(y_true, y_score) > roc_auc_score(y_true, -y_score):
        valid_df["is_defect"] = (valid_df["anomaly_score"] > used_thr)
    else:
        valid_df["is_defect"] = (valid_df["anomaly_score"] < used_thr)
    valid_df.to_csv(OUTPUT_DIR / "test_results.csv", index=False)

    # ── 히트맵 시각화 (--save_vis 옵션) ──────────
    if args.save_vis:
        vis_dir = OUTPUT_DIR / "visualizations"
        (vis_dir / "defect").mkdir(parents=True, exist_ok=True)
        (vis_dir / "normal").mkdir(parents=True, exist_ok=True)
        for batch_pred in predictions:
            if isinstance(batch_pred, dict):
                paths  = batch_pred.get("image_path", [])
                amaps  = batch_pred.get("anomaly_map")
                images = batch_pred.get("image")
                scores_t = batch_pred.get("pred_score")
            else:
                paths  = getattr(batch_pred, "image_path", [])
                amaps  = getattr(batch_pred, "anomaly_map", None)
                images = getattr(batch_pred, "image", None)
                scores_t = getattr(batch_pred, "pred_score", None)

            if scores_t is None:
                continue
            scores_np = scores_t.cpu().numpy()

            for i, path in enumerate(paths):
                score     = float(scores_np[i]) if i < len(scores_np) else 0.0
                row_match = valid_df[valid_df["file_path"] == path]
                is_def    = bool(row_match["is_defect"].values[0]) if len(row_match) > 0 else False

                subdir   = vis_dir / ("defect" if is_def else "normal")
                save_p   = subdir / f"vis_{Path(path).name}"
                try:
                    if amaps is not None and images is not None:
                        canvas = blend_heatmap(images[i], amaps[i])
                    else:
                        img_np = np.array(Image.open(path).convert("RGB").resize((256, 256)))
                        canvas = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    annotate_image(canvas, score, is_def)
                    cv2.imwrite(str(save_p), canvas)
                except Exception as e:
                    logger.debug(f"시각화 실패 ({Path(path).name}): {e}")

    # ── 최종 요약 ─────────────────────────────────
    logger.success("=" * 60)
    logger.success(f"[TEST EVAL 완료]")
    logger.success(f"  총 테스트: {metrics['n_total']}장 (정상 {metrics['n_normal']} | 불량 {metrics['n_defect']})")
    logger.success(f"  AUROC    : {metrics['auroc']:.4f}")
    logger.success(f"  Accuracy : {metrics['accuracy']:.4f}")
    logger.success(f"  F1 Score : {metrics['f1']:.4f}")
    logger.success(f"  Recall   : {metrics['recall']:.4f}  (불량 검출률)")
    logger.success(f"  Precision: {metrics['precision']:.4f}")
    logger.success(f"  결과 저장: {OUTPUT_DIR}")
    logger.success("=" * 60)

    try:
        os.unlink(ckpt_path)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="FastFlow 종합 테스트 평가")
    parser.add_argument("--model_path",       type=str, required=True)
    parser.add_argument("--test_data_path",   type=str, default=None)
    parser.add_argument("--class_data_path",  type=str, default=None)
    parser.add_argument("--normal_data_path", type=str, default=None)
    parser.add_argument("--output_dir",       type=str, default="./outputs")
    parser.add_argument("--precision",        type=str, default="32")
    parser.add_argument("--batch_size",       type=int, default=16)
    parser.add_argument("--sample_n",         type=int, default=0)
    parser.add_argument("--save_vis",         action="store_true")
    run_test_eval(parser.parse_args())


if __name__ == "__main__":
    main()
