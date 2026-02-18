# -*- coding: utf-8 -*-
"""
배터리 불량 분류 - 전처리 파이프라인 (Azure ML CPU Cluster용)

Flow:
  1. Azure Blob에서 원본 이미지 ZIP 다운로드 + 라벨 JSON ZIP 다운로드
  2. CSV에 있는 대상 이미지만 추출
  3. JSON 라벨로 배터리 외곽선 마스킹 → Crop/Rotate → CLAHE → Letterbox 256x256
  4. 전처리된 이미지 + CSV를 output_dir에 저장 (→ 학습 step에서 사용)

출력 구조:
  output_dir/
    images/          ← 전처리된 이미지 (.png)
    labels.csv       ← 분류 라벨 CSV (복사)
"""
import argparse
import os
import shutil
import tempfile
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
from loguru import logger

# ============================================================
# Azure Blob 설정
# ============================================================
BLOB_ACCOUNT = "batterydata8ai6team"
BLOB_CONTAINER = "battery-data-zip"
BLOB_BASE = "103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터/Training"

IMAGE_ZIPS = [
    f"{BLOB_BASE}/01.원천데이터/TS_Exterior_Img_Datasets_images_1.zip",
    f"{BLOB_BASE}/01.원천데이터/TS_Exterior_Img_Datasets_images_2.zip",
    f"{BLOB_BASE}/01.원천데이터/TS_Exterior_Img_Datasets_images_3.zip",
    f"{BLOB_BASE}/01.원천데이터/TS_Exterior_Img_Datasets_images_4.zip",
]
LABEL_ZIP = f"{BLOB_BASE}/02.라벨링데이터/TL_Exterior_Img_Datasets_label.zip"


# ============================================================
# 전처리 함수들 (pipeline.py에서 이식)
# ============================================================
def get_outline_points(json_path):
    """JSON에서 battery_outline 좌표 추출"""
    import json

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        outline = data.get("swelling", {}).get("battery_outline", [])
        if not outline or len(outline) < 6:
            return None
        return np.array(
            [(outline[i], outline[i + 1]) for i in range(0, len(outline), 2)],
            dtype=np.int32,
        )
    except Exception:
        return None


def apply_mask(img, pts):
    """배터리 외곽선 내부만 남기고 배경 제거"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(img, img, mask=mask)


def crop_and_rotate(img, pts, padding=15):
    """마스킹 + minAreaRect 기반 회전/크롭"""
    img_masked = apply_mask(img, pts)

    rect = cv2.minAreaRect(pts.astype(np.float32))
    center, (w, h), angle = rect

    if w > h:
        angle += 90
        w, h = h, w

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_h, img_w = img.shape[:2]
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(img_h * sin_a + img_w * cos_a)
    new_h = int(img_h * cos_a + img_w * sin_a)
    M[0, 2] += (new_w - img_w) / 2
    M[1, 2] += (new_h - img_h) / 2

    # Orientation fix
    up_vector = np.array([0, -1])
    rotated_up = M[:2, :2] @ up_vector
    if rotated_up[1] > 0:
        M = cv2.getRotationMatrix2D(center, angle + 180, 1.0)
        M[0, 2] += (new_w - img_w) / 2
        M[1, 2] += (new_h - img_h) / 2

    rotated = cv2.warpAffine(
        img_masked, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    nc = np.array([center[0], center[1], 1.0])
    new_cx, new_cy = M[0].dot(nc), M[1].dot(nc)

    crop_w, crop_h = int(w) + padding * 2, int(h) + padding * 2
    x1 = max(0, int(new_cx - crop_w / 2))
    y1 = max(0, int(new_cy - crop_h / 2))
    x2 = min(new_w, x1 + crop_w)
    y2 = min(new_h, y1 + crop_h)

    return rotated[y1:y2, x1:x2]


def resize_letterbox(img, target_size=(256, 256)):
    """비율 유지 + 패딩 리사이즈"""
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    dx, dy = (tw - nw) // 2, (th - nh) // 2
    if len(resized.shape) == 3:
        canvas[dy : dy + nh, dx : dx + nw, :] = resized
    else:
        canvas[dy : dy + nh, dx : dx + nw, :] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return canvas


def preprocess_single(img, pts, target_size=(256, 256)):
    """단일 이미지 전처리: Mask → Crop/Rotate → CLAHE → Letterbox"""
    cropped = crop_and_rotate(img, pts)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    final = resize_letterbox(gray_clahe, target_size)
    return final


# ============================================================
# Blob 다운로드 & 추출
# ============================================================
def download_and_extract_from_zips(
    sas_token, target_files, raw_dir, label_dir
):
    """Blob ZIP에서 대상 이미지 + 라벨 JSON 추출"""
    from adlfs import AzureBlobFileSystem

    fs = AzureBlobFileSystem(account_name=BLOB_ACCOUNT, sas_token=sas_token)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # --- 이미지 ZIP 다운로드 ---
    remaining = set(target_files)
    existing = set(os.listdir(raw_dir))
    remaining -= existing
    total_img = 0

    logger.info(f"Image download: {len(target_files)} target, {len(existing)} existing, {len(remaining)} remaining")

    if remaining:
        for zip_idx, zip_blob in enumerate(IMAGE_ZIPS):
            if not remaining:
                break
            full_path = f"{BLOB_CONTAINER}/{zip_blob}"
            zip_name = zip_blob.split("/")[-1]
            logger.info(f"  [{zip_idx + 1}/4] {zip_name}")

            try:
                with fs.open(full_path, "rb") as f:
                    with zipfile.ZipFile(f, "r") as z:
                        name_map = {}
                        for n in z.namelist():
                            basename = os.path.basename(n)
                            if basename in remaining:
                                name_map[n] = basename

                        if not name_map:
                            logger.info(f"    No matching files, skip.")
                            continue

                        logger.info(f"    Found {len(name_map)} files, extracting...")
                        for zn, bn in name_map.items():
                            with z.open(zn) as src:
                                with open(os.path.join(raw_dir, bn), "wb") as dst:
                                    dst.write(src.read())
                            remaining.discard(bn)
                            total_img += 1
            except Exception as e:
                logger.error(f"    Image ZIP error: {e}")

    logger.info(f"  Images extracted: {total_img} new, {len(os.listdir(raw_dir))} total")

    if remaining:
        logger.warning(f"  {len(remaining)} images not found in any ZIP")

    # --- 라벨 JSON ZIP 다운로드 ---
    target_jsons = {f.replace(".png", ".json") for f in target_files}
    existing_labels = set(os.listdir(label_dir))
    remaining_labels = target_jsons - existing_labels
    total_label = 0

    logger.info(f"Label download: {len(target_jsons)} target, {len(existing_labels)} existing, {len(remaining_labels)} remaining")

    if remaining_labels:
        full_path = f"{BLOB_CONTAINER}/{LABEL_ZIP}"
        logger.info(f"  Label ZIP: {LABEL_ZIP.split('/')[-1]}")

        try:
            with fs.open(full_path, "rb") as f:
                with zipfile.ZipFile(f, "r") as z:
                    name_map = {}
                    for n in z.namelist():
                        basename = os.path.basename(n)
                        if basename in remaining_labels:
                            name_map[n] = basename

                    logger.info(f"    Found {len(name_map)} label files, extracting...")
                    for zn, bn in name_map.items():
                        with z.open(zn) as src:
                            with open(os.path.join(label_dir, bn), "wb") as dst:
                                dst.write(src.read())
                        total_label += 1
        except Exception as e:
            logger.error(f"    Label ZIP error: {e}")

    logger.info(f"  Labels extracted: {total_label} new, {len(os.listdir(label_dir))} total")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Battery Classification Preprocessing")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV data asset path (mounted)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--target_size", type=int, default=256, help="Output image size")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Battery Classification Preprocessing (Azure ML)")
    logger.info(f"  CSV path: {args.csv_path}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Target size: {args.target_size}x{args.target_size}")
    logger.info("=" * 60)

    # --- SAS token from environment ---
    sas_token = os.environ.get("BLOB_SAS_TOKEN")
    if not sas_token:
        raise RuntimeError(
            "BLOB_SAS_TOKEN 환경변수가 설정되지 않았습니다. "
            "Azure ML job YAML에 environment_variables로 설정하거나 "
            "Key Vault에서 가져오세요."
        )

    # --- CSV 읽기 ---
    # csv_path가 폴더일 수 있음 (data asset mount)
    csv_path = args.csv_path
    if os.path.isdir(csv_path):
        csvs = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {csv_path}")
        csv_path = os.path.join(csv_path, csvs[0])
        logger.info(f"  CSV auto-detected: {csv_path}")

    df = pd.read_csv(csv_path)
    target_files = list(df["file_name"].dropna().unique())
    logger.info(f"  Target images: {len(target_files)}")

    # --- 임시 디렉토리에 다운로드 & 추출 ---
    work_dir = tempfile.mkdtemp(prefix="battery_preprocess_")
    raw_dir = os.path.join(work_dir, "raw_images")
    label_dir = os.path.join(work_dir, "labels")

    logger.info(f"  Work dir: {work_dir}")

    download_and_extract_from_zips(sas_token, target_files, raw_dir, label_dir)

    # --- 전처리 ---
    output_img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(output_img_dir, exist_ok=True)

    target_size = (args.target_size, args.target_size)
    success, fail, skip = 0, 0, 0

    logger.info(f"\nPreprocessing {len(target_files)} images...")

    for i, fname in enumerate(target_files):
        img_path = os.path.join(raw_dir, fname)
        json_path = os.path.join(label_dir, fname.replace(".png", ".json"))
        out_path = os.path.join(output_img_dir, fname)

        # 이미 처리된 경우 skip
        if os.path.exists(out_path):
            skip += 1
            continue

        if not os.path.exists(img_path):
            fail += 1
            continue

        if not os.path.exists(json_path):
            fail += 1
            continue

        try:
            img = cv2.imread(img_path)
            pts = get_outline_points(json_path)
            if img is None or pts is None:
                fail += 1
                continue

            final = preprocess_single(img, pts, target_size)
            cv2.imwrite(out_path, final)
            success += 1
        except Exception as e:
            logger.warning(f"  Failed {fname}: {e}")
            fail += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(target_files):
            logger.info(
                f"  [{i + 1}/{len(target_files)}] "
                f"success={success}, skip={skip}, fail={fail} "
                f"({time.time() - t0:.0f}s)"
            )

    # --- CSV 복사 ---
    out_csv = os.path.join(args.output_dir, "labels.csv")
    shutil.copy2(csv_path, out_csv)
    logger.info(f"  CSV copied to {out_csv}")

    # --- 임시 파일 정리 ---
    shutil.rmtree(work_dir, ignore_errors=True)
    logger.info(f"  Temp dir cleaned: {work_dir}")

    # --- 결과 ---
    elapsed = time.time() - t0
    final_count = len(os.listdir(output_img_dir))
    logger.info("=" * 60)
    logger.info(f"Preprocessing complete in {elapsed:.0f}s")
    logger.info(f"  Output: {final_count} images in {output_img_dir}")
    logger.info(f"  Success: {success}, Skip: {skip}, Fail: {fail}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
