# -*- coding: utf-8 -*-
"""
V5 Dataset Preprocessor - Combine V3 defects + Normal crops from ZIP
Runs as first step in Azure ML job before train_entry_v5.py

Input:
  --defect_data : V3 defect crops (images/ + labels.csv)
  --zip_path    : Mounted blob dir containing TS ZIP files
  --output_dir  : Combined dataset output

Output:
  output_dir/
    images/      <- V3 defects + Normal crops (256x256)
    labels.csv   <- Combined labels (4-class)
"""
import argparse
import csv
import os
import shutil
import zipfile

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

CROP_REGIONS = [
    (0.0, 0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5, 1.0),
    (0.5, 1.0, 0.0, 0.5),
    (0.5, 1.0, 0.5, 1.0),
    (0.1, 0.6, 0.1, 0.6),
    (0.25, 0.75, 0.0, 0.5),
    (0.25, 0.75, 0.5, 1.0),
    (0.1, 0.9, 0.2, 0.8),
]
TARGET_SIZE = 256


def find_zip_files(zip_path):
    """Find TS ZIP files in mounted directory (recursively)."""
    zip_files = []
    for root, dirs, files in os.walk(zip_path):
        for f in sorted(files):
            if f.startswith("TS_Exterior") and f.endswith(".zip"):
                zip_files.append(os.path.join(root, f))
    if not zip_files:
        # Fallback: any zip file
        for root, dirs, files in os.walk(zip_path):
            for f in sorted(files):
                if f.endswith(".zip"):
                    zip_files.append(os.path.join(root, f))
    return zip_files


def crop_patch(image, idx):
    """Crop a 256x256 patch from image using predefined regions."""
    h, w = image.shape[:2]
    region = CROP_REGIONS[idx % len(CROP_REGIONS)]
    y1 = int(h * region[0])
    y2 = int(h * region[1])
    x1 = int(w * region[2])
    x2 = int(w * region[3])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="V5 Dataset Preprocessor")
    parser.add_argument("--defect_data", required=True, help="V3 defect crops path")
    parser.add_argument("--zip_path", required=True, help="Mounted blob dir with ZIP files")
    parser.add_argument("--output_dir", required=True, help="Combined dataset output")
    args = parser.parse_args()

    output_images = os.path.join(args.output_dir, "images")
    os.makedirs(output_images, exist_ok=True)

    labels_rows = []

    # ========================================
    # Step 1: Copy V3 defect data
    # ========================================
    print("=" * 60)
    print("[1/3] Copying V3 defect data...")

    v3_csv = os.path.join(args.defect_data, "labels.csv")
    v3_images = os.path.join(args.defect_data, "images")

    if not os.path.exists(v3_csv):
        csvs = [f for f in os.listdir(args.defect_data) if f.endswith(".csv")]
        if csvs:
            v3_csv = os.path.join(args.defect_data, csvs[0])

    if not os.path.isdir(v3_images):
        pngs = [f for f in os.listdir(args.defect_data) if f.lower().endswith((".png", ".jpg"))]
        if pngs:
            v3_images = args.defect_data

    v3_df = pd.read_csv(v3_csv)
    copied = 0

    for _, row in v3_df.iterrows():
        src = os.path.join(v3_images, row["file_name"])
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(output_images, row["file_name"]))
        labels_rows.append({
            "file_name": row["file_name"],
            "has_defect": True,
            "defect_types": row.get("defect_types", ""),
            "label": row["label"],
        })
        copied += 1

    print(f"  Copied {copied} defect crops")
    from collections import Counter
    for label, count in Counter(r["label"] for r in labels_rows).most_common():
        print(f"    {label}: {count}")

    # ========================================
    # Step 2: Find normal images in ZIPs
    # ========================================
    print("\n[2/3] Extracting normal images from ZIPs...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    normal_csv = os.path.join(script_dir, "normal_download_list.csv")

    if not os.path.exists(normal_csv):
        print("  WARNING: normal_download_list.csv not found, skipping normals")
        save_csv(labels_rows, args.output_dir)
        return

    normal_df = pd.read_csv(normal_csv)
    target_files = set(normal_df["file_name"].values)
    print(f"  Target normal files: {len(target_files)}")

    # Explore mounted directory structure
    print(f"\n  Exploring mounted path: {args.zip_path}")
    try:
        root_items = os.listdir(args.zip_path)
        for item in sorted(root_items)[:15]:
            full = os.path.join(args.zip_path, item)
            item_type = "DIR" if os.path.isdir(full) else "FILE"
            size = ""
            if os.path.isfile(full):
                size = f" ({os.path.getsize(full) / 1024 / 1024:.0f}MB)"
            print(f"    [{item_type}] {item}{size}")
        if len(root_items) > 15:
            print(f"    ... and {len(root_items) - 15} more")
    except Exception as e:
        print(f"  Cannot list root: {e}")

    # Find ZIP files
    zip_files = find_zip_files(args.zip_path)
    print(f"\n  Found {len(zip_files)} ZIP files:")
    for zf in zip_files:
        print(f"    {os.path.basename(zf)}")

    if not zip_files:
        print("  ERROR: No ZIP files found! Saving defect-only dataset.")
        save_csv(labels_rows, args.output_dir)
        return

    # Extract normal images from ZIPs
    remaining = set(target_files)
    saved = 0
    errors = 0

    for zip_path in zip_files:
        if not remaining:
            break

        zip_name = os.path.basename(zip_path)
        print(f"\n  Processing: {zip_name}")

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                # Build name mapping
                name_map = {}
                for n in z.namelist():
                    basename = os.path.basename(n)
                    if basename in remaining:
                        name_map[basename] = n

                if not name_map:
                    print(f"    No matching files, skip.")
                    continue

                print(f"    Found {len(name_map)} matching files, extracting...")

                for fname, zip_entry in name_map.items():
                    try:
                        data = z.read(zip_entry)
                        arr = np.frombuffer(data, np.uint8)
                        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        del data, arr

                        if image is None:
                            errors += 1
                            remaining.discard(fname)
                            continue

                        crop = crop_patch(image, saved)
                        del image

                        if crop is None:
                            errors += 1
                            remaining.discard(fname)
                            continue

                        stem = Path(fname).stem
                        out_name = f"normal_{saved:05d}_{stem}.png"
                        cv2.imwrite(os.path.join(output_images, out_name), crop)

                        labels_rows.append({
                            "file_name": out_name,
                            "has_defect": False,
                            "defect_types": "Normal",
                            "label": "Normal",
                        })
                        saved += 1
                        remaining.discard(fname)

                    except Exception as e:
                        errors += 1
                        remaining.discard(fname)

                    if saved % 200 == 0 and saved > 0:
                        print(f"      Saved {saved} normal crops...")

        except Exception as e:
            print(f"    ZIP error: {e}")

    print(f"\n  Normal extraction complete: saved={saved}, errors={errors}, not_found={len(remaining)}")

    # ========================================
    # Step 3: Save combined labels CSV
    # ========================================
    print(f"\n[3/3] Saving combined dataset...")
    save_csv(labels_rows, args.output_dir)


def save_csv(labels_rows, output_dir):
    csv_out = os.path.join(output_dir, "labels.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_name", "has_defect", "defect_types", "label",
        ])
        writer.writeheader()
        writer.writerows(labels_rows)

    from collections import Counter
    total = len(labels_rows)
    print(f"\n{'='*60}")
    print(f"  Combined Dataset: {total} images")
    for label, count in Counter(r["label"] for r in labels_rows).most_common():
        print(f"    {label}: {count}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
