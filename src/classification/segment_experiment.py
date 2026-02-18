# -*- coding: utf-8 -*-
"""
배터리 세그멘테이션 비교 실험 (Azure ML GPU Job)

5가지 방법으로 배터리 아웃라인을 자동 추출하고 비교:
  1. SAM (center_point) - 이미지 중심점 프롬프트
  2. SAM (center_box)   - 통계적 bbox 프롬프트
  3. Otsu + Morphology   - 이진화 → 최대 contour
  4. Canny + Contour     - 에지 검출 → contour
  5. GrabCut             - bbox 초기화 전경/배경 분리

출력:
  output_dir/
    masks/               ← 각 방법별 바이너리 마스크
    outlines/            ← 폴리곤 좌표 (JSON)
    visual_comparisons/  ← 비교 이미지
    report.txt           ← IoU/Dice 지표 요약
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from loguru import logger
from pathlib import Path


# ============================================================
# Ground Truth (JSON 라벨이 있는 경우)
# ============================================================
def get_gt_mask_and_outline(img_shape, json_path):
    """JSON battery_outline → (binary_mask, polygon_points)"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        outline = data.get("swelling", {}).get("battery_outline", [])
        if not outline or len(outline) < 6:
            return None, None
        pts = np.array(
            [(outline[i], outline[i + 1]) for i in range(0, len(outline), 2)],
            dtype=np.int32,
        )
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask, pts
    except Exception:
        return None, None


# ============================================================
# Mask → Outline (폴리곤 좌표 추출)
# ============================================================
def mask_to_outline(mask, epsilon_ratio=0.005):
    """바이너리 마스크 → 폴리곤 좌표 리스트"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    # approxPolyDP로 폴리곤 단순화
    perimeter = cv2.arcLength(largest, True)
    epsilon = epsilon_ratio * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist()


# ============================================================
# SAM Segmentation
# ============================================================
def init_sam(model_type="vit_b", device="cuda"):
    """SAM 모델 초기화 (체크포인트 자동 다운로드)"""
    from segment_anything import SamPredictor, sam_model_registry

    # 체크포인트 경로
    ckpt_map = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }
    ckpt_name = ckpt_map[model_type]
    ckpt_path = os.path.join(os.getcwd(), ckpt_name)

    if not os.path.exists(ckpt_path):
        url = f"https://dl.fbaipublicfiles.com/segment_anything/{ckpt_name}"
        logger.info(f"Downloading SAM checkpoint: {url}")
        import urllib.request
        urllib.request.urlretrieve(url, ckpt_path)
        logger.info(f"Downloaded to {ckpt_path}")

    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def segment_sam_point(predictor, img_rgb):
    """SAM center point 프롬프트"""
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    predictor.set_image(img_rgb)
    h, w = img_rgb.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )
    best_mask = masks[np.argmax(scores)]
    return (best_mask * 255).astype(np.uint8)


def segment_sam_box(predictor, img_rgb):
    """SAM bbox 프롬프트 (이미지 크기 기반 추정)"""
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    predictor.set_image(img_rgb)
    h, w = img_rgb.shape[:2]
    # 배터리가 이미지 중앙에 위치한다고 가정한 bbox
    margin_x, margin_y = int(w * 0.3), int(h * 0.03)
    box = np.array([margin_x, margin_y, w - margin_x, h - margin_y])
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    best_mask = masks[np.argmax(scores)]
    return (best_mask * 255).astype(np.uint8)


# ============================================================
# Classical CV Segmentation
# ============================================================
def segment_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def segment_hsv_s(img):
    """HSV S채널(채도) 기반 Otsu - 금속 배터리 vs 배경 분리"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(s_channel, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def segment_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def segment_grabcut(img):
    h, w = img.shape[:2]
    max_dim = 480
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        small = img
    sh, sw = small.shape[:2]
    margin_x, margin_y = int(sw * 0.3), int(sh * 0.05)
    rect = (margin_x, margin_y, sw - 2 * margin_x, sh - 2 * margin_y)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    mask = np.zeros((sh, sw), dtype=np.uint8)
    try:
        cv2.grabCut(small, mask, rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if scale < 1.0:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    return clean


# ============================================================
# Metrics
# ============================================================
def calc_metrics(mask_pred, mask_gt):
    p = mask_pred > 0
    g = mask_gt > 0
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else 0.0
    precision = intersection / p.sum() if p.sum() > 0 else 0.0
    recall = intersection / g.sum() if g.sum() > 0 else 0.0
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


# ============================================================
# Visualization
# ============================================================
def create_comparison_image(img, method_results, gt_outline=None):
    """원본 + 각 방법의 마스크 오버레이 비교 이미지"""
    h, w = img.shape[:2]
    n_methods = len(method_results)
    panel_w = 300
    panel_h = int(panel_w * h / w)
    canvas_w = panel_w * (n_methods + 1) + 10 * n_methods
    canvas_h = panel_h + 40
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Original
    orig_small = cv2.resize(img, (panel_w, panel_h))
    if gt_outline is not None:
        scale_x, scale_y = panel_w / w, panel_h / h
        gt_scaled = (gt_outline * [scale_x, scale_y]).astype(np.int32)
        cv2.polylines(orig_small, [gt_scaled], True, (0, 255, 0), 2)
    canvas[30:30 + panel_h, 0:panel_w] = orig_small
    cv2.putText(canvas, "Original+GT", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    x = panel_w + 10
    for name, mask in method_results:
        # 마스크 오버레이
        img_small = cv2.resize(img, (panel_w, panel_h))
        mask_small = cv2.resize(mask, (panel_w, panel_h))

        # 아웃라인 그리기 (빨간색)
        contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_small, contours, -1, (0, 0, 255), 2)

        # 마스크 영역 반투명 파란색
        overlay = img_small.copy()
        overlay[mask_small > 0] = (overlay[mask_small > 0] * 0.7 + np.array([255, 144, 30]) * 0.3).astype(np.uint8)

        canvas[30:30 + panel_h, x:x + panel_w] = overlay
        cv2.putText(canvas, name, (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x += panel_w + 10

    return canvas


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Battery Segmentation Comparison")
    parser.add_argument("--data_path", type=str, required=True, help="Test data path")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--sam_model", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--skip_sam", action="store_true", help="Skip SAM (CPU only)")
    parser.add_argument("--resize", type=int, default=0, help="Resize long edge before segmentation (0=원본)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    outlines_dir = os.path.join(output_dir, "outlines")
    vis_dir = os.path.join(output_dir, "visual_comparisons")
    for d in [masks_dir, outlines_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    # --- 이미지 찾기 ---
    data_path = args.data_path
    # 디버그: 데이터 에셋 구조 출력
    logger.info(f"data_path: {data_path}")
    logger.info(f"data_path contents: {os.listdir(data_path)}")
    for item in os.listdir(data_path):
        full = os.path.join(data_path, item)
        if os.path.isdir(full):
            sub_items = os.listdir(full)[:10]
            logger.info(f"  {item}/ ({len(os.listdir(full))} files): {sub_items}")

    img_dir = data_path
    # PNG 또는 JPG 이미지 검색
    for sub in ["originnal_image", "original_image", "good_images", "images", "test_100/images", "test_100"]:
        candidate = os.path.join(data_path, sub)
        if os.path.isdir(candidate):
            pngs = [f for f in os.listdir(candidate) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if pngs:
                img_dir = candidate
                break

    pngs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    logger.info(f"Found {len(pngs)} images in {img_dir}")

    # --- JSON 라벨 찾기 (있으면 GT 비교용) ---
    json_dir = img_dir  # 같은 폴더에 있을 수 있음
    for sub in ["labels", "../labels"]:
        candidate = os.path.join(data_path, sub) if not sub.startswith("..") else os.path.join(img_dir, sub)
        if os.path.isdir(candidate):
            jsons = [f for f in os.listdir(candidate) if f.endswith(".json")]
            if jsons:
                json_dir = candidate
                logger.info(f"Found {len(jsons)} JSON labels in {json_dir}")
                break

    # --- SAM 초기화 ---
    import torch
    device = "cuda" if torch.cuda.is_available() and not args.skip_sam else "cpu"
    sam_predictor = None

    if not args.skip_sam:
        logger.info(f"Initializing SAM ({args.sam_model}) on {device}...")
        try:
            sam_predictor = init_sam(model_type=args.sam_model, device=device)
            logger.info("SAM initialized successfully")
        except Exception as e:
            logger.warning(f"SAM init failed: {e}, skipping SAM methods")

    # --- 방법 정의 ---
    methods = {}
    if sam_predictor is not None:
        methods["SAM_point"] = lambda img: segment_sam_point(sam_predictor, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        methods["SAM_box"] = lambda img: segment_sam_box(sam_predictor, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    methods["Otsu"] = segment_otsu
    methods["HSV_S"] = segment_hsv_s
    methods["Canny"] = segment_canny
    methods["GrabCut"] = segment_grabcut

    # --- 실행 ---
    results = {name: {"iou": [], "dice": [], "precision": [], "recall": [], "time": []} for name in methods}
    all_outlines = {}
    per_image_rows = []  # per-image CSV 수집

    for i, fname in enumerate(pngs):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 리사이징 전처리 (장축 기준, 비율 유지)
        resize_scale = 1.0
        if args.resize > 0:
            h_orig, w_orig = img.shape[:2]
            resize_scale = args.resize / max(h_orig, w_orig)
            if resize_scale < 1.0:
                img = cv2.resize(img, (int(w_orig * resize_scale), int(h_orig * resize_scale)))

        # GT (있으면) — 원본 해상도로 생성 후 리사이징
        json_path = os.path.join(json_dir, fname.replace(".png", ".json"))
        gt_mask, gt_outline = None, None
        if os.path.exists(json_path):
            orig_shape = (int(img.shape[0] / resize_scale), int(img.shape[1] / resize_scale), 3) if resize_scale < 1.0 else img.shape
            gt_mask, gt_outline = get_gt_mask_and_outline(orig_shape, json_path)
            if gt_mask is not None and resize_scale < 1.0:
                gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                if gt_outline is not None:
                    gt_outline = (gt_outline * resize_scale).astype(np.int32)

        method_results_for_vis = []
        file_outlines = {}

        for name, method in methods.items():
            t0 = time.time()
            pred_mask = method(img)
            elapsed = time.time() - t0
            results[name]["time"].append(elapsed)

            # 아웃라인 추출
            outline = mask_to_outline(pred_mask)
            if outline:
                file_outlines[name] = outline

            # GT 비교
            if gt_mask is not None:
                m = calc_metrics(pred_mask, gt_mask)
                for k, v in m.items():
                    results[name][k].append(v)
                per_image_rows.append(f"{fname},{name},{m['iou']:.4f},{m['dice']:.4f},{m['precision']:.4f},{m['recall']:.4f}")

            # 마스크 저장
            mask_path = os.path.join(masks_dir, f"{Path(fname).stem}_{name}.png")
            cv2.imwrite(mask_path, pred_mask)

            method_results_for_vis.append((name, pred_mask))

        # 아웃라인 JSON 저장
        all_outlines[fname] = file_outlines
        outline_path = os.path.join(outlines_dir, fname.replace(".png", ".json"))
        with open(outline_path, "w") as f:
            json.dump(file_outlines, f)

        # 비교 이미지 (매 5장)
        if i < 5 or i % 5 == 0:
            vis = create_comparison_image(img, method_results_for_vis, gt_outline)
            cv2.imwrite(os.path.join(vis_dir, f"compare_{i:03d}_{fname}"), vis)

        if (i + 1) % 10 == 0 or (i + 1) == len(pngs):
            logger.info(f"  [{i + 1}/{len(pngs)}] processed")

    # --- Report ---
    logger.info("\n" + "=" * 70)
    logger.info("Battery Segmentation Comparison Results")
    logger.info("=" * 70)

    header = f"{'Method':<14} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8} {'Time(ms)':>10}"
    logger.info(header)
    logger.info("-" * 70)

    resize_info = f" (resized: {args.resize})" if args.resize > 0 else " (original)"
    report_lines = [
        f"Battery Segmentation Comparison Results{resize_info}",
        f"Images: {len(pngs)}",
        "=" * 70,
        header,
        "-" * 70,
    ]

    best_method, best_iou = None, 0
    for name in methods:
        ious = results[name]["iou"]
        dices = results[name]["dice"]
        precs = results[name]["precision"]
        recs = results[name]["recall"]
        times = results[name]["time"]

        mean_iou = np.mean(ious) if ious else 0
        mean_dice = np.mean(dices) if dices else 0
        mean_prec = np.mean(precs) if precs else 0
        mean_rec = np.mean(recs) if recs else 0
        mean_time = np.mean(times) * 1000

        line = f"{name:<14} {mean_iou:>8.4f} {mean_dice:>8.4f} {mean_prec:>8.4f} {mean_rec:>8.4f} {mean_time:>10.1f}"
        logger.info(line)
        report_lines.append(line)

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_method = name

    report_lines.append("")
    report_lines.append(f"BEST: {best_method} (mean IoU: {best_iou:.4f})")

    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\nBEST: {best_method} (mean IoU: {best_iou:.4f})")

    # per-image CSV 저장
    if per_image_rows:
        csv_path = os.path.join(output_dir, "per_image_metrics.csv")
        with open(csv_path, "w") as f:
            f.write("filename,method,iou,dice,precision,recall\n")
            f.write("\n".join(per_image_rows) + "\n")
        logger.info(f"Per-image metrics saved to {csv_path}")

    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
