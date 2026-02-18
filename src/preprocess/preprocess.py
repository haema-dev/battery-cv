import json
import cv2
import numpy as np
from pathlib import Path


def get_outline_points(json_path):
    """JSON 라벨에서 battery_outline 좌표를 추출"""
    try:
        with open(str(json_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        outline = data.get("swelling", {}).get("battery_outline", [])
        if not outline or len(outline) < 6:
            return None
        return np.array([(outline[i], outline[i+1]) for i in range(0, len(outline), 2)], dtype=np.int32)
    except Exception:
        return None


def apply_mask(img, pts):
    """배터리 외곽선 내부만 남기고 배경을 검은색으로 채움"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(img, img, mask=mask)


def crop_and_rotate(img, pts, padding=15):
    """마스킹 → 회전 → 크롭 (배터리가 세로로 정렬되도록)"""
    img_masked = apply_mask(img, pts)

    rect = cv2.minAreaRect(pts.astype(np.float32))
    center, (w, h), angle = rect

    if w > h:
        angle += 90
        w, h = h, w

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_h, img_w = img.shape[:2]
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(img_h * sin + img_w * cos)
    new_h = int(img_h * cos + img_w * sin)
    M[0, 2] += (new_w - img_w) / 2
    M[1, 2] += (new_h - img_h) / 2

    rotated = cv2.warpAffine(img_masked, M, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    new_center = np.array([center[0], center[1], 1.0])
    new_cx = M[0].dot(new_center)
    new_cy = M[1].dot(new_center)

    crop_w, crop_h = int(w) + padding * 2, int(h) + padding * 2
    x1 = max(0, int(new_cx - crop_w / 2))
    y1 = max(0, int(new_cy - crop_h / 2))
    x2 = min(new_w, x1 + crop_w)
    y2 = min(new_h, y1 + crop_h)

    return rotated[y1:y2, x1:x2]


def apply_clahe(img):
    """
    Applies CLAHE to the image. 
    Converts to Grayscale, applies CLAHE, then converts back to BGR.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    # Convert back to BGR for consistency with model input expectations (usually 3 channels)
    return cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)

def resize_letterbox(img, target_size=(256, 256)):
    """
    Resizes image to target_size while maintaining aspect ratio using letterbox padding.
    target_size: (width, height)
    """
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Create black canvas
    canvas = np.zeros((th, tw, 3) if len(img.shape)==3 else (th, tw), dtype=np.uint8)
    dx, dy = (tw - nw) // 2, (th - nh) // 2
    
    if len(img.shape) == 3:
        canvas[dy:dy+nh, dx:dx+nw, :] = resized
    else:
        canvas[dy:dy+nh, dx:dx+nw] = resized
        
    return canvas

def preprocess_image(image_path, target_size=(256, 256), force=False, json_path=None):
    """
    Loads image, optionally crops using JSON outline, applies CLAHE, and resizes.
    Pipeline: [Crop+Rotate (if JSON)] → CLAHE → Letterbox Resize
    """
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path
    else:
        raise ValueError("Unsupported image type")

    # [Smart Skip] 이미 전처리가 완료된 해상도(256x256)라면 연산을 건너뜁니다.
    h, w = img.shape[:2]
    if (w, h) == target_size and not force:
        return img

    # Crop + Rotate (JSON 라벨이 있는 경우)
    if json_path is not None:
        pts = get_outline_points(json_path)
        if pts is not None:
            img = crop_and_rotate(img, pts)

    # Apply CLAHE (returns BGR)
    img_clahe = apply_clahe(img)

    # Resize Letterbox
    img_final = resize_letterbox(img_clahe, target_size)

    return img_final
