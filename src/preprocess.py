import cv2
import numpy as np
from pathlib import Path

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

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Loads image, applies CLAHE, and resizes to target_size with letterbox.
    Returns: numpy array (H, W, 3) in BGR format and the loaded original image (for visualization if needed, ensuring size match) but here we return processed.
    """
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path
    else:
        raise ValueError("Unsupported image type")
    
    # Apply CLAHE (returns BGR)
    img_clahe = apply_clahe(img)
    
    # Resize Letterbox
    img_final = resize_letterbox(img_clahe, target_size)
    
    return img_final
