import sys
import os
import torch
import warnings
from torch.utils.data import DataLoader
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType
import cv2
import numpy as np
from torchvision.transforms.v2 import Resize, Compose, ToDtype, Normalize
import torchvision.transforms.v2.functional as F

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
CHECKPOINT_PATH = r"C:\Users\EL98\Downloads\BatterySample\results_cv1_purified\Patchcore\battery_gate\latest\weights\lightning\model.ckpt"
IMAGE_SIZE = (256, 256)

def load_model():
    print(f"[*] Loading model from: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[!] Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)
        
    model = Patchcore.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    
    # [CRITICAL FIX] Force Disable Internal Normalization
    # The checkpoint has broken stats (min=max) causing all scores to become 1.0 or 0.0.
    # We strip these to get RAW anomaly distances.
    if hasattr(model, "normalization_metrics"):
        model.normalization_metrics = None
    if hasattr(model, "image_threshold"):
        model.image_threshold = None
    if hasattr(model, "pixel_threshold"):
        model.pixel_threshold = None
    if hasattr(model, "min_max"):
        model.min_max = None
        
    return model

def preprocess_image(image_path, target_size=None):
    # print(f"[Debug] Reading image: {image_path}")
    # Read with CV2 (Unicode Path Support)
    try:
        if not os.path.exists(image_path):
            print(f"[!] Path does not exist: {image_path}")
            return None
            
        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        stream.close()
    except Exception as e:
        print(f"[!] Error reading file: {e}")
        return None
        
    if img is None:
        return None
    
    # print("[Debug] Image read success. Converting to tensor...")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Manual Transform
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
    if target_size is not None:
        # target_size is (H, W) for F.resize
        tensor = F.resize(tensor, target_size, interpolation=F.InterpolationMode.BICUBIC)
    
    # Standard ImageNet normalization (PatchCore default)
    tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return tensor.unsqueeze(0) # Batch dim

def infer_single_pass(model, input_tensor):
    with torch.no_grad():
        raw_output = model.model(input_tensor)
        
    score = 0.0
    if isinstance(raw_output, tuple) and len(raw_output) >= 2:
        score = raw_output[0].max().item() # Use Map Max
    elif isinstance(raw_output, dict):
         if "anomaly_map" in raw_output:
            score = raw_output["anomaly_map"].max().item()
         elif "pred_score" in raw_output:
            score = raw_output["pred_score"].item()
    elif isinstance(raw_output, torch.Tensor):
        score = raw_output.max().item()
    return score

def infer(image_path, model, threshold):
    print(f"[*] Analyzing: {os.path.basename(image_path)}")
    
    # [FIX] Lock resolution to Training Size (256x256)
    # The model triggers False Positives on OOD resolutions (Original/224)
    target_size = (256, 256)
    
    input_tensor = preprocess_image(image_path, target_size=target_size)
    if input_tensor is None: 
        print("[!] Failed to preprocess image.")
        return

    final_score = infer_single_pass(model, input_tensor)
    
    # ---------------------------------------------------------
    # DECISION LOGIC
    # ---------------------------------------------------------
    is_anomaly = final_score >= threshold
    label = "DEFECT (불량)" if is_anomaly else "NORMAL (정상)"
    
    result_color = "\033[91m" if is_anomaly else "\033[92m" # Red / Green
    reset_color = "\033[0m"
    
    print("\n" + "="*40)
    print(f" >> RESULT: {result_color}{label}{reset_color}")
    print(f" >> Final Score : {final_score:.4f}")
    print(f" >> Threshold   : {threshold:.4f}")
    print("="*40)

# --- SELF CALIBRATION ---
def calibrate_threshold(model):
    print("\n[*] Calibrating Threshold with Normal Images...")
    normal_dir = r"C:\Users\EL98\Downloads\BatterySample\dataset_cv1\train\good"
    if not os.path.exists(normal_dir):
        print(f"[!] Warning: Normal dir not found at {normal_dir}. Using default threshold 0.5")
        return 0.5

    files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("[!] No normal images found for calibration.")
        return 0.5
        
    # [OPTIMIZATION] Reduce sample size for speed (20 -> 4)
    files = files[:4]
    
    scores = []
    print(f"    Scanning {len(files)} normal samples (Fast Mode)...")
    
    for p in files:
        # Calibration relies on 256x256 as baseline
        input_tensor = preprocess_image(p, target_size=(256, 256))
        if input_tensor is None: continue
        
        score = infer_single_pass(model, input_tensor)
        scores.append(score)
        
    if not scores:
        print("[!] Failed to extract scores from calibration images.")
        return 0.5
        
    max_score = max(scores)
    mean_score = sum(scores) / len(scores)
    
    # Heuristic: Threshold = Max Normal + Gap
    # [Adjustment] Found outliers up to 6.4 in validation. Increasing margin.
    calibrated_threshold = max_score * 1.7 
    if calibrated_threshold == 0: calibrated_threshold = 0.5 # Safety for perfect 0

    print(f"    [Calibration] Normal Scores - Max: {max_score:.4f}, Mean: {mean_score:.4f}")
    print(f"    [Calibration] New Dynamic Threshold: {calibrated_threshold:.4f}")
    
    return calibrated_threshold

if __name__ == "__main__":
    model = load_model()
    
    # [RESTOREED] Fast Calibration (4 samples)
    DYNAMIC_THRESHOLD = calibrate_threshold(model)
    # DYNAMIC_THRESHOLD = 8.0
    # print(f"\n[*] Using Fixed Threshold: {DYNAMIC_THRESHOLD} (Fast Mode)")

    # [Fix Removed] sys.stdin reconfigure might cause issues. Removed.

    if len(sys.argv) < 2:
        print("\n[*] Interactive Mode Started")
        print("    (Tip: Drag & Drop file, then PRESS ENTER!)")
        
    if len(sys.argv) < 2:
        print("\n[*] Interactive Mode Started", flush=True)
        
        while True:
            print("\n" + "-"*40, flush=True)
            try:
                # Cleaner input prompt
                print("[?] Drop image file here (or type 'q' to quit) > ", end="", flush=True)
                user_input = input().strip()
            except EOFError:
                break
            
            if not user_input:
                continue
                
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
                
            print(f"    >> Input received: {user_input[:20]}...", flush=True)

            path = user_input.strip('"\'')
            if not os.path.exists(path):
                print(f"[!] File not found: {path}", flush=True)
                continue
                
            infer(path, model, DYNAMIC_THRESHOLD)
    else:
        path = sys.argv[1]
        infer(path, model, DYNAMIC_THRESHOLD)
        input("Press Enter to close...")
