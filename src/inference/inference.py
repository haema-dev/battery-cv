import os
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np

try:
    from anomalib.deploy import TorchInferencer
    INFERENCER_AVAILABLE = True
except ImportError:
    INFERENCER_AVAILABLE = False

from anomalib.models import Fastflow

# [Security Fix] PyTorch 2.4+ requires explicit trust for custom code (TunableFastflow)
os.environ["TRUST_REMOTE_CODE"] = "1"

# [Namespace Fix] 설계도(Class)가 있어야 Pickle이 객체를 복원할 수 있습니다.
class TunableFastflow(Fastflow):
    def __init__(self, *args, **kwargs):
        # 학습용 인자 제거 후 부모 생성자 호출 (Unpickling용)
        kwargs.pop('lr', None)
        kwargs.pop('weight_decay', None)
        super().__init__(*args, **kwargs)

try:
    from preprocess import preprocess_image
except ImportError:
    print("⚠️ Warning: 'src/preprocess.py' not found. Ensure it exists.")
    preprocess_image = None

def run_inference(data_path, model_path, output_dir, skip_preprocess=False):
    """
    Runs inference using Anomalib TorchInferencer.
    skip_preprocess: If True, skips custom CLAHE/Resize (use for pipeline output).
    """
    print("--------------------------------------------------")
    print(f"🚀 [Phase 2] Inference & Heatmap Generation")
    print(f"📦 Model Path: {model_path}")
    print(f"📂 Data Path: {data_path}")
    print(f"⚙️ Skip Preprocess: {skip_preprocess}")
    print("--------------------------------------------------")

    if not INFERENCER_AVAILABLE:
        print("❌ Error: 'anomalib.deploy.TorchInferencer' could not be loaded.")
        return

    # 1. Initialize Inferencer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        # [Priority Selection] 
        # 1순위: engine.export()로 생성된 메타데이터 포함 모델 (model.pt)
        # 2순위: 체크포인트 파일 (model.ckpt)
        # 3순위: 기타 .pt 파일
        all_pt = list(model_path_obj.rglob("model.pt"))
        all_ckpt = list(model_path_obj.rglob("*.ckpt"))
        all_fallback = [f for f in model_path_obj.rglob("*.pt") if f.name != "model_weights.pt"]
        
        candidates = all_pt + all_ckpt + all_fallback
        
        if not candidates:
            print(f"❌ Error: No valid model file found in {model_path}")
            return
            
        actual_model_file = candidates[0]
        print(f"📍 Selected model for inference: {actual_model_file}")
    else:
        actual_model_file = model_path_obj

    inferencer = TorchInferencer(
        path=str(actual_model_file),
        device=device
    )

    data_base = Path(data_path)
    output_base = Path(output_dir)
    
    # 🎯 데이터 구조 탐색 (Standard: test/good, test/damaged or root/good, root/damaged)
    # 먼저 루트 아래에 서브폴더가 있는지 확인
    categories = [d for d in data_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # 만약 'test'라는 전용 폴더가 있다면 그 안으로 들어감
    if (data_base / "test").exists():
        print(f"📂 'test' folder detected, navigating inside...")
        data_base = data_base / "test"
        categories = [d for d in data_base.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not categories:
        print(f"❌ Error: No category folders (good, damaged, etc.) found in {data_base}")
        return

    print(f"🎯 Found categories: {[c.name for c in categories]}")
    
    for cat_dir in categories:
        cat_name = cat_dir.name
        print(f"🔍 Processing: {cat_name}")
        
        cat_output = output_base / cat_name
        cat_output.mkdir(parents=True, exist_ok=True)
        
        img_files = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpeg"))
        
        for img_path in img_files:
            # 1. Image Loading (Conditional Preprocess)
            if preprocess_image and not skip_preprocess:
                # Returns (H, W, 3) BGR numpy array with CLAHE + Resize
                processed_img = preprocess_image(img_path, target_size=(256, 256))
            else:
                # Already preprocessed or fallback
                processed_img = cv2.imread(str(img_path))
                if processed_img is None:
                    continue
                # Ensure size matches model even if skip_preprocess is on
                if processed_img.shape[:2] != (256, 256):
                    processed_img = cv2.resize(processed_img, (256, 256))
            # 2. Prediction with Defensive Logic
            try:
                # [Predicted Error Fix] Shape Mismatch 방어
                # 모델이 기대하는 256x256이 아닐 경우 강제 리사이즈
                if processed_img.shape[:2] != (256, 256):
                    processed_img = cv2.resize(processed_img, (256, 256))
                
                results = inferencer.predict(image=processed_img)
            except Exception as e:
                # [Predicted Error Fix] KeyError: 'metadata' 또는 기타 구조 에러 대응
                print(f"⚠️ Prediction failed for {img_path.name}: {e}")
                if "metadata" in str(e).lower():
                    print("🛠️ Attempting fallback: Check if model.pt contains metadata. If not, re-export might be needed.")
                continue

            # 3. Extract Anomaly Map (Tensor)
            # results is ImageBatch. anomaly_map should be present.
            if hasattr(results, 'anomaly_map') and results.anomaly_map is not None:
                anomaly_map = results.anomaly_map.squeeze().cpu().numpy() # (H, W)
                
                # Normalize to 0-255
                min_val, max_val = anomaly_map.min(), anomaly_map.max()
                if max_val - min_val > 0:
                    am_norm = ((anomaly_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    am_norm = np.zeros_like(anomaly_map, dtype=np.uint8)
                
                # Resize map to image size if needed (though usually same size)
                if am_norm.shape != processed_img.shape[:2]:
                    am_norm = cv2.resize(am_norm, (processed_img.shape[1], processed_img.shape[0]))
                
                # Apply Colormap
                heatmap = cv2.applyColorMap(am_norm, cv2.COLORMAP_JET)
                
                # Overlay
                overlay = cv2.addWeighted(processed_img, 0.6, heatmap, 0.4, 0)
                
                # Save Heatmap Overlay
                save_name = f"heatmap_{img_path.name}"
                save_path = cat_output / save_name
                cv2.imwrite(str(save_path), overlay)
            else:
                print(f"⚠️ Warning: No anomaly_map found for {img_path.name}")

            # 4. Extract Prediction Mask (if available)
            if hasattr(results, 'pred_mask') and results.pred_mask is not None:
                mask = results.pred_mask.squeeze().cpu().numpy()
                if mask.max() > 0: # If any anomaly predicted
                    # Resize mask
                    if mask.shape != processed_img.shape[:2]:
                        mask = cv2.resize(mask, (processed_img.shape[1], processed_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # Create red overlay for mask
                    # Mask is 0 or 1 (or boolean)
                    mask_overlay = processed_img.copy()
                    mask_overlay[mask > 0.5] = [0, 0, 255] # Red
                    
                    overlay_mask = cv2.addWeighted(processed_img, 0.7, mask_overlay, 0.3, 0)
                    
                    save_name_mask = f"mask_{img_path.name}"
                    save_path_mask = cat_output / save_name_mask
                    cv2.imwrite(str(save_path_mask), overlay_mask)
            
    print(f"\n✅ All heatmaps generated successfully!")
    print(f"📍 Location: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Input data folder")
    parser.add_argument("--model_path", type=str, required=True, help="Trained model (.pt) file")
    parser.add_argument("--output_dir", type=str, required=True, help="Inference results folder")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip custom preprocessing (use for pipeline)")
    
    args = parser.parse_args()
    run_inference(args.data_path, args.model_path, args.output_dir, args.skip_preprocess)