import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import cv2
import numpy as np

from anomalib.engine import Engine
from anomalib.models import Fastflow
# [Security Fix] PyTorch 2.4+ requires explicit trust for custom code
os.environ["TRUST_REMOTE_CODE"] = "1"

# [Backward Compatibility] Old models pickled with TunableFastflow
class TunableFastflow(Fastflow):
    def __init__(self, *args, **kwargs):
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
    Runs inference using Anomalib Engine.predict() API.
    """
    print("--------------------------------------------------")
    print(f"🚀 [Phase 2] Inference & Heatmap Generation")
    print(f"📦 Model Path: {model_path}")
    print(f"📂 Data Path: {data_path}")
    print(f"⚙️ Skip Preprocess: {skip_preprocess}")
    print("--------------------------------------------------")

    # 1. Device and Model Resolution
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        # Priority: model.pt (exported) -> model.ckpt (checkpoint)
        all_pt = list(model_path_obj.rglob("model.pt"))
        all_ckpt = list(model_path_obj.rglob("*.ckpt"))
        candidates = all_pt + all_ckpt
        if not candidates:
            print(f"❌ Error: No valid model file found in {model_path}")
            return
        actual_model_file = candidates[0]
        print(f"📍 Selected model for inference: {actual_model_file}")
    else:
        actual_model_file = model_path_obj

    # 2. Load model
    print(f"📥 Loading model from: {actual_model_file}")

    # Load based on file type
    if str(actual_model_file).endswith('.ckpt'):
        # Lightning checkpoint
        model = Fastflow.load_from_checkpoint(str(actual_model_file))
    else:
        # Torch export (.pt)
        checkpoint = torch.load(str(actual_model_file), map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        else:
            print(f"❌ Unexpected checkpoint format")
            return

    model.eval()
    print(f"✅ Model loaded successfully")

    # 3. Setup Engine
    engine = Engine(
        accelerator="auto",
        devices=1,
        default_root_dir=str(output_dir)
    )

    data_base = Path(data_path)
    output_base = Path(output_dir)

    # Category discovery
    categories = [d for d in data_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if (data_base / "test").exists():
        data_base = data_base / "test"
        categories = [d for d in data_base.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not categories:
        print(f"❌ Error: No category folders found in {data_base}")
        return

    print(f"🎯 Found categories: {[c.name for c in categories]}")

    # Forward hook to capture pre-normalization raw anomaly maps.
    # Anomalib's predict_step calls self.model(batch) (inner model) directly,
    # NOT self(batch), so the LightningModule's forward() is never called.
    # We must hook the inner model (model.model) to intercept anomaly maps.
    raw_prenorm_maps = []

    def _capture_prenorm(module, input, output):
        has_amap = hasattr(output, 'anomaly_map') and output.anomaly_map is not None
        if has_amap:
            paths = output.image_path if hasattr(output, 'image_path') else []
            for i in range(output.anomaly_map.shape[0]):
                amap = output.anomaly_map[i].detach().cpu().numpy()
                if amap.ndim == 3:
                    amap = amap.squeeze(0)
                path = paths[i] if i < len(paths) else f"unknown_{len(raw_prenorm_maps)}"
                raw_prenorm_maps.append({'path': path, 'anomaly_map': amap})
        else:
            print(f"  ⚠️ [Hook] output type={type(output).__name__}, "
                  f"has anomaly_map={hasattr(output, 'anomaly_map')}")

    # Hook inner model (model.model) first; fallback to model itself
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        hook_target = model.model
    else:
        hook_target = model
    hook_handle = hook_target.register_forward_hook(_capture_prenorm)
    print(f"📌 Pre-norm hook target: {type(hook_target).__name__}")

    # 4. Process each category
    for cat_dir in categories:
        cat_name = cat_dir.name
        cat_output = output_base / cat_name
        cat_output.mkdir(parents=True, exist_ok=True)

        print(f"🔍 Processing: {cat_name}")

        # Engine.predict with data_path
        try:
            predictions = engine.predict(
                model=model,
                data_path=str(cat_dir),
                return_predictions=True
            )
        except Exception as e:
            print(f"⚠️ Prediction failed for {cat_name}: {e}")
            continue

        # Extract results from list of batches
        for batch in predictions:
            # batch is ImageBatch
            paths = batch.image_path  # List of paths
            anomaly_maps = batch.anomaly_map  # (B, H, W) or (B, 1, H, W)
            pred_masks = batch.pred_mask if hasattr(batch, 'pred_mask') else None

            batch_size = len(paths)
            for i in range(batch_size):
                img_path = Path(paths[i])

                # Original image for overlay (read from disk)
                orig_img = cv2.imread(str(img_path))
                if orig_img is None:
                    continue

                # Anomaly map processing
                if anomaly_maps is not None:
                    amap = anomaly_maps[i].detach().cpu().numpy()
                    if amap.ndim == 3:  # (1, H, W)
                        amap = amap.squeeze(0)

                    # Save raw anomaly map (.npy) for downstream analysis
                    raw_dir = cat_output / "anomaly_maps_raw"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    np.save(str(raw_dir / f"{img_path.stem}.npy"), amap)

                    # Normalize to 0-255
                    min_val, max_val = amap.min(), amap.max()
                    if max_val - min_val > 0:
                        amap_norm = ((amap - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        amap_norm = np.zeros_like(amap, dtype=np.uint8)

                    # Resize to match original image if needed
                    if amap_norm.shape != orig_img.shape[:2]:
                        amap_norm = cv2.resize(amap_norm, (orig_img.shape[1], orig_img.shape[0]))

                    # Apply colormap and overlay
                    heatmap = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
                    cv2.imwrite(str(cat_output / f"heatmap_{img_path.name}"), overlay)

                # Pred mask processing
                if pred_masks is not None:
                    mask = pred_masks[i].detach().cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask.squeeze(0)

                    if mask.max() > 0:
                        if mask.shape != orig_img.shape[:2]:
                            mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

                        mask_overlay = orig_img.copy()
                        mask_overlay[mask > 0.5] = [0, 0, 255]  # Red
                        overlay_mask = cv2.addWeighted(orig_img, 0.7, mask_overlay, 0.3, 0)
                        cv2.imwrite(str(cat_output / f"mask_{img_path.name}"), overlay_mask)

        # Save pre-normalization raw anomaly maps for this category
        if raw_prenorm_maps:
            raw_prenorm_dir = cat_output / "anomaly_maps_raw_prenorm"
            raw_prenorm_dir.mkdir(parents=True, exist_ok=True)
            for entry in raw_prenorm_maps:
                stem = Path(entry['path']).stem
                np.save(str(raw_prenorm_dir / f"{stem}.npy"), entry['anomaly_map'])
            print(f"  📊 Saved {len(raw_prenorm_maps)} pre-normalization raw maps")
            raw_prenorm_maps.clear()

    hook_handle.remove()

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
