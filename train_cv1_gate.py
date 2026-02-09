import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

# [설정] 데이터셋 경로
DATASET_ROOT = "dataset_cv1"
RESULT_PATH = "results_cv1_purified"

def train_gate_model():
    print("========================================")
    print("      CV-1 Gate Model Training (Prototype)")
    print("      Model: PatchCore (State-of-the-Art)")
    print("========================================")

    # 1. Check Data
    train_good_path = os.path.join(DATASET_ROOT, "train", "good")
    if not os.path.exists(train_good_path):
        print(f"[!] Error: Data folder not found: {train_good_path}")
        print("    Please run 'organize_dataset.py' first.")
        return

    num_images = len(os.listdir(train_good_path))
    print(f"[*] Training Images (Normal): {num_images} found.")
    
    if num_images < 5:
        print("[!] Warning: Too few images. Need at least 5-10 normal images.")
        # Just proceed for demo
    
    # 2. Setup Data Module
    print("[*] Setting up Data Module...")
    datamodule = Folder(
        name="battery_gate",
        root=DATASET_ROOT,
        normal_dir="train/good",
        normal_test_dir="test/good", 
        abnormal_dir="test/defect",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        val_split_mode="none" # [FIX] Skip val split (No defects available)
    )
    datamodule.setup()

    # 3. Setup Model
    print("[*] Initializing PatchCore Model...")
    model = Patchcore(
        backbone="resnet18", 
        pre_trained=True,
    )

    # 4. Train
    print("[*] Starting Training... (This may take a while)")
    engine = Engine(
        default_root_dir=RESULT_PATH,
        max_epochs=1, 
        accelerator="auto", 
        devices=1,
        limit_val_batches=0, # [FIX] Skip validation loop
        limit_test_batches=0, # [FIX] Skip test loop
        num_sanity_val_steps=0, # [FIX] Skip sanity check
        enable_progress_bar=False, # [FIX] Avoid CP949 encoding error
        enable_model_summary=False,
        logger=False, # [FIX] Disable wandb logic
    )
    
    # Train
    engine.fit(datamodule=datamodule, model=model)
    
    print("\n[+] Training Complete!")
    print(f"    Results saved to: {RESULT_PATH}")

if __name__ == "__main__":
    train_gate_model()
