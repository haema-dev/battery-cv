import os
import sys
import io
import shutil
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

# [Fix] Windows Encoding Issue (CP949)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# [설정] 데이터셋 경로
DATASET_ROOT = "dataset_cv1"
RESULT_PATH = "results_cv1_fixed"
CHECKPOINT_PATH = r"C:\Users\EL98\Downloads\BatterySample\results_cv1\Patchcore\battery_gate\latest\weights\lightning\model.ckpt"

def validate_and_fix():
    print("========================================")
    print("      Model Validation & Fix Tool")
    print("========================================")

    # 1. Setup Data Module
    print("[*] Setting up Data Module...")
    datamodule = Folder(
        name="battery_gate",
        root=DATASET_ROOT,
        normal_dir="train/good",
        normal_test_dir="test/good", 
        abnormal_dir="test/defect",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0 
    )
    datamodule.setup()

    # 2. Load Model
    print(f"[*] Loading Checkpoint: {CHECKPOINT_PATH}")
    model = Patchcore.load_from_checkpoint(CHECKPOINT_PATH)
    
    # 3. Validate (This calculates thresholds and stats)
    print("[*] Starting Validation to Compute Stats...")
    engine = Engine(
        default_root_dir=RESULT_PATH,
        accelerator="auto",
        devices=1,
    )
    
    # Force validation
    engine.test(model=model, datamodule=datamodule)
    
    # [Fix] Explicitly save the updated model with calculated stats
    print("[*] Saving repaired model...")
    save_path = os.path.join(RESULT_PATH, "fixed_model.ckpt")
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    engine.trainer.save_checkpoint(save_path)
    
    print("\n[+] Validation Complete!")
    print(f"    Check the new checkpoint in {save_path}")

if __name__ == "__main__":
    validate_and_fix()
