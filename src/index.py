
import argparse
import os
import glob
import shutil
import zipfile
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

# Anomalib 2.2.0 imports
try:
    from anomalib.data import Folder
    from anomalib.models import Fastflow
    from anomalib.engine import Engine
except ImportError as e:
    print(f"[!] Import Error: {e}")
    print("    Ensure anomalib==2.2.0 is installed.")
    raise e

import extractor # Imported from local src/extractor.py

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted Azure Blob data")
    parser.add_argument("--sas_token", type=str, default="", help="SAS Token for Blob Access (Optional)")
    parser.add_argument("--blob_path", type=str, default="*.zip", help="Target Blob Path or Pattern (e.g. 'TS_....zip')")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Path to save outputs")
    parser.add_argument("--filter_csv", type=str, default=None, help="Path to good_list.csv (Optional)")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Prepare Data (using extractor module)
    local_dataset_root = "/tmp/dataset" 
    if os.path.exists(local_dataset_root):
        shutil.rmtree(local_dataset_root)
        
    # Call the external extractor logic
    # Note: 'blob_path' acts as the file pattern for local mounting
    extractor.extract_and_organize(
        source_root=args.data_path, 
        target_root=local_dataset_root, 
        file_pattern=args.blob_path,
        filter_csv=args.filter_csv
    )
    
    # 2. Data Module (Anomalib 2.2 Compatible)
    # Using 'Folder' for custom dataset structure
    datamodule = Folder(
        name="battery_cloud",
        root=local_dataset_root,
        normal_dir="train/good",   
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        train_batch_size=32,
        eval_batch_size=32,
        image_size=(256, 256),
        num_workers=os.cpu_count(),
        task="segmentation" # Explicitly define task for 2.x
    )
    
    # 3. Model (FastFlow)
    model = Fastflow(
        backbone="resnet18",
        flow_steps=8,
    )
    
    # 4. Train
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    engine = Engine(
        default_root_dir=output_dir,
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        task="segmentation",
    )
    
    print("[*] Starting Training with Anomalib Engine...")
    engine.fit(datamodule=datamodule, model=model)
    
    # 5. Test (metrics)
    print("[*] Running Evaluation...")
    try:
        engine.test(datamodule=datamodule, model=model)
    except Exception as e:
        print(f"[!] Test failed (possibly no test data): {e}")
        
    print("[*] Done. Check 'Outputs + Logs' in Azure ML Studio.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[!] Critical Error in Cloud Script: {e}")
        raise e
