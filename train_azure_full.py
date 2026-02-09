
import sys
import os
import shutil
import zipfile
import glob
from azure.storage.blob import ContainerClient
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch.callbacks import Callback

# ==========================================
# CONFIGURATION
# ==========================================
# [Azure Settings]
# We will use Azure ML Workspace & Datastore
# No manual SAS URL needed anymore!

# [Training Settings]
TEMP_DATA_DIR = os.path.join(os.getcwd(), "temp_dataset") 
RESULT_PATH = "outputs"

def download_and_extract_zips(target_dir):
    """
    Downloads data using Azure ML Datastore (verified method).
    """
    print(f"[*] Connecting to Azure ML Workspace...")
    try:
        from azureml.core import Workspace, Datastore, Dataset
        ws = Workspace.from_config()
        ds = Datastore.get(ws, 'battery_storage')
        
        # Define path (Update this if you want to download ALL zips or specific ones)
        # For full training, we usually want all zips in the folder or a specific list.
        # Here we target the specific folder path we found earlier.
        print("[*] Accessing Datastore 'battery_storage'...")
        
        # Example: Download everything under the 'raw_zips' container or specific path
        # If you uploaded zips to a specific path in the container, simplify it here.
        # NOTE: Adjust 'path' pattern to match your blob structure.
        # e.g. '/*/*.zip' or specific path
        target_path = '103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/*.zip'
        
        print(f"[*] Downloading ZIPs from: {target_path}")
        dataset = Dataset.File.from_files(path=(ds, target_path))
        
        # Download
        download_paths = dataset.download(target_path=target_dir, overwrite=True)
        print(f"[*] Downloaded {len(download_paths)} files.")
        
        # Extract
        for zip_path in download_paths:
            print(f"    Extracting {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            # Remove zip to save space
            os.remove(zip_path)
            
        print("[*] Extraction Complete.")
            
    except ImportError:
        print("[!] Azure ML SDK not installed or configured.")
    except Exception as e:
        print(f"[!] Error during download: {e}")

def run_training():
    print("========================================")
    print("      Azure Large-Scale Training")
    print("      Model: PatchCore (Optimized)")
    print("========================================")

    # 1. Prepare Data
    if not os.path.exists(TEMP_DATA_DIR) or len(os.listdir(TEMP_DATA_DIR)) == 0:
        download_and_extract_zips(TEMP_DATA_DIR)
    else:
        print("[*] Data directory already exists. Skipping download.")
    
    # 2. Organize Data Paths (Auto-discovery)
    # The Azure folder structure is deep (103.Battery.../...)
    # We need to find where the actual images are.
    
    # Let's assume after unzip we have many folders. 
    # We need to consolidate them or point anomalib to the right place.
    # For simplicity in this script, we'll try to find the 'good' folder.
    
    data_root = find_image_root(TEMP_DATA_DIR)
    print(f"[*] Detected Data Root: {data_root}")

    # 3. Setup Anomalib Data Module
    # Note: Anomalib expects root/normal_dir structure.
    # If the structure is complex, we might need a custom CSV, but let's try Folder first.
    
    datamodule = Folder(
        name="battery_full",
        root=data_root,
        normal_dir="good",      # Assumes we found the parent of 'good'
        abnormal_dir="defect",  # Placeholder
        normal_test_dir=None,   # Use split if needed
        train_batch_size=32,       
        eval_batch_size=32,
        num_workers= os.cpu_count() or 4,
        image_size=(256, 256)
    )
    
    # 4. Model (Memory Optimized)
    model = Patchcore(
        backbone="wide_resnet50_2", 
        pre_trained=True,
        coreset_sampling_ratio=0.01 
    )

    # 5. Engine
    engine = Engine(
        default_root_dir=RESULT_PATH,
        max_epochs=1,
        accelerator="auto", # GPU if available
        devices=1,
        enable_progress_bar=True
    )
    
    engine.fit(datamodule=datamodule, model=model)

if __name__ == "__main__":
    # For local test, create dummy data
    if not os.path.exists(TEMP_DATA_DIR):
        os.makedirs(os.path.join(TEMP_DATA_DIR, "train/good"), exist_ok=True)
        
    run_training()
