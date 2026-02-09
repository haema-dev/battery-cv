import pandas as pd
import zipfile
import os
import shutil
import sys
import random

# --- CONFIG ---
CSV_FILE = "sample_data_v1/TS_Exterior_Img_Datasets_images_1_test_500.csv" # Default, can be overridden
ZIP_FILE_PATH = r"C:\Users\EL98\Downloads\TS_Exterior_Img_Datasets_images_csv.zip" # Update if needed, or ask user. 
# actually user just extracted csv, the big zip is:
# C:\Users\EL98\Downloads\TS_Exterior_Img_Datasets_images_csv.zip based on previous screenshot
# Wait, user screenshot says "battery_id, data_type...". The file name in "Archive Inspector" was TS_Exterior_Img_Datasets_images_1_test_500.csv
# But the ZIP they inspected was 97GB C:\Users\EL98\Downloads\TS_Exterior_Img_Datasets_images_csv.zip
# I will try to auto-detect the big zip or ask user.
# For now, I will scan the download dir for the big zip.

BASE_OUTPUT_DIR = "dataset_cv1"

def find_large_zip():
    # Heuristic: look for the largest zip in Downloads
    dl_dir = r"C:\Users\EL98\Downloads"
    candidates = []
    print(f"[*] Searching for 97GB Archive in {dl_dir}...")
    for f in os.listdir(dl_dir):
        if f.lower().endswith(".zip"):
            path = os.path.join(dl_dir, f)
            try:
                size_gb = os.path.getsize(path) / (1024**3)
                if size_gb > 10: # Larger than 10GB
                    candidates.append((path, size_gb))
            except:
                pass
    
    if not candidates:
        print("[!] Could not auto-find the large ZIP.")
        return input("Please enter full path to the 97GB ZIP file: ").strip().strip('"')
    
    # Return largest
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0][0]
    print(f"[*] Found large archive: {best} ({candidates[0][1]:.2f} GB)")
    return best

def build_dataset():
    # 1. Setup paths
    zip_path = find_large_zip()
    if not os.path.exists(zip_path):
        print(f"[!] Zip file not found: {zip_path}")
        return

    # User input for CSV if not found
    csv_path = CSV_FILE
    if not os.path.exists(csv_path):
        # Look in current dir
        files = [f for f in os.listdir(".") if f.endswith(".csv")]
        if files:
            csv_path = files[0]
        else:
            # Look in sample_data_v1
            if os.path.exists("sample_data_v1"):
                files = [f for f in os.listdir("sample_data_v1") if f.endswith(".csv")]
                if files:
                    csv_path = os.path.join("sample_data_v1", files[0])
    
    if not os.path.exists(csv_path):
        print("[!] CSV file not found. Please drag and drop the CSV file here:")
        csv_path = input().strip().strip('"')

    print(f"[*] Using CSV: {csv_path}")
    
    # 2. Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[!] Error reading CSV: {e}")
        return

    # Check columns
    required_cols = ['file_name', 'is_normal']
    for c in required_cols:
        if c not in df.columns:
            print(f"[!] Missing column '{c}' in CSV. Found: {list(df.columns)}")
            return

    # 3. Prepare ZIP map
    print(f"[*] Reading ZIP file structure (this may take a moment)...")
    zf = zipfile.ZipFile(zip_path, 'r')
    # Map filename (basename) to full path in zip
    zip_files_map = {os.path.basename(f): f for f in zf.namelist()}
    
    print(f"[*] Zip contains {len(zip_files_map)} files.")

    # 4. Process
    print("[*] Processing images...")
    
    # Counters
    counts = {"train_good": 0, "val_good": 0, "test_good": 0, "val_defect": 0, "test_defect": 0}
    
    # Clean old dataset?
    # shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True) 
    
    success_count = 0
    missing_count = 0
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        fname = row['file_name']
        is_normal = row['is_normal'] # Expecting TRUE/FALSE or Boolean or 0/1
        
        # Normalize boolean
        if isinstance(is_normal, str):
            is_normal = is_normal.upper() == 'TRUE'
        
        # Find in zip
        if fname not in zip_files_map:
            missing_count += 1
            if missing_count < 10:
                print(f"    [!] Missing in Zip: {fname}")
            continue
            
        full_zip_path = zip_files_map[fname]
        
        # Decide Target
        if is_normal:
            # 80% Train, 10% Val, 10% Test
            rand = random.random()
            if rand < 0.8:
                target_subnet = "train/good"
                key = "train_good"
            elif rand < 0.9:
                target_subnet = "val/good"
                key = "val_good"
            else:
                target_subnet = "test/good"
                key = "test_good"
        else:
            # 50% Val, 50% Test (No train for anomaly detection)
            rand = random.random()
            if rand < 0.5:
                target_subnet = "val/defect"
                key = "val_defect"
            else:
                target_subnet = "test/defect"
                key = "test_defect"
        
        # Extract
        target_dir = os.path.join(BASE_OUTPUT_DIR, target_subnet)
        os.makedirs(target_dir, exist_ok=True)
        
        target_path = os.path.join(target_dir, fname)
        
        if not os.path.exists(target_path):
            with zf.open(full_zip_path) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
        
        counts[key] += 1
        success_count += 1
        
        if success_count % 100 == 0:
            print(f"    Processed {success_count}/{total_rows} images...")

    zf.close()
    
    print("\n" + "="*30)
    print("      DATASET BUILD COMPLETE      ")
    print("="*30)
    print(f"Total Processed: {success_count}")
    print(f"Missing Files:   {missing_count}")
    print("-" * 20)
    print(f"Train Good:   {counts['train_good']}")
    print(f"Val Good:     {counts['val_good']}")
    print(f"Test Good:    {counts['test_good']}")
    print("-" * 20)
    print(f"Val Defect:   {counts['val_defect']}")
    print(f"Test Defect:  {counts['test_defect']}")
    print("="*30)
    print("You can now run 'run_modeling.bat' again!")

if __name__ == "__main__":
    build_dataset()
