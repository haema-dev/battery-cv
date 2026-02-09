
import os
import glob
import shutil
import zipfile
import fnmatch

def extract_and_organize(source_root, target_root, file_pattern, filter_csv=None):
    """
    Extracts ZIP files matching 'file_pattern' from 'source_root' to 'target_root'.
    Organizes them into Train/Test based on filename conventions (TS_, VS_).
    
    Args:
        source_root (str): Path to mounted Azure Blob data.
        target_root (str): Local path to extract data to.
        file_pattern (str): Glob pattern for ZIP selection (e.g. '*_4.zip').
        filter_csv (str, optional): Path to 'good_list.csv' for filtering. (Not implemented yet)
    """
    print(f"[*] Extracting data from {source_root} to {target_root}...")
    print(f"[*] Filter Pattern: {file_pattern}")
    
    if filter_csv:
        print(f"[!] Warning: CSV filtering with '{filter_csv}' is mentioned in design but not fully implemented. Proceeding with pattern match.")
    
    # 1. Setup folders
    train_dir = os.path.join(target_root, "train", "good")
    test_good_dir = os.path.join(target_root, "test", "good")
    test_defect_dir = os.path.join(target_root, "test", "defect")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defect_dir, exist_ok=True)
    
    # 2. Find Zips with Pattern
    # Use glob to find all zips, then filter by filename pattern
    all_zips = glob.glob(os.path.join(source_root, "**/*.zip"), recursive=True)
    zips = [z for z in all_zips if fnmatch.fnmatch(os.path.basename(z), file_pattern)]
    
    print(f"[*] Found {len(zips)} zip files matching pattern (out of {len(all_zips)} total).")
    
    for zip_path in zips:
        filename = os.path.basename(zip_path)
        print(f"    Processing: {filename}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Decide target based on filename (TS=Train, VS=Validation/Test)
                if "TS_" in filename:
                    # Training Data (All Normal)
                    zip_ref.extractall(train_dir)
                
                elif "VS_" in filename:
                    # Validation Data (Normal + Defect)
                    temp_vs = os.path.join(target_root, "temp_vs")
                    zip_ref.extractall(temp_vs)
                    
                    for root, _, files in os.walk(temp_vs):
                        for f in files:
                            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                                continue
                            full_path = os.path.join(root, f)
                            lower_path = full_path.lower()
                            
                            if "normal" in lower_path or "양품" in lower_path:
                                shutil.move(full_path, os.path.join(test_good_dir, f))
                            else:
                                shutil.move(full_path, os.path.join(test_defect_dir, f))
                                
                    shutil.rmtree(temp_vs) # Cleanup
                    
        except Exception as e:
            print(f"[!] Error extracting {filename}: {e}")

    # Check counts
    n_train = len(os.listdir(train_dir))
    n_test_good = len(os.listdir(test_good_dir))
    n_test_defect = len(os.listdir(test_defect_dir))
    
    # [Auto-Split Logic]
    # If we have TS data but NO VS data, the model training might fail or we can't validate.
    # We will move 10% of Train -> Test/Good to allow validation to run.
    if n_train > 0 and n_test_good == 0 and n_test_defect == 0:
        print("[!] No Validation (VS) data found. Executing Auto-Split (10% of Train -> Test/Good)...")
        train_files = os.listdir(train_dir)
        import random
        # Move 10% or at least 1 file
        split_count = max(1, int(len(train_files) * 0.1))
        move_files = random.sample(train_files, split_count)
        
        for f in move_files:
            shutil.move(os.path.join(train_dir, f), os.path.join(test_good_dir, f))
            
        n_train -= split_count
        n_test_good += split_count
        print(f"    -> Moved {split_count} files to Test/Good.")

    print(f"[*] Data Prepared:")
    print(f"    Train (Good): {n_train}")
    print(f"    Test (Good) : {n_test_good}")
    print(f"    Test (Defect): {n_test_defect}")
    
    if n_train == 0:
        # If absolutely no data, that's an error
        raise RuntimeError("No training data found! Check 'TS_' logic or file pattern.")
        
    return target_root
