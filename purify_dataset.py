
import os
import shutil
import pandas as pd
import glob

# Config
TRUTH_FILE = "truth.csv"
DATASET_ROOT = "dataset_cv1"
EXCLUDED_ROOT = "dataset_cv1_excluded"

# Load Ground Truth
if not os.path.exists(TRUTH_FILE):
    print(f"[!] Truth file {TRUTH_FILE} not found.")
    exit(1)

df = pd.read_csv(TRUTH_FILE)
# Ensure clean string IDs
valid_ids = set(df['battery_id'].astype(str))
print(f"[*] Loaded {len(valid_ids)} verified IDs from Ground Truth.")

# Target Folders to Purify (Only Good folders, as we want to clean the training set)
target_dirs = [
    os.path.join(DATASET_ROOT, "train", "good"),
    os.path.join(DATASET_ROOT, "test", "good"),
]

moved_count = 0
if not os.path.exists(EXCLUDED_ROOT):
    os.makedirs(EXCLUDED_ROOT)

print(f"[*] Scanning for unverified data to quarantine...")

for folder in target_dirs:
    if not os.path.exists(folder):
        continue
        
    # Recursive search not needed for this structure, just simple glob
    files = glob.glob(os.path.join(folder, "*.png"))
    print(f"    Scanning {folder} ({len(files)} items)...")
    
    for f_path in files:
        fname = os.path.basename(f_path)
        # Format: RGB_cell_cylindrical_0515_002.png
        parts = fname.split('_')
        
        should_move = False
        
        if len(parts) >= 5:
            # Extract ID usually at index 3: ['RGB', 'cell', 'cylindrical', '0515', '002.png']
            bat_id_str = parts[3]
            bat_id_int = str(int(bat_id_str)) # 0515 -> 515
            
            if bat_id_int not in valid_ids:
                should_move = True
                # print(f"Quarantining {fname} (ID {bat_id_int} not in whitelist)")
        
        if should_move:
            # Mirror structure in excluded root
            # parent folder name: 'good'
            # grand parent: 'train' or 'test'
            parent = os.path.basename(os.path.dirname(f_path))
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(f_path)))
            
            dest_dir = os.path.join(EXCLUDED_ROOT, grandparent, parent)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            dest_path = os.path.join(dest_dir, fname)
            try:
                shutil.move(f_path, dest_path)
                moved_count += 1
                if moved_count % 50 == 0:
                    print(f"    Moved {moved_count} files...", end='\r')
            except Exception as e:
                print(f"    [!] Error moving {fname}: {e}")

print(f"\n[+] Purification Complete.")
print(f"    Total files moved to {EXCLUDED_ROOT}: {moved_count}")
print(f"    The dataset {DATASET_ROOT} contains ONLY verified IDs now.")
