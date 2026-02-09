
import os
import pandas as pd
import glob

# Load Ground Truth
truth_file = "truth.csv"
if not os.path.exists(truth_file):
    print("[!] Truth file not found.")
    exit()

df = pd.read_csv(truth_file)
# Ensure string type for matching (e.g. "0515" vs 515)
valid_ids = set(df['battery_id'].astype(str))
print(f"[*] Loaded {len(valid_ids)} valid (NORMAL) battery IDs.")

# Search Paths
paths = {
    "Train Good": "dataset_cv1/train/good",
    "Test Good": "dataset_cv1/test/good",
    "Test Defect": "dataset_cv1/test/defect"
}

print("\n[*] Checking Dataset Consistency...")

errors = 0
for label, folder in paths.items():
    if not os.path.exists(folder):
        continue
        
    files = glob.glob(os.path.join(folder, "*.png"))
    print(f"    Checking {label} ({len(files)} images)...")
    
    for f in files:
        fname = os.path.basename(f)
        # Format: "RGB_cell_cylindrical_0515_002.png"
        parts = fname.split('_')
        if len(parts) >= 5:
            # Extract ID part (index 3 usually, e.g. 0515)
            # parts: ['RGB', 'cell', 'cylindrical', '0515', '002.png']
            bat_id_str = parts[3]
            bat_id_int = str(int(bat_id_str)) # "0515" -> "515"
            
            is_valid_id = bat_id_int in valid_ids
            
            # Logic Check
            # 1. "Defect" folder should NOT contain Good IDs (Already checked)
            if "Defect" in label and is_valid_id:
                print(f"      [!] MISLABEL (False Defect): {fname} is in DEFECT folder, but ID {bat_id_int} is in Good List!")
                errors += 1
                
            # 2. "Good" folder should ONLY contain Good IDs (New Constraint)
            # If an ID is NOT in the whitelist, it might be a Defect or Unknown. 
            # Training on it is risky.
            if "Good" in label and not is_valid_id:
                 print(f"      [!] POLLUTION (Risk): {fname} (ID {bat_id_int}) is in Good folder but NOT in Whitelist!")
                 errors += 1

if errors == 0:
    print("\n[+] Dataset is CLEAN! All Good files match the Whitelist, and no Good files are in Defect.")
else:
    print(f"\n[!] Found {errors} label issues.")
