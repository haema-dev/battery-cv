
import os
import json
import glob

# Paths
JSON_ROOT = r"C:\Users\EL98\Downloads\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터"
IMAGE_ROOT = r"C:\Users\EL98\Downloads\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터\Training\01.원천데이터\TS_Exterior_Img_Datasets_images_4"

# User has these IDs based on list_dir
TARGET_IDS = ["1154", "1155", "1156"]

def check_user_dataset_for_defects():
    print(f"[*] Checking JSON labels for Image IDs: {TARGET_IDS}")
    
    label_files = glob.glob(os.path.join(JSON_ROOT, "*.json"))
    print(f"[*] Total Labels Available: {len(label_files)}")
    
    defects_in_subset = []
    
    for jpath in label_files:
        filename = os.path.basename(jpath)
        
        # Check if this JSON belongs to the user's subset
        is_target = False
        for tid in TARGET_IDS:
            if tid in filename:
                is_target = True
                break
        
        if not is_target:
            continue
            
        # Parse JSON
        try:
            with open(jpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            info = data.get("data_info", {})
            is_normal = True
            
            # Check different possible locations for flag
            if "is_normal" in info:
                is_normal = info["is_normal"]
            elif "is_normal" in data:
                is_normal = data["is_normal"]
            
            # Additional check: 'defects' list not empty
            defects_list = data.get("defects", [])
            if defects_list and len(defects_list) > 0:
                 # Ensure it's not just null
                 if defects_list != [None]:
                     is_normal = False

            if is_normal is False:
                print(f"[!] FOUND DEFECT in User Set: {filename}")
                defects_in_subset.append(filename)
                
        except Exception as e:
            pass

    print("-" * 30)
    if len(defects_in_subset) > 0:
        print(f"SUCCESS: Found {len(defects_in_subset)} defects in your current download.")
        print("Files to test:")
        for f in defects_in_subset[:10]:
            print(f" - {f.replace('.json', '.png')}")
    else:
        print("RESULT: ZERO defects found in the 1154, 1155, 1156 series.")
        print("It seems 'images_4.zip' contains ONLY Normal batteries.")
        print("You may need to download 'images_1.zip' (contains 0001 series) to get defects.")

if __name__ == "__main__":
    check_user_dataset_for_defects()
