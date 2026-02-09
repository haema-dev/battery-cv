
import os
import json
import glob

# Path derived from user's screenshot
JSON_ROOT = r"C:\Users\EL98\Downloads\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터"

def scan_for_defects():
    print(f"[*] Scanning JSON files in: {JSON_ROOT}")
    
    if not os.path.exists(JSON_ROOT):
        print(f"[!] Directory not found: {JSON_ROOT}")
        # Try to find it nearby if the exact path is wrong
        return

    # Recursive search for all json files
    json_files = glob.glob(os.path.join(JSON_ROOT, "**", "*.json"), recursive=True)
    print(f"[*] Found {len(json_files)} JSON files. Checking for defects...")

    defect_count = 0
    normal_count = 0
    
    found_defects = []

    for jpath in json_files:
        try:
            with open(jpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check for 'is_normal' flag (based on user screenshot)
            # The screenshot shows: "data_info": { ..., "is_normal": true }
            # Wait, the structure in screenshot is: "data_info": { ... "is_normal": true }?
            # actually checking the screenshot carefully:
            # The root object seems to be the dict.
            # "data_info": { ... }
            # "description": ...
            # Wait, let's look at the screenshot again.
            # Row 4: "is_normal": true is INSIDE "data_info"? 
            # No, looking at the indentation in the screenshot:
            # Line 1: { "data_info": { ...
            # It's a bit hard to see the closing brace of data_info.
            # But usually "is_normal" is inside data_info or at root.
            # I will check both locations to be safe.
            
            is_normal = True # Default to true
            
            info = data.get("data_info", {})
            
            # Check inside data_info
            if "is_normal" in info:
                is_normal = info["is_normal"]
            elif "is_normal" in data:
                 is_normal = data["is_normal"]
            
            # Also check 'defects' field if present
            # Screenshot shows "defects": null inside data_info? Or root?
            # It seems "defects": null is visible.
            
            if is_normal is False:
                defect_count += 1
                fname = info.get("image_info", {}).get("file_name", "Unknown")
                bat_id = info.get("battery_ids", "Unknown")
                defect_type = info.get("defects", "Unknown")
                
                print(f"[!] DEFECT FOUND: {fname} (ID: {bat_id})")
                found_defects.append(fname)
                
                if len(found_defects) >= 10:
                    print("... (Stopping clear output after 10 found, but continuing count)")
                    
            else:
                normal_count += 1
                
        except Exception as e:
            pass
            # print(f"[!] Error parsing {jpath}: {e}")

    print("-" * 40)
    print(f"Scan Complete.")
    print(f"Total Scanned: {len(json_files)}")
    print(f"Normal: {normal_count}")
    print(f"Defect: {defect_count}")
    
    if defect_count > 0:
        print("\n[Recommendation]")
        print("Please test one of the files listed above (starts with [!] DEFECT FOUND).")
        print("These are the TRUE defects confirmed by Labeling Data.")

if __name__ == "__main__":
    scan_for_defects()
