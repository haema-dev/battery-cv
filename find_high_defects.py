
import os
import json
import glob

# Paths
JSON_ROOT = r"C:\Users\EL98\Downloads\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터"

def find_high_id_defects():
    print(f"[*] Scanning all JSONs for high-ID defects (>1000)...")
    
    label_files = glob.glob(os.path.join(JSON_ROOT, "*.json"))
    print(f"[*] Total Labels: {len(label_files)}")
    
    high_id_defects = []
    max_defect_id = 0
    
    for jpath in label_files:
        try:
            with open(jpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            info = data.get("data_info", {})
            is_normal = True
            
            # Check normal flag
            if "is_normal" in info:
                is_normal = info["is_normal"]
            elif "is_normal" in data:
                is_normal = data["is_normal"]
            
            # Check defects list
            defects_list = data.get("defects", [])
            if defects_list and len(defects_list) > 0 and defects_list != [None]:
                 is_normal = False

            if is_normal is False:
                # Check ID
                bat_id = info.get("battery_ids", 0)
                try:
                    bat_id_num = int(bat_id)
                except:
                    continue
                    
                if bat_id_num > max_defect_id:
                    max_defect_id = bat_id_num
                    
                if bat_id_num >= 1000:
                    fname = os.path.basename(jpath)
                    print(f"[!] HIGH ID DEFECT: {fname} (ID: {bat_id_num})")
                    high_id_defects.append(fname)
                    if len(high_id_defects) >= 10:
                        print("... Found 10 high IDs, stopping specific print but finding max.")
                        # Don't break, keep finding max
        except:
            pass

    print("-" * 30)
    print(f"Max Defect ID Found: {max_defect_id}")
    print(f"Total High ID Defects (>1000): {len(high_id_defects)}")
    
    if len(high_id_defects) > 0:
        print("Sample High ID Defects:")
        for f in high_id_defects[:5]:
            print(f" - {f}")

if __name__ == "__main__":
    find_high_id_defects()
