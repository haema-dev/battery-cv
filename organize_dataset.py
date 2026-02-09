import os
import shutil

# 목표 폴더 구조
# dataset_cv1/
#   train/
#     good/   (정상 이미지 학습용)
#   test/
#     good/   (정상 이미지 테스트용)
#     defect/ (불량 이미지 테스트용)

BASE_DIR = "dataset_cv1"
SOURCE_SAMPLE_DIR = "sample_data_v1" # 아까 압축 푼 곳

def create_structure():
    print(f"[*] Creating folder structure in '{BASE_DIR}'...")
    
    paths = [
        os.path.join(BASE_DIR, "train", "good"),
        os.path.join(BASE_DIR, "test", "good"),
        os.path.join(BASE_DIR, "test", "defect")
    ]
    
    for p in paths:
        os.makedirs(p, exist_ok=True)
        
    print("[*] Looking for source images...")
    if os.path.exists(SOURCE_SAMPLE_DIR):
        files = [f for f in os.listdir(SOURCE_SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg'))]
        print(f"    Found {len(files)} images in {SOURCE_SAMPLE_DIR}")
        
        print("\n[Action Required] We need to separate OK(Normal) vs NG(Defect).")
        print("Since we don't have labels yet, I will move ALL images to 'train/good' for now.")
        print("Please MANUALLY check them later and remove any bad images from 'train/good'.\n")
        
        # Move to train/good
        for f in files:
            src = os.path.join(SOURCE_SAMPLE_DIR, f)
            dst = os.path.join(BASE_DIR, "train", "good", f)
            shutil.copy(src, dst)
            
        # [Fix] Anomalib strict validation needs more data to split (ZeroDivisionError fix)
        # We will copy 5 images to test/good and 1 to test/defect
        if len(files) > 0:
            # 1. Fill test/good with up to 5 images
            num_test = min(5, len(files))
            for i in range(num_test):
                img_name = files[i]
                src = os.path.join(SOURCE_SAMPLE_DIR, img_name)
                dst = os.path.join(BASE_DIR, "test", "good", img_name)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
            print(f"    [Fix] Populated test/good with {num_test} images")

            # 2. Fill test/defect with 1 image (Fake Defect)
            first_img = files[0]
            src = os.path.join(SOURCE_SAMPLE_DIR, first_img)
            dst_test_defect = os.path.join(BASE_DIR, "test", "defect", first_img)
            if not os.path.exists(dst_test_defect):
                shutil.copy(src, dst_test_defect)
                print(f"    [Fix] Created placeholder test/defect: {first_img}")

            # 3. Fill val/good and val/defect (To bypass auto-split error)
            # We copy the same test images to val set
            dst_val_good = os.path.join(BASE_DIR, "val", "good", first_img)
            os.makedirs(os.path.dirname(dst_val_good), exist_ok=True)
            if not os.path.exists(dst_val_good):
                shutil.copy(src, dst_val_good)
                print(f"    [Fix] Created placeholder val/good: {first_img}")

            dst_val_defect = os.path.join(BASE_DIR, "val", "defect", first_img)
            os.makedirs(os.path.dirname(dst_val_defect), exist_ok=True)
            if not os.path.exists(dst_val_defect):
                shutil.copy(src, dst_val_defect)
                print(f"    [Fix] Created placeholder val/defect: {first_img}")

        print("\n[+] Setup Complete!")
        print(f"    Go to '{BASE_DIR}/train/good' and verify images are NORMAL.")
    else:
        print(f"[!] Source folder '{SOURCE_SAMPLE_DIR}' not found. Did you run the inspector?")

if __name__ == "__main__":
    create_structure()
