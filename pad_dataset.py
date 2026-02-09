import os
import shutil

TARGET_DIR = r"C:\Users\EL98\Downloads\BatterySample\dataset_cv1\test\defect"

def pad_images():
    if not os.path.exists(TARGET_DIR):
        print(f"Directory not found: {TARGET_DIR}")
        return

    files = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} images in {TARGET_DIR}")
    
    if len(files) == 0:
        print("No images to duplicate.")
        return

    if len(files) >= 10:
        print("Enough images exist. No padding needed.")
        return

    # Duplicate to reach at least 10
    needed = 10 - len(files)
    print(f"Padding with {needed} duplicates...")
    
    src = os.path.join(TARGET_DIR, files[0])
    base_name, ext = os.path.splitext(files[0])
    
    for i in range(needed):
        new_name = f"{base_name}_pad_{i}{ext}"
        dst = os.path.join(TARGET_DIR, new_name)
        shutil.copy(src, dst)
        print(f"Created: {new_name}")

if __name__ == "__main__":
    pad_images()
