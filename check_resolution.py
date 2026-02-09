
import os
from PIL import Image

path = r"C:\Users\EL98\Downloads\BatterySample\dataset_cv1\train\good"
files = os.listdir(path)
if files:
    img_path = os.path.join(path, files[0])
    with Image.open(img_path) as img:
        print(f"Training Image Size: {img.size}")
