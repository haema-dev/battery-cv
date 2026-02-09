import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Paths
BASE_DIR = r"c:/Users/EL98/Downloads/BatterySample/Sample"
IMG_DIR = os.path.join(BASE_DIR, "01.원천데이터/CT_Datasets/images")
LBL_DIR = os.path.join(BASE_DIR, "02.라벨링데이터/CT_Datasets/label")

def get_ground_truth(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Check if 'defects' list is not empty or 'is_normal' flag
    # Based on previous file view: "is_normal": false, "defects": [...]
    is_normal = data.get('image_info', {}).get('is_normal', True)
    
    # Also double check defects list just in case
    defects = data.get('defects', [])
    if defects is None: defects = []
    
    return 0 if is_normal and not defects else 1 # 0: Normal, 1: Defect

def load_data(max_samples=200):
    image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    
    data = []
    labels = []
    
    print(f"Found {len(image_paths)} images.")
    if len(image_paths) == 0:
        return [], []

    count = 0
    for img_path in image_paths:
        if count >= max_samples: break
        
        basename = os.path.basename(img_path)
        json_name = os.path.splitext(basename)[0] + ".json"
        json_path = os.path.join(LBL_DIR, json_name)
        
        if os.path.exists(json_path):
            try:
                label = get_ground_truth(json_path)
                data.append(img_path)
                labels.append(label)
                count += 1
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                
    return data, np.array(labels)

def extract_features(image_paths):
    # Load Pretrained ResNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the classification layer (fc) to get feature vectors
    model.fc = nn.Identity()
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    print("Extracting features...")
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            if i % 20 == 0: print(f"Processing {i}/{len(image_paths)}...")
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img).unsqueeze(0)
                feat = model(img_t)
                features.append(feat.numpy().flatten())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return np.array(features)

def run_clustering():
    # 1. Load Data
    print("Loading data...")
    image_paths, y_true = load_data(max_samples=300) # Use up to 300 samples
    
    if len(image_paths) == 0:
        print("No data found!")
        return

    print(f"Loaded {len(image_paths)} samples.")
    print(f"Normal: {np.sum(y_true==0)}, Defect: {np.sum(y_true==1)}")

    # 2. Extract Features
    X_features = extract_features(image_paths)
    
    # 3. K-Means Clustering
    print("Running K-Means...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_features)
    
    # 4. Evaluation
    # Note: Cluster labels (0, 1) might be swapped relative to true labels (0, 1).
    # We check which mapping gives higher accuracy.
    
    # Option A: Cluster 0 -> Normal (0), Cluster 1 -> Defect (1)
    acc_a = np.mean(y_pred == y_true)
    
    # Option B: Cluster 0 -> Defect (1), Cluster 1 -> Normal (0)
    y_pred_b = 1 - y_pred
    acc_b = np.mean(y_pred_b == y_true)
    
    final_pred = y_pred if acc_a > acc_b else y_pred_b
    final_acc = max(acc_a, acc_b)
    
    print("\n" + "="*40)
    print("CLUSTERING RESULT")
    print("="*40)
    print(f"Accuracy (Alignment corrected): {final_acc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, final_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, final_pred, target_names=['Normal', 'Defect']))
    
    # PCA Visualization
    print("\nGenerating PCA visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_features)
    
    plt.figure(figsize=(10, 5))
    
    # Plot Ground Truth
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[y_true==0, 0], X_pca[y_true==0, 1], c='blue', label='Normal', alpha=0.6)
    plt.scatter(X_pca[y_true==1, 0], X_pca[y_true==1, 1], c='red', label='Defect', alpha=0.6)
    plt.title("Ground Truth")
    plt.legend()
    
    # Plot K-Means Result
    plt.subplot(1, 2, 2)
    # Use final_pred to color
    plt.scatter(X_pca[final_pred==0, 0], X_pca[final_pred==0, 1], c='blue', label='Cluster A (Pred Normal)', alpha=0.6)
    plt.scatter(X_pca[final_pred==1, 0], X_pca[final_pred==1, 1], c='red', label='Cluster B (Pred Defect)', alpha=0.6)
    plt.title(f"K-Means Clustering (Acc: {final_acc:.2f})")
    plt.legend()
    
    save_path = "clustering_result.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    run_clustering()
