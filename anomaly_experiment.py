import os
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import random

# Settings
BASE_DIR = r"c:/Users/EL98/Downloads/BatterySample/Sample"
IMG_DIR = os.path.join(BASE_DIR, "01.원천데이터/CT_Datasets/images")
LBL_DIR = os.path.join(BASE_DIR, "02.라벨링데이터/CT_Datasets/label")
SEED = 42
N_TRAIN = 100
N_TEST_NORMAL = 30
N_TEST_DEFECT = 30

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_ground_truth(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    is_normal = data.get('image_info', {}).get('is_normal', True)
    defects = data.get('defects', [])
    if defects is None: defects = []
    # 0 = Normal, 1 = Defect
    return 0 if is_normal and not defects else 1

def load_data():
    image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    normal_samples = []
    defect_samples = []
    
    print(f"Scanning {len(image_paths)} files...")
    
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        json_name = os.path.splitext(basename)[0] + ".json"
        json_path = os.path.join(LBL_DIR, json_name)
        
        if os.path.exists(json_path):
            try:
                label = get_ground_truth(json_path)
                if label == 0:
                    normal_samples.append(img_path)
                else:
                    defect_samples.append(img_path)
            except:
                pass
                
    print(f"Found Normal: {len(normal_samples)}, Defect: {len(defect_samples)}")
    
    # Split Data
    random.shuffle(normal_samples)
    train_x = normal_samples[:N_TRAIN]
    test_normal = normal_samples[N_TRAIN : N_TRAIN + N_TEST_NORMAL]
    
    # For defects, just take N_TEST_DEFECT
    random.shuffle(defect_samples)
    test_defect = defect_samples[:N_TEST_DEFECT]
    
    test_x = test_normal + test_defect
    test_y = [0]*len(test_normal) + [1]*len(test_defect)
    
    return train_x, test_x, test_y

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use simple ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.layer1 = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.eval()
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3

def run_experiment():
    print("Loading Data...")
    train_files, test_files, test_labels = load_data()
    print(f"Train (Normal): {len(train_files)}, Test: {len(test_files)} (Normal={test_labels.count(0)}, Defect={test_labels.count(1)})")
    
    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    model = ResNetFeatureExtractor()
    
    # 1. Extract Features from Training Data (Normal Only)
    print("Training (Extracting Normal Features)...")
    train_features = [] # List of (C, H, W) tensors
    
    # Hook for features? No, just manual forward
    extracted_features = []
    
    with torch.no_grad():
        for i, fpath in enumerate(train_files):
            img = Image.open(fpath).convert('RGB')
            x = preprocess(img).unsqueeze(0)
            f1, f2, f3 = model(x)
            
            # Upsample locally to match largest map (f1 is largest ~56x56)
            # ResNet18: L1=56, L2=28, L3=14
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            f3 = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            
            # Concat depth-wise: 64 + 128 + 256 = 448 dim
            embedding = torch.cat([f1, f2, f3], dim=1) # (1, 448, 56, 56)
            extracted_features.append(embedding)
            
    # Stack: (N, 448, 56, 56)
    train_outputs = torch.cat(extracted_features, dim=0) 
    
    # 2. Gaussian Modeling (PaDiM)
    # Calculate Mean and Covariance for EACH pixel position (i, j)
    # To save memory/time, we can subsample embeddings (d=448 -> 100) using Random Projection if needed, but 448 is small enough for cpu.
    
    B, C, H, W = train_outputs.shape
    # Reshape: (H*W, B, C) -> For each pixel, we have B samples of dim C
    embedding_vectors = train_outputs.permute(2, 3, 0, 1).reshape(H*W, B, C)
    
    print("Computing Gaussian Stats (Mean/Cov)...")
    # Mean: (H*W, C)
    means = torch.mean(embedding_vectors, dim=1)
    
    # Covariance: (H*W, C, C) - this is heavy. 
    # Simplified PaDiM: Use regularized Identity covariance or diagonal only if full is too slow?
    # Full covariance for 448x448 is heavy.
    # Let's use Random Dimensionality Reduction to ~100 dims to make it fast.
    
    msg = "Reducing dimensionality..."
    idx = torch.randperm(C)[:100] # Random select 100 features
    embedding_vectors = embedding_vectors[:, :, idx]
    means = torch.mean(embedding_vectors, dim=1)
    C = 100
    
    # Cov: (H*W, C, C)
    # shape: (HW, B, C) - mean (HW, 1, C)
    x_centered = embedding_vectors - means.unsqueeze(1)
    covs = torch.bmm(x_centered.permute(0, 2, 1), x_centered) / (B - 1) # Batch Matrix Mul
    
    # Add epsilon for invertibility
    identity = torch.eye(C).to(covs.device).unsqueeze(0).repeat(H*W, 1, 1)
    covs += 0.01 * identity
    
    inv_covs = torch.inverse(covs) # (H*W, C, C)
    
    print("Model Trained.")
    
    # 3. Inference on Test Data
    print("Testing...")
    anomaly_scores = []
    
    with torch.no_grad():
        for i, fpath in enumerate(test_files):
            img = Image.open(fpath).convert('RGB')
            x = preprocess(img).unsqueeze(0)
            f1, f2, f3 = model(x)
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            f3 = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            embedding = torch.cat([f1, f2, f3], dim=1) # (1, 448, 56, 56)
            
            # Select same random indices
            embedding = embedding[:, idx, :, :] # (1, 100, 56, 56)
            
            # Flatten to pixels: (HW, 1, C)
            test_vecs = embedding.permute(2, 3, 0, 1).reshape(H*W, 1, C)
            
            # Calculate Mahalanobis Distance for each pixel
            # Dist = sqrt( (x-u)T * inv_cov * (x-u) )
            diff = test_vecs - means.unsqueeze(1) # (HW, 1, C)
            
            # (HW, 1, C) @ (HW, C, C) -> (HW, 1, C)
            # Efficient: (HW, 1, C) * (HW, C, C) 
            left = torch.bmm(diff, inv_covs) # (HW, 1, C)
            # (HW, 1, C) * (HW, C, 1) -> (HW, 1, 1)
            dist = torch.bmm(left, diff.permute(0, 2, 1))
            dist = torch.sqrt(dist).squeeze() # (HW,)
            
            # Anomaly Score for image = Max pixel distance
            score = torch.max(dist).item()
            anomaly_scores.append(score)
            
            if i % 20 == 0:
                # Visualize heatmap for check
                amap = dist.reshape(H, W).cpu().numpy()
                amap = gaussian_filter(amap, sigma=4)
                
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(img.resize((224,224)))
                plt.title(f"Label: {'Defect' if test_labels[i]==1 else 'Normal'}")
                plt.subplot(1,2,2)
                plt.imshow(amap, cmap='jet')
                plt.title(f"Score: {score:.2f}")
                plt.savefig(f"anomaly_{i}.png")
                plt.close()

    # Evaluation
    auroc = roc_auc_score(test_labels, anomaly_scores)
    print(f"\nFinal AUROC: {auroc:.4f}")
    
    # Find simplified threshold for acc
    # Sort scores, find split that maximizes accuracy
    z = sorted(zip(anomaly_scores, test_labels))
    best_acc = 0
    best_thresh = 0
    
    for i in range(len(z)):
        thresh = z[i][0]
        # Predict 1 if score > thresh
        preds = [1 if s > thresh else 0 for s in anomaly_scores]
        acc = np.mean(np.array(preds) == np.array(test_labels))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    print(f"Best Accuracy: {best_acc:.4f} (Threshold: {best_thresh:.2f})")

if __name__ == "__main__":
    run_experiment()
