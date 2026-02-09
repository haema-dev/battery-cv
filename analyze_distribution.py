
import os
import torch
import sys
import numpy as np
from run_inference import load_model, preprocess_image, infer_single_pass

def analyze():
    print("Loading model...")
    model = load_model()
    
    # Directories
    train_good_dir = r"C:\Users\EL98\Downloads\BatterySample\dataset_cv1\train\good"
    test_good_dir = r"C:\Users\EL98\Downloads\BatterySample\dataset_cv1\test\good"
    test_defect_dir = r"C:\Users\EL98\Downloads\BatterySample\dataset_cv1\test\defect"
    
    report = []
    
    def get_scores(directory, label):
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []
            
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.png')]
        scores = []
        print(f"Analyzing {label}: {len(files)} images...")
        
        for i, p in enumerate(files):
            input_tensor = preprocess_image(p, target_size=(256, 256))
            if input_tensor is None: continue
            score = infer_single_pass(model, input_tensor)
            scores.append(score)
            if i % 10 == 0: print(f"{i}/{len(files)}", end="\r")
            
        if not scores: return []
        
        scores_np = np.array(scores)
        stats = (
            f"\n--- {label} ---\n"
            f"Count: {len(scores)}\n"
            f"Min  : {scores_np.min():.4f}\n"
            f"Max  : {scores_np.max():.4f}\n"
            f"Mean : {scores_np.mean():.4f}\n"
            f"Std  : {scores_np.std():.4f}\n"
            f"95th : {np.percentile(scores_np, 95):.4f}\n"
            f"99th : {np.percentile(scores_np, 99):.4f}\n"
        )
        report.append(stats)
        return scores

    train_scores = get_scores(train_good_dir, "TRAIN (Good)")
    test_scores = get_scores(test_good_dir, "TEST (Good)")
    defect_scores = get_scores(test_defect_dir, "TEST (Defect)")
    
    with open("distribution_report.txt", "w") as f:
        f.write("\n".join(report))
        
    print("\nAnalysis Complete. Check distribution_report.txt")

if __name__ == "__main__":
    analyze()
