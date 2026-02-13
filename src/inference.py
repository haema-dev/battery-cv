import os
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np

# Anomalib 공식 문서 추천: 추론 전용으로 설계된 TorchInferencer 사용
# Reference: https://anomalib.readthedocs.io/en/latest/guides/inference.html
try:
    from anomalib.deploy import TorchInferencer
    INFERENCER_AVAILABLE = True
except ImportError:
    # v1.x 일부 버전 또는 환경에 따라 경로가 다를 수 있음 대비
    INFERENCER_AVAILABLE = False

def run_inference(data_path, model_path, output_dir):
    """
    Anomalib TorchInferencer를 사용한 고수준 추론 로직.
    임의의 로직 대신 라이브러리 제공 기능을 최우선으로 사용합니다.
    """
    print("--------------------------------------------------")
    print(f"🚀 [Phase 2] Heatmap Generation Starting")
    print(f"📦 Model: {model_path}")
    print(f"📂 Data: {data_path}")
    print("--------------------------------------------------")

    if not INFERENCER_AVAILABLE:
        print("❌ Error: 'anomalib.deploy.TorchInferencer'를 로드할 수 없습니다. 패키지 설치를 확인하세요.")
        return

    # 1. Inferencer 초기화 (CPU/GPU 자동 감지)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    # .pt 파일을 직접 로드하여 추론 준비 (내부적으로 Pre-processing 설정 포함함)
    inferencer = TorchInferencer(
        path=model_path,
        device=device
    )

    data_base = Path(data_path)
    output_base = Path(output_dir)
    
    # 4대 카테고리 (damaged, good, pollution, damaged&pollution)
    # 데이터셋 구조: data_path/test/[category]/*.jpg
    test_root = data_base / "test"
    if not test_root.exists():
        # 대체 경로 탐색 (resized 폴더 등이 포함된 경우)
        possible_paths = list(data_base.glob("**/test"))
        if possible_paths:
            test_root = possible_paths[0]
        else:
            print(f"❌ Error: 'test' 폴더를 찾을 수 없습니다. (Base: {data_path})")
            return

    print(f"🎯 Found test root: {test_root}")
    
    categories = [d for d in test_root.iterdir() if d.is_dir()]
    
    for cat_dir in categories:
        cat_name = cat_dir.name
        print(f"🔍 Processing: {cat_name}")
        
        # 카테고리별 출력 폴더 생성
        cat_output = output_base / cat_name
        cat_output.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 스캔
        img_files = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpeg"))
        
        for img_path in img_files:
            # 2. Prediction 수행 (Anomalib 표준 API)
            # predict()는 PredictionResults 객체를 반환하며, 
            # 여기에는 시각화된 heatmapImage가 포함됩니다.
            results = inferencer.predict(image=str(img_path))
            
            # 3. 히트맵 시각화 데이터 추출
            # predictions.heatmap은 오버레이된 BGR 이미지(numpy)입니다.
            heatmap_img = results.heatmap
            
            # 4. 저장
            save_name = f"heatmap_{img_path.name}"
            save_path = cat_output / save_name
            cv2.imwrite(str(save_path), heatmap_img)
            
    print(f"\n✅ All heatmaps generated successfully!")
    print(f"📍 Location: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Input data folder")
    parser.add_argument("--model_path", type=str, required=True, help="Trained model (.pt) file")
    parser.add_argument("--output_dir", type=str, required=True, help="Inference results folder")
    
    args = parser.parse_args()
    run_inference(args.data_path, args.model_path, args.output_dir)