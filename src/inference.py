
import os
# Trigger: Phase 2 Heatmap Generation Run
import torch
import argparse
import mlflow
import json
import time
import cv2
import numpy as np
from loguru import logger
from anomalib.models import Fastflow
from pathlib import Path
from torchvision.transforms.v2 import Resize
from PIL import Image

def get_heatmap(anomaly_map):
    """지도를 컬러맵(Jet)으로 변환합니다."""
    # 정규화 (0~1)
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    # 0~255 변환
    anomaly_map = (anomaly_map * 255).astype(np.uint8)
    # Jet 컬러맵 적용
    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    return heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model.pt")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs")
    
    args = parser.parse_args()
    base_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_base = Path(args.output_dir)
    
    logger.info("==================================================")
    logger.info("🎨 Phase 2: Heatmap Generation (Hyper-Robust Inference)")
    logger.info(f"📍 마운트 루트: {base_path}")
    logger.info(f"⚖️ 모델 경로: {model_path}")
    logger.info("==================================================")

    # 1. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ 사용 장치: {device}")
    
    model = Fastflow(backbone="resnet18", flow_steps=8)
    try:
        # Azure ML에서 로컬로 다운로드되거나 마운트된 경로에서 로드
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.success("✅ 모델 가중치 로드 성공!")
    except Exception as e:
        logger.error(f"❌ 모델 로드 중 오류 발생: {e}")
        raise

    # 2. 테스트 디렉토리 탐색 (4대 카테고리 기점)
    # 목표: 'test' 폴더 아래의 {damaged&pollution, damaged, good, pollution} 찾기
    test_root = None
    
    # [우선순위 1] 명시적 구조 (datasets/resized/test)
    explicit_test = base_path / "datasets" / "resized" / "test"
    if explicit_test.exists():
        test_root = explicit_test
        logger.info(f"✅ [P1] 명시적 테스트 경로 발견: {test_root}")

    # [우선순위 2] 이름이 'test'인 폴더 탐색
    if not test_root:
        for root, dirs, files in os.walk(base_path):
            if Path(root).name.lower() == "test":
                test_root = Path(root)
                logger.info(f"🎯 [P2] 탐색으로 'test' 폴더 발견: {test_root}")
                break

    if not test_root:
        # [우분순위 3] 카테고리 이름 중 하나라도 들어있는 폴더 찾기
        target_cats = ["damaged", "pollution", "good"]
        for root, dirs, files in os.walk(base_path):
            if any(cat in Path(root).name.lower() for cat in target_cats):
                test_root = Path(root).parent
                logger.info(f"🎯 [P3] 카테고리 기반 부모 폴더 발견: {test_root}")
                break

    if not test_root:
        raise FileNotFoundError(f"❌ '{base_path}' 내부에서 'test' 폴더 구조를 찾을 수 없습니다.")

    categories = [d for d in os.listdir(test_root) if os.path.isdir(test_root / d)]
    logger.info(f"📂 감지된 카테고리: {categories}")

    transform = Resize((256, 256))

    # 3. 인퍼런스 및 히트맵 생성
    with torch.no_grad():
        for cat in categories:
            cat_path = test_root / cat
            save_path = output_base / cat
            save_path.mkdir(parents=True, exist_ok=True)
            
            img_files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not img_files: continue
            
            logger.info(f"🖼️ [{cat}] 이미지 {len(img_files)}장 히트맵 생성 중...")
            
            for f in img_files:
                img_path = cat_path / f
                # 이미지 로드 (RGB)
                input_img = Image.open(img_path).convert("RGB")
                input_tensor = transform(input_img)
                # 텐서 변환 및 정규화
                input_tensor = (torch.from_numpy(np.array(input_tensor)).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
                
                # 모델 추론
                output = model(input_tensor)
                anomaly_map = output[0].cpu().numpy().squeeze()
                
                # 히트맵 생성 (ColorMap)
                heatmap = get_heatmap(anomaly_map)
                
                # 원본 시각화용 변환 (OpenCV BGR 포맷)
                orig_img_cv = cv2.cvtColor(np.array(input_img.resize((256, 256))), cv2.COLOR_RGB2BGR)
                
                # 합성 (오버레이)
                overlay = cv2.addWeighted(orig_img_cv, 0.6, heatmap, 0.4, 0)
                
                # 저장
                cv2.imwrite(str(save_path / f"heatmap_{f}"), overlay)

    logger.success(f"🎉 모든 히트맵이 '{output_base}' 폴더에 카테고리별로 저장되었습니다.")

if __name__ == "__main__":
    main()
