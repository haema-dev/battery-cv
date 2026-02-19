
import os
import torch
import argparse
import mlflow
import json
import time
import cv2
from loguru import logger
from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.engine import Engine
from pathlib import Path
import torch.nn as nn
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
import numpy as np

class BatteryMaskTransform(nn.Module):
    """
    배경 노이즈를 근본적으로 제거하기 위한 마스킹 변환기입니다.
    배터리 영역(중앙부) 외의 배경을 0(검정색)으로 처리합니다.
    """
    def forward(self, image):
        # image logic: 0.5 이상의 밝기를 가진 영역 혹은 중앙부 GrabCut 마스킹
        # 여기서는 Anomalib 파이프라인 내에서 텐서 연산으로 수행하거나,
        # CPU 변환 과정에서 OpenCV를 활용합니다.
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)

        mask = np.zeros(img_np.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        # 배터리가 주로 위치하는 중앙 영역을 타겟으로 GrabCut 수행
        rect = (10, 10, img_np.shape[1]-20, img_np.shape[0]-20)
        
        try:
            cv2.grabCut(img_np, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            img_masked = img_np * mask2[:, :, np.newaxis]
            
            # 다시 텐서로 변환하여 반환
            return ToImage()(img_masked)
        except Exception:
            # 실패 시 원본 반환 (안정성 확보)
            return image

def main():
    # ================== 1. Input/Output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=1) # PatchCore는 1에폭으로 충분 (Coreset 추출이 핵심)

    args = parser.parse_args()
    base_path = Path(args.data_path)
    
    logger.info("==================================================")
    logger.info("🚀 S1_PatchCore_Training: [Extreme Precision Mode]")
    logger.info(f"📍 마운트 루트: {base_path}")
    logger.info("==================================================")

    # ... (데이터 경로 탐색 로직 동일) ...
    dataset_root = None
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        if root_path.name.lower() == "good" and root_path.parent.name.lower() == "train":
            dataset_root = root_path
            break
    
    if not dataset_root:
        dataset_root = base_path # Fallback

    # ================== 2. MLflow & Output 설정 ==================== #
    mlflow.start_run()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ 사용 장치: {device}")

    try:
        # [Extreme Precision] 고해상도 + 마스킹 변환 적용
        transforms = Compose([
            Resize((512, 512)),
            BatteryMaskTransform(), # 배경 박멸
            ToDtype(torch.float32, scale=True)
        ])

        # 데이터 세트의 최상위 루트를 안전하게 설정합니다.
        if dataset_root is not None and dataset_root != base_path:
            # train/good 구조가 발견된 경우
            data_root_dir = dataset_root.parent.parent
        else:
            # 못 찾았거나 flat한 경우
            data_root_dir = base_path
        
        datamodule = Folder(
            name="battery_extreme",
            root=str(data_root_dir),
            normal_dir="train/good",
            normal_test_dir="test/good", 
            abnormal_dir="test/bad",    
            test_split_mode="from_dir",
            train_batch_size=4, # [Stability] WideResNet50 + 512x512 메모리 최적화
            eval_batch_size=1,
            num_workers=4,
            transform=transforms,
        )

        # [Backbone Upgrade] ResNet18 -> WideResNet50
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"], # WideResNet은 layer2, 3 조합이 가장 강력함
            coreset_sampling_ratio=0.1,
        )

        engine = Engine(
            max_epochs=args.epochs, 
            accelerator="auto", 
            devices=1, 
            default_root_dir=str(OUTPUT_DIR),
            task="segmentation"
        )

        # ================== 4. 모델 학습 & 평가 ==================== #
        logger.info(f"💎 S1 WideResNet PatchCore 학습 시작...")
        engine.fit(model=model, datamodule=datamodule)
        
        logger.info("📊 초정밀 성능 평가(AUROC) 및 히트맵 생성...")
        engine.test(model=model, datamodule=datamodule)
        
        logger.success(f"✅ 초정밀 학습 및 평가 완료!")

        # ================== 5. 결과 저장 ==================== #
        torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
        info = {
            "model": "patchcore",
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "resolution": 512,
            "coreset_ratio": 0.1,
            "masked": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(OUTPUT_DIR / "info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        mlflow.log_params(info)
        mlflow.log_artifact(str(OUTPUT_DIR))
        logger.success("🎉 모든 고도화 산출물이 저장되었습니다.")

    except Exception as e:
        logger.error(f"❌ 학습 중 오류 발생: {e}")
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()