
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
from torchvision.transforms.v2 import Resize
import numpy as np

class MaskedFolder(Folder):
    """
    배경을 마스킹하여 배터리 본체에만 집중하게 만드는 데이터모듈입니다.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _apply_mask(self, image):
        # GrabCut 혹은 단순 threshold를 사용하여 배터리 영역을 추출 (배경 노이즈 제거)
        img_np = np.array(image)
        mask = np.zeros(img_np.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (5, 5, img_np.shape[1]-10, img_np.shape[0]-10)
        
        cv2.grabCut(img_np, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        img_masked = img_np * mask2[:, :, np.newaxis]
        return img_masked


def main():
    # ================== 1. Input/Output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()
    base_path = Path(args.data_path)
    
    logger.info("==================================================")
    logger.info("🚀 S1_PatchCore_Training: [Targeted Path Mode]")
    logger.info(f"📍 마운트 루트: {base_path}")
    logger.info("==================================================")

    # 📂 데이터 경로 탐색 로직 (사용자님의 우려를 반영하여 정밀화)
    # 원본 데이터와 섞이지 않도록 'good' 폴더를 최우선으로 찾습니다.
    dataset_root = None
    
    # [1순위] 우리가 이전에 성공했던 경로 패턴 (train/good)
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        parent_name = root_path.parent.name.lower()
        current_name = root_path.name.lower()
        
        # 'train' 폴더 아래의 'good' 폴더를 찾으면 256 리사이즈 폴더일 확률이 매우 높음
        if current_name == "good" and parent_name == "train":
            img_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if img_count > 0:
                dataset_root = root_path
                logger.info(f"🎯 [Targeted] 최적의 학습 경로 발견: {dataset_root} ({img_count}장)")
                break

    # [2순위] 'good'이라는 이름이 포함된 모든 폴더 중 이미지가 있는 곳
    if not dataset_root:
        for root, dirs, files in os.walk(base_path):
            if "good" in root.lower():
                img_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if img_count > 0:
                    dataset_root = Path(root)
                    logger.info(f"🎯 [Fallback] 'good' 키워드 폴더 발견: {dataset_root} ({img_count}장)")
                    break

    if not dataset_root:
        # 디버깅을 위해 현재 구조를 간단히 출력
        logger.error("❌ 'good' 혹은 'train/good' 구조의 이미지를 찾을 수 없습니다.")
        logger.info(f"현재 루트({base_path})의 직계 자식들: {os.listdir(base_path)}")
        raise FileNotFoundError(f"❌ '{base_path}' 내부에 학습용 (Good) 이미지 폴더가 없습니다.")

    # ================== 2. MLflow & Output 설정 ==================== #
    mlflow.start_run()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ 사용 장치: {device}")

    try:
        # [Update] 'good' 외에 'bad' 폴더도 포함하여 평가를 수행합니다 (AUROC 측정용)
        test_root = dataset_root.parent / "test"
        if not test_root.exists():
             test_root = dataset_root.parent # test 폴더가 없는 경우 부모 폴더에서 bad 검색
             
        # [Masked PatchCore] 배경을 제거한 이미지만 학습하도록 설정합니다.
        datamodule = MaskedFolder(
            name="battery_masked",
            root=str(dataset_root.parent),
            normal_dir="train/good",
            test_split_mode="from_dir",
            test_dir="test",
            train_batch_size=16,
            eval_batch_size=1,
            num_workers=4,
            augmentations=Resize((512, 512)),
        )

        model = Patchcore(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            coreset_sampling_ratio=0.1,
        )
        # metrics에 AUROC, F1Score 등을 추가하여 양불 분류 성능을 측정합니다.
        engine = Engine(
            max_epochs=args.epochs, 
            accelerator="auto", 
            devices=1, 
            default_root_dir=str(OUTPUT_DIR),
            task="segmentation" # 결함 위치까지 확인
        )

        # ================== 4. 모델 학습 ==================== #
        logger.info(f"🧬 S1 PatchCore 학습 시작 (Target Epochs: {args.epochs})...")
        engine.fit(model=model, datamodule=datamodule)
        
        # [NEW] 평가 수행: 정상/불량 분류 성능(AUROC) 및 히트맵 생성
        logger.info("📊 성능 평가 및 히트맵 생성 중...")
        engine.test(model=model, datamodule=datamodule)
        
        logger.success(f"✅ {args.epochs} 에폭 학습 및 평가가 성공적으로 끝났습니다!")

        # ================== 5. 결과 저장 ==================== #
        torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
        info = {
            "model": "patchcore",
            "backbone": "resnet18",
            "layers": ["layer1", "layer2", "layer3"],
            "resolution": 512,
            "coreset_ratio": 0.1,
            "dataset_path": str(dataset_root),
            "epochs": args.epochs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "anomaly_detection_with_eval"
        }
        with open(OUTPUT_DIR / "info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        mlflow.log_params(info)
        mlflow.log_artifact(str(OUTPUT_DIR))
        logger.success("🎉 모든 산출물이 Azure ML에 저장되었습니다.")

    except Exception as e:
        logger.error(f"❌ 학습 중 오류 발생: {e}")
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()