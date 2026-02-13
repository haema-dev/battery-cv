
import os
import torch
import argparse
import mlflow
import json
import time
from loguru import logger
from anomalib.models import Fastflow
from anomalib.data import Folder
from anomalib.engine import Engine
from pathlib import Path
from torchvision.transforms.v2 import Resize

def main():
    # ================== 1. Input/Output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()
    
    # 에저 스토리지 연동 경로 (양성 데이터셋 집중)
    base_path = Path(args.data_path).resolve()
    dataset_root = base_path / "datasets" / "resized" / "train" / "good"

    logger.info("==================================================")
    logger.info("🚀 S1_FastFlow_Training: [Training Only Mode]")
    logger.info(f"📍 학습 데이터 경로: {dataset_root}")
    logger.info("==================================================")

    if not dataset_root.exists():
        logger.warning(f"⚠️ {dataset_root} 경로가 직접적으로 발견되지 않았습니다. 검색을 시도합니다.")
        potential = list(base_path.rglob("resized/train/good")) 
        if potential:
            dataset_root = potential[0]
            logger.info(f"✅ 실제 경로 발견: {dataset_root}")
        else:
            raise FileNotFoundError(f"❌ '{base_path}' 내부에 'resized/train/good' 폴더가 없습니다.")

    # ================== 2. MLflow & Output 설정 ==================== #
    mlflow.start_run()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ 사용 장치: {device}")

    try:
        # ================== 3. Anomalib 데이터 구성 ==================== #
        logger.info("📥 Anomalib 데이터 모듈 구성 중...")
        
        # 이미 전처리/리사이즈된 데이터이므로 최소한의 transform 적용
        transform = Resize((256, 256))
        
        datamodule = Folder(
            name="battery_resized",
            root=str(dataset_root),
            normal_dir=".",
            train_batch_size=32,
            eval_batch_size=8,
            num_workers=4,
            augmentations=transform,
        )

        # 모델 초기화 (FastFlow)
        model = Fastflow(
            backbone="resnet18",
            flow_steps=8,
            evaluator=False 
        )

        # Engine 설정
        engine = Engine(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            default_root_dir=str(OUTPUT_DIR),
            enable_checkpointing=True,
        )

        # ================== 4. 모델 학습 ==================== #
        logger.info(f"🧬 모델 학습 진행 (Epochs: {args.epochs})...")
        engine.fit(model=model, datamodule=datamodule)
        logger.success("✅ 학습 프로세스 성공적으로 완료!")

        # ================== 5. 모델 가중치 저장 ==================== #
        # 이 가중치가 저장되어야 나중에 별도의 스크립트로 추론(히트맵 생성)이 가능합니다.
        model_save_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"💾 모델 가중치 저장 완료: {model_save_path}")

        # 작업 정보 기록
        info = {
            "experiment": "Battery_S1_AnomalyDetection",
            "mode": "training_only",
            "model": "FastFlow",
            "backbone": "resnet18",
            "dataset_path": str(dataset_root),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(OUTPUT_DIR / "info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        mlflow.log_params(info)
        mlflow.log_artifact(str(OUTPUT_DIR))
        logger.success("🎉 Azure ML 결과 저장 및 실험 종료!")

    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()