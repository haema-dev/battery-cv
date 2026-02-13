
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
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()
    base_path = Path(args.data_path)
    
    logger.info("==================================================")
    logger.info("🚀 S1_FastFlow_Training: [Full Training Mode]")
    logger.info(f"📍 마운트 루트: {base_path}")
    logger.info("==================================================")

    # 📂 데이터 경로 탐색 로직 (이전 성공한 Robust logic 유지)
    dataset_root = None
    
    # [우선순위 1] 명시적 경로
    explicit_path = base_path / "datasets" / "resized" / "train" / "good"
    if explicit_path.exists() and any(explicit_path.iterdir()):
        dataset_root = explicit_path
        logger.info(f"✅ 명시적 경로 발견: {dataset_root}")

    # [우선순위 2] 키워드 조합 검색
    if not dataset_root:
        for root, dirs, files in os.walk(base_path):
            root_path = Path(root)
            parts = [p.lower() for p in root_path.parts]
            if "resized" in parts and "good" in parts:
                img_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if img_count > 0:
                    dataset_root = root_path
                    logger.info(f"🎯 자동 탐색으로 경로 발견: {dataset_root} ({img_count}장)")
                    break

    if not dataset_root:
        raise FileNotFoundError(f"❌ '{base_path}' 내부에서 학습용 이미지 폴더를 찾을 수 없습니다.")

    # ================== 2. MLflow & Output 설정 ==================== #
    mlflow.start_run()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ 사용 장치: {device}")

    try:
        # ================== 3. Anomalib 데이터 구성 ==================== #
        logger.info(f"📥 데이터셋 로드: {dataset_root}")
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

        model = Fastflow(backbone="resnet18", flow_steps=8, evaluator=False)
        engine = Engine(max_epochs=args.epochs, accelerator="auto", devices=1, default_root_dir=str(OUTPUT_DIR))

        # ================== 4. 모델 학습 ==================== #
        logger.info(f"🧬 모델 학습 시작 (Target Epochs: {args.epochs})...")
        engine.fit(model=model, datamodule=datamodule)
        logger.success(f"✅ {args.epochs} 에폭 학습을 무사히 완료했습니다!")

        # ================== 5. 결과 저장 ==================== #
        model_save_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"💾 모델 가중치 저장 완료: {model_save_path}")

        info = {
            "experiment": "Battery_S1_AnomalyDetection",
            "epochs": args.epochs,
            "dataset": str(dataset_root),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(OUTPUT_DIR / "info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        mlflow.log_params(info)
        mlflow.log_artifact(str(OUTPUT_DIR))
        logger.success("🎉 모든 프로세스가 성공적으로 종료되었습니다!")

    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()