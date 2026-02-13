
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

def list_directory_contents(path, depth=2):
    """디버깅을 위해 디렉토리 구조를 로깅합니다."""
    try:
        path = Path(path)
        logger.info(f"📂 [DEBUG] Listing {path}:")
        for root, dirs, files in os.walk(path):
            level = len(Path(root).relative_to(path).parts)
            if level < depth:
                indent = "  " * level
                logger.info(f"{indent}📁 {os.path.basename(root)}/ ({len(files)} files)")
    except Exception as e:
        logger.error(f"❌ Directory listing failed: {e}")

def main():
    # ================== 1. Input/Output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()
    
    # Path resolve() 대신 직접 사용 (마운트 지점에서 가끔 이슈 발생 방지)
    base_path = Path(args.data_path)
    
    logger.info("==================================================")
    logger.info("🚀 S1_FastFlow_Training: [Robust Path Search Mode]")
    logger.info(f"📍 마운트 루트: {base_path}")
    logger.info("==================================================")

    # 디버깅: 루드 디렉토리 내용 출력
    list_directory_contents(base_path, depth=3)

    # 📂 데이터 경로 자동 탐색 (가장 정확한 'good' 폴더 찾기)
    dataset_root = None
    
    # 1. 명시적 경로 확인
    explicit_path = base_path / "datasets" / "resized" / "train" / "good"
    if explicit_path.exists():
        dataset_root = explicit_path
        logger.info(f"✅ 명시적 경로 발견: {dataset_root}")
    else:
        # 2. os.walk를 이용한 유연한 검색 (resized + train + good 키워드 조합)
        logger.warning("⚠️ 명시적 경로를 찾지 못해 전체 검색을 시작합니다.")
        for root, dirs, files in os.walk(base_path):
            root_path = Path(root)
            parts = [p.lower() for p in root_path.parts]
            
            # 'resized', 'train', 'good'이 모두 경로에 포함되면서 이미지가 있는 폴더 탐색
            if "resized" in parts and "train" in parts and "good" in parts:
                img_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if img_count > 0:
                    dataset_root = root_path
                    logger.info(f"🎯 유연한 검색으로 경로 발견: {dataset_root} (이미지 {img_count}장)")
                    break

    if not dataset_root:
        # 3. 최후의 수단: 'good' 폴더 중 이미지가 100장 이상인 곳 탐색
        for root, dirs, files in os.walk(base_path):
            if os.path.basename(root).lower() == "good":
                img_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if img_count > 100:
                    dataset_root = Path(root)
                    logger.info(f"🔄 최후의 수단으로 경로 발견: {dataset_root} ({img_count}장)")
                    break

    if not dataset_root:
        raise FileNotFoundError(f"❌ '{base_path}' 내부에서 학습용 'good' 이미지 폴더를 찾을 수 없습니다.")

    # ================== 2. MLflow & Output 설정 ==================== #
    mlflow.start_run()
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ 사용 장치: {device}")

    try:
        # ================== 3. Anomalib 데이터 구성 ==================== #
        logger.info("📥 Anomalib 데이터 모듈 구성 중...")
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

        # ================== 5. 모델 가직치 저장 ==================== #
        model_save_path = OUTPUT_DIR / "model.pt"
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"💾 모델 가중치 저장 완료: {model_save_path}")

        # 작업 정보 기록
        info = {
            "experiment": "Battery_S1_AnomalyDetection",
            "mode": "training_only",
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