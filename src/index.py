
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

def list_all_contents(path, max_files=10):
    """디버깅을 위해 마운트된 모든 구조를 샅샅이 로깅합니다."""
    try:
        path = Path(path)
        logger.info(f"� [DEEP DEBUG] Full structure of {path}:")
        for root, dirs, files in os.walk(path):
            level = len(Path(root).relative_to(path).parts)
            indent = "  " * level
            logger.info(f"{indent}📁 {os.path.basename(root) or '/'} ({len(files)} files)")
            # 파일 일부 출력
            for f in files[:max_files]:
                logger.info(f"{indent}  - 📄 {f}")
            if len(files) > max_files:
                logger.info(f"{indent}  - ... and {len(files)-max_files} more files")
    except Exception as e:
        logger.error(f"❌ Deep listing failed: {e}")

def main():
    # ================== 1. Input/Output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, required=True, help="Path to mounted data asset")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()
    base_path = Path(args.data_path)
    
    logger.info("==================================================")
    logger.info("🚀 S1_FastFlow_Training: [Hyper-Robust Mode]")
    logger.info(f"📍 마운트 루트: {base_path}")
    logger.info("==================================================")

    # 1. 일단 다 찍어보기 (원인 파악용)
    list_all_contents(base_path)

    # 📂 데이터 경로 탐색 로직 (우선순위)
    dataset_root = None
    
    # [우선순위 1] 명시적 경로
    explicit_path = base_path / "datasets" / "resized" / "train" / "good"
    if explicit_path.exists() and any(explicit_path.iterdir()):
        dataset_root = explicit_path
        logger.info(f"✅ [Priority 1] 명시적 경로 발견: {dataset_root}")

    # [우선순위 2] 키워드 조합 검색
    if not dataset_root:
        for root, dirs, files in os.walk(base_path):
            root_path = Path(root)
            parts = [p.lower() for p in root_path.parts]
            if "resized" in parts and "good" in parts:
                img_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if img_count > 0:
                    dataset_root = root_path
                    logger.info(f"✅ [Priority 2] 키워드 검색 발견: {dataset_root} ({img_count}장)")
                    break

    # [우선순위 3] 이름이 'good'인 폴더 중 이미지가 있는 곳
    if not dataset_root:
        for root, dirs, files in os.walk(base_path):
            if os.path.basename(root).lower() == "good":
                img_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if img_count > 0:
                    dataset_root = Path(root)
                    logger.info(f"✅ [Priority 3] 'good' 폴더 발견: {dataset_root} ({img_count}장)")
                    break

    # [우선순위 4] 그냥 이미지가 가장 많은 폴더 (최후의 보루)
    if not dataset_root:
        max_imgs = 0
        best_path = None
        for root, dirs, files in os.walk(base_path):
            img_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if img_count > max_imgs:
                max_imgs = img_count
                best_path = Path(root)
        if best_path and max_imgs > 0:
            dataset_root = best_path
            logger.info(f"✅ [Priority 4] 최대 이미지 폴더 선택: {dataset_root} ({max_imgs}장)")

    if not dataset_root:
        # 정말 아무것도 없으면 루트 자체라도 시도 (파일이 루트에 있을 수도 있음)
        img_count = len([f for f in os.listdir(base_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if img_count > 0:
            dataset_root = base_path
            logger.info(f"✅ [Priority 5] 루트 디렉토리 선택: {dataset_root} ({img_count}장)")

    if not dataset_root:
        raise FileNotFoundError(f"❌ '{base_path}' 내부에서 어떤 이미지 형식도 찾을 수 없습니다. 자산 구성을 확인해주세요.")

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
        logger.info(f"🧬 모델 학습 진행 (Epochs: {args.epochs})...")
        engine.fit(model=model, datamodule=datamodule)
        logger.success("✅ 학습 완료!")

        # ================== 5. 결과 저장 ==================== #
        torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
        info = {"dataset": str(dataset_root), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(OUTPUT_DIR / "info.json", 'w') as f: json.dump(info, f, indent=2)

        mlflow.log_params(info)
        mlflow.log_artifact(str(OUTPUT_DIR))
        logger.success("🎉 모든 프로세스 종료!")

    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()