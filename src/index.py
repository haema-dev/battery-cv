import os, torch, argparse, mlflow, json, time
from loguru import logger
from anomalib.models import Fastflow
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from pathlib import Path
import numpy as np, cv2
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
import adlfs
import fsspec


def main():
    # Trigger training for battery-data-CLAHE-gray

    # ================== 1. input/output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, help="Path to mounted data asset")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=10)    
    parser.add_argument("--model", type=str, default="patchcore", choices=["patchcore", "fastflow"], help="Model to train")

    args = parser.parse_args()
    base_path = Path(args.data_path)

    # ==========================================
    # 🔍 마운트 경로 확인 (압축 해제된 이미지 사용)
    # ==========================================
    logger.info(f"📍 마운트 루트 확인: {args.data_path}")
    
    if os.path.exists(args.data_path):
        import subprocess
        # 폴더 구조를 2단계까지 싹 훑어서 로그에 남깁니다.
        result = subprocess.run(['ls', '-R', args.data_path], capture_output=True, text=True)
        # logger.info(f"📂 실제 마운트된 파일 구조:\n{result.stdout[:2000]}")
        
        # 이미지 파일 수 확인
        image_count = len([f for f in os.listdir(args.data_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        logger.info(f"📷 마운트된 이미지 수: {image_count}개")
    else:
        raise FileNotFoundError(f"마운트 경로를 찾을 수 없습니다: {args.data_path}")
    
    # ========================================== Mlflow ==========================================
    mlflow.start_run()
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"📂 {os.path.abspath(OUTPUT_DIR)}")
    
    try:
        # ================== 2. 이상탐지 작업 ==================== #
        
        logger.info(f"📥 {args.model.upper()} 모델 및 데이터셋 구성")
        
        import anomalib
        logger.info(f"📦 Anomalib Version: {anomalib.__version__}")

        # 데이터셋 구성 (마운트된 압축해제 이미지 사용)
        dataset_root = str(base_path)  # 마운트된 경로 직접 사용
        logger.info(f"📂 학습 데이터 경로: {dataset_root}")
        
        # Transform 정의 (Anomalib 2.2.0 호환)
        # image_size 인자 대신 explicit transform 사용
        transform = Compose([
            Resize((1024, 320)),
            ToImage(), 
            ToDtype(torch.float32, scale=True),
        ])

        datamodule = Folder(
            name="battery",
            root=dataset_root,
            normal_dir=".", 
            train_batch_size=4,
            eval_batch_size=8,
            num_workers=4,
            train_augmentations=transform,
            val_augmentations=transform,
            test_augmentations=transform,
        )
        
        # 모델 초기화
        if args.model == "fastflow":
            model = Fastflow(
                backbone="resnet18",
                pre_trained=True,
                flow_steps=8, # Default flow steps
            )
        else:
            model = Patchcore(
                backbone="resnet18",
                pre_trained=True,
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.01,  # Reduced to 0.01 for high-res (320x1024) inputs
            )

        # ---------------------------------------------------------
        # 🔧 [Fix] 모델 내부 리사이징 로직 강제 수정
        # 모델이 기본적으로 256x256으로 리사이징하려는 것을 방지하고,
        # 우리가 전처리한 1024x320 해상도를 유지하도록 강제합니다.
        # ---------------------------------------------------------
        if hasattr(model, "pre_processor") and hasattr(model.pre_processor, "transform"):
            model.pre_processor.transform = Compose([
                Resize((1024, 320)),
                ToImage(), 
                ToDtype(torch.float32, scale=True),
            ])
            logger.info("🔧 모델 내부 PreProcessor를 1024x320으로 강제 설정했습니다.")
        
        # 엔진 설정 및 학습
        engine = Engine(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            default_root_dir=OUTPUT_DIR,
            enable_checkpointing=True,
            precision="16-mixed", # Mixed Precision for memory optimization
        )
        
        # 💉 [Optim] 메모리 단편화 방지 환경변수 설정 (경고 메시지 반영)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        logger.info("🚀 학습 시작...")
        engine.fit(model=model, datamodule=datamodule)
        logger.success("✅ 학습 완료!")
        
        # ================== 3. 모델 및 결과 저장 ==================== #
        
        # 모델 저장
        model_path = f"{OUTPUT_DIR}/model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 모델 저장: {model_path}")
        
        # 메타데이터 저장
        info = {
            "model": args.model,
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "epochs": args.epochs,
            "image_size": (1024, 320),
            "anomalib_version": anomalib.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(f"{OUTPUT_DIR}/info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 메타데이터 저장: {OUTPUT_DIR}/info.json")

        # MLflow 아티팩트 로깅
        mlflow.log_artifact(OUTPUT_DIR)
        logger.success("✅ 결과 Blob 업로드 완료!")
                
    except Exception as e:
        logger.error(f"❌ {e}")
        raise

    mlflow.end_run()
    logger.success("🎉 완료!")

if __name__ == "__main__": main()
#성공기원