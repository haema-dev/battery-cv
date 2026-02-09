import os, torch, argparse, mlflow, json, time
from loguru import logger
from anomalib.models import Fastflow
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from pathlib import Path
import numpy as np, cv2
import adlfs
import fsspec


def main():

    # ================== 1. input/output 설정 ==================== #
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_path", type=str, help="Path to mounted data asset")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument("--epochs", type=int, default=10)    

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
        logger.info(f"📂 실제 마운트된 파일 구조:\n{result.stdout[:2000]}")
        
        # 이미지 파일 수 확인
        image_count = len([f for f in os.listdir(args.data_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        logger.info(f"📷 마운트된 이미지 수: {image_count}개")
    else:
        raise FileNotFoundError(f"마운트 경로를 찾을 수 없습니다: {args.data_path}")
    
    # ==========================================
    # [나중 사용] ZIP 기반 데이터 추출 코드 (현재 비활성화)
    # ==========================================
    # zip_folder_rel = "3.개방데이터/1.데이터/Training/01.원천데이터"
    # zip_dir = base_path / zip_folder_rel
    # zip_file = zip_dir / "TS_Exterior_Img_Datasets_images_3.zip"
    # csv_file = base_path / "good_list.csv"
    # check_targets = {"데이터 디렉토리": zip_dir, "ZIP 파일": zip_file, "CSV 데이터": csv_file}
    # for label, path in check_targets.items():
    #     if path.exists():
    #         logger.info(f"✅ {label} 확인 완료!: {path}")
    #     else:
    #         logger.error(f"❌ {label}을(를) 찾을 수 없음: {path}")
    #         if path == zip_file or path == csv_file:
    #             raise FileNotFoundError(f"필수 파일 '{label}'이(가) 없습니다.")
    
    # ========================================== Mlflow ==========================================
    mlflow.start_run()
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"📂 {os.path.abspath(OUTPUT_DIR)}")
    
    try:
        # ================== 2. 이상탐지 작업 ==================== #
        
        # ====== PatchCore 학습 ====== 
        logger.info("📥 PatchCore 모델 및 데이터셋 구성")
        
        # 데이터셋 구성 (마운트된 압축해제 이미지 사용)
        # battery-data-unzip 컨테이너에서 마운트된 이미지 사용
        dataset_root = str(base_path)  # 마운트된 경로 직접 사용
        logger.info(f"📂 학습 데이터 경로: {dataset_root}")
        
        datamodule = Folder(
            name="battery",
            root=dataset_root,
            normal_dir=".",  # 이미지가 루트에 직접 있음
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
        )
        
        # 모델 초기화
        model = Patchcore(
            backbone="resnet18",
            pre_trained=True,
            layers=["layer2", "layer3"],
        )
        
        # 엔진 설정 및 학습
        engine = Engine(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            default_root_dir=OUTPUT_DIR,
            enable_checkpointing=True,
        )
        
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
            "model": "PatchCore",
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "epochs": args.epochs,
            "image_size": 256,
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