import os, torch, argparse, mlflow, json, time
from loguru import logger
from anomalib.models import Fastflow
from anomalib.data import Folder
from anomalib.engine import Engine
import numpy as np, cv2

# 독립 모듈 임포트
from extractor import run_selective_extraction

def main():

    # ================== 1. input/output 설정 ==================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    # 데이터 추출을 위한 추가 인자
    parser.add_argument("--account_name", type=str, default="batterydata8ai6team")
    parser.add_argument("--sas_token", type=str, required=True)
    parser.add_argument("--container", type=str, default="battery-data-zip")
    parser.add_argument("--blob_path", type=str, default="103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_Exterior_Img_Datasets_images_3.zip")
    parser.add_argument("--good_list_path", type=str, default="good_list.csv")
    parser.add_argument("--epochs", type=int, default=10)
    
    args = parser.parse_args()
    
    mlflow.start_run()
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"📂 {os.path.abspath(OUTPUT_DIR)}")
    
    try:
        # ================== 2. 이상탐지 작업 ==================== #
        
        # ====== 전처리: 데이터 자동 추출 ====== 
        dataset_root = "./temp_datasets"
        normal_dir = os.path.join(dataset_root, "normal")
        
        success = run_selective_extraction(
            account_name=args.account_name,
            sas_token=args.sas_token,
            container=args.container,
            blob_path=args.blob_path,
            good_list_path=args.good_list_path,
            output_dir=normal_dir
        )

        if not success:
            raise RuntimeError("학습 데이터 준비(추출) 실패")

        # ====== Anomalib FastFlow 학습 ====== 
        logger.info("🚀 Fastflow 학습 프로세스 시작")
        datamodule = Folder(
            name="battery_anomaly",
            root=dataset_root,
            normal_dir="normal",
            train_batch_size=4,
            num_workers=4,
        )

        model = Fastflow(backbone="resnet18", flow_steps=8)

        engine = Engine(
            max_epochs=args.epochs,
            accelerator="gpu",
            devices=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            default_root_dir=OUTPUT_DIR
        )

        engine.fit(datamodule=datamodule, model=model)

        # 결과 변수 설정 (기존 템플릿 호환용)
        score = 0.0 # 학습용이므로 더미값
        label = "N/A"
        result = np.zeros((100, 100, 3), dtype=np.uint8) # 더미 이미지
        # ====== 여기까지 =======

        # mlflow 에 추가할 결과들이 있으면 추가해도 됨. 없으면 삭제.
        cv2.imwrite(f"{OUTPUT_DIR}/result.jpg", result)
        model_path = f"{OUTPUT_DIR}/model.pt"
        torch.save(model.state_dict(), model_path)
        with open(f"{OUTPUT_DIR}/info.json", 'w') as f:
            json.dump({
                "model": "FastFlow",
                "backbone": "resnet18",
                "zip_source": args.blob_path,
                "finish_time": time.ctime()
            }, f)


        # ================== 3. output blob mount ==================== #
        logger.success(f"✅ {score:.3f} ({label})")
        mlflow.log_artifact(OUTPUT_DIR)
                
    except Exception as e:
        logger.error(f"❌ {e}")
        raise

    mlflow.end_run()
    logger.success("🎉 완료!")

if __name__ == "__main__": main()
