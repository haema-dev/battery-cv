import os
import argparse
from loguru import logger
from ultralytics.models import YOLO
import mlflow

def main():
    parser = argparse.ArgumentParser(description="YOLO Training/Inference")
    
    # ================== 1. input/output 데이터 세팅 ==================
    parser.add_argument('--output_dir', type=str, default='./outputs', help='결과 저장 경로')
    args = parser.parse_args()
    
    # Azure ML Job 실행 시 자동으로 트래킹 서버와 연결
    mlflow.start_run()

    OUTPUT_DIR = args.output_dir
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info(f"📂 Output Dir: {os.path.abspath(OUTPUT_DIR)}")

    try:

        # ================== 2. YOLO 작업 ==================
        logger.info("📥 YOLO 모델 로드 중..")
        model = YOLO("yolov8n.pt")
        
        logger.info("🔍 샘플 추론 중..")
        results = model.predict(source="https://ultralytics.com/images/zidane.jpg", conf=0.25)
        
        # 결과 이미지 저장
        for i, result in enumerate(results):
            save_path = os.path.join(MODEL_DIR, f"result_{i}.jpg")
            result.save(filename=save_path)
            # MLflow에 개별 파일 로깅 (선택사항)
            mlflow.log_artifact(save_path, artifact_path="predictions")
        
        # 모델 저장
        model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
        model.save(model_path)
        logger.success("✅ 모델 저장 완료!")

        # ================== YOLO 작업 끝 ==================


        # ================== 3. 모델 등록 ==================
        mlflow.log_artifact(model_path, artifact_path="weights")

    except Exception as e:
        logger.error(f"❌ YOLO 테스트 실패: {e}")
        raise e # 에러를 다시 던져서 Job이 'Failed' 상태가 되게 함
    
    mlflow.end_run()
    logger.success("🎉 모든 프로세스 완료!")

if __name__ == "__main__":
    main()