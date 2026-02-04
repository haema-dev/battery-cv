import os
import argparse
from loguru import logger

# azure module
from azureml.core.run import Run
from ultralytics.models import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO")
    
    # ================== 1. config 세팅 ==================
    parser.add_argument('--output_dir', type=str, default='./outputs', help='결과 저장 경로')
    
    args = parser.parse_args()
    
    # ================== 1. 경로 및 환경 설정 ==================
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"📂 Output Root: {os.path.abspath(OUTPUT_DIR)}")

    # 모델 저장 디렉토리
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    logger.info(f"📂 작업 디렉토리 설정 완료:")
    logger.info(f"   - Model Save Dir: {os.path.abspath(MODEL_DIR)}")
    logger.info(f"   - Log Save Dir: {os.path.abspath(LOG_DIR)}")

    # ================== 2. YOLO ==================
    print("===================")
    print("로직 작성 자유롭게")
    print("===================")

    try:
        
        # ✅ 테스트 1: YOLO 모델 로드 (사전학습 모델)
        logger.info("📥 YOLO 모델 로드 중...")
        model = YOLO("yolov8n.pt")  # nano 버전 (가장 빠름)
        logger.success("✅ YOLO 모델 로드 완료!")
        
        # ✅ 테스트 2: 샘플 이미지로 추론
        logger.info("🔍 샘플 추론 중...")
        results = model.predict(source="https://ultralytics.com/images/zidane.jpg", conf=0.25)
        
        # 결과 저장
        for i, result in enumerate(results):
            result.save(filename=os.path.join(MODEL_DIR, f"result_{i}.jpg"))
        
        logger.success(f"✅ 추론 완료! 결과: {len(results)}개")
        
        # ✅ 테스트 3: 모델 저장
        logger.info("💾 모델 저장 중...")
        model.save(os.path.join(MODEL_DIR, "yolov8n.pt"))
        logger.success("✅ 모델 저장 완료!")
        
    except Exception as e:
        logger.error(f"❌ YOLO 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    # ================== 3. Azure 업로드 ==================
    try:
        run = Run.get_context()
        run.upload_folder(name="outputs", path=OUTPUT_DIR)
        logger.success("✅ Outputs uploaded to Azure ML!")
    except Exception as e:
        logger.warning(f"⚠️ Upload failed (로컬 실행인 경우 무시): {e}")

    logger.success("🎉 모든 테스트 완료!")


if __name__ == "__main__":
    main()
