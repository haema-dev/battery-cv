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

    # ZIP 및 CSV 경로 설정
    zip_folder_rel = "3.개방데이터/1.데이터/Training/01.원천데이터"
    zip_dir = base_path / zip_folder_rel
    zip_file = zip_dir / "TS_Exterior_Img_Datasets_images_3.zip"
    csv_file = base_path / "good_list.csv"

    # ==========================================
    # 🔍 근본 해결: 구조 확인 + 존재 여부 검증
    # ==========================================
    logger.info(f"📍 마운트 루트 확인: {args.data_path}")
    
    if os.path.exists(args.data_path):
        import subprocess
        # 폴더 구조를 2단계까지 싹 훑어서 로그에 남깁니다. (경로가 꼬였는지 눈으로 확인용)
        result = subprocess.run(['ls', '-R', args.data_path], capture_output=True, text=True)
        logger.info(f"📂 실제 마운트된 파일 구조:\n{result.stdout[:2000]}") # 넉넉하게 출력
    
    # 실제 파일 존재 여부 체크 (이게 없으면 나중에 터짐)
    check_targets = {"데이터 디렉토리": zip_dir, "ZIP 파일": zip_file, "CSV 데이터": csv_file}
    for label, path in check_targets.items():
        if path.exists():
            logger.info(f"✅ {label} 확인 완료!: {path}")
        else:
            logger.error(f"❌ {label}을(를) 찾을 수 없음: {path}")
            # 필수 파일이 없으면 여기서 즉시 멈춰야 합니다.
            if path == zip_file or path == csv_file:
                raise FileNotFoundError(f"필수 파일 '{label}'이(가) 없습니다. 'ls -R' 로그를 보고 경로를 수정하세요.")
    
    # ========================================== Mlflow ==========================================
    logger.info(f"🔍 Mlflow 시작")
    with mlflow.start_run() as run:
        # 이제 run.info를 직접 참조하므로 절대 에러가 나지 않습니다.
        run_id = run.info.run_id
        logger.info(f"🚀 Azure ML Run ID (Database Key): {run_id}")
        
        # MLOps 추적을 위한 태그 설정 (비용 산정 및 데이터 리니지용)
        mlflow.set_tags({
            "Project_ID": "Test_Github_v1",
            "Model_Version": "v1.0.0",
            "Azure_Env_Name": "env-yolo",
            "Azure_Env_Version": "v21"
        })

        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"📂 {os.path.abspath(OUTPUT_DIR)}")
        
        try:
            # ================== 2. 이상탐지 작업 ==================== #
            logger.info("📥 Patchcore 로드")
            model = Patchcore(backbone="resnet18", pre_trained=True)

            # (실제 학습 로직...)
            img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
            cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), 3)
            score = np.random.random() * 0.3 + 0.2
            result = img.copy()
            label, color = ("ANOMALY", (0,0,255)) if score > 0.4 else ("NORMAL", (0,255,0))
            cv2.putText(result, f"{label} {score:.3f}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 결과 저장 및 아티팩트 로그
            cv2.imwrite(f"{OUTPUT_DIR}/result.jpg", result)
            model_path = f"{OUTPUT_DIR}/model.pt"
            torch.save(model.state_dict(), model_path)
            
            with open(f"{OUTPUT_DIR}/info.json", 'w') as f:
                json.dump({"backbone": "resnet18", "score": float(score)}, f)

            logger.success(f"✅ {score:.3f} ({label})")
            mlflow.log_artifact(OUTPUT_DIR)
            
        except Exception as e:
            logger.error(f"❌ {e}")
            # 에러가 나도 with 구문 덕분에 MLflow는 자동으로 'FAILED' 처리됩니다.
            raise

    # mlflow.end_run()은 이제 필요 없습니다. (with 구문이 끝날 때 자동 실행)
    logger.success("🎉 모든 프로세스 완료!")

if __name__ == "__main__": main()