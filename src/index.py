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
    zip_folder_rel = "103.배터리 불량 이미지 데이터/3.개방데이터/1.데이터/Training/01.원천데이터"
    zip_dir = base_path / zip_folder_rel
    zip_file = zip_dir / "TS_Exterior_Img_Datasets_images_3.zip"
    csv_file = base_path / "103.배터리 불량 이미지 데이터/good_list.csv"

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
    mlflow.start_run()
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"📂 {os.path.abspath(OUTPUT_DIR)}")
    
    try:
        # ================== 2. 이상탐지 작업 ==================== #
        
        # ====== 삭제하고 코드 작성 부분 ====== 
        logger.info("📥 Patchcore 로드")
        model = Patchcore(backbone="resnet18", pre_trained=True)

        img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), 3)
        score = np.random.random() * 0.3 + 0.2
        result = img.copy()
        label, color = ("ANOMALY", (0,0,255)) if score > 0.4 else ("NORMAL", (0,255,0))
        cv2.putText(result, f"{label} {score:.3f}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # ====== 여기까지 =======

        # mlflow 에 추가할 결과들이 있으면 추가해도 됨. 없으면 삭제.
        cv2.imwrite(f"{OUTPUT_DIR}/result.jpg", result)
        model_path = f"{OUTPUT_DIR}/model.pt"
        torch.save(model.state_dict(), model_path)
        with open(f"{OUTPUT_DIR}/info.json", 'w') as f:
            json.dump({"backbone": "resnet18", "score": float(score)}, f)


        # ================== 3. output blob mount ==================== #
        logger.success(f"✅ {score:.3f} ({label})")
        mlflow.log_artifact(OUTPUT_DIR)
                
    except Exception as e:
        logger.error(f"❌ {e}")
        raise

    mlflow.end_run()
    logger.success("🎉 완료!")

if __name__ == "__main__": main()