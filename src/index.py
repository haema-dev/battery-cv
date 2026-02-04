import os, torch, argparse, mlflow, json
from loguru import logger
from anomalib.models import Patchcore
import numpy as np, cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()
    
    mlflow.start_run()
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"📂 {os.path.abspath(OUTPUT_DIR)}")
    
    try:
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
        mlflow.log_artifact(f"{OUTPUT_DIR}/result.jpg")
        mlflow.log_metric("anomaly_score", score)
        mlflow.log_param("status", label)

        model_path = f"{OUTPUT_DIR}/model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        with open(f"{OUTPUT_DIR}/info.json", 'w') as f:
            json.dump({"backbone": "resnet18", "score": float(score)}, f)
        mlflow.log_artifact(f"{OUTPUT_DIR}/info.json")

        logger.success(f"✅ {score:.3f} ({label})")
                
    except Exception as e:
        logger.error(f"❌ {e}")
        raise
    
    mlflow.end_run()
    logger.success("🎉 완료!")

if __name__ == "__main__": main()
