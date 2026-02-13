import argparse
import os
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

# 사용자님의 요청에 따른 '일관성' 확보: 추론 전용 라이브러리인 TorchInferencer도 함께 준비
try:
    from anomalib.deploy import TorchInferencer
    HAS_INFERENCER = True
except ImportError:
    HAS_INFERENCER = False

def run_pipeline(data_path, output_dir, epochs):
    print("--------------------------------------------------")
    print(f"🚀 [Stage 1] FastFlow Training Pipeline (v2: 100e)")
    print(f"📍 Data: {data_path}")
    print(f"⏲️ Target Epochs: {epochs}")
    print(f"🛠️ Inferencer Ready: {HAS_INFERENCER}")
    print("--------------------------------------------------")

    # 1. 데이터 모듈 설정 (Anomalib 1.x 최신 API 대응)
    # 로그 확인 결과 'test_dir' 인자가 지원되지 않으므로 'normal_test_dir'로 수정
    datamodule = Folder(
        name="battery",
        root=data_path,
        normal_dir="train/good",
        normal_test_dir="test/good",    # test_dir 대신 구체적인 경로 지정
        test_split_mode="from_dir",
        task="classification",
        image_size=(256, 256)
    )

    # 2. 모델 설정 (FastFlow)
    model = Fastflow(backbone="resnet18", flow_steps=8)

    # 3. 엔진 설정 (T4 GPU 사용)
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto",
        task="classification"
    )

    # 4. 학습 시작
    print("⏳ Starting training...")
    engine.fit(model=model, datamodule=datamodule)
    
    # 5. 결과물 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 추론 단계(Stage 2)에서 바로 사용할 수 있도록 가중치 저장
    model_save_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    
    print(f"✅ Training completed. Weights saved: {model_save_path}")

    # 6. [일관성 검증] TorchInferencer로 로드 가능한지 확인
    if HAS_INFERENCER:
        try:
            print("🔍 Verifying model consistency with TorchInferencer...")
            # 검증 시에는 cpu로 로드 테스트
            inferencer = TorchInferencer(path=model_save_path, device="cpu")
            print("✨ Success: Model is compatible with TorchInferencer API.")
        except Exception as e:
            print(f"⚠️ Note: Inferencer verification skipped or errored: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    
    args = parser.parse_args()
    run_pipeline(args.data_path, args.output_dir, args.epochs)