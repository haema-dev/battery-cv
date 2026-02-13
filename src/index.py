import argparse
import os
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

# TorchInferencer consistency
try:
    from anomalib.deploy import TorchInferencer
    HAS_INFERENCER = True
except ImportError:
    HAS_INFERENCER = False

def find_data_root(base_path):
    """'train/good' 폴더가 포함된 최적의 경로를 찾습니다."""
    base = Path(base_path)
    # 1. 바로 아래에 있는 경우
    if (base / "train/good").exists():
        return base
    
    # 2. datasets/resized/ 하위에 있는 경우 (사용자 스크린샷 구조)
    possible_sub = base / "datasets/resized"
    if (possible_sub / "train/good").exists():
        return possible_sub
    
    # 3. 더 깊이 있는 경우 검색
    found = list(base.glob("**/train/good"))
    if found:
        return found[0].parent.parent
        
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("--------------------------------------------------")
    print(f"🚀 [Stage 1] FastFlow Training Pipeline (v2: 100e)")
    print(f"📍 Raw Data Path: {data_path}")
    
    optimized_root = find_data_root(data_path)
    print(f"📁 Optimized Root: {optimized_root}")
    print(f"⏲️ Target Epochs: {epochs}")
    print(f"🛠️ Inferencer Ready: {HAS_INFERENCER}")
    print("--------------------------------------------------")

    # 1. 데이터 모듈 설정 (Anomalib 1.x 최소 사양 규격)
    # 'test_dir', 'task'에 이어 'image_size'까지 1.x 최신 버전에서는 지원되지 않음 확인
    # 가장 필수적인 인자들로만 구성하여 호환성 극대화
    datamodule = Folder(
        name="battery",
        root=str(optimized_root),
        normal_dir="train/good",
        normal_test_dir="test",
        test_split_mode="from_dir"
    )

    # 2. 모델 설정 (FastFlow)
    # FastFlow 모델에서 기본적으로 256x256 등을 처리하므로 데이터에서 뺄 수 있습니다.
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
    model_save_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Training completed. Weights saved: {model_save_path}")

    # 6. 일관성 검증
    if HAS_INFERENCER:
        try:
            print("🔍 Verifying model consistency with TorchInferencer...")
            inferencer = TorchInferencer(path=model_save_path, device="cpu")
            print("✨ Success: Model is compatible with TorchInferencer API.")
        except Exception as e:
            print(f"⚠️ Note: Inferencer verification failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run_pipeline(args.data_path, args.output_dir, args.epochs)