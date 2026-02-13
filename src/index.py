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
    if (base / "train/good").exists():
        return base
    possible_sub = base / "datasets/resized"
    if (possible_sub / "train/good").exists():
        return possible_sub
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

    # 1. 데이터 모듈 설정 (Anomalib 1.x 초정밀 다이어트)
    # 모든 버전 민감 인자(task, image_size, test_dir 등)를 제거하고 최소 필수값만 유지
    datamodule = Folder(
        name="battery",
        root=str(optimized_root),
        normal_dir="train/good",
        normal_test_dir="test",
        test_split_mode="from_dir"
    )

    # 2. 모델 설정 (FastFlow)
    model = Fastflow(backbone="resnet18", flow_steps=8)

    # 3. 엔진 설정
    # 'task' 인자가 Trainer까지 넘어가 에러를 유발하므로 과감히 제거 (기본값 활용)
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto"
    )

    # 4. 학습 시작
    print("⏳ Starting training...")
    # datamodule에서 데이터 셋을 가져오면 엔진이 자동으로 구조 파악
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