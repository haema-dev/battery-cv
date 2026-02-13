import argparse
import os
import sys
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

# v2.2: Comprehensive Path Discovery & Diagnostic Logging
def diagnostic_ls(path, depth=3):
    """디렉토리 구조를 재귀적으로 출력하여 로그에 남깁니다."""
    print(f"\n🔍 [Diagnostic] Listing structure of: {path}")
    base = Path(path)
    if not base.exists():
        print(f"❌ Error: Path {path} does not exist.")
        return
    
    for p in base.rglob('*'):
        rel = p.relative_to(base)
        if len(rel.parts) > depth:
            continue
        indent = "  " * (len(rel.parts) - 1)
        suffix = "/" if p.is_dir() else ""
        print(f"{indent}- {rel.name}{suffix}")

def find_anomalib_root(base_path):
    """'train/good' 폴더가 있는 위치를 찾아 Anomalib root를 반환합니다."""
    base = Path(base_path)
    print(f"🔎 Searching for training data root starting from: {base}")
    
    # 1. 재귀적으로 'train/good' 폴더 찾기 (최대 깊이 제한으로 성능 확보)
    for path in base.rglob('train/good'):
        if path.is_dir():
            root = path.parent.parent
            print(f"✅ Found Anomalib root candidate: {root}")
            # 해당 위치에 test 폴더도 있는지 가볍게 확인
            if (root / "test").exists():
                print(f"✨ Verified root with 'test' folder: {root}")
                return root
            return root
            
    # 2. 'train' 폴더만이라도 찾기
    for path in base.rglob('train'):
        if path.is_dir():
            root = path.parent
            print(f"⚠️ Found 'train' but no 'good' subdir? Using root: {root}")
            return root
            
    print("❌ Failed to find a valid training structure. Falling back to base path.")
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("--------------------------------------------------")
    print(f"🚀 [Stage 1] FastFlow Training Pipeline (v2.2)")
    print(f"📍 Raw Data Path: {data_path}")
    
    # 디버깅용 로그: 현재 마운트된 데이터의 3단계 깊이까지 출력
    try:
        diagnostic_ls(data_path, depth=3)
    except Exception as e:
        print(f"⚠️ Warning: Diagnostic logging failed: {e}")

    # 데이터 구조 최적화 탐색
    optimized_root = find_anomalib_root(data_path)
    print(f"📁 Final Optimized Root: {optimized_root}")
    print(f"⏲️ Target Epochs: {epochs}")
    print("--------------------------------------------------")

    # 1. 데이터 모듈 설정
    # Anomalib 1.1.3 기준 가장 안전한 설정
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
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto"
    )

    # 4. 학습 시작
    print("⏳ Starting training (Engine.fit)...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        print("\n❌ Training Failed during engine.fit!")
        print(f"Error details: {e}")
        # 실패 시 다시 한 번 상세 경로 출력 (디버깅용)
        diagnostic_ls(optimized_root, depth=4)
        raise e
    
    # 5. 결과물 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_save_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Training completed successfully.")
    print(f"� Weights saved: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    # Azure ML 환경에서는 간혹 로그 전달이 늦어지므로 즉시 출력 강제
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    run_pipeline(args.data_path, args.output_dir, args.epochs)