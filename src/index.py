import argparse
import os
import sys
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

# v3.0: Ultimate Robust Stabilization
def print_directory_tree(path, max_depth=4):
    """디렉토리 구조를 끝까지 파헤쳐서 로그에 남깁니다. (디버깅의 핵심)"""
    print(f"\n� [File System Check] Root: {path}")
    base = Path(path)
    if not base.exists():
        print(f"❌ Error: {path} does not exist!")
        return

    for root, dirs, files in os.walk(base):
        level = root.replace(str(base), '').count(os.sep)
        if level > max_depth:
            continue
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        # 파일이 너무 많을 수 있으므로 5개까지만 출력
        for f in files[:5]:
            print(f"{sub_indent}{f}")
        if len(files) > 5:
            print(f"{sub_indent}... and {len(files)-5} more files")

def find_anomalib_root(base_path):
    """'train'과 'test' 폴더가 공존하는 최적의 지점을 찾습니다."""
    base = Path(base_path)
    print(f"\n🔎 Searching for data root in: {base_path}")
    
    # 대소문자 무시하고 'train' 폴더 찾기
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            root_candidate = p.parent
            # 해당 root에 'test' 폴더도 있는지 확인
            test_dir = root_candidate / "test"
            if test_dir.exists() and test_dir.is_dir():
                print(f"✨ Perfect Match Found: {root_candidate}")
                return root_candidate
            
            # test 폴더 이름이 대소문자가 다를 수 있으니 한 번 더 확인
            for sub in root_candidate.iterdir():
                if sub.is_dir() and sub.name.lower() == "test":
                    print(f"✨ Match Found (Case-insensitive test): {root_candidate}")
                    return root_candidate
                    
    # 못 찾으면 'train' 폴더의 부모라도 반환
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            print(f"⚠️ Only 'train' found. Returning parent: {p.parent}")
            return p.parent
            
    print("❌ No 'train' folder found anywhere. Using base path.")
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("==================================================")
    print("🚀 STAGE 1 TRAINING: DEFINITIVE STABILIZATION V3")
    print("==================================================")
    
    # 0. 시스템 환경 및 파일 구조 출력
    print(f"🐍 Python version: {sys.version}")
    print(f"📍 Input Data Path: {data_path}")
    try:
        print_directory_tree(data_path)
    except Exception as e:
        print(f"⚠️ Directory listing failed: {e}")

    # 1. 데이터 루트 탐색
    optimized_root = find_anomalib_root(data_path)
    
    # 2. 데이터 모듈 설정 (Anomalib 1.1.3 최적화 가이드)
    # normal_dir과 normal_test_dir은 root 아래의 상대 경로여야 합니다.
    # 스크린샷 구조상 root 아래에 바로 train/good이 있을 것으로 예상됩니다.
    datamodule = Folder(
        name="battery",
        root=str(optimized_root),
        normal_dir="train/good",
        normal_test_dir="test",
        test_split_mode="from_dir"
    )

    # 3. 모델 설정
    model = Fastflow(backbone="resnet18", flow_steps=8)

    # 4. 엔진 설정
    # TrainerTypeError(task)를 방지하기 위해 task 인자 완전 제거
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto"
    )

    # 5. 실행
    print(f"\n⏳ Starting Engine.fit (Epochs: {epochs})...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE during fit: {e}")
        # 실패 시 경로 재확인 (로그 추적용)
        print_directory_tree(optimized_root, max_depth=2)
        raise e

    # 6. 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_save_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ SUCCESS: Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    # 출력 즉시 로그 전송
    sys.stdout.reconfigure(line_buffering=True)
    
    run_pipeline(args.data_path, args.output_dir, args.epochs)