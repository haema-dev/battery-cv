import argparse
import os
import sys
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

# v3.1: Definitive Path Fix (No-Test Fallback)
def print_directory_tree(path, max_depth=4):
    """디렉토리 구조를 상세히 출력하여 로그에 남깁니다."""
    print(f"\n📂 [File System Check] Root: {path}")
    base = Path(path)
    if not base.exists():
        print(f"❌ Error: {path} does not exist!")
        return

    for root, dirs, files in os.walk(base):
        level = root.replace(str(base), '').count(os.sep)
        if level > max_depth:
            continue
        indent = ' ' * 4 * level
        basename = os.path.basename(root)
        if not basename: # root의 경우 basename이 비어있을 수 있음
            basename = str(root)
        print(f"{indent}{basename}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files[:3]: # 파일은 3개만 출력
            print(f"{sub_indent}{f}")
        if len(files) > 3:
            print(f"{sub_indent}... and {len(files)-3} more files")

def find_anomalib_root(base_path):
    """'train' 폴더가 포함된 최적의 경로를 찾습니다."""
    base = Path(base_path)
    print(f"\n🔎 Searching for data root in: {base_path}")
    
    # 1단계: 재귀적으로 'train' 폴더 찾기
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            root_candidate = p.parent
            print(f"✅ Found data root candidate: {root_candidate}")
            return root_candidate
            
    print("❌ No 'train' folder found anywhere. Falling back to base path.")
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("==================================================")
    print("🚀 STAGE 1 TRAINING: DEFINITIVE STABILIZATION V3.1")
    print("==================================================")
    
    # 0. 시스템 환경 및 파일 구조 출력
    print(f"🐍 Python version: {sys.version}")
    print(f"📍 Raw Mount Path: {data_path}")
    try:
        print_directory_tree(data_path)
    except Exception as e:
        print(f"⚠️ Directory listing failed: {e}")

    # 1. 데이터 루트 탐색
    optimized_root = find_anomalib_root(data_path)
    
    # 2. 데이터 유효성 검증 (test 폴더가 선택적임을 반영)
    train_dir = optimized_root / "train"
    test_dir = optimized_root / "test" # 대문자 Test일 가능성도 고려하여 체크할 수 있지만 rglob이 base를 잡아줌
    
    if not train_dir.exists():
        # rglob으로 못 찾았을 경우를 대비한 최후의 보루
        print(f"❌ Error: 'train' directory not found even in {optimized_root}")
        # 여기서 죽기 전에 전체 리스트 한 번 더 출력
        print_directory_tree(data_path, max_depth=5)
        sys.exit(1)

    # 3. 데이터 모듈 설정 (Anomalib 1.1.3 최적화)
    # 이번 에러의 핵심: test 폴더가 없으면 인자에서 제외합니다.
    datamodule_args = {
        "name": "battery",
        "root": str(optimized_root),
        "normal_dir": "train/good"
    }

    if test_dir.exists() and test_dir.is_dir():
        print(f"📁 'test' folder found at {test_dir}. Enabling validation mode.")
        datamodule_args["normal_test_dir"] = "test"
        datamodule_args["test_split_mode"] = "from_dir"
    else:
        print(f"⚠️ 'test' folder NOT found. Proceeding with 'train-only' configuration.")
        # test_split_mode를 지정하지 않으면 Anomalib이 내부적으로 split하거나 학습만 진행함

    datamodule = Folder(**datamodule_args)

    # 4. 모델 설정
    model = Fastflow(backbone="resnet18", flow_steps=8)

    # 5. 엔진 설정
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto"
    )

    # 6. 실행
    print(f"\n⏳ Starting Engine.fit (Target Epochs: {epochs})...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE during fit: {e}")
        # 실패 시 로그 분석의 정석: 경로 재확인
        print_directory_tree(optimized_root, max_depth=3)
        raise e

    # 7. 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_save_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ SUCCESS: Stage 1 Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    # 출력 강제 동기화
    sys.stdout.reconfigure(line_buffering=True)
    
    run_pipeline(args.data_path, args.output_dir, args.epochs)