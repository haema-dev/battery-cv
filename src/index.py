import argparse
import os
import sys
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

# v3.2: Definitive Metric & Task Fix (Addressing gt_mask error)
def print_directory_tree(path, max_depth=4):
    """디렉토리를 탐색하여 상세 구조를 로그에 남깁니다."""
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
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files[:2]: # 요약 출력
            print(f"{sub_indent}{f}")

def find_anomalib_root(base_path):
    """'train' 폴더가 있는 지점을 찾아 root로 반환합니다."""
    base = Path(base_path)
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            return p.parent
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("==================================================")
    print("🚀 STAGE 1 TRAINING: DEFINITIVE STABILIZATION V3.2")
    print("==================================================")
    
    # 0. 디렉토리 구조 출력
    print_directory_tree(data_path)

    # 1. 데이터 루트 탐색
    optimized_root = find_anomalib_root(data_path)
    print(f"🔎 Final Data Root: {optimized_root}")

    # 2. 데이터 모듈 설정 (Anomalib 1.1.3)
    # 로그에서 확인된 실제 폴더 구조를 기반으로 경로를 정교하게 매핑합니다.
    # [중요] normal_test_dir과 abnormal_test_dir을 명확히 분리하여 
    # 분류(Classification) 태스크에 필요한 Ground Truth를 확보합니다.
    datamodule = Folder(
        name="battery",
        root=str(optimized_root),
        normal_dir="train/good",
        normal_test_dir="test/normal",   # 정상 테스트 이미지
        abnormal_test_dir="test/damaged", # 불량 테스트 이미지 (최소 하나 필요)
        test_split_mode="from_dir"
    )

    # 3. 모델 설정
    model = Fastflow(backbone="resnet18", flow_steps=8)

    # 4. 엔진 설정
    # [핵심解決] 'classification' 태스크임을 명시하여 mask(gt_mask)를 찾지 않도록 합니다.
    # 또한 pixel_metrics를 None으로 설정하여 gt_mask 누락 에러를 완벽히 차단합니다.
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto",
        task="classification",
        pixel_metrics=None
    )

    # 5. 실행
    print(f"\n⏳ Starting Engine.fit (Target Epochs: {epochs})...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        print(f"\n❌ FAILURE during fit: {e}")
        # 실패 시 상세 로그 출력
        print_directory_tree(optimized_root, max_depth=3)
        raise e

    # 6. 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_save_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ SUCCESS: Training completed and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    # 출력 즉시 로깅
    sys.stdout.reconfigure(line_buffering=True)
    run_pipeline(args.data_path, args.output_dir, args.epochs)