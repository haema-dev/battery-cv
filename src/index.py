import argparse
import os
import sys
import inspect
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

# v3.4: Definitive Stabilization (The "Zero-Regression" Fix)
def find_anomalib_root(base_path):
    base = Path(base_path)
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            return p.parent
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("==================================================")
    print("🚀 STAGE 1: DEFINITIVE STABILIZATION V3.4")
    print("==================================================")
    
    # 1. 데이터 루트 탐색
    optimized_root = find_anomalib_root(data_path)
    print(f"🔎 Final Data Root: {optimized_root}")

    # 2. Folder 동적 인자 설정 (V3.3에서 검증된 로직)
    sig_folder = inspect.signature(Folder)
    dm_args = {
        "name": "battery",
        "root": str(optimized_root),
        "normal_dir": "train/good",
        "test_split_mode": "from_dir"
    }
    if "normal_test_dir" in sig_folder.parameters: 
        dm_args["normal_test_dir"] = "test/normal"
    
    # abnormal_dir 명칭 자동 대응
    for k in ["abnormal_dir", "abnormal_test_dir", "test_abnormal_dir"]:
        if k in sig_folder.parameters:
            dm_args[k] = "test/damaged"
            break
    
    print(f"🛠️ Built Datamodule Args: {dm_args}")
    datamodule = Folder(**dm_args)

    # 3. 모델 설정
    model = Fastflow(backbone="resnet18", flow_steps=8)
    # gt_mask 에러를 방지하기 위해 모델 레벨에서 태스크를 설정
    if hasattr(model, "task"):
        model.task = "classification"

    # 4. 엔진 설정 (TypeError 방지를 위한 초슬림화)
    # v1.1.3 Engine은 __init__에서 'task'를 받으면 내부 Trainer로 넘기는데, 
    # 정작 Trainer는 'task' 인자를 몰라서 에러가 납니다.
    # 따라서 __init__에서는 제거하고, 객체 생성 후에 속성으로 설정합니다.
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto"
    )
    
    # 인스턴스 생성 후 태스크 설정 (가장 안전한 방법)
    if hasattr(engine, "task"):
        engine.task = "classification"
    
    # 픽셀 메트릭 에러(gt_mask) 원천 차단
    if hasattr(engine, "pixel_metrics"):
        engine.pixel_metrics = None

    # 5. 실행
    print(f"\n⏳ Starting Engine.fit...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        print(f"\n❌ FAILURE: {e}")
        raise e

    # 6. 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_save_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ SUCCESS: Stage 1 Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    run_pipeline(args.data_path, args.output_dir, args.epochs)