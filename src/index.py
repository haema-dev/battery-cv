import argparse
import os
import sys
import inspect
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

# v3.3: "God Mode" Dynamic API Inspector (No More TypeErrors!)
def print_directory_tree(path, max_depth=3):
    print(f"\n📂 [File System Check] Root: {path}")
    base = Path(path)
    if not base.exists():
        print(f"❌ Error: {path} does not exist!")
        return
    for root, dirs, files in os.walk(base):
        level = root.replace(str(base), '').count(os.sep)
        if level > max_depth: continue
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files[:2]: print(f"{sub_indent}{f}")

def find_anomalib_root(base_path):
    base = Path(base_path)
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            return p.parent
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("==================================================")
    print("🚀 STAGE 1: ULTIMATE DYNAMIC STABILIZATION V3.3")
    print("==================================================")
    
    # 1. 데이터 루트 탐색
    optimized_root = find_anomalib_root(data_path)
    print(f"🔎 Final Data Root: {optimized_root}")

    # 2. "God Mode" 동적 파라미터 빌더
    # Anomalib 버전마다 다른 인자명(abnormal_dir vs abnormal_test_dir 등)을 
    # 런타임에 직접 검사해서 맞춰줍니다. 이제 TypeError는 원천 차단됩니다.
    sig = inspect.signature(Folder)
    params = sig.parameters
    print(f"🧬 Detected Folder API signature: {sig}")

    datamodule_args = {
        "name": "battery",
        "root": str(optimized_root),
        "normal_dir": "train/good",
        "test_split_mode": "from_dir"
    }

    # 정상 테스트 경로 설정
    if "normal_test_dir" in params:
        datamodule_args["normal_test_dir"] = "test/normal"
    
    # 불량 테스트 경로 설정 (가장 에러가 잦은 부분 동적 처리)
    if "abnormal_dir" in params:
        datamodule_args["abnormal_dir"] = "test/damaged"
    elif "abnormal_test_dir" in params:
        datamodule_args["abnormal_test_dir"] = "test/damaged"
    elif "test_abnormal_dir" in params:
        datamodule_args["test_abnormal_dir"] = "test/damaged"
    
    print(f"🛠️ Built Datamodule Args: {datamodule_args}")
    datamodule = Folder(**datamodule_args)

    # 3. 모델 설정
    model = Fastflow(backbone="resnet18", flow_steps=8)

    # 4. 엔진 설정
    # gt_mask 에러를 방지하기 위해 classification 태스크임을 명시
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto",
        task="classification",
        pixel_metrics=None
    )

    # 5. 실행
    print(f"\n⏳ Starting Engine.fit (Epochs: {epochs})...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE: {e}")
        # 실패 시 즉시 파일 시스템 구조 출력 (마지막 수단)
        print_directory_tree(data_path, max_depth=4)
        raise e

    # 6. 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_save_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ SUCCESS: Training completed. Model saved at {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    # 로그 버퍼링 해제
    sys.stdout.reconfigure(line_buffering=True)
    run_pipeline(args.data_path, args.output_dir, args.epochs)