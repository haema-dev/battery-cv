import argparse
import os
import sys
import inspect
from pathlib import Path
import torch
import lightning.pytorch.trainer.trainer as trainer_module

# v3.5: "The Surgeon" - Tactical Monkey-Patch & Dynamic API Fix
# Anomalib 1.1.3의 치명적 설계 결함을 정밀 수술로 해결합니다.

# [핵심 수술 1] Trainer가 'task' 인자를 받고 죽는 것을 방지
# Engine은 task가 필요하지만, Trainer는 이를 모르기에 중간에서 가로채서 제거합니다.
original_trainer_init = trainer_module.Trainer.__init__
def patched_trainer_init(self, *args, **kwargs):
    if "task" in kwargs:
        print(f"🩹 [Surgeon] Intercepted and removed 'task' argument from Trainer: {kwargs['task']}")
        kwargs.pop("task")
    return original_trainer_init(self, *args, **kwargs)
trainer_module.Trainer.__init__ = patched_trainer_init

from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

def print_directory_tree(path, max_depth=3):
    print(f"\n📂 [File System Check] Root: {path}")
    base = Path(path)
    if not base.exists(): return
    for root, dirs, files in os.walk(base):
        level = root.replace(str(base), '').count(os.sep)
        if level > max_depth: continue
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:2]: print(f"{' ' * 4 * (level + 1)}{f}")

def find_anomalib_root(base_path):
    base = Path(base_path)
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            return p.parent
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("==================================================")
    print("🚀 STAGE 1: DEFINITIVE STABILIZATION V3.5 (SURGEON)")
    print("==================================================")
    
    # 1. 데이터 루트 탐색
    optimized_root = find_anomalib_root(data_path)
    print(f"🔎 Final Data Root: {optimized_root}")

    # 2. Folder 동적 인자 설정 (V3.3 검증 완료)
    sig_folder = inspect.signature(Folder)
    dm_args = {
        "name": "battery",
        "root": str(optimized_root),
        "normal_dir": "train/good",
        "test_split_mode": "from_dir"
    }
    if "normal_test_dir" in sig_folder.parameters: 
        dm_args["normal_test_dir"] = "test/normal"
    
    for k in ["abnormal_dir", "abnormal_test_dir", "test_abnormal_dir"]:
        if k in sig_folder.parameters:
            dm_args[k] = "test/damaged"
            break
    
    print(f"🛠️ Built Datamodule Args: {dm_args}")
    datamodule = Folder(**dm_args)

    # 3. 모델 설정
    model = Fastflow(backbone="resnet18", flow_steps=8)

    # 4. 엔진 설정 (이제 'task'를 수술로 해결했으므로 당당하게 넘깁니다)
    # task="classification"이 들어가야 gt_mask 에러가 나지 않습니다.
    # pixel_metrics=None을 통해 픽셀 단위 계산을 원천 차단합니다.
    print("⚙️ Initializing Engine with Classification task...")
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
        print(f"\n❌ FINAL FAILURE: {e}")
        print_directory_tree(data_path, max_depth=4)
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