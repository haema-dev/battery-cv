import argparse
import os
import sys
import inspect
from pathlib import Path
import torch
import lightning.pytorch.trainer.trainer as trainer_module

# v3.6: "The Master Surgeon" - Ultimate Argument Filtering Fix
# Anomalib 1.1.3's Engine leaks internal arguments (task, pixel_metrics, etc.) 
# into the PyTorch Lightning Trainer, causing continuous TypeErrors.
# This patch dynamically filters out any arguments that the Trainer doesn't recognize.

original_trainer_init = trainer_module.Trainer.__init__
# Trainer가 실제로 받을 수 있는 인자 목록을 미리 파악합니다.
TRAINER_ALLOWED_PARAMS = set(inspect.signature(original_trainer_init).parameters.keys())

def patched_trainer_init(self, *args, **kwargs):
    # Trainer가 모르는 인자들은 모두 가지치기(Filter) 합니다.
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in TRAINER_ALLOWED_PARAMS}
    
    removed_params = set(kwargs.keys()) - TRAINER_ALLOWED_PARAMS
    if removed_params:
        print(f"🩹 [Master Surgeon] Filtered out invalid Trainer arguments: {removed_params}")
        
    return original_trainer_init(self, *args, **filtered_kwargs)

trainer_module.Trainer.__init__ = patched_trainer_init

from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine

def find_anomalib_root(base_path):
    base = Path(base_path)
    for p in base.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            return p.parent
    return base

def run_pipeline(data_path, output_dir, epochs):
    print("==================================================")
    print("🚀 STAGE 1: DEFINITIVE STABILIZATION V3.6 (MASTER)")
    print("==================================================")
    
    # 1. 데이터 루트 탐색
    optimized_root = find_anomalib_root(data_path)
    print(f"🔎 Final Data Root: {optimized_root}")

    # 2. Folder 동적 인자 설정 (검증 완료된 로직)
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

    # 4. 엔진 설정
    # 이제 'Master Surgeon'이 모르는 인자는 자동으로 깎아내므로 
    # 마음 편히 필요한 인자들을 전달합니다.
    print("⚙️ Initializing Engine with Classification task...")
    engine = Engine(
        max_epochs=epochs,
        default_root_dir=output_dir,
        devices=1,
        accelerator="auto",
        task="classification",
        pixel_metrics=None # gt_mask 에러 방지용
    )

    # 5. 실행
    print(f"\n⏳ Starting Engine.fit (Epochs: {epochs})...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        print(f"\n❌ FINAL FAILURE: {e}")
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