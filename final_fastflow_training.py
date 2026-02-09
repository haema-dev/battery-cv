import os
import glob
import random
import pandas as pd
from PIL import Image
import torch
import gc
from torchvision import transforms
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Fastflow # 호환성 왕

# [0] 메모리 초기화
try:
    gc.collect()
    torch.cuda.empty_cache()
except: pass

print("========================================")
print("      [Final All-in-One] 학습 + 실전 테스트")
print("========================================")

# 1. 경로 설정 (리사이징된 폴더 사용)
# 주의: 이 경로는 Azure ML Compute Instance 내부 경로입니다.
resized_path = "./smoke_data_resized"
target_root = None
normal_folder_name = None

# 리사이징된 이미지가 있는지 확인
for root, dirs, files in os.walk(resized_path):
    if any(f.lower().endswith('.png') for f in files):
        target_root = os.path.dirname(root)
        normal_folder_name = os.path.basename(root)
        break

if not target_root: 
    print("🚨 리사이징된 이미지가 없습니다! (리사이징 단계 필요)")
    # raise ValueError("이미지 없음") # 로컬 테스트용으로 주석 처리

print(f"[*] 학습 경로: {target_root}")

# 2. 데이터셋 (배치 4)
datamodule = Folder(
    name="battery_final",
    root=target_root,
    normal_dir=normal_folder_name,
    train_batch_size=4,
    num_workers=0,
)

# 3. 모델 생성 (FastFlow)
print("[*] 모델 로딩 중...")
model = Fastflow(backbone="resnet18", flow_steps=8)

# 4. 학습 (중간 평가 스킵 -> 속도 향상 & 에러 방지)
engine = Engine(
    max_epochs=1, 
    accelerator="gpu", 
    devices=1,
    limit_val_batches=0,     # 중간 평가 금지
    num_sanity_val_steps=0   # 시작 전 검증 금지
)

print("\n[*] 모델 학습 시작... (FastFlow)")
engine.fit(datamodule=datamodule, model=model)

print("\n✅ 학습 완료! 바로 실전 테스트로 넘어갑니다...")

# 5. 실전 테스트 (CSV 무시하고 폴더 파일 직접 평가)
print("---------------------------------------------------------------")
print("       파일명 (랜덤 20개)       |  이상 점수  |  AI 판단")
print("---------------------------------------------------------------")

# 실제 파일 리스트 확보
all_files = []
for root, dirs, files in os.walk(resized_path):
    for f in files:
        if f.lower().endswith('.png'):
            all_files.append(os.path.join(root, f))

# 랜덤하게 20개만 뽑기
if len(all_files) > 0:
    test_files = random.sample(all_files, min(20, len(all_files)))

    # 모델 준비
    model.eval()
    if torch.cuda.is_available(): model.cuda()

    # 수동 변환 도구
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for fpath in test_files:
        try:
            # 이미지 로드 & 변환
            image_pil = Image.open(fpath).convert("RGB")
            image_tensor = val_transform(image_pil)
            if torch.cuda.is_available(): image_tensor = image_tensor.cuda()
            
            # 예측
            with torch.no_grad():
                out = model(image_tensor.unsqueeze(0))
                
                # 점수 추출
                if isinstance(out, dict) and "pred_scores" in out:
                    score = out["pred_scores"].item()
                elif isinstance(out, tuple):
                    score = out[1].item()
                else:
                    score = out.item() if hasattr(out, "item") else 0.5
                
                # 판단 (임계값 0.5 기준)
                result = "🔴 불량의심" if score >= 0.5 else "🟢 정상"
                
            fname = os.path.basename(fpath)
            fname_short = (fname[:20] + '..') if len(fname) > 20 else fname
            
            print(f" {fname_short:22s} |   {score:.4f}   | {result}")

        except Exception as e:
            print(f" 에러 발생: {e}")

    print("---------------------------------------------------------------")
    print("✅ 점수가 위처럼 출력되면 모든 과정 성공입니다!")
else:
    print("❌ 테스트할 파일이 없습니다.")
