import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from preprocess import preprocess_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Input raw data folder")
    parser.add_argument("--output_path", type=str, required=True, help="Output preprocessed data folder")
    parser.add_argument("--target_size", type=int, default=256, help="Target resize dimension (height and width)")
    parser.add_argument("--force", action="store_true", help="Force re-preprocessing even if output exists")
    
    args = parser.parse_args()
    
    input_root = Path(args.data_path)
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 📝 전처리 로직 및 설정 출력 (검토를 위해 상단에 노출)
    print("=" * 60)
    print("🚀 [Preprocessing Step] Batch Processing Initialized")
    print("-" * 60)
    print(f"📍 Configuration:")
    print(f"   - Input Directory:  {input_root}")
    print(f"   - Output Directory: {output_root}")
    print(f"   - Target Resolution: {args.target_size}x{args.target_size} (Letterbox)")
    print(f"   - CLAHE Applied:    Yes (ClipLimit: 2.0, Grid: 8x8)")
    print(f"   - Skip Existing:    {'No (Force Rerun)' if args.force else 'Yes (Idempotent mode)'}")
    print("=" * 60)
    
    # 지원하는 이미지 확장자
    exts = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")
    
    # 모든 이미지 파일 수집
    all_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(exts):
                all_files.append(Path(root) / file)
                
    total_images = len(all_files)
    print(f"📊 Total images discovered in raw data: {total_images}")
    
    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "categories": {} # 하위 폴더별 분포 확인용
    }
    
    # 전처리 루프
    for img_path in tqdm(all_files, desc="🔄 Preprocessing Progress"):
        # 카테고리(부모 폴더명) 통계 수집
        category = img_path.parent.name
        stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        # 출력 경로 결정 (입력의 상대 경로 구조 그대로 복사)
        rel_path = img_path.relative_to(input_root)
        save_path = output_root / rel_path
        
        # [중복 작업 방지 로직] 이미 파일이 존재하면 스킵 (Force 옵션 없을 시)
        if save_path.exists() and not args.force:
            stats["skipped"] += 1
            continue
            
        try:
            # 상위 디렉토리 생성
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 1. 원본 로드 및 전처리 (CLAHE + Resize)
            # preprocess.py의 preprocess_image 함수를 재사용합니다.
            processed_img = preprocess_image(img_path, target_size=(args.target_size, args.target_size))
            
            # 2. 결과 저장
            cv2.imwrite(str(save_path), processed_img)
            stats["processed"] += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            stats["failed"] += 1
            
    # 최종 결과 보고 (AML 로그에서 확인 가능)
    print("\n" + "=" * 60)
    print("✅ Preprocessing Complete!")
    print("-" * 60)
    print(f"✨ Newly Processed: {stats['processed']}")
    print(f"⏩ Skipped Existing: {stats['skipped']}")
    print(f"⚠️ Failed:           {stats['failed']}")
    print("-" * 60)
    print("📁 Category Distribution (Raw):")
    for cat, count in sorted(stats["categories"].items()):
        print(f"   - {cat}: {count:4d} images")
    print("=" * 60)

if __name__ == "__main__":
    main()
