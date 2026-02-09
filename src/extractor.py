import os
import re
import pandas as pd
from adlfs import AzureBlobFileSystem
import zipfile
from loguru import logger

def extract_battery_id(filename):
    """파일명에서 배터리 ID를 추출하는 유틸리티 함수"""
    match = re.search(r'cylindrical_(\d+)_', filename)
    return int(match.group(1)) if match else None

def run_selective_extraction(account_name, sas_token, container, blob_path, good_list_path, output_dir):
    """
    Azure Blob Storage의 ZIP 파일로부터 정상 배터리 이미지만을 선택적으로 추출합니다.
    
    Args:
        account_name (str): Azure 스토리지 계정 이름
        sas_token (str): SAS 토큰
        container (str): 컨테이너 이름
        blob_path (str): ZIP 파일 경로
        good_list_path (str): 정상 배터리 ID 목록 CSV 경로
        output_dir (str): 로컬 추출 경로 (예: ./temp_datasets/normal)
    """
    logger.info("📦 선택적 이미지 추출 모듈 시작")
    
    if not os.path.exists(good_list_path):
        logger.error(f"정상 목록 파일을 찾을 수 없습니다: {good_list_path}")
        return False
    
    # 1. 대상 배터리 ID 로드
    good_df = pd.read_csv(good_list_path)
    good_ids = set(good_df['battery_id'].unique())
    logger.info(f"정상 배터리 목록 로드 완료: {len(good_ids)}개 ID")

    try:
        # 2. ZIP 파일 오픈 (로컬 마운트 vs 원격 Blob)
        if not account_name or not sas_token:
            # 로컬 파일 시스템 직접 접근 (마운트 등)
            logger.info(f"로컬 파일 시스템 접근 시도: {blob_path}")
            zip_context = open(blob_path, "rb")
        else:
            # 원격 Blob 접근 (SAS 기반 직구)
            if sas_token.startswith('?'):
                sas_token = sas_token[1:]
            fs = AzureBlobFileSystem(account_name=account_name, sas_token=sas_token)
            full_blob_path = f"{container}/{blob_path}"
            logger.info(f"원격 ZIP 파일 연결 시도 (adlfs): {full_blob_path}")
            zip_context = fs.open(full_blob_path, "rb")
        
        with zip_context as f:
            with zipfile.ZipFile(f, 'r') as z:
                all_names = z.namelist()
                
                # 3. ZIP 내부 ID 필터링
                zip_ids = {extract_battery_id(n) for n in all_names if extract_battery_id(n)}
                match_ids = sorted(list(zip_ids.intersection(good_ids)))
                
                if not match_ids:
                    logger.warning("가져올 수 있는 정상 배터리 ID가 ZIP 내에 없습니다.")
                    return False
                
                logger.info(f"매칭된 배터리 ID 발견: {len(match_ids)}개 ({match_ids[0]} ~ {match_ids[-1]})")
                
                # 추출 대상 필터링
                files_to_extract = [n for n in all_names if extract_battery_id(n) in match_ids]
                logger.info(f"대상 이미지 총 {len(files_to_extract)}개 추출 준비 중...")

                # 4. 순차적 추출 및 저장
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                for i, filename in enumerate(files_to_extract):
                    save_path = os.path.join(output_dir, os.path.basename(filename))
                    if not os.path.exists(save_path):
                        with z.open(filename) as source, open(save_path, "wb") as target:
                            target.write(source.read())
                    
                    if (i + 1) % 1000 == 0 or (i + 1) == len(files_to_extract):
                        logger.info(f"진행 상황: [{i+1}/{len(files_to_extract)}] 파일 추출 완료")
        
        logger.success("✅ 선택적 추출 작업이 성공적으로 종료되었습니다.")
        return True

    except Exception as e:
        logger.error(f"추출 작업 중 치명적 오류 발생: {e}")
        return False
