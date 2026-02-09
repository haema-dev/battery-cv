
import sys
import os
import torch
import cv2
import numpy as np
import warnings
from run_inference import load_model, preprocess_image, infer_single_pass

# 경고 무시
warnings.filterwarnings("ignore")

# --- 설정 ---
# 모델 체크포인트 경로 (run_inference.py와 동일)
CHECKPOINT_PATH = r"C:\Users\EL98\Downloads\BatterySample\results_cv1_purified\Patchcore\battery_gate\latest\weights\lightning\model.ckpt"
IMAGE_SIZE = (256, 256)

def generate_heatmap(image_path, model):
    print(f"[*] 이미지 분석 중: {os.path.basename(image_path)}")
    
    # 1. 이미지 읽기
    if not os.path.exists(image_path):
        print(f"[!] 파일을 찾을 수 없습니다: {image_path}")
        return

    # 한글 경로 지원을 위한 cv2 읽기
    stream = open(image_path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    original_img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    stream.close()
    
    if original_img is None:
        print("[!] 이미지 디코딩 실패")
        return

    # 2. 전처리 및 추론
    # 모델 학습 해상도인 256x256으로 고정
    input_tensor = preprocess_image(image_path, target_size=(256, 256))
    
    if input_tensor is None:
        print("[!] 전처리 실패")
        return

    # 원본 이미지 크기 (나중에 사용)
    h, w = original_img.shape[:2]

    # 추론 (결과 맵 추출)
    with torch.no_grad():
        output = model.model(input_tensor)

    print(f"\n[Debug] Model Output Type: {type(output)}")
    if isinstance(output, tuple):
        print(f"[Debug] Output Shape (Tuple): {[x.shape if hasattr(x, 'shape') else x for x in output]}")
    elif hasattr(output, 'shape'):
        print(f"[Debug] Output Shape (Tensor): {output.shape}")
    elif isinstance(output, dict):
        print(f"[Debug] Output Keys: {output.keys()}")
        
    # Anomaly Map 추출
    if isinstance(output, tuple):
        anomaly_map = output[0] # PatchCore는 보통 tuple의 첫번째가 map
    elif isinstance(output, dict) and "anomaly_map" in output:
        anomaly_map = output["anomaly_map"]
    else:
        # 구조가 다를 경우 확인 필요, 임시로 텐서 자체라고 가정
        anomaly_map = output

    # 텐서를 넘파이로 변환
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.cpu().numpy()
        
    print(f"\n[Debug] Input Tensor Stats - Min: {input_tensor.min():.4f}, Max: {input_tensor.max():.4f}, Mean: {input_tensor.mean():.4f}")
    print(f"[Debug] Raw Anomaly Map Stats:")
    print(f"    Min: {anomaly_map.min():.6f}")
    print(f"    Max: {anomaly_map.max():.6f}")
    print(f"    Mean: {anomaly_map.mean():.6f}")
    print(f"    StdDev: {anomaly_map.std():.6f}")
    print(f"    Shape: {anomaly_map.shape}")

    print(f"    [Debug] Map Stats - Min: {anomaly_map.min():.8f}, Max: {anomaly_map.max():.8f}, Mean: {anomaly_map.mean():.6f}")
    if anomaly_map.max() - anomaly_map.min() < 1e-6:
        print("    [!] Warning: Map is effectively CONSTANT.")
    
    # 배치 차원 제거 (1, 1, H, W) -> (H, W)
    if anomaly_map.ndim == 4:
        anomaly_map = anomaly_map[0, 0]
    elif anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[0]

    # 3. 히트맵 시각화 생성
    # [수정] 상대적 정규화(Min-Max) 대신 절대적 정규화 사용
    # 이유: 상대적 정규화를 하면, 정상 이미지(점수 4점)도 가장 높은 부분이 빨간색(255)으로 표시되어 불량처럼 보임.
    # 해결: 고정된 최대값(예: 20점)을 기준으로 정규화하여, 낮은 점수는 파란색, 높은 점수는 빨간색으로 고정.
    
    # [수정] 하이브리드 정규화 (Hybrid Normalization) 적용
    # 이유: 0081_196 이미지처럼 배경 자체가 이상 점수가 높은 경우(Min=18), 기존 절대 평가(Max=20)로는 전체가 빨간색이 됨.
    # 해결: "기본 임계값(15점)"과 "현재 이미지의 최소값" 중 더 큰 값을 0(파란색)으로 기준 잡음.
    #       즉, 배경 점수가 아무리 높아도, 그 배경보다는 점수가 더 높아야 빨간색으로 표시됨.
    
    score_min = anomaly_map.min()
    score_max = anomaly_map.max()
    THRESHOLD_START = 15.0  # 이 점수 이하는 무조건 정상(파란색)으로 간주
    
    # 시각화의 기준점 설정
    # 배경이 18점이면, 18점부터 23점 사이를 0~1로 늘려서 보여줌.
    # 배경이 2점이면, 15점까지는 0으로 묶고, 15점~23점 사이를 늘려서 보여줌.
    # [수정] Constant Map (전체 균일 점수) 처리
    score_min = anomaly_map.min()
    score_max = anomaly_map.max()
    score_std = anomaly_map.std()
    
    print(f"    [Debug] Checking Constant: Diff={score_max - score_min:.6f}, Std={score_std:.6f}")
    
    is_constant_high = False
    
    # 허용 오차를 1e-6 -> 0.01로 완화하고, 표준편차 체크 추가
    # 0081_196의 경우 std가 0.000000임
    # 허용 오차를 1e-6 -> 0.01로 완화하고, 표준편차 체크 추가
    # 0081_196의 경우 std가 0.000000임
    if score_max > THRESHOLD_START and (score_std < 0.05 or (score_max - score_min) < 0.05):
        is_constant_high = True
        print(f"    [!] 전체 이미지 균일 불량 감지 (Diff={score_max - score_min:.6f}, Std={score_std:.6f}) -> 테두리 강조 모드 전환")
        
        # 히트맵 대신 원본 사용
        overlay = original_img.copy()
        
        # 테두리 (Red, 30px)
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 30)
        
        # 텍스트 표시 (배경 박스 추가)
        text = f"Global Anomaly (Score: {score_max:.2f})"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
        cv2.rectangle(overlay, (20, 40), (20 + text_w + 20, 40 + text_h + 40), (255, 255, 255), -1)
        cv2.putText(overlay, text, (30, 40 + text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)
        
    else:
        # 기존 로직 (국소 불량 시각화)
        
        # [수정] Peak-Focus 정규화 (최상위 점수 집중)
        vis_max = score_max
        vis_min = max(THRESHOLD_START, score_max - FOCUS_RANGE)
        
        # 만약 범위가 너무 좁으면(노이즈) 늘림
        if vis_max - vis_min < 1e-6:
             vis_min = vis_max - FOCUS_RANGE

        diff = vis_max - vis_min
        anomaly_map_norm = (anomaly_map - vis_min) / diff
        anomaly_map_norm = np.clip(anomaly_map_norm, 0, 1.0) # 0~1 사이로 고정
        
        # 0~255로 변환
        anomaly_map_uint8 = (anomaly_map_norm * 255).astype(np.uint8)

        # 원본 크기로 리사이즈
        h, w = original_img.shape[:2]
        anomaly_map_resized = cv2.resize(anomaly_map_uint8, (w, h))
        
        # 컬러맵 적용 (Jet)
        heatmap = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)

        # 원본과 오버레이 (점수 기반 적응형 투명도)
        # Alpha Map 생성
        alpha_map = 0.3 + (anomaly_map_norm * 0.6)
        alpha_map = np.clip(alpha_map, 0, 1.0)
        
        # 차원 맞추기
        # 주의: anomaly_map_norm은 리사이즈 전 크기일 수 있음 -> 리사이즈 필요
        if anomaly_map_norm.shape[:2] != (h, w):
             alpha_map_resized = cv2.resize(alpha_map, (w, h))
        else:
             alpha_map_resized = alpha_map
             
        alpha_map_expanded = np.expand_dims(alpha_map_resized, axis=2)
        
        # 블렌딩
        overlay = (original_img * (1 - alpha_map_expanded) + heatmap * alpha_map_expanded).astype(np.uint8)

    # 4. 저장
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    save_path = f"{name}_heatmap_fixed.png"
    
    cv2.imwrite(save_path, overlay)
    
    print("\n" + "="*40)
    print(f" >> 히트맵 저장 완료: {save_path}")
    print(f" >> 최대 이상 점수 (Max Score): {anomaly_map.max():.4f}")
    print("="*40)

if __name__ == "__main__":
    print("[*] 모델 로딩 중...")
    model = load_model()
    print("[*] 모델 로딩 완료\n")

    if len(sys.argv) < 2:
        print("[*] 사용법: python visualize_heatmap.py [이미지경로]")
        print("    (팁: 이미지를 이 창에 드래그 앤 드롭 하세요)")
        
        while True:
            print("\n" + "-"*40)
            try:
                user_input = input("[?] 이미지 파일 경로 입력 ('q' 종료) > ").strip()
            except EOFError:
                break
                
            if not user_input or user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            # 따옴표 제거
            path = user_input.strip('"\'')
            generate_heatmap(path, model)
    else:
        path = sys.argv[1]
        generate_heatmap(path, model)
