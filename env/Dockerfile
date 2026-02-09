FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 1. 시스템 의존성 설치 (비대화형 모드 및 정식 버전 지향)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

# 2. uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. 가상환경 전용 폴더 설정 및 의존성 설치
WORKDIR /venv_pool
COPY pyproject.toml ./

# --no-cache 플래그를 사용하여 uv의 내부 캐시 사용을 금지
RUN UV_CACHE_DIR=/tmp/uv-cache uv sync \
    --no-cache \
    --no-dev \
    --python python3.11

# 4. 소스코드 복사
WORKDIR /app
COPY . .

# 5. 환경변수 설정
ENV VIRTUAL_ENV="/venv_pool/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# 6. 설치 확인 (adlfs 존재 여부를 실시간 검증)
RUN uv pip list | grep adlfs

# 7. 실행
CMD ["python", "index.py"]