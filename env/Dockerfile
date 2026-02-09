FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 1. 시스템 패키지 설치 (Python 3.11로 변경)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 3. 의존성 파일 복사
# 주의: pyproject.toml 변경 후 uv.lock도 다시 생성되어야 합니다.
COPY pyproject.toml ./

# 4. ✅ uv로 Python 3.11 환경 구축 및 의존성 설치
# --python python3.11 플래그를 명시하여 시스템 버전과 혼선을 방지합니다.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --python python3.11

# 5. 소스코드 복사
COPY . /app

# 6. 환경변수 설정
# .venv/bin을 PATH 맨 앞으로 두어 설치된 패키지들이 우선적으로 참조되게 합니다.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Azure ML Job 실행을 위한 기본 명령
CMD ["python", "index.py"]