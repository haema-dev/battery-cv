# Build trigger: NumPy 1.x strict lock - v3
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

# 1. 시스템 의존성 설치 (OpenGL 및 필수 라이브러리)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

# 2. uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. 환경 구성 (가상환경)
WORKDIR /venv_pool
COPY pyproject.toml ./

# uv sync를 통해 pyproject.toml에 명시된 의존성 설치
RUN UV_CACHE_DIR=/tmp/uv-cache uv sync \
    --no-cache \
    --no-dev \
    --python python3.11

# 4. 환경변수 설정 (가상환경 활성화 및 Azure ML 호환성)
ENV VIRTUAL_ENV="/venv_pool/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# 5. Azure ML 작업 디렉토리 설정
WORKDIR /app