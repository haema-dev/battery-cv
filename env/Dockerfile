FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 1. 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# 2. uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 3. 의존성 파일 복사
COPY pyproject.toml uv.lock ./

# 4. ✅ uv로 모든 의존성 설치 (torch도 포함)
# 이전 torch 설치 명령어는 완전히 제거!
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --python python3.10

# 5. 소스코드 복사
COPY . /app

# 6. 환경변수
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python"]
