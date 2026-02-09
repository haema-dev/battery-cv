FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 1. System packages (Python 3.11)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. uv install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 3. Copy dependencies
# Note: pyproject.toml changes require uv.lock regeneration.
COPY pyproject.toml .

# 4. Install dependencies with uv (Python 3.11)
# --python python3.11 flag prevents system version confusion.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --python python3.11

# 5. Copy source code
COPY . /app

# 6. Environment variables
# Prepend .venv/bin to PATH so installed packages are preferred.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Azure ML Job execution command default
CMD ["python", "index.py"]
