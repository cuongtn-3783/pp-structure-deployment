# GPU image for the PPStructureV3 FastAPI service (single stage).
#
# CUDA target is adjustable at build time:
#   docker build --build-arg CUDA_INDEX=cu129 \
#                --build-arg CUDA_TAG=12.9.0-cudnn-runtime-ubuntu24.04 .
ARG CUDA_TAG=13.0.0-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

# Paddle CUDA wheel index (matches CUDA_TAG). Versions are pinned below.
ARG CUDA_INDEX=cu130
ARG PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/${CUDA_INDEX}/

ENV DEBIAN_FRONTEND=noninteractive

# OpenCV (a PaddleOCR dep) needs libGL / libglib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv (also provides a managed Python 3.12 so the base image needs no python).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1. Install locked CPU/runtime deps (fastapi, uvicorn, pydantic) + the project.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# 2. Install pinned GPU deps from the Paddle CUDA index (kept out of the lock so
#    the index stays a build ARG and CPU CI never pulls them).
RUN uv pip install \
        --index-url https://pypi.org/simple \
        --extra-index-url ${PADDLE_INDEX_URL} \
        "paddlepaddle-gpu==3.2.1" \
        "paddleocr==3.7.0"

COPY main.py ./
COPY scripts ./scripts

# 3. Bake model weights into the image (CPU instantiation; no GPU at build time).
RUN uv run python scripts/warm_weights.py

ENV PORT=2603
EXPOSE ${PORT}

CMD ["uv", "run", "python", "main.py"]
