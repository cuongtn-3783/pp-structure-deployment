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

# Pin the managed Python for every uv command. requires-python allows up to 3.14,
# but Paddle only ships wheels through cp313, and .python-version is dockerignored
# (so `uv sync` would otherwise pick the newest 3.14 and fail the ABI match).
ENV UV_PYTHON=3.12

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
#    unsafe-best-match: paddleocr also appears on the Paddle index (older
#    version), so the default first-index strategy never sees 3.7.0 on PyPI.
#    paddlex[ocr]: PP-StructureV3 needs paddlex's OCR extra (extra model deps);
#    it's unversioned so the resolver pins it to the paddlex that paddleocr==3.7.0
#    requires and just adds the missing extra.
RUN uv pip install \
        --index-strategy unsafe-best-match \
        --index-url https://pypi.org/simple \
        --extra-index-url ${PADDLE_INDEX_URL} \
        "paddlepaddle-gpu==3.2.1" \
        "paddleocr==3.7.0" \
        "paddlex[ocr]"

COPY main.py ./
COPY scripts ./scripts

# 3. Bake model weights into the image (CPU instantiation; no GPU at build time).
#    The paddlepaddle-gpu wheel links libcuda.so.1 (the NVIDIA driver), loaded at
#    `import paddle` even with device="cpu". The runtime CUDA base image ships no
#    libcuda at all, and the container runtime injects the real driver only at
#    `--gpus` runtime, not during build. Install NVIDIA's cuda-compat package — a
#    real, loadable libcuda.so.1 that needs no kernel driver — purely to satisfy
#    the import; the host driver overrides it when the container runs on a GPU.
#    Package name is derived from CUDA_VERSION (e.g. 13.0.0 -> cuda-compat-13-0).
RUN set -eux; \
    compat="cuda-compat-$(echo "${CUDA_VERSION}" | cut -d. -f1)-$(echo "${CUDA_VERSION}" | cut -d. -f2)"; \
    apt-get update && apt-get install -y --no-install-recommends "$compat"; \
    rm -rf /var/lib/apt/lists/*; \
    lib="$(find / -name 'libcuda.so*' 2>/dev/null | head -n1)"; \
    [ -n "$lib" ] || { echo "no libcuda.so found after installing $compat" >&2; exit 1; }; \
    dir="$(dirname "$lib")"; \
    [ -e "$dir/libcuda.so.1" ] || ln -sf "$lib" "$dir/libcuda.so.1"; \
    LD_LIBRARY_PATH="$dir:${LD_LIBRARY_PATH}" uv run python scripts/warm_weights.py

ENV PORT=2603
EXPOSE ${PORT}

CMD ["uv", "run", "python", "main.py"]
