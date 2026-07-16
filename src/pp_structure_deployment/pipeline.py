"""Single shared PPStructureV3 pipeline: init, serialized inference, I/O helpers.

This is the only module that touches PaddleOCR. `paddleocr` is imported lazily
inside init_pipeline() so the module (and the FastAPI app) import cleanly on a
machine without the GPU wheel, which is what lets the test suite mock the
pipeline on CPU-only CI.
"""

import asyncio
import base64
import binascii
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

# The single pipeline instance for this process. Never construct more than one.
_pipeline: Any | None = None

# Serializes GPU access: PaddleOCR predict() is not thread-safe, so concurrent
# HTTP requests must not run into the shared instance at the same time.
_lock = asyncio.Lock()


class ModelNotReadyError(RuntimeError):
    """Raised when predict is attempted before the pipeline is initialized."""


def init_pipeline(device: str = "gpu") -> None:
    """Construct the single PPStructureV3 instance and load its weights.

    Called once at container startup (device="gpu"). The build-time weight-bake
    step calls it with device="cpu" purely to trigger the model download into an
    image layer (no GPU is available at build time).
    """
    global _pipeline
    if _pipeline is not None:
        raise RuntimeError("pipeline already initialized")

    from paddleocr import PPStructureV3  # pyright: ignore[reportMissingImports]

    # Default PP-OCRv5 recognition model covers EN + JA in one model; no language
    # kwarg is required.
    _pipeline = PPStructureV3(device=device)


def is_ready() -> bool:
    """True once the pipeline instance exists."""
    return _pipeline is not None


def _suffix_for(data: bytes, file_type: str | None) -> str:
    """Pick a temp-file suffix from an explicit hint or the content's magic bytes.

    PDFs must carry a .pdf suffix so PaddleOCR uses its PDF reader; images are
    read by content, so any image suffix works.
    """
    if file_type == "pdf":
        return ".pdf"
    if file_type == "image":
        return ".png"
    if data[:4] == b"%PDF":
        return ".pdf"
    return ".png"


def _decode_to_tempfile(file_base64: str, file_type: str | None) -> Path:
    """Decode a base64 document to a temp file and return its path.

    Raises ValueError if the input is not valid base64.
    """
    try:
        data = base64.b64decode(file_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 input") from exc

    suffix = _suffix_for(data, file_type)
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


async def run_predict(
    file_base64: str, predict_kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Decode, run serialized GPU inference, and return one JSON dict per page.

    Raises ModelNotReadyError if the pipeline is not initialized and ValueError
    if the base64 input is invalid.
    """
    if _pipeline is None:
        raise ModelNotReadyError("pipeline not initialized")

    path = _decode_to_tempfile(file_base64, file_type=None)
    try:
        async with _lock:
            # predict() is blocking; offload to a thread so the event loop stays
            # responsive while the lock still serializes GPU access.
            results = await asyncio.to_thread(
                _pipeline.predict, input=str(path), **predict_kwargs
            )
            return [page.json for page in results]
    finally:
        os.unlink(path)
