"""Tests for the pipeline wrapper (pipeline mocked; no GPU, no real weights)."""

import asyncio
import base64
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pp_structure_deployment import pipeline
from pp_structure_deployment.pipeline import (
    ModelNotReadyError,
    _decode_to_tempfile,
    run_predict,
)


class FakePage:
    """Stands in for a PaddleOCR page result object exposing `.json`."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def json(self) -> dict[str, Any]:
        return self._data


class RecordingPipeline:
    """Mock pipeline recording predict() calls and their execution windows."""

    def __init__(self, pages: list[dict[str, Any]], sleep: float = 0.0) -> None:
        self._pages = pages
        self._sleep = sleep
        self.calls: list[dict[str, Any]] = []
        self.windows: list[tuple[float, float]] = []

    def predict(self, input: str, **kwargs: Any) -> list[FakePage]:  # noqa: A002
        start = time.perf_counter()
        if self._sleep:
            time.sleep(self._sleep)
        end = time.perf_counter()
        self.calls.append({"input": input, "kwargs": kwargs})
        self.windows.append((start, end))
        return [FakePage(p) for p in self._pages]


@pytest.fixture
def set_pipeline() -> Any:
    """Install a mock pipeline instance and reset it afterwards."""

    def _set(instance: Any) -> None:
        pipeline._pipeline = instance

    yield _set
    pipeline._pipeline = None


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_decode_valid_png_creates_tempfile_with_suffix() -> None:
    path = _decode_to_tempfile(_b64(b"\x89PNG\r\n\x1a\n"), None)
    try:
        assert path.exists()
        assert path.suffix == ".png"
    finally:
        path.unlink()


def test_decode_detects_pdf_from_magic_bytes() -> None:
    path = _decode_to_tempfile(_b64(b"%PDF-1.7 rest"), None)
    try:
        assert path.suffix == ".pdf"
    finally:
        path.unlink()


def test_decode_invalid_base64_raises_valueerror() -> None:
    with pytest.raises(ValueError):
        _decode_to_tempfile("not base64 !!!", None)


def test_run_predict_forwards_only_set_params(set_pipeline: Callable[[Any], None]) -> None:
    mock = RecordingPipeline(pages=[{"page": 1}])
    set_pipeline(mock)

    result = asyncio.run(run_predict(_b64(b"\x89PNG"), {"use_table_recognition": True}))

    assert result == [{"page": 1}]
    assert len(mock.calls) == 1
    call = mock.calls[0]
    assert call["kwargs"] == {"use_table_recognition": True}
    assert call["input"].endswith(".png")


def test_run_predict_returns_one_json_per_page(set_pipeline: Callable[[Any], None]) -> None:
    mock = RecordingPipeline(pages=[{"page": 1}, {"page": 2}])
    set_pipeline(mock)

    result = asyncio.run(run_predict(_b64(b"%PDF-1.7"), {}))

    assert result == [{"page": 1}, {"page": 2}]


def test_run_predict_deletes_tempfile(set_pipeline: Callable[[Any], None]) -> None:
    created: list[str] = []
    real_decode = pipeline._decode_to_tempfile

    def _spy(file_base64: str, file_type: str | None) -> Path:
        path = real_decode(file_base64, file_type)
        created.append(str(path))
        return path

    pipeline._decode_to_tempfile = _spy  # type: ignore[assignment]
    try:
        set_pipeline(RecordingPipeline(pages=[{"page": 1}]))
        asyncio.run(run_predict(_b64(b"\x89PNG"), {}))
    finally:
        pipeline._decode_to_tempfile = real_decode  # type: ignore[assignment]

    assert len(created) == 1
    assert not Path(created[0]).exists()


def test_run_predict_without_init_raises(set_pipeline: Callable[[Any], None]) -> None:
    # Fixture leaves _pipeline as None (not installed).
    with pytest.raises(ModelNotReadyError):
        asyncio.run(run_predict(_b64(b"\x89PNG"), {}))


def test_init_pipeline_forwards_device_and_guards_double_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    seen: dict[str, Any] = {}

    class FakePPStructureV3:
        def __init__(self, device: str) -> None:
            seen["device"] = device

    fake_module = types.ModuleType("paddleocr")
    fake_module.PPStructureV3 = FakePPStructureV3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    monkeypatch.setattr(pipeline, "_pipeline", None)

    try:
        pipeline.init_pipeline(device="cpu")
        assert seen["device"] == "cpu"
        assert pipeline.is_ready()
        # Second init must refuse: never more than one instance per process.
        with pytest.raises(RuntimeError):
            pipeline.init_pipeline()
    finally:
        pipeline._pipeline = None


def test_concurrent_predicts_are_serialized(set_pipeline: Callable[[Any], None]) -> None:
    mock = RecordingPipeline(pages=[{"page": 1}], sleep=0.05)
    set_pipeline(mock)

    async def _run_two() -> None:
        await asyncio.gather(
            run_predict(_b64(b"\x89PNG"), {}),
            run_predict(_b64(b"\x89PNG"), {}),
        )

    asyncio.run(_run_two())

    assert len(mock.windows) == 2
    first, second = sorted(mock.windows, key=lambda w: w[0])
    # The second call must not start until the first has finished (lock held).
    assert second[0] >= first[1]
