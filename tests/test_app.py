"""Tests for the FastAPI app (pipeline mocked; no GPU, no real weights)."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pp_structure_deployment import app as app_module
from pp_structure_deployment import pipeline


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient with the pipeline fully mocked.

    Lifespan init flips a fake readiness flag instead of loading a real model.
    """
    ready = {"value": False}

    def fake_init(device: str = "gpu") -> None:
        ready["value"] = True

    monkeypatch.setattr(pipeline, "init_pipeline", fake_init)
    monkeypatch.setattr(pipeline, "is_ready", lambda: ready["value"])
    with TestClient(app_module.app) as c:
        yield c


def test_health_ready_after_startup(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_health_initializing_before_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    # No lifespan entered (no context manager), so is_ready stays False.
    monkeypatch.setattr(pipeline, "is_ready", lambda: False)
    client = TestClient(app_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "initializing"}


def test_predict_returns_result_list(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(file_base64: str, predict_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"page": 1}]

    monkeypatch.setattr(pipeline, "run_predict", fake_run)
    resp = client.post("/predict", json={"file_base64": "Zm9v"})
    assert resp.status_code == 200
    assert resp.json() == [{"page": 1}]


def test_predict_forwards_only_set_params(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def fake_run(file_base64: str, predict_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        seen.update(predict_kwargs)
        return []

    monkeypatch.setattr(pipeline, "run_predict", fake_run)
    resp = client.post(
        "/predict",
        json={"file_base64": "Zm9v", "use_table_recognition": True},
    )
    assert resp.status_code == 200
    assert seen == {"use_table_recognition": True}


def test_predict_missing_file_base64_returns_422(client: TestClient) -> None:
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_invalid_base64_returns_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(file_base64: str, predict_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        raise ValueError("invalid base64 input")

    monkeypatch.setattr(pipeline, "run_predict", fake_run)
    resp = client.post("/predict", json={"file_base64": "!!!"})
    assert resp.status_code == 400
    assert "detail" in resp.json()
