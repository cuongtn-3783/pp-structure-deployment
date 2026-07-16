"""FastAPI application: startup model init, /health, and /predict."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pp_structure_deployment import pipeline
from pp_structure_deployment.schemas import HealthResponse, PredictRequest


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the PPStructureV3 pipeline onto the GPU before any request is served."""
    pipeline.init_pipeline()
    yield


app = FastAPI(lifespan=lifespan, title="PPStructureV3 Service")


@app.get("/health")
def health() -> HealthResponse:
    """Liveness + model-readiness. Reports 'ready' only after startup init."""
    return HealthResponse(status="ready" if pipeline.is_ready() else "initializing")


@app.post("/predict", response_model=None)
async def predict(request: PredictRequest) -> list[dict[str, Any]] | JSONResponse:
    """Run PPStructureV3 on the base64 document; return one JSON result per page."""
    try:
        return await pipeline.run_predict(request.file_base64, request.predict_kwargs())
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except pipeline.ModelNotReadyError:
        return JSONResponse(status_code=503, content={"detail": "model not ready"})
