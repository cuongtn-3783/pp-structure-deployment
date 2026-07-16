# spec.md — PPStructureV3 GPU Docker Deployment

## Assumptions (all verified or user-confirmed)

- **PP-OCRv5 unified model covers EN + JA in one model.** Verified: the default `PP-OCRv5_server_rec` / `PP-OCRv5_mobile_rec` recognition models support Simplified Chinese, Traditional Chinese, English, and Japanese in a single model ([PaddleOCR docs](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)). No per-language model switching is required to satisfy the EN+JA requirement.
- **Target GPU server**: NVIDIA driver 580.95.05, CUDA 13.0 capable (user-provided `nvidia-smi`).
- **paddlepaddle-gpu `cu130` stable wheels exist** (verified: `https://www.paddlepaddle.org.cn/packages/stable/` lists `cu130`). PPStructureV3 requires PaddlePaddle ≥ 3.2.1.
- **API input** (user-confirmed): single JSON request body containing a base64-encoded document plus all prediction parameters.
- **API output** (user-confirmed): structured JSON only (no Markdown, no visualization images).
- **Model weights** (user-confirmed): downloaded and baked into the image at Docker **build** time; container **startup** loads them into the GPU via a FastAPI lifespan event (satisfies "initialize at startup, not lazily on first request").
- **Deliverable is the project code + Docker configuration**, not a live deployment on the server. [ASSUMPTION — no contrary signal]
- **No authentication / API key** on the endpoint (internal service). [ASSUMPTION]
- **Single pipeline instance, serialized GPU inference** (PaddleOCR predict is not thread-safe; one shared instance guarded so requests do not run concurrently on the GPU). [ASSUMPTION]
- **Server models** (not mobile) are used, for GPU accuracy. [ASSUMPTION]

## Problem statement

- **Current behavior**: The repository is a bare `uv` scaffold. `main.py:1-6` prints a hello message; `src/` and `docs/` are empty; `pyproject.toml` declares only dev tooling (`ruff`, `pyright`, `pyrefly`). There is no application, no model, no API, no Docker configuration.
- **Desired behavior**: A FastAPI service wrapping PaddleOCR's PPStructureV3 document-parsing pipeline, packaged in a GPU-enabled Docker image. The service loads models at container startup and exposes an HTTP endpoint that accepts a base64 document + prediction parameters in the JSON body and returns the structured parse result as JSON.
- **Why it matters**: Enables reproducible, GPU-accelerated document structure extraction (layout, tables, text; EN + JA) as a deployable network service.

## Scope (in)

1. FastAPI application (Python 3.12) exposing:
   - `POST /predict` — accepts JSON `{ "file_base64": str, "file_type"?: str, ...prediction_params }`; returns structured JSON.
   - `GET /health` — liveness + model-readiness status.
2. A shared `PPStructureV3` pipeline instance initialized once at startup via FastAPI **lifespan**, on GPU (`device="gpu"`), using the default PP-OCRv5 recognition model (EN + JA).
3. A typed Pydantic request model exposing the documented **predict-time** prediction parameters (see Acceptance Criteria §5), all optional; only params the client sets are forwarded to `predict()`.
4. Base64 decoding of the input document (image or PDF), inference, and serialization of each page's result via PaddleOCR's own JSON representation.
5. Serialized GPU access (async lock + threadpool offload) so concurrent HTTP requests do not race on the single pipeline instance.
6. `Dockerfile` targeting CUDA 13.0:
   - Base image `nvidia/cuda:13.0.*-cudnn-runtime-ubuntu24.04` (ships Python 3.12).
   - `uv`-based dependency install; `paddlepaddle-gpu` from the `cu130` index; `paddleocr`.
   - Model weights downloaded during build (a build step that instantiates the pipeline so weights are cached into an image layer).
   - `ENV PORT=2603`; `EXPOSE ${PORT}`; server bound to `0.0.0.0:${PORT}`.
   - CUDA wheel index exposed as a build `ARG` (default `cu130`) so the target CUDA can be changed without editing the Dockerfile body.
7. Dependency additions to `pyproject.toml`: `fastapi`, `uvicorn[standard]`, `paddleocr`, `paddlepaddle-gpu`, `pydantic`, plus `pytest` in the dev group.
8. `ruff` and `pyright` clean; `pytest` unit tests covering request validation, base64 decode/error paths, param passthrough, and the `/health` contract (pipeline mocked — no GPU required for the test suite).
9. A `.dockerignore` and a short run/usage section in `README.md`.

## Scope (out)

- Live deployment, orchestration (k8s/compose beyond a documented `docker run` example), TLS, ingress.
- Authentication, rate limiting, quotas, multi-tenancy.
- Markdown / Word / visualization-image outputs (`save_to_markdown`, `save_to_word`, annotated images).
- Batch/async job queue, streaming, or webhook callbacks.
- Horizontal scaling, multi-GPU sharding, multi-worker concurrency of the model.
- CPU-only fallback path.
- Model fine-tuning, custom model swapping beyond exposing `text_recognition_model_name` if trivially supported.
- Persisting outputs to disk/object storage.
- GPU integration tests in CI (require physical GPU); test suite mocks the pipeline.

## Affected components

| Component | Path | Change | Evidence |
|---|---|---|---|
| Entry/app | `src/pp_structure_deployment/app.py` (new) | FastAPI app + lifespan model init | none exists today (`src/` empty) |
| Schemas | `src/pp_structure_deployment/schemas.py` (new) | Pydantic request/response models | — |
| Inference wrapper | `src/pp_structure_deployment/pipeline.py` (new) | pipeline holder, lock, predict helper | — |
| Server entrypoint | `main.py` (rewrite) | launch uvicorn | current `main.py:1-6` is placeholder |
| Deps | `pyproject.toml` | add runtime deps + pytest | current has dev-only tooling |
| Docker | `Dockerfile`, `.dockerignore` (new) | GPU image, port ENV, build-time weights | none exist |
| Docs | `README.md` | build/run/API usage | current is 2 lines |
| Tests | `tests/` (new) | pytest suite | none exist |

## Acceptance criteria

1. **GPU image builds**: `docker build` completes on a `cu130` base and produces an image containing PPStructureV3 weights (verifiable: build log shows weight download; image runs without runtime model download).
2. **Startup init, not lazy**: Given the container starts, When the lifespan event completes, Then the `PPStructureV3` instance exists on GPU **before** any request is served; `GET /health` returns `{"status":"ready"}` only after init; the first `/predict` triggers **no** model download.
3. **Port via ENV**: The Dockerfile defines `ENV PORT=2603`, uses `EXPOSE ${PORT}`, and the server binds `0.0.0.0:$PORT`; `docker run -p 2603:2603` serves the API.
4. **JSON base64 input**: Given a valid JSON body `{"file_base64": "<b64 of a PNG/JPG/PDF>"}`, When `POST /predict`, Then the response is HTTP 200 with a JSON body containing one structured result per input page (keys as produced by PaddleOCR's result `.json`).
5. **All prediction params in body**: Each of the following documented predict-time params is accepted as an optional field in the JSON body and forwarded to `predict()` only when set: `use_doc_orientation_classify`, `use_doc_unwarping`, `use_textline_orientation`, `use_seal_recognition`, `use_table_recognition`, `use_formula_recognition`, `use_chart_recognition`, `use_region_detection`, `layout_threshold`, `layout_nms`, `layout_unclip_ratio`, `layout_merge_bboxes_mode`, `text_det_limit_side_len`, `text_det_limit_type`, `text_det_thresh`, `text_det_box_thresh`, `text_det_unclip_ratio`, `text_rec_score_thresh`. (Verifiable: unit test asserts a set param appears in the mocked `predict` call kwargs and an unset param does not.)
6. **EN + JA**: Given an image containing English text and an image containing Japanese text, When parsed, Then recognized text is returned for both without any language configuration change (default PP-OCRv5 model). (Manual/GPU check; documented, not in CI.)
7. **Bad input rejected**: Given a body with missing `file_base64` or a string that is not valid base64, When `POST /predict`, Then HTTP 422 (validation) or 400 with a JSON error message; the process does not crash.
8. **Concurrency safety**: Given two simultaneous `/predict` requests, When both are handled, Then GPU inference is serialized (no interleaved calls into the single pipeline instance) and both return correct results. (Verifiable: unit test with a mocked pipeline asserts non-overlapping execution under the lock.)
9. **Quality gates**: `uv run ruff check` passes with no errors; `uv run pyright` reports no errors on `src/` and `main.py`; `uv run pytest` passes and requires no GPU.
10. **PDF multi-page**: Given a base64 multi-page PDF, When `POST /predict`, Then the response contains one structured result object per page in page order.

## Constraints & invariants

- **Python 3.12 syntax**; `uv` for env/dependency management (CLAUDE.md).
- **Single model instance**: never construct more than one `PPStructureV3` per process; never call into it concurrently.
- **No runtime network dependency for weights**: after build, container startup must not fetch models (weights baked in layer).
- **paddlepaddle-gpu and paddlepaddle CPU must not both be installed** (Paddle constraint) — only the GPU wheel.
- **Port is ENV-driven**; changing `PORT` must not require editing application code.
- **Test suite runs on CPU-only CI** (pipeline mocked); GPU-dependent behavior is documented as manual verification.
- Backward compatibility: none required (greenfield); the placeholder `main.py` may be freely replaced.

## Risks & unknowns

- **`cu130` wheel × base-image minor-version drift**: exact CUDA 13.0 cuDNN runtime tag and matching `paddlepaddle-gpu` version must align. Mitigation: pin both in the Dockerfile; CUDA index is a build `ARG` for easy adjustment; driver 580 is backward-compatible with older CUDA runtimes if a specific `cu130` combo proves unstable (fallback `cu129`).
- **Image size**: server models + CUDA runtime + baked weights yield a multi-GB image. Accepted per "bake at build" decision; noted in README.
- **PaddleOCR result JSON shape** varies by enabled sub-pipelines. Mitigation: serialize via the library's own `.json` representation rather than a hand-rolled schema; response schema documented as pass-through.
- **PDF input handling**: `predict()` accepts a file path; base64 PDFs are decoded to a temp file for inference and cleaned up. Mitigation: temp-file lifecycle handled in the wrapper; covered by criterion §10.
- **Build-time weight caching**: instantiating the pipeline at build to cache weights requires the build stage to run Paddle import successfully (CPU import is sufficient for download). Mitigation: run the download step tolerant of no-GPU-at-build.

## Open questions

_None. All blockers resolved._

---
Spec is final and self-contained. Ready for Planning phase.
