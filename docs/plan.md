# plan.md — PPStructureV3 GPU Docker Deployment

Derived from `docs/spec.md` (final). Confirmed implementation choices:
- **Docker**: single-stage `nvidia/cuda:*-cudnn-runtime-ubuntu24.04`.
- **Versions**: pinned exact (`paddlepaddle-gpu==3.2.1`, `paddleocr` pinned).
- **Package**: `src/` layout installed as a real package via a `hatchling` build backend.

Conventions used by every step below:
- Package import root is `pp_structure_deployment` (under `src/`).
- All quality commands run via `uv run`.
- The pipeline is **always mocked** in tests — no GPU, no real weights in CI.

---

## Step 1: Convert the project to an installable `src/` package with runtime + test deps

**Description**
Make `pyproject.toml` a proper application package so `from pp_structure_deployment import ...` works and `uv run` resolves runtime deps. Add:
- `[build-system]` using `hatchling` (`requires = ["hatchling"]`, `build-backend = "hatchling.build"`).
- `[tool.hatch.build.targets.wheel]` with `packages = ["src/pp_structure_deployment"]`.
- `[project.dependencies]`: `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `python-multipart` is **not** needed (JSON body only — do not add it).
- `paddleocr` and `paddlepaddle-gpu` are declared but must resolve from the Paddle index only at Docker build time. To keep `uv sync` / `uv run pytest` working on a CPU-only dev/CI machine **without** pulling GPU wheels, place `paddleocr` and `paddlepaddle-gpu` in an **optional-dependency group** `gpu` (`[project.optional-dependencies] gpu = ["paddleocr==<pin>", "paddlepaddle-gpu==3.2.1"]`), NOT in the base `dependencies`. Base runtime deps (`fastapi`, `uvicorn[standard]`, `pydantic`) stay in `[project.dependencies]`. The application code must import `paddleocr` lazily (inside the pipeline module's init function, not at module top level) so the app package imports cleanly without Paddle installed — this is what lets the test suite mock it on CPU CI.
- Add `pytest` to the existing `[dependency-groups] dev` list (keep `ruff`, `pyright`, `pyrefly`).
- Create the package directory `src/pp_structure_deployment/__init__.py` (empty).
- Configure tooling: add `[tool.pytest.ini_options]` with `testpaths = ["tests"]`; add `[tool.pyright]` with `include = ["src", "main.py", "tests"]`; ruff needs no config beyond defaults.

Pin `paddleocr` to the latest stable version verified to support PP-StructureV3 with PaddlePaddle 3.2.1 (resolve exact value during execution from `https://pypi.org/project/paddleocr/`; record the pinned value in the file).

**Depends on** none.

**Changes**
- Modify: `pyproject.toml`
- Create: `src/pp_structure_deployment/__init__.py`
- Regenerate: `uv.lock` (via `uv lock`)

**Out of bounds**
No application logic, no FastAPI code, no Dockerfile. Do not add `paddleocr`/`paddlepaddle-gpu` to base `dependencies`.

**Acceptance criteria**
1. `uv sync` succeeds on the CPU-only dev machine **without** installing `paddlepaddle-gpu` (the `gpu` extra is not selected by default). Verify: `uv run python -c "import paddleocr" ` fails with ModuleNotFoundError (proves GPU deps are opt-in), while `uv run python -c "import pp_structure_deployment"` succeeds.
2. `uv run python -c "import fastapi, uvicorn, pydantic"` succeeds.
3. `uv run ruff check` and `uv run pyright` both exit 0 (empty package is trivially clean).

**Rollback** `git checkout pyproject.toml uv.lock && rm -rf src/`.

---

## Step 2: Define Pydantic request/response schemas with all prediction params

**Description**
Create `src/pp_structure_deployment/schemas.py`:
- `PredictRequest(BaseModel)` with:
  - `file_base64: str` (required).
  - `file_type: Literal["image", "pdf"] | None = None` (optional hint; if absent, inferred during decode).
  - All 18 optional predict-time params from spec §5, each typed and defaulting to `None`:
    - bools: `use_doc_orientation_classify`, `use_doc_unwarping`, `use_textline_orientation`, `use_seal_recognition`, `use_table_recognition`, `use_formula_recognition`, `use_chart_recognition`, `use_region_detection` → `bool | None = None`.
    - `layout_threshold: float | None = None`
    - `layout_nms: bool | None = None`
    - `layout_unclip_ratio: float | None = None` (accept scalar; PaddleOCR also allows tuple, but expose scalar per least-code — document limitation)
    - `layout_merge_bboxes_mode: str | None = None`
    - `text_det_limit_side_len: int | None = None`
    - `text_det_limit_type: str | None = None`
    - `text_det_thresh: float | None = None`
    - `text_det_box_thresh: float | None = None`
    - `text_det_unclip_ratio: float | None = None`
    - `text_rec_score_thresh: float | None = None`
  - `model_config = ConfigDict(extra="forbid")` so unknown fields yield 422.
  - A method `predict_kwargs(self) -> dict[str, Any]` returning **only** the prediction params that are not `None` (excludes `file_base64` and `file_type`). This is the single source of truth for param passthrough (spec §5).
- `HealthResponse(BaseModel)` with `status: Literal["ready", "initializing"]`.
- Response for `/predict` is pass-through JSON (`list[dict[str, Any]]`), so no strict response model — document as `list` in the route.

**Depends on** Step 1.

**Changes**
- Create: `src/pp_structure_deployment/schemas.py`
- Create: `tests/__init__.py`, `tests/test_schemas.py`

**Out of bounds**
No FastAPI app, no pipeline, no base64 decoding logic here (schema only).

**Acceptance criteria**
1. `tests/test_schemas.py` asserts: a `PredictRequest` with only `file_base64` set → `predict_kwargs()` returns `{}`.
2. Setting `use_table_recognition=True` and `text_rec_score_thresh=0.5` → `predict_kwargs()` == `{"use_table_recognition": True, "text_rec_score_thresh": 0.5}` (spec §5 verifiable: set param present, unset absent).
3. Missing `file_base64` → `ValidationError`. Unknown extra field → `ValidationError` (extra=forbid).
4. `uv run pytest tests/test_schemas.py` passes; `uv run ruff check`, `uv run pyright` clean.

**Rollback** `git checkout . && rm src/pp_structure_deployment/schemas.py tests/test_schemas.py`.

---

## Step 3: Build the pipeline wrapper (lazy import, single instance, serialized access, base64→temp-file)

**Description**
Create `src/pp_structure_deployment/pipeline.py` — the only module that touches PaddleOCR. It must import `paddleocr` **lazily** (inside functions) so the module is importable and mockable on CPU CI.

Contents:
- Module-level singleton holder: `_pipeline: Any | None = None` and `_lock: asyncio.Lock` (created in `init_pipeline`, or a module-level `asyncio.Lock()`).
- `def init_pipeline() -> None`: imports `from paddleocr import PPStructureV3`, constructs the single instance with `device="gpu"` and default PP-OCRv5 recognition model (EN+JA — no language kwarg needed per spec assumption), assigns to `_pipeline`. Raises if already initialized (guard against >1 instance, spec invariant). This is called by the lifespan (Step 4) and by the Docker build-time warm step (Step 6).
- `def is_ready() -> bool`: returns `_pipeline is not None`.
- `def _decode_to_tempfile(file_base64: str, file_type: str | None) -> Path`: base64-decode (`base64.b64decode(..., validate=True)`; on failure raise `ValueError`); detect type — if `file_type` given use it, else sniff magic bytes (`%PDF` prefix → pdf; PNG/JPEG magic → image) and default to image; write to a `NamedTemporaryFile` with the right suffix (`.pdf`/`.png`); return the path.
- `async def run_predict(file_base64: str, predict_kwargs: dict[str, Any]) -> list[dict[str, Any]]`:
  - decode to temp file (raises `ValueError` on bad base64 — caller maps to 400).
  - acquire `_lock` (serialize GPU access, spec §8) and run the **blocking** `_pipeline.predict(input=path, **predict_kwargs)` inside `asyncio.to_thread(...)` so the event loop is not blocked while still serialized by the lock.
  - iterate the result generator/list; for each page result call its `.json` property (PaddleOCR result object exposes `.json` dict) and collect into a list.
  - `finally`: delete the temp file.
  - return the list.
- Keep a `ModelNotReadyError(RuntimeError)` raised by `run_predict` if `_pipeline is None`.

Because tests mock the pipeline, expose the singleton via a setter used by tests, or have tests monkeypatch `pipeline._pipeline` and `pipeline.asyncio.to_thread` targets — design `run_predict` so a mock object with a `.predict` returning objects that have `.json` works without a real import.

**Depends on** Step 2.

**Changes**
- Create: `src/pp_structure_deployment/pipeline.py`
- Create: `tests/test_pipeline.py`

**Out of bounds**
No FastAPI routes. Do not import `paddleocr` at module top level. No real GPU calls in tests.

**Acceptance criteria**
1. `tests/test_pipeline.py` — base64 decode: valid base64 of PNG magic bytes → temp file created with expected suffix; invalid base64 string → `ValueError`.
2. Param passthrough: with `pipeline._pipeline` monkeypatched to a mock, `await run_predict("<valid b64>", {"use_table_recognition": True})` calls `mock.predict` with `use_table_recognition=True` and `input=<path>`; kwargs not set are absent (spec §5).
3. Result serialization: mock `predict` returns two page objects each with a `.json` dict → `run_predict` returns a 2-element list of those dicts (spec §10 shape — one per page).
4. Concurrency: a mock `predict` that records enter/exit timestamps (with a small `time.sleep`) invoked via two concurrent `run_predict` tasks shows **non-overlapping** execution windows (lock serialization, spec §8).
5. Temp file is deleted after `run_predict` returns (assert path no longer exists).
6. `uv run pytest tests/test_pipeline.py` passes; `ruff`, `pyright` clean.

**Rollback** `git checkout . && rm src/pp_structure_deployment/pipeline.py tests/test_pipeline.py`.

---

## Step 4: Build the FastAPI app with lifespan init, `/predict`, and `/health`

**Description**
Create `src/pp_structure_deployment/app.py`:
- `@asynccontextmanager async def lifespan(app)`: call `pipeline.init_pipeline()` on startup (loads weights onto GPU **before** serving — spec §2). No teardown needed beyond letting the process exit.
- `app = FastAPI(lifespan=lifespan, title="PPStructureV3 Service")`.
- `GET /health` → `HealthResponse`: `{"status": "ready"}` if `pipeline.is_ready()` else `{"status": "initializing"}` (spec §2 contract).
- `POST /predict` → accepts `PredictRequest`, calls `await pipeline.run_predict(req.file_base64, req.predict_kwargs())`, returns the list as JSON (HTTP 200). Error mapping:
  - Pydantic validation (missing/extra field) → FastAPI auto-422 (spec §7).
  - `ValueError` from bad base64 → catch, return `JSONResponse(status_code=400, {"detail": "invalid base64 input"})` (spec §7).
  - `ModelNotReadyError` → 503 `{"detail":"model not ready"}`.
- `def create_app()` factory optional; a module-level `app` is sufficient for `uvicorn pp_structure_deployment.app:app`.

Tests use FastAPI `TestClient` with `pipeline.run_predict` / `pipeline.init_pipeline` / `pipeline.is_ready` monkeypatched so no GPU/weights are needed. For lifespan, monkeypatch `init_pipeline` to a no-op that sets a fake ready flag.

**Depends on** Step 3.

**Changes**
- Create: `src/pp_structure_deployment/app.py`
- Create: `tests/test_app.py`

**Out of bounds**
No uvicorn launch code here (that's `main.py`, Step 5). No Dockerfile.

**Acceptance criteria**
1. `tests/test_app.py` (TestClient, mocked pipeline):
   - `GET /health` before init → `{"status":"initializing"}`; after lifespan init → `{"status":"ready"}` (spec §2).
   - `POST /predict` with valid `{"file_base64": "<b64>"}` and mocked `run_predict` returning `[{...}]` → 200, body is that list (spec §4).
   - `POST /predict` missing `file_base64` → 422 (spec §7).
   - `POST /predict` with `run_predict` raising `ValueError` → 400 with JSON `detail` (spec §7).
   - A set param (e.g. `use_table_recognition=true`) reaches the mocked `run_predict`'s `predict_kwargs` argument; an unset one does not (spec §5, end-to-end through the route).
2. `uv run pytest tests/test_app.py` passes; `ruff`, `pyright` clean; process does not crash on bad input.

**Rollback** `git checkout . && rm src/pp_structure_deployment/app.py tests/test_app.py`.

---

## Step 5: Rewrite `main.py` to launch uvicorn bound to `0.0.0.0:$PORT`

**Description**
Replace the placeholder `main.py` with an entrypoint that reads `PORT` from env (default `2603`) and runs `uvicorn.run("pp_structure_deployment.app:app", host="0.0.0.0", port=int(os.environ.get("PORT","2603")), workers=1)`. `workers=1` is mandatory (single model instance, spec invariant). Keep `if __name__ == "__main__":`.

**Depends on** Step 4.

**Changes**
- Modify: `main.py`

**Out of bounds**
No multi-worker config. No app logic (import the app by string path only).

**Acceptance criteria**
1. `uv run pyright` and `uv run ruff check` clean on `main.py`.
2. Static check: `main.py` reads `PORT` from env with default `2603` and passes `host="0.0.0.0"`, `workers=1` (grep-verifiable; spec §3).
3. `uv run python -c "import ast,sys; ast.parse(open('main.py').read())"` succeeds (valid syntax). Do **not** actually start the server in CI (it would try to init the GPU pipeline).

**Rollback** `git checkout main.py`.

---

## Step 6: Author the GPU Dockerfile (single-stage, ENV PORT, build-time weight bake) + `.dockerignore`

**Description**
Create `Dockerfile` (single-stage):
- `ARG CUDA_INDEX=cu130` and `ARG CUDA_TAG=13.0.0-cudnn-runtime-ubuntu24.04` (both build-time adjustable per spec risk mitigation; fallback `cu129` documented in README).
- `FROM nvidia/cuda:${CUDA_TAG}` (ships Python 3.12 on ubuntu24.04).
- Install `uv` (copy from `ghcr.io/astral-sh/uv:latest` or pip). Set `WORKDIR /app`.
- Copy `pyproject.toml`, `uv.lock`, `README.md`, `src/`, `main.py`.
- Install deps **including the `gpu` extra**, pulling `paddlepaddle-gpu` from the Paddle index:
  `RUN uv sync --frozen --extra gpu --index paddle=https://www.paddlepaddle.org.cn/packages/stable/${CUDA_INDEX}/` (or `uv pip install` with `--index-url`/`--extra-index-url` for the paddle wheel; choose the form that works with the pinned versions — verify during execution).
- **Build-time weight bake**: `RUN uv run python -c "from pp_structure_deployment.pipeline import init_pipeline; init_pipeline()"` — but `init_pipeline` uses `device="gpu"` which will fail without a GPU at build. Mitigation per spec §Risks: add a build-only warm script `scripts/warm_weights.py` (or a `PP_WARM_CPU=1` env branch in `init_pipeline`) that instantiates `PPStructureV3` with `device="cpu"` **solely to trigger weight download into the image layer**, without keeping the instance. Runtime lifespan still uses `device="gpu"`. Ensure weights download to a path baked into the image (default `~/.paddlex` / `PADDLE_PDX_MODEL_SOURCE`); set `ENV` for the cache dir if needed so runtime finds them.
- `ENV PORT=2603` then `EXPOSE ${PORT}`.
- `CMD ["uv", "run", "python", "main.py"]` (main.py binds `0.0.0.0:$PORT`).

Add a build-only helper: `scripts/warm_weights.py` importing and instantiating the pipeline on CPU to cache weights (kept out of the runtime path).

Create `.dockerignore`: `.git`, `.venv`, `__pycache__`, `tests/`, `docs/`, `*.md` except README (keep README for the build copy), `.claude/`, local caches.

**Depends on** Step 5.

**Changes**
- Create: `Dockerfile`, `.dockerignore`, `scripts/warm_weights.py`

**Out of bounds**
No docker-compose, no k8s, no CI pipeline, no TLS. No changes to app/pipeline logic beyond an optional CPU-warm branch (if that branch is added, keep it in `pipeline.py` and covered by a trivial test).

**Acceptance criteria**
1. `docker build -t ppstructure:test .` completes successfully on a machine with Docker (build log shows `paddlepaddle-gpu==3.2.1` install from the `cu130` index and weight-download output during the warm step) — spec §1. *(Manual/host with Docker; not CPU-CI unit-testable.)*
2. Static/grep checks (CI-runnable without building): `Dockerfile` contains `ENV PORT=2603`, `EXPOSE ${PORT}`, base `nvidia/cuda:*cudnn-runtime-ubuntu24.04`, `ARG CUDA_INDEX=cu130` (spec §3, §6).
3. `docker run --gpus all -p 2603:2603 ppstructure:test` starts; `GET /health` returns `{"status":"ready"}` and the **first** `/predict` triggers no model download (weights baked) — spec §2, §3. *(Manual GPU verification, documented.)*
4. No `paddlepaddle` (CPU) wheel is installed alongside `paddlepaddle-gpu` (spec invariant): `docker run ... uv run pip list | grep paddlepaddle` shows only the `-gpu` wheel. *(Manual.)*

**Rollback** `rm Dockerfile .dockerignore scripts/warm_weights.py`.

---

## Step 7: Write README run/usage section and verify full quality gates

**Description**
Update `README.md` with: overview, build command (`docker build`, with `--build-arg CUDA_INDEX=cu129` fallback note), run command (`docker run --gpus all -p 2603:2603`), `/health` and `/predict` request/response examples (JSON body with `file_base64` + sample params), the EN+JA note (default PP-OCRv5, no config), image-size caveat, and local dev/test instructions (`uv sync`, `uv run pytest`, `uv run ruff check`, `uv run pyright`). Document the scalar-only `layout_unclip_ratio` limitation.

Then run the complete quality gate over the whole repo.

**Depends on** Steps 1–6.

**Changes**
- Modify: `README.md`

**Out of bounds**
No code changes (docs + verification only). If a gate fails, fix in the owning step, not here.

**Acceptance criteria**
1. `uv run ruff check` → 0 errors (spec §9).
2. `uv run pyright` → 0 errors on `src/`, `main.py`, `tests/` (spec §9).
3. `uv run pytest` → all pass, no GPU required (spec §9).
4. `README.md` contains build, run (`-p 2603:2603 --gpus all`), and a `POST /predict` JSON body example (spec §9 deliverable §9).

**Rollback** `git checkout README.md`.

---

## Verification matrix

| Spec acceptance criterion | Covered by step(s) | Proving test / check |
|---|---|---|
| §1 GPU image builds, weights baked | Step 6 | `docker build` succeeds; build log shows weight download (manual) |
| §2 Startup init, not lazy; `/health` gating | Step 4 (health/lifespan), Step 6 (baked weights) | `test_app.py` health-before/after init; manual first-`/predict` no download |
| §3 Port via ENV, `-p 2603:2603` | Step 5 (`main.py` bind), Step 6 (`ENV PORT`/`EXPOSE`) | grep Dockerfile + main.py; manual `docker run` |
| §4 JSON base64 input → 200 structured JSON | Step 3 (serialize), Step 4 (route) | `test_pipeline.py` result list; `test_app.py` 200 body |
| §5 All 18 predict params optional & forwarded | Step 2 (`predict_kwargs`), Step 3, Step 4 | `test_schemas.py`, `test_pipeline.py`, `test_app.py` set/unset assertions |
| §6 EN + JA | Step 3 (default PP-OCRv5), Step 7 (documented) | Manual GPU check; README note |
| §7 Bad input → 422/400, no crash | Step 2 (validation), Step 4 (400 mapping) | `test_schemas.py`, `test_app.py` 422 & 400 cases |
| §8 Concurrency serialized | Step 3 (async lock) | `test_pipeline.py` non-overlapping windows |
| §9 Quality gates (ruff/pyright/pytest, no GPU) | Every step + Step 7 | `uv run ruff check` / `pyright` / `pytest` |
| §10 PDF multi-page → one result per page | Step 3 (per-page `.json` list) | `test_pipeline.py` 2-page mock returns 2-element list |

Every spec criterion maps to at least one step. Plan complete.

## Execution order

Strictly sequential by dependency (each step leaves the repo green):

1. **Step 1** — package + deps (foundation).
2. **Step 2** — schemas. *(depends 1)*
3. **Step 3** — pipeline wrapper. *(depends 2)*
4. **Step 4** — FastAPI app. *(depends 3)*
5. **Step 5** — `main.py` entrypoint. *(depends 4)*
6. **Step 6** — Dockerfile + `.dockerignore` + warm script. *(depends 5)*
7. **Step 7** — README + full quality gate. *(depends 1–6)*

No steps are parallelizable — each builds on the prior module's import surface. Steps 2, 3, 4 could be drafted in parallel by separate agents only if interfaces are frozen first, but sequential execution is recommended for the always-green invariant.
