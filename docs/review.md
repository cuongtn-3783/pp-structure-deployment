# review.md — PPStructureV3 GPU Docker Deployment (whole-task review)

**Verdict: APPROVED WITH NITS**

Independent re-verification of all CPU-verifiable acceptance criteria passed. Two
MINOR, non-blocking findings (both plan-consistent). GPU/Docker-build criteria
(§1, §6, and the runtime halves of §2/§3/§4/§10) cannot be executed in this
environment (no GPU, no Docker) and are correctly disclosed by the coder as
manual verification.

---

## Evidence log

| # | Check | What I ran / read | Result |
|---|---|---|---|
| 1 | Scope compliance | `git status`; file inventory; each file vs plan Changes lists | PASS — every file traces to a step; no out-of-bounds edits; no drive-by refactors. `.gitignore`/`.python-version`/`CLAUDE.md` are pre-existing untracked, not created by this work. |
| 2 | ruff (spec §9) | `uv run ruff check` | PASS — "All checks passed!" |
| 2 | pyright (spec §9) | `uv run pyright` | PASS — 0 errors, 0 warnings |
| 2 | pytest (spec §9) | `uv run pytest -q` | PASS — 20 passed, 1 warning (harmless httpx/starlette deprecation) |
| 2 | GPU deps absent on CPU (spec constraint) | `uv run python -c "import paddleocr"` | PASS — ModuleNotFoundError (opt-out honored) |
| 2 | App imports without paddle | `uv run python -c "from pp_structure_deployment import app"` | PASS — lazy import confirmed |
| 3 | Spec traceability | Walked verification matrix vs executed tests | PASS for all CI-verifiable ACs (see matrix below) |
| 4 | Concurrency non-flaky (§8) | Ran serialization test 3× | PASS — 3/3 passed, no flake |
| 5 | Regression / lock cleanliness | `grep -c paddle uv.lock` | PASS — 0 paddle refs; lock is CPU-clean |
| 6/7 | Conventions & simplicity | Read all source | PASS — consistent style, no speculative abstractions |
| 8 | Hygiene | Read diff | PASS — no debug output, TODOs, secrets, or new dead code; lazy-import pyright suppression is scoped to one line |
| 9 | Report honesty | Re-verified all 4 disclosed deviations + manual-AC claims | PASS — all reproduce (see below) |

### Highest-risk hypothesis probed (and cleared)
The Dockerfile installs GPU deps via `uv pip install` (out of lock), then runs the
warm step and CMD via `uv run`. If `uv run` pruned out-of-lock packages, the image
would be broken at runtime. **Empirically tested** (uv 0.11.24): a package installed
via `uv pip install` survives a subsequent default `uv run`. `uv run` installs
missing locked deps but does not prune extras (only `uv sync` is exact). **Not a bug.**

### Spec AC → evidence matrix
| Spec AC | Verified how | Status |
|---|---|---|
| §1 GPU image builds, weights baked | Docker unavailable in env | MANUAL (not verifiable here) |
| §2 startup init / health gating | `test_app.py` health ready/initializing | PASS (CI part); GPU runtime MANUAL |
| §3 port via ENV | `main.py` + Dockerfile grep (`ENV PORT=2603`, `EXPOSE ${PORT}`, `host=0.0.0.0`, `workers=1`) | PASS (static); `docker run` MANUAL |
| §4 JSON base64 → 200 structured | `test_app` 200 body + `test_pipeline` result list | PASS (mocked); real inference MANUAL |
| §5 all 18 params optional & forwarded | `test_schemas`, `test_pipeline`, `test_app` set/unset assertions | PASS |
| §6 EN + JA | documented, default PP-OCRv5 | MANUAL (GPU) |
| §7 bad input → 422/400, no crash | `test_schemas` 422, `test_app` 400 | PASS |
| §8 concurrency serialized | `test_pipeline` non-overlapping windows (3× non-flaky) | PASS |
| §9 quality gates | ruff / pyright / pytest | PASS |
| §10 PDF multi-page → one result per page | `test_pipeline` 2-page mock → 2-element list | PASS (mocked); real PDF MANUAL |

### Disclosed deviations — all re-verified as accurate
1. GPU deps out of lock (installed in Dockerfile via ARG index) — confirmed: `uv.lock` has 0 paddle refs; Dockerfile lines 35–39 pin `paddlepaddle-gpu==3.2.1`/`paddleocr==3.7.0`. Consistent with spec §Risks ("pin both in the Dockerfile; CUDA index is a build ARG").
2. `httpx` added to dev deps — confirmed needed by `TestClient`; present in `pyproject.toml:21`.
3. `response_model=None` on `/predict` — confirmed required for the `list | JSONResponse` union; `app.py:30`.
4. `device` param on `init_pipeline` — confirmed; covered by `test_pipeline.py` init test.

---

## Findings

### Finding 1 — MINOR — `file_type` request field is accepted but has no effect
**File:** `src/pp_structure_deployment/pipeline.py:94`, `src/pp_structure_deployment/schemas.py:43`, `README.md`
`run_predict` calls `_decode_to_tempfile(file_base64, file_type=None)`, hardcoding
`None`, so the request's `file_type` value never reaches suffix selection —
detection is purely magic-byte based (`_suffix_for`). The schema exposes
`file_type` and the README documents it as a hint, but it is functionally dead.
**Impact:** none in practice — valid PDFs always begin with `%PDF` (detected), and
images are read by content regardless of extension. **Plan status:** consistent
with the plan (Step 3 `run_predict` signature omits `file_type`; Step 4 does not
pass it). **Correct target state:** either thread `request.file_type` through
`run_predict` → `_decode_to_tempfile` so the advertised hint is honored, or drop
the field from the schema/README to avoid advertising a no-op parameter. Non-blocking.

### Finding 2 — MINOR — `/predict` maps all `ValueError`s to HTTP 400
**File:** `src/pp_structure_deployment/app.py:35`
The route catches `ValueError` and returns 400. Only bad-base64 raises `ValueError`
today, but a `ValueError` raised inside `paddle.predict()` (e.g. an out-of-range
param value) would also surface as a 400 with paddle's message, categorizing a
server-side inference failure as a client error. **Impact:** low — within spec §7's
"bad input" intent, and the detail message is the actual exception text (not a
hardcoded mismatch). **Correct target state:** scope the 400 to decode failures
(e.g. raise a dedicated decode error from `_decode_to_tempfile`, catch that for
400, and let other exceptions become 500). Non-blocking.

---

## Spec/plan defects
None. The plan's "gpu extra" wording was superseded during coding by installing
pinned GPU deps in the Dockerfile with the CUDA index as a build ARG — this better
realizes spec §Risks and was correctly disclosed. No document amendment required.

## Questions for the human (unverifiable in this environment)
These require a GPU host with Docker and are **not** folded into the approval:
1. Does `docker build` succeed on the `cu130` base and bake weights (spec §1)?
2. Does the container start, `GET /health` return `ready`, and the first `/predict`
   trigger no model download (spec §2/§3)?
3. Do EN and JA images both return recognized text with no config change (spec §6)?
4. Does the running image contain only the `-gpu` paddle wheel, not CPU paddle
   (spec invariant)?

---

**Verdict: APPROVED WITH NITS.** No blocking findings. Safe to merge; the two MINOR
findings may be addressed in a follow-up. GPU/Docker acceptance criteria remain to
be confirmed on a GPU host (questions 1–4 above).
