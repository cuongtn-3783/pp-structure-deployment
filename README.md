# pp-structure-deployment

FastAPI service wrapping PaddleOCR's **PPStructureV3** document-parsing pipeline,
packaged as a GPU-enabled Docker image. Models are loaded onto the GPU at
container startup (not lazily on first request); the HTTP API accepts a
base64-encoded document plus prediction parameters and returns the structured
parse result as JSON.

English + Japanese are both supported with **no configuration change** — the
default PP-OCRv5 recognition model covers Simplified/Traditional Chinese,
English, and Japanese in a single model.

## API

### `GET /health`
Returns model readiness:

```json
{ "status": "ready" }        // after startup init completes
{ "status": "initializing" } // before the pipeline is loaded
```

### `POST /predict`
Request body — `file_base64` is required; every prediction parameter is optional
and only forwarded to `predict()` when set:

```json
{
  "file_base64": "<base64 of a PNG / JPG / PDF>",
  "file_type": "pdf",
  "use_table_recognition": true,
  "use_formula_recognition": false,
  "text_rec_score_thresh": 0.5,
  "layout_threshold": 0.5
}
```

Accepted prediction parameters: `use_doc_orientation_classify`,
`use_doc_unwarping`, `use_textline_orientation`, `use_seal_recognition`,
`use_table_recognition`, `use_formula_recognition`, `use_chart_recognition`,
`use_region_detection`, `layout_threshold`, `layout_nms`, `layout_unclip_ratio`
(scalar only), `layout_merge_bboxes_mode`, `text_det_limit_side_len`,
`text_det_limit_type`, `text_det_thresh`, `text_det_box_thresh`,
`text_det_unclip_ratio`, `text_rec_score_thresh`. `file_type` is an optional hint
(`"image"` / `"pdf"`); when omitted the type is detected from the content.

Response — HTTP 200 with a JSON array holding one structured result per input
page (multi-page PDFs yield one object per page, in order). The shape is
PaddleOCR's own result `.json` representation and varies with which
sub-pipelines are enabled.

Errors: missing/unknown fields → **422**; invalid base64 → **400**; requests
before the model is ready → **503**.

Example:

```bash
curl -s http://localhost:2603/predict \
  -H 'Content-Type: application/json' \
  -d "{\"file_base64\": \"$(base64 -w0 sample.png)\"}"
```

## Build

```bash
docker build -t ppstructure:latest .
```

The build installs `paddlepaddle-gpu==3.2.1` from the CUDA `cu130` wheel index
and bakes the PPStructureV3 model weights into the image, so runtime performs no
model download.

**Different CUDA target** (e.g. the `cu129` fallback — driver 580 is backward
compatible with older CUDA runtimes):

```bash
docker build -t ppstructure:latest \
  --build-arg CUDA_INDEX=cu129 \
  --build-arg CUDA_TAG=12.9.0-cudnn-runtime-ubuntu24.04 .
```

> The image is multi-GB (CUDA runtime + server models + baked weights). This is
> expected given the build-time weight bake.

## Run

```bash
docker run --gpus all -p 2603:2603 ppstructure:latest
```

The port is ENV-driven inside the container (`ENV PORT=2603`); override with
`-e PORT=<port>` and adjust the `-p` mapping accordingly. Check readiness with
`curl http://localhost:2603/health`.

## Local development

GPU dependencies are **not** installed locally — the test suite mocks the
pipeline and runs on CPU.

```bash
uv sync                # install runtime + dev deps (no GPU wheels)
uv run pytest          # unit tests (no GPU required)
uv run ruff check      # lint
uv run pyright         # type-check
```
