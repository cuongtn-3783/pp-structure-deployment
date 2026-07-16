"""Pydantic request/response models for the PPStructureV3 service."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Predict-time parameter names forwarded to PPStructureV3.predict(). Kept as an
# explicit tuple so predict_kwargs() has a single source of truth and cannot
# accidentally forward transport fields (file_base64, file_type).
_PREDICT_PARAM_NAMES: tuple[str, ...] = (
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_textline_orientation",
    "use_seal_recognition",
    "use_table_recognition",
    "use_formula_recognition",
    "use_chart_recognition",
    "use_region_detection",
    "layout_threshold",
    "layout_nms",
    "layout_unclip_ratio",
    "layout_merge_bboxes_mode",
    "text_det_limit_side_len",
    "text_det_limit_type",
    "text_det_thresh",
    "text_det_box_thresh",
    "text_det_unclip_ratio",
    "text_rec_score_thresh",
)


class PredictRequest(BaseModel):
    """Request body for POST /predict.

    Carries the base64-encoded document plus every documented predict-time
    parameter. All prediction params are optional; only those explicitly set by
    the client are forwarded to predict() (see predict_kwargs).
    """

    model_config = ConfigDict(extra="forbid")

    file_base64: str
    file_type: Literal["image", "pdf"] | None = None

    # Sub-pipeline toggles.
    use_doc_orientation_classify: bool | None = None
    use_doc_unwarping: bool | None = None
    use_textline_orientation: bool | None = None
    use_seal_recognition: bool | None = None
    use_table_recognition: bool | None = None
    use_formula_recognition: bool | None = None
    use_chart_recognition: bool | None = None
    use_region_detection: bool | None = None

    # Layout detection params.
    layout_threshold: float | None = None
    layout_nms: bool | None = None
    # Exposed as a scalar only; PaddleOCR also accepts a tuple, which this API
    # does not surface.
    layout_unclip_ratio: float | None = None
    layout_merge_bboxes_mode: str | None = None

    # Text detection / recognition params.
    text_det_limit_side_len: int | None = None
    text_det_limit_type: str | None = None
    text_det_thresh: float | None = None
    text_det_box_thresh: float | None = None
    text_det_unclip_ratio: float | None = None
    text_rec_score_thresh: float | None = None

    def predict_kwargs(self) -> dict[str, Any]:
        """Return only the prediction params the client explicitly set.

        Transport fields (file_base64, file_type) are never included. A param
        left at its None default is omitted so predict() falls back to its own
        pipeline default.
        """
        return {
            name: value
            for name in _PREDICT_PARAM_NAMES
            if (value := getattr(self, name)) is not None
        }


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["ready", "initializing"]
