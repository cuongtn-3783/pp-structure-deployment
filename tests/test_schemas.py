"""Tests for the Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from pp_structure_deployment.schemas import PredictRequest


def test_predict_kwargs_empty_when_only_file_given() -> None:
    req = PredictRequest(file_base64="Zm9v")
    assert req.predict_kwargs() == {}


def test_predict_kwargs_includes_only_set_params() -> None:
    req = PredictRequest(
        file_base64="Zm9v",
        use_table_recognition=True,
        text_rec_score_thresh=0.5,
    )
    assert req.predict_kwargs() == {
        "use_table_recognition": True,
        "text_rec_score_thresh": 0.5,
    }


def test_predict_kwargs_excludes_transport_fields() -> None:
    req = PredictRequest(file_base64="Zm9v", file_type="pdf", use_seal_recognition=False)
    kwargs = req.predict_kwargs()
    assert "file_base64" not in kwargs
    assert "file_type" not in kwargs
    # A param set to False is still explicitly set, so it must be forwarded.
    assert kwargs == {"use_seal_recognition": False}


def test_missing_file_base64_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictRequest()  # type: ignore[call-arg]


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(file_base64="Zm9v", not_a_real_param=1)  # type: ignore[call-arg]
