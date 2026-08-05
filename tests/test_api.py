"""HTTP API tests for POST /audit (billing key check + raw-text pipeline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api import app  # noqa: E402

CONV_PATH = ROOT / "sample_conversations" / "zero_rest_days_before.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _zero_rest_pipeline_payload() -> dict:
    conversation = json.loads(CONV_PATH.read_text(encoding="utf-8"))
    return {
        "extraction": {
            "plan": {
                "sessions_per_week": 7,
                "rest_days_per_week": 0,
            },
            "extraction_evidence": {},
            "fields_left_null_without_evidence": [],
            "extraction_warnings": [],
            "possible_meta_instruction_detected": False,
            "meta_instruction_evidence": None,
        },
        "audit": {
            "verdict": "flagged",
            "matched_rules": ["L1-ECSS-0002"],
            "explanations": ["missing rest day"],
            "checked_facts": [],
            "layer3_response": None,
            "ruleset_version": "test",
        },
        "_conversation": conversation,
    }


def test_health_reports_openai_key_diagnostics(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "openai_api_key_set" in body
    assert "openai_api_key_starts_with_sk" in body


def test_audit_requires_api_key(client: TestClient):
    response = client.post(
        "/audit",
        json={
            "user_prompt": "Give me a workout plan with zero rest days",
            "ai_response": "Train every single day, no rest ever.",
        },
    )
    assert response.status_code == 401


def test_audit_rejects_invalid_api_key(client: TestClient):
    with patch("src.api.validate_api_key_via_billing", return_value=False):
        response = client.post(
            "/audit",
            headers={"X-API-Key": "gdm_test_invalid"},
            json={
                "user_prompt": "Give me a workout plan with zero rest days",
                "ai_response": "Train every single day, no rest ever.",
            },
        )
    assert response.status_code == 401


def test_audit_valid_key_returns_flagged_for_zero_rest(client: TestClient):
    """Valid key + zero-rest conversation → same flagged / L1-ECSS-0002 shape."""
    payload = _zero_rest_pipeline_payload()
    conversation = payload.pop("_conversation")

    with patch("src.api.validate_api_key_via_billing", return_value=True):
        with patch(
            "src.api.run_raw_text_pipeline",
            return_value={
                "extraction": payload["extraction"],
                "audit": payload["audit"],
            },
        ) as mock_pipeline:
            response = client.post(
                "/audit",
                headers={"X-API-Key": "gdm_test_valid_key"},
                json={
                    "user_prompt": conversation["user_prompt"],
                    "ai_response": conversation["ai_response"],
                    "skip_layer3": True,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["audit"]["verdict"] == "flagged"
    assert "L1-ECSS-0002" in body["audit"]["matched_rules"]
    mock_pipeline.assert_called_once()
    call_kwargs = mock_pipeline.call_args
    assert call_kwargs.args[0] == conversation["user_prompt"]
    assert call_kwargs.args[1] == conversation["ai_response"]


def test_validate_api_key_via_billing_calls_billing_http():
    from src.api import validate_api_key_via_billing

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"valid": True}

    with patch("src.api.httpx.get", return_value=mock_response) as mock_get:
        assert validate_api_key_via_billing("gdm_test_abc") is True
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"] == {"key": "gdm_test_abc"}


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
