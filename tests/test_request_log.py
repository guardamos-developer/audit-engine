"""Tests for structured request logging (metering / latency)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api import app  # noqa: E402
from src.request_log import (  # noqa: E402
    REQUEST_LOGGER,
    build_request_log_record,
    emit_request_log,
    hash_api_key,
    resolve_ruleset_ids,
)


def test_hash_api_key_is_stable_and_does_not_embed_raw_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GUARDAMOS_LOG_PEPPER", "unit-test-pepper")
    key = "gdm_test_super_secret_value_xyz"
    h1 = hash_api_key(key)
    h2 = hash_api_key(key)
    assert h1 == h2
    assert len(h1) == 32
    assert key not in h1
    assert "super_secret" not in h1
    # Different pepper → different hash
    monkeypatch.setenv("GUARDAMOS_LOG_PEPPER", "other-pepper")
    assert hash_api_key(key) != h1


def test_resolve_ruleset_ids_by_population_and_matched_rules():
    assert resolve_ruleset_ids(
        effective_population="older_adult_healthy",
        matched_rules=["L1-RT-NSCA-0001"],
    ) == ["L1-RT-NSCA2019-v1"]
    assert resolve_ruleset_ids(
        effective_population="healthy_adult_18plus",
        matched_rules=["L1-RTT-0001", "L1-ECSS-0002"],
    ) == [
        "L1-RT-ACSM2026-v1",
        "L1-CSCCa-ReturnToTraining-v1",
    ]


def test_build_and_emit_request_log_has_no_plan_fields():
    record = build_request_log_record(
        api_key_hash="abc123",
        verdict="rejected",
        ruleset_ids=["L1-RT-ACSM2026-v1"],
        total_latency_ms=1200,
        pipeline_latency_ms=1100,
        billing_validate_ms=80,
        extraction_ms=900,
        stage1_extraction_ms=400,
        stage2_extraction_ms=500,
        layer3_ms=None,
        skip_layer3=True,
    )
    forbidden = {
        "user_prompt",
        "ai_response",
        "plan",
        "explanations",
        "extraction",
        "checked_facts",
    }
    assert forbidden.isdisjoint(record.keys())
    assert record["event"] == "audit_request"
    assert record["pipeline_latency_ms"] == 1100
    assert record["total_latency_ms"] == 1200
    assert record["billing_validate_ms"] == 80
    assert record["extraction_ms"] == 900
    assert record["stage1_extraction_ms"] == 400
    assert record["stage2_extraction_ms"] == 500
    assert record["extraction_ms"] == (
        record["stage1_extraction_ms"] + record["stage2_extraction_ms"]
    )

    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, log_record: logging.LogRecord) -> None:
            captured.append(log_record.getMessage())

    handler = _Capture()
    REQUEST_LOGGER.addHandler(handler)
    try:
        emit_request_log(record)
    finally:
        REQUEST_LOGGER.removeHandler(handler)

    assert len(captured) == 1
    payload = json.loads(captured[0])
    assert payload["api_key_hash"] == "abc123"
    assert "knee" not in captured[0].lower()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_audit_emits_structured_log_and_strips_timing_from_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GUARDAMOS_LOG_PEPPER", "api-test-pepper")

    pipeline_payload = {
        "extraction": {
            "plan": {"sessions_per_week": 3},
            "extraction_evidence": {},
            "fields_left_null_without_evidence": [],
            "extraction_warnings": [],
            "possible_meta_instruction_detected": False,
            "meta_instruction_evidence": None,
        },
        "audit": {
            "verdict": "pass",
            "summary": "1 checks passed, 0 flagged.",
            "matched_rules": [],
            "explanations": [],
            "checked_facts": [{"rule_id": "L1-RT-0002", "text": "ok"}],
            "layer3_response": None,
            "ruleset_version": "test",
            "_timing": {
                "extraction_ms": 50,
                "stage1_extraction_ms": 30,
                "stage2_extraction_ms": 20,
                "layer3_ms": None,
                "pipeline_latency_ms": 60,
                "effective_population": "healthy_adult_18plus",
            },
        },
    }

    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, log_record: logging.LogRecord) -> None:
            captured.append(log_record.getMessage())

    handler = _Capture()
    REQUEST_LOGGER.addHandler(handler)
    try:
        with patch("src.api.validate_api_key_via_billing", return_value=True), patch(
            "src.api.run_raw_text_pipeline", return_value=pipeline_payload
        ):
            response = client.post(
                "/audit",
                headers={"X-API-Key": "gdm_test_logging_check"},
                json={
                    "user_prompt": "I want a plan with a secret medical condition",
                    "ai_response": "Do 3 sessions with secret details",
                    "skip_layer3": True,
                },
            )
    finally:
        REQUEST_LOGGER.removeHandler(handler)

    assert response.status_code == 200
    body = response.json()
    assert "_timing" not in body["audit"]
    assert body["audit"]["verdict"] == "pass"
    assert body["audit"]["checked_facts"]

    assert captured, "expected a guardamos.request log line"
    log_line = captured[-1]
    logged = json.loads(log_line)
    assert logged["verdict"] == "pass"
    assert logged["skip_layer3"] is True
    assert logged["ruleset_ids"] == ["L1-RT-ACSM2026-v1"]
    assert logged["extraction_ms"] == 50
    assert logged["stage1_extraction_ms"] == 30
    assert logged["stage2_extraction_ms"] == 20
    assert logged["pipeline_latency_ms"] == 60
    assert logged["billing_validate_ms"] is not None
    assert logged["total_latency_ms"] >= logged["pipeline_latency_ms"]
    # PII / plan content must never appear in the metering line.
    assert "secret medical" not in log_line
    assert "secret details" not in log_line
    assert "gdm_test_logging_check" not in log_line
