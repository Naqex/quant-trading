import hashlib

import pytest

from quant_trading.theory_receipts import (
    THEORY_RECEIPT_SCHEMA,
    TheoryHypothesisReceiptV1,
    TheoryReceiptError,
    json_sha256,
)


def _receipt(**overrides):
    payload = {
        "theory_id": "chaos.bds-surrogate",
        "theory_version": "1.0.0",
        "hypothesis": "A volatility regime changes dependence structure.",
        "source_hash": "a" * 64,
        "data_hash": "b" * 64,
        "model_hash": "c" * 64,
        "cutoff_timestamp": 1_700_000_000,
        "known_at": 1_700_000_000,
        "seed": 42,
        "assumptions": {"minimum_rows": 500, "fixed_bins": [0, 1, 2]},
        "metrics": {"p_value": 0.21, "statistic": 1.2},
        "baseline": {"name": "iid-surrogate", "p_value": 0.20},
        "result_status": "inconclusive",
        "rollback_ref": "research/chaos-bds/2026-09-27",
    }
    payload.update(overrides)
    return TheoryHypothesisReceiptV1(**payload)


def test_receipt_is_deterministic_serializable_and_hashable() -> None:
    first = _receipt()
    second = TheoryHypothesisReceiptV1.from_mapping(first.as_dict())

    assert first.as_dict() == second.as_dict()
    assert first.receipt_hash == second.receipt_hash
    assert first.as_dict()["schema"] == THEORY_RECEIPT_SCHEMA
    assert first.as_dict()["execution_capability"] is False
    assert first.as_dict()["provider_access"] is False
    assert first.as_dict()["broker_access"] is False


def test_hash_helper_is_canonical_and_matches_sha256() -> None:
    value_a = {"b": [2, 1], "a": "stable"}
    value_b = {"a": "stable", "b": [2, 1]}
    expected = hashlib.sha256(b'{"a":"stable","b":[2,1]}').hexdigest()
    assert json_sha256(value_a) == json_sha256(value_b) == expected


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"known_at": 1_700_000_001}, "known_at cannot be later"),
        ({"data_hash": "not-a-hash"}, "data_hash"),
        ({"metrics": {"score": float("nan")}}, "finite"),
        ({"execution_capability": True}, "execution_capability"),
        ({"provider_access": True}, "provider_access"),
        ({"broker_access": True}, "broker_access"),
        ({"result_status": "promote"}, "result_status"),
    ],
)
def test_receipt_rejects_unsafe_or_noncausal_values(overrides, message: str) -> None:
    with pytest.raises(TheoryReceiptError, match=message):
        _receipt(**overrides)


def test_receipt_rejects_tampered_hash() -> None:
    receipt = _receipt()
    payload = receipt.as_dict()
    payload["metrics"]["p_value"] = 0.01
    with pytest.raises(TheoryReceiptError, match="receipt_hash"):
        TheoryHypothesisReceiptV1.from_mapping(payload)
