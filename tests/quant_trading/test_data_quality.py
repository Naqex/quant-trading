from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_trading.data_quality import (
    DataQualityError,
    OHLCVQualityPolicy,
    audit_ohlcv,
    require_ohlcv_quality,
)


def _frame(*, periods: int = 4, step: str = "1h") -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq=step, tz="UTC")
    close = np.arange(periods, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(periods, 1_000.0),
        }
    )


def test_contiguous_artifact_has_deterministic_receipt() -> None:
    frame = _frame()
    policy = OHLCVQualityPolicy(expected_interval_seconds=3_600, max_gap_count=0)
    first = audit_ohlcv(frame, policy=policy)
    second = audit_ohlcv(frame.copy(), policy=policy)

    assert first.quality_status == "accepted"
    assert first.gap_count == 0
    assert first.observed_interval_seconds == 3_600.0
    assert first.data_sha256 == second.data_sha256
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.execution_capability is False
    json.dumps(first.as_dict(), allow_nan=False)


def test_missing_candle_is_explicit_warning_by_default() -> None:
    frame = _frame()
    frame = frame.drop(index=2).reset_index(drop=True)
    receipt = audit_ohlcv(frame, policy=OHLCVQualityPolicy(expected_interval_seconds=3_600))

    assert receipt.quality_status == "accepted"
    assert receipt.gap_count == 1
    assert receipt.warnings == ("timestamp_gaps_detected",)
    require_ohlcv_quality(frame, policy=OHLCVQualityPolicy(expected_interval_seconds=3_600))


def test_gap_budget_can_fail_closed_without_repairing_rows() -> None:
    frame = _frame().drop(index=2).reset_index(drop=True)
    receipt = audit_ohlcv(
        frame,
        policy=OHLCVQualityPolicy(expected_interval_seconds=3_600, max_gap_count=0),
    )

    assert receipt.quality_status == "rejected"
    assert receipt.errors == ("gap_budget_exceeded",)
    with pytest.raises(DataQualityError, match="gap_budget_exceeded"):
        require_ohlcv_quality(
            frame,
            policy=OHLCVQualityPolicy(expected_interval_seconds=3_600, max_gap_count=0),
        )


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("open", 0.0, "non_positive_price"),
        ("volume", -1.0, "negative_volume"),
        ("high", 1.0, "inconsistent_high_low"),
    ],
)
def test_malformed_numeric_rows_are_rejected(column: str, value: float, error: str) -> None:
    frame = _frame()
    frame.loc[1, column] = value
    receipt = audit_ohlcv(frame)

    assert receipt.quality_status == "rejected"
    assert error in receipt.errors
    with pytest.raises(DataQualityError):
        require_ohlcv_quality(frame)


def test_duplicate_and_reordered_timestamps_are_rejected() -> None:
    frame = _frame()
    frame.loc[2, "open_time"] = frame.loc[1, "open_time"]
    frame.loc[1, "open_time"] = frame.loc[3, "open_time"]
    receipt = audit_ohlcv(frame)
    assert receipt.quality_status == "rejected"
    assert "duplicate_timestamp" in receipt.errors
    assert "timestamp_order" in receipt.errors


def test_nonfinite_and_missing_schema_are_rejected() -> None:
    frame = _frame()
    frame.loc[0, "close"] = np.inf
    receipt = audit_ohlcv(frame)
    assert receipt.quality_status == "rejected"
    assert "non_finite_numeric" in receipt.errors

    missing = _frame().drop(columns=["volume"])
    missing_receipt = audit_ohlcv(missing)
    assert missing_receipt.quality_status == "rejected"
    assert missing_receipt.data_sha256 is None
    assert missing_receipt.errors == ("missing_required_columns:volume",)


@pytest.mark.parametrize("column", ["open_time", "open", "volume"])
def test_duplicate_column_labels_are_rejected_before_pandas_coercion(column: str) -> None:
    frame = _frame()
    frame.insert(1, column, frame[column].to_numpy(copy=True), allow_duplicates=True)

    receipt = audit_ohlcv(frame)

    assert receipt.quality_status == "rejected"
    assert receipt.data_sha256 is None
    assert receipt.errors == ("duplicate_columns",)
    with pytest.raises(DataQualityError, match="duplicate_columns"):
        require_ohlcv_quality(frame)


def test_policy_rejects_ambiguous_values() -> None:
    with pytest.raises(DataQualityError):
        OHLCVQualityPolicy(expected_interval_seconds=0)
    with pytest.raises(DataQualityError):
        OHLCVQualityPolicy(max_gap_count=-1)
    with pytest.raises(DataQualityError, match="max_gap_count.*expected_interval_seconds"):
        OHLCVQualityPolicy(max_gap_count=0)
    with pytest.raises(DataQualityError):
        audit_ohlcv([], policy=OHLCVQualityPolicy())


def test_receipt_constructor_enforces_canonical_and_coherent_evidence() -> None:
    accepted = audit_ohlcv(
        _frame(),
        policy=OHLCVQualityPolicy(expected_interval_seconds=3_600, max_gap_count=0),
    )

    normalized = replace(accepted, observed_interval_seconds="3600")
    assert normalized.observed_interval_seconds == 3_600.0

    with pytest.raises(DataQualityError, match="schema"):
        replace(accepted, schema="other-schema")
    with pytest.raises(DataQualityError, match="canonical UTC timestamp"):
        replace(accepted, start_utc="2026-01-01T00:00:00Z")
    with pytest.raises(DataQualityError, match="min_interval_seconds"):
        replace(accepted, min_interval_seconds=4_000.0)
    with pytest.raises(DataQualityError, match="requires data_sha256"):
        replace(accepted, data_sha256=None)


def test_rejected_receipt_cannot_hide_missing_diagnostics() -> None:
    accepted = audit_ohlcv(_frame())
    with pytest.raises(DataQualityError, match="at least one error"):
        replace(accepted, quality_status="rejected")


@pytest.mark.parametrize(
    "field", ["observed_interval_seconds", "min_interval_seconds", "max_interval_seconds"]
)
def test_receipt_rejects_negative_interval_metrics(field: str) -> None:
    accepted = audit_ohlcv(_frame())
    with pytest.raises(DataQualityError, match="finite and >= 0"):
        replace(accepted, **{field: -1.0}, receipt_sha256=None)
