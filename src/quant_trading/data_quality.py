"""Fail-closed OHLCV quality preflight for offline research.

The preflight runs on an already captured candle artifact.  It does not fill,
sort, interpolate, or otherwise repair rows.  A missing candle is a diagnostic
warning by default because crypto fixtures can contain legitimate gaps; a
caller may opt into a hard gap budget.  The returned receipt is advisory and
cannot grant provider, broker, wallet, or execution access.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

DATA_QUALITY_SCHEMA = "ohlcv-quality-receipt-v1"
REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QualityStatus = Literal["accepted", "rejected"]


class DataQualityError(ValueError):
    """Raised when a quality policy or a required preflight is unsafe."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DataQualityError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DataQualityError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataQualityError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise DataQualityError(f"{name} must be finite")
    return result


def _text(value: Any, name: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataQualityError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise DataQualityError(f"{name} must be at most {maximum} characters")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DataQualityError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DataQualityError("quality receipt must contain finite JSON data") from exc


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


@dataclass(frozen=True)
class OHLCVQualityPolicy:
    """Explicit preflight policy; no gap assumption is implicit."""

    expected_interval_seconds: int | None = None
    max_gap_count: int | None = None

    def __post_init__(self) -> None:
        interval = self.expected_interval_seconds
        if interval is not None:
            interval = _strict_int(interval, "expected_interval_seconds", minimum=1)
        limit = self.max_gap_count
        if limit is not None:
            limit = _strict_int(limit, "max_gap_count", minimum=0)
        object.__setattr__(self, "expected_interval_seconds", interval)
        object.__setattr__(self, "max_gap_count", limit)

    def as_dict(self) -> dict[str, int | None]:
        return {
            "expected_interval_seconds": self.expected_interval_seconds,
            "max_gap_count": self.max_gap_count,
        }


@dataclass(frozen=True)
class OHLCVQualityReceipt:
    """Immutable result of one deterministic OHLCV preflight."""

    quality_status: QualityStatus
    rows: int
    start_utc: str | None
    end_utc: str | None
    data_sha256: str | None
    observed_interval_seconds: float | None
    min_interval_seconds: float | None
    max_interval_seconds: float | None
    gap_count: int | None
    policy: OHLCVQualityPolicy
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    schema: str = DATA_QUALITY_SCHEMA
    status: str = "PREP_ONLY"
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.quality_status not in {"accepted", "rejected"}:
            raise DataQualityError("quality_status is unsupported")
        if self.status != "PREP_ONLY":
            raise DataQualityError("quality receipt status must remain PREP_ONLY")
        rows = _strict_int(self.rows, "rows", minimum=0)
        if not isinstance(self.policy, OHLCVQualityPolicy):
            raise DataQualityError("policy must be OHLCVQualityPolicy")
        if self.data_sha256 is not None:
            _sha256(self.data_sha256, "data_sha256")
        for name in ("warnings", "errors"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
                raise DataQualityError(f"{name} must be a tuple of non-empty strings")
        for name in ("observed_interval_seconds", "min_interval_seconds", "max_interval_seconds"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise DataQualityError(f"{name} must be finite and >= 0")
        if self.gap_count is not None:
            gap_count = _strict_int(self.gap_count, "gap_count", minimum=0)
            if gap_count > max(rows - 1, 0):
                raise DataQualityError("gap_count cannot exceed interval count")
            object.__setattr__(self, "gap_count", gap_count)
        if self.quality_status == "accepted" and self.errors:
            raise DataQualityError("accepted receipt cannot contain errors")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise DataQualityError(f"{name} must remain false for offline quality")
        object.__setattr__(self, "rows", rows)
        core = self._core_dict()
        expected = _hash_json(core)
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise DataQualityError("receipt_sha256 does not match canonical receipt")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "quality_status": self.quality_status,
            "rows": self.rows,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "data_sha256": self.data_sha256,
            "observed_interval_seconds": self.observed_interval_seconds,
            "min_interval_seconds": self.min_interval_seconds,
            "max_interval_seconds": self.max_interval_seconds,
            "gap_count": self.gap_count,
            "policy": self.policy.as_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["receipt_sha256"] = self.receipt_sha256
        return result


def _canonical_frame(frame: pd.DataFrame, timestamps: pd.Series, numeric: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for index in range(len(frame)):
        row = [_normalize_timestamp(timestamps.iloc[index])]
        row.extend(format(float(numeric[column].iloc[index]), ".17g") for column in _NUMERIC_COLUMNS)
        rows.append(row)
    return _hash_json({"schema": "ohlcv-canonical-v1", "columns": list(REQUIRED_COLUMNS), "rows": rows})


def audit_ohlcv(frame: pd.DataFrame, *, policy: OHLCVQualityPolicy | None = None) -> OHLCVQualityReceipt:
    """Audit an artifact without mutating or repairing it."""

    if policy is None:
        policy = OHLCVQualityPolicy()
    if not isinstance(policy, OHLCVQualityPolicy):
        raise DataQualityError("policy must be OHLCVQualityPolicy")
    if not isinstance(frame, pd.DataFrame):
        raise DataQualityError("frame must be a pandas DataFrame")

    errors: list[str] = []
    warnings: list[str] = []
    if not frame.columns.is_unique:
        errors.append("duplicate_columns")
    missing = tuple(column for column in REQUIRED_COLUMNS if column not in frame.columns)
    if missing:
        errors.append("missing_required_columns:" + ",".join(missing))
    if errors:
        return OHLCVQualityReceipt(
            quality_status="rejected",
            rows=len(frame),
            start_utc=None,
            end_utc=None,
            data_sha256=None,
            observed_interval_seconds=None,
            min_interval_seconds=None,
            max_interval_seconds=None,
            gap_count=None,
            policy=policy,
            errors=tuple(errors),
        )
    if frame.empty:
        errors.append("empty_frame")
        return OHLCVQualityReceipt(
            quality_status="rejected",
            rows=0,
            start_utc=None,
            end_utc=None,
            data_sha256=None,
            observed_interval_seconds=None,
            min_interval_seconds=None,
            max_interval_seconds=None,
            gap_count=None,
            policy=policy,
            errors=tuple(errors),
        )

    timestamps = pd.to_datetime(frame["open_time"], errors="coerce", utc=True)
    if timestamps.isna().any():
        errors.append("invalid_timestamp")
    if not timestamps.is_unique:
        errors.append("duplicate_timestamp")
    if not timestamps.is_monotonic_increasing:
        errors.append("timestamp_order")

    numeric = frame.loc[:, _NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        errors.append("non_finite_numeric")
    if (numeric.loc[:, ("open", "high", "low", "close")] <= 0).any().any():
        errors.append("non_positive_price")
    if (numeric["volume"] < 0).any():
        errors.append("negative_volume")
    high_floor = numeric[["open", "close"]].max(axis=1)
    low_ceiling = numeric[["open", "close"]].min(axis=1)
    if (numeric["high"] < high_floor).any() or (numeric["low"] > low_ceiling).any():
        errors.append("inconsistent_high_low")

    start_utc = end_utc = None
    data_sha256 = None
    observed = minimum = maximum = None
    gap_count: int | None = None
    if not errors:
        start_utc = _normalize_timestamp(timestamps.iloc[0])
        end_utc = _normalize_timestamp(timestamps.iloc[-1])
        data_sha256 = _canonical_frame(frame, timestamps, numeric)
        if len(timestamps) > 1:
            seconds = np.diff(timestamps.astype("int64").to_numpy(dtype=np.int64)) / 1_000_000_000
            if not np.isfinite(seconds).all() or (seconds <= 0).any():
                errors.append("invalid_timestamp_interval")
            else:
                observed = float(np.median(seconds))
                minimum = float(np.min(seconds))
                maximum = float(np.max(seconds))
                if policy.expected_interval_seconds is not None:
                    expected = float(policy.expected_interval_seconds)
                    gap_count = int(np.count_nonzero(seconds > expected))
                    if gap_count:
                        warnings.append("timestamp_gaps_detected")
                    if policy.max_gap_count is not None and gap_count > policy.max_gap_count:
                        errors.append("gap_budget_exceeded")
        else:
            warnings.append("single_row_interval_unknown")
    quality_status: QualityStatus = "accepted" if not errors else "rejected"
    return OHLCVQualityReceipt(
        quality_status=quality_status,
        rows=len(frame),
        start_utc=start_utc,
        end_utc=end_utc,
        data_sha256=data_sha256,
        observed_interval_seconds=observed,
        min_interval_seconds=minimum,
        max_interval_seconds=maximum,
        gap_count=gap_count,
        policy=policy,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def require_ohlcv_quality(
    frame: pd.DataFrame, *, policy: OHLCVQualityPolicy | None = None
) -> OHLCVQualityReceipt:
    """Return an accepted receipt or fail closed with its error codes."""

    receipt = audit_ohlcv(frame, policy=policy)
    if receipt.quality_status != "accepted":
        raise DataQualityError("OHLCV quality rejected: " + ",".join(receipt.errors))
    return receipt


__all__ = [
    "DATA_QUALITY_SCHEMA",
    "DataQualityError",
    "OHLCVQualityPolicy",
    "OHLCVQualityReceipt",
    "audit_ohlcv",
    "require_ohlcv_quality",
]
