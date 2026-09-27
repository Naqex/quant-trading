"""Offline split-conformal intervals with a causal research contract.

The module calibrates a point predictor on a chronological holdout and
produces finite-sample, distribution-free symmetric intervals.  It deliberately
does not fit a predictor, fetch data, call a provider, or grant execution
capability.  Timestamps and the data digest are part of the contract so an
interval cannot silently be evaluated with a future calibration row.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_trading.discrete_optimizer import ResearchCutoffs

CONFORMAL_SCHEMA = "split-conformal-receipt-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ConformalError(ValueError):
    """Raised when an offline conformal contract is malformed."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConformalError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConformalError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConformalError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ConformalError(f"{name} must be finite")
    return result


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ConformalError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _numeric_vector(value: Sequence[float] | np.ndarray, name: str) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConformalError(f"{name} must be a non-empty one-dimensional numeric sequence") from exc
    if array.ndim != 1 or array.size == 0:
        raise ConformalError(f"{name} must be a non-empty one-dimensional numeric sequence")
    if not np.all(np.isfinite(array)):
        raise ConformalError(f"{name} must contain only finite values")
    return tuple(float(item) for item in array.tolist())


def _timestamps(value: Sequence[int] | np.ndarray, name: str) -> tuple[int, ...]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ConformalError(f"{name} must be a non-empty timestamp sequence") from exc
    if not values:
        raise ConformalError(f"{name} must be a non-empty timestamp sequence")
    normalized = tuple(_strict_int(item, f"{name}[{index}]") for index, item in enumerate(values))
    if any(left >= right for left, right in zip(normalized, normalized[1:], strict=False)):
        raise ConformalError(f"{name} must be strictly increasing")
    return normalized


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConformalError("receipt must be finite JSON data") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _residual_digest(residuals: Sequence[float]) -> str:
    return _sha256_json({"schema": "absolute-residuals-v1", "residuals": list(residuals)})


def _calibration_digest(timestamps: Sequence[int], residuals: Sequence[float]) -> str:
    return _sha256_json(
        {
            "schema": "conformal-calibration-v1",
            "timestamps_utc": list(timestamps),
            "residuals": list(residuals),
        }
    )


def _residuals(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[float, ...]:
    actual = np.asarray(_numeric_vector(y_true, "y_true"), dtype=np.float64)
    predicted = np.asarray(_numeric_vector(y_pred, "y_pred"), dtype=np.float64)
    if actual.size != predicted.size:
        raise ConformalError("y_true and y_pred must have the same length")
    with np.errstate(over="ignore", invalid="raise"):
        try:
            errors = np.abs(actual - predicted)
        except FloatingPointError as exc:
            raise ConformalError("absolute residuals overflowed or are invalid") from exc
    if not np.all(np.isfinite(errors)):
        raise ConformalError("absolute residuals overflowed or are invalid")
    return tuple(float(item) for item in errors.tolist())


@dataclass(frozen=True)
class PredictionInterval:
    """One point prediction and its calibrated interval."""

    timestamp_utc: int
    point: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        timestamp = _strict_int(self.timestamp_utc, "timestamp_utc")
        point = _finite_float(self.point, "point")
        lower = _finite_float(self.lower, "lower")
        upper = _finite_float(self.upper, "upper")
        if lower > point or point > upper:
            raise ConformalError("interval must contain its point prediction")
        object.__setattr__(self, "timestamp_utc", timestamp)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class ConformalCalibration:
    """Immutable split-conformal calibration artifact.

    ``cutoffs.train_end_utc`` is the model's information cutoff.  Calibration
    observations must arrive strictly after it and no later than validation;
    predictions must be strictly after validation and no later than test.
    """

    alpha: float
    quantile: float
    calibration_count: int
    calibration_end_utc: int
    residual_sha256: str
    calibration_sha256: str
    data_sha256: str
    model_cutoff_utc: int
    cutoffs: ResearchCutoffs
    execution_capability: bool = False

    def __post_init__(self) -> None:
        alpha = _finite_float(self.alpha, "alpha")
        if not 0.0 < alpha < 1.0:
            raise ConformalError("alpha must be strictly between 0 and 1")
        quantile = _finite_float(self.quantile, "quantile")
        if quantile < 0:
            raise ConformalError("quantile must be >= 0")
        count = _strict_int(self.calibration_count, "calibration_count", minimum=1)
        end = _strict_int(self.calibration_end_utc, "calibration_end_utc")
        model_cutoff = _strict_int(self.model_cutoff_utc, "model_cutoff_utc")
        _validate_sha256(self.residual_sha256, "residual_sha256")
        _validate_sha256(self.calibration_sha256, "calibration_sha256")
        _validate_sha256(self.data_sha256, "data_sha256")
        if not isinstance(self.cutoffs, ResearchCutoffs):
            raise ConformalError("cutoffs must be ResearchCutoffs")
        if model_cutoff != self.cutoffs.train_end_utc:
            raise ConformalError("model_cutoff_utc must equal cutoffs.train_end_utc")
        if not model_cutoff < end <= self.cutoffs.validation_end_utc:
            raise ConformalError("calibration_end_utc must be after train and no later than validation")
        if self.execution_capability is not False:
            raise ConformalError("conformal calibration cannot grant execution capability")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "calibration_count", count)
        object.__setattr__(self, "calibration_end_utc", end)
        object.__setattr__(self, "model_cutoff_utc", model_cutoff)

    def predict(
        self,
        point_predictions: Sequence[float] | np.ndarray,
        timestamps_utc: Sequence[int] | np.ndarray,
    ) -> tuple[PredictionInterval, ...]:
        """Apply the calibration to future, strictly ordered predictions."""

        points = _numeric_vector(point_predictions, "point_predictions")
        timestamps = _timestamps(timestamps_utc, "prediction_timestamps_utc")
        if len(points) != len(timestamps):
            raise ConformalError("point_predictions and prediction timestamps must have the same length")
        if any(timestamp <= self.cutoffs.validation_end_utc for timestamp in timestamps):
            raise ConformalError("prediction timestamps must be after validation cutoff")
        if any(timestamp > self.cutoffs.test_end_utc for timestamp in timestamps):
            raise ConformalError("prediction timestamps must not exceed test cutoff")
        intervals: list[PredictionInterval] = []
        for timestamp, point in zip(timestamps, points, strict=True):
            with np.errstate(over="ignore", invalid="raise"):
                try:
                    lower = float(point - self.quantile)
                    upper = float(point + self.quantile)
                except FloatingPointError as exc:
                    raise ConformalError("interval bounds overflowed or are invalid") from exc
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ConformalError("interval bounds overflowed or are invalid")
            intervals.append(PredictionInterval(timestamp, point, lower, upper))
        return tuple(intervals)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFORMAL_SCHEMA,
            "status": "PREP_ONLY",
            "method": "split_conformal_absolute_residual",
            "alpha": self.alpha,
            "quantile": self.quantile,
            "calibration_count": self.calibration_count,
            "calibration_end_utc": self.calibration_end_utc,
            "residual_sha256": self.residual_sha256,
            "calibration_sha256": self.calibration_sha256,
            "data_sha256": self.data_sha256,
            "model_cutoff_utc": self.model_cutoff_utc,
            "cutoffs": self.cutoffs.as_dict(),
            "execution_capability": self.execution_capability,
            "provider_access": False,
            "broker_access": False,
        }


def fit_split_conformal(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    timestamps_utc: Sequence[int] | np.ndarray,
    *,
    alpha: float,
    data_sha256: str,
    cutoffs: ResearchCutoffs,
) -> ConformalCalibration:
    """Fit a chronological split-conformal absolute-residual calibrator.

    This function only calibrates an already-produced point forecast.  It does
    not fit or alter the predictor and therefore cannot manufacture an edge.
    """

    if not isinstance(cutoffs, ResearchCutoffs):
        raise ConformalError("cutoffs must be ResearchCutoffs")
    alpha = _finite_float(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ConformalError("alpha must be strictly between 0 and 1")
    data_sha256 = _validate_sha256(data_sha256, "data_sha256")
    actual = _numeric_vector(y_true, "y_true")
    predicted = _numeric_vector(y_pred, "y_pred")
    timestamps = _timestamps(timestamps_utc, "calibration_timestamps_utc")
    if len(actual) != len(predicted) or len(actual) != len(timestamps):
        raise ConformalError("calibration values and timestamps must have the same length")
    if any(timestamp <= cutoffs.train_end_utc for timestamp in timestamps):
        raise ConformalError("calibration timestamps must be after train cutoff")
    if any(timestamp > cutoffs.validation_end_utc for timestamp in timestamps):
        raise ConformalError("calibration timestamps must not exceed validation cutoff")
    residuals = _residuals(actual, predicted)
    ordered = np.sort(np.asarray(residuals, dtype=np.float64))
    # The finite-sample split-conformal rank is ceil((n + 1)(1 - alpha));
    # clipping to n keeps the smallest calibration set conservative.
    rank = min(len(ordered), max(1, math.ceil((len(ordered) + 1) * (1.0 - alpha))))
    quantile = float(ordered[rank - 1])
    return ConformalCalibration(
        alpha=alpha,
        quantile=quantile,
        calibration_count=len(residuals),
        calibration_end_utc=timestamps[-1],
        residual_sha256=_residual_digest(residuals),
        calibration_sha256=_calibration_digest(timestamps, residuals),
        data_sha256=data_sha256,
        model_cutoff_utc=cutoffs.train_end_utc,
        cutoffs=cutoffs,
    )


def empirical_coverage(
    actual: Sequence[float] | np.ndarray,
    intervals: Sequence[PredictionInterval],
) -> float:
    """Return observed interval coverage for an evaluation-only sample."""

    values = _numeric_vector(actual, "actual")
    interval_values = tuple(intervals)
    if not interval_values or len(values) != len(interval_values):
        raise ConformalError("actual and intervals must have the same non-empty length")
    if any(not isinstance(interval, PredictionInterval) for interval in interval_values):
        raise ConformalError("intervals must contain PredictionInterval values")
    inside = sum(
        interval.lower <= value <= interval.upper
        for value, interval in zip(values, interval_values, strict=True)
    )
    return inside / len(values)


__all__ = [
    "CONFORMAL_SCHEMA",
    "ConformalCalibration",
    "ConformalError",
    "PredictionInterval",
    "empirical_coverage",
    "fit_split_conformal",
]
