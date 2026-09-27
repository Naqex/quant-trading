"""Offline selective-prediction and abstention contracts.

This module calibrates a deterministic uncertainty threshold against a frozen
chronological calibration window.  It reports the retained coverage and mean
absolute error (selective risk) for calibration and future evaluation rows.
The contract is intentionally provider-free: it does not fit a predictor,
fetch data, call a broker, make a trading decision, or grant execution access.
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

from .discrete_optimizer import ResearchCutoffs

SELECTIVE_SCHEMA = "selective-abstention-receipt-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SelectivePredictionError(ValueError):
    """Raised when a selective-prediction contract is malformed or unsafe."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SelectivePredictionError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SelectivePredictionError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectivePredictionError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise SelectivePredictionError(f"{name} must be finite")
    return parsed


def _nonnegative(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0.0:
        raise SelectivePredictionError(f"{name} must be >= 0")
    return parsed


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SelectivePredictionError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SelectivePredictionError("receipt must contain finite JSON data") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _numeric_vector(value: Sequence[float] | np.ndarray, name: str) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SelectivePredictionError(
            f"{name} must be a non-empty one-dimensional numeric sequence"
        ) from exc
    if array.ndim != 1 or array.size == 0:
        raise SelectivePredictionError(f"{name} must be a non-empty one-dimensional numeric sequence")
    if not np.all(np.isfinite(array)):
        raise SelectivePredictionError(f"{name} must contain only finite values")
    return tuple(float(item) for item in array.tolist())


def _uncertainty_vector(value: Sequence[float] | np.ndarray, name: str) -> tuple[float, ...]:
    values = _numeric_vector(value, name)
    if any(item < 0.0 for item in values):
        raise SelectivePredictionError(f"{name} must contain only non-negative values")
    return values


def _timestamps(value: Sequence[int] | np.ndarray, name: str) -> tuple[int, ...]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise SelectivePredictionError(f"{name} must be a non-empty timestamp sequence") from exc
    if not values:
        raise SelectivePredictionError(f"{name} must be a non-empty timestamp sequence")
    normalized = tuple(_strict_int(item, f"{name}[{index}]") for index, item in enumerate(values))
    if any(left >= right for left, right in zip(normalized, normalized[1:], strict=False)):
        raise SelectivePredictionError(f"{name} must be strictly increasing")
    return normalized


def _absolute_errors(actual: Sequence[float], predicted: Sequence[float]) -> tuple[float, ...]:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(predicted, dtype=np.float64)
    if left.size != right.size:
        raise SelectivePredictionError("actual and predictions must have the same length")
    with np.errstate(over="ignore", invalid="raise"):
        try:
            errors = np.abs(left - right)
        except FloatingPointError as exc:
            raise SelectivePredictionError("absolute errors overflowed or are invalid") from exc
    if not np.all(np.isfinite(errors)):
        raise SelectivePredictionError("absolute errors overflowed or are invalid")
    return tuple(float(item) for item in errors.tolist())


def _mean(values: Sequence[float], name: str) -> float:
    if not values:
        raise SelectivePredictionError(f"{name} requires at least one value")
    with np.errstate(over="ignore", invalid="raise"):
        try:
            result = float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))
        except FloatingPointError as exc:
            raise SelectivePredictionError(f"{name} overflowed or is invalid") from exc
    if not math.isfinite(result):
        raise SelectivePredictionError(f"{name} overflowed or is invalid")
    return result


@dataclass(frozen=True)
class SelectiveDecision:
    """One future prediction and the deterministic keep/abstain decision."""

    timestamp_utc: int
    point: float
    uncertainty: float
    accepted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_utc", _strict_int(self.timestamp_utc, "timestamp_utc"))
        object.__setattr__(self, "point", _finite(self.point, "point"))
        object.__setattr__(self, "uncertainty", _nonnegative(self.uncertainty, "uncertainty"))
        if type(self.accepted) is not bool:
            raise SelectivePredictionError("accepted must be a boolean")

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "point": self.point,
            "uncertainty": self.uncertainty,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class SelectiveCalibration:
    """Immutable uncertainty gate calibrated on a chronological window.

    The threshold is the largest observed uncertainty whose retained sample
    meets both ``target_coverage`` and ``max_risk``.  This is a calibration
    diagnostic, not a distribution-free guarantee; future risk must be
    measured separately with :func:`evaluate_selective`.
    """

    uncertainty_threshold: float
    target_coverage: float
    max_risk: float
    calibration_count: int
    calibration_accepted_count: int
    calibration_coverage: float
    calibration_risk: float
    calibration_end_utc: int
    model_cutoff_utc: int
    data_sha256: str
    calibration_sha256: str
    cutoffs: ResearchCutoffs
    schema: str = SELECTIVE_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        threshold = _nonnegative(self.uncertainty_threshold, "uncertainty_threshold")
        target = _finite(self.target_coverage, "target_coverage")
        if not 0.0 < target <= 1.0:
            raise SelectivePredictionError("target_coverage must be in (0, 1]")
        max_risk = _nonnegative(self.max_risk, "max_risk")
        count = _strict_int(self.calibration_count, "calibration_count", minimum=1)
        accepted_count = _strict_int(self.calibration_accepted_count, "calibration_accepted_count", minimum=1)
        if accepted_count > count:
            raise SelectivePredictionError("calibration_accepted_count cannot exceed calibration_count")
        coverage = _finite(self.calibration_coverage, "calibration_coverage")
        if not 0.0 < coverage <= 1.0 or not math.isclose(
            coverage, accepted_count / count, rel_tol=0.0, abs_tol=1e-15
        ):
            raise SelectivePredictionError("calibration_coverage must match accepted count")
        risk = _nonnegative(self.calibration_risk, "calibration_risk")
        if risk > max_risk + 1e-12:
            raise SelectivePredictionError("calibration_risk must not exceed max_risk")
        calibration_end = _strict_int(self.calibration_end_utc, "calibration_end_utc")
        model_cutoff = _strict_int(self.model_cutoff_utc, "model_cutoff_utc")
        if not isinstance(self.cutoffs, ResearchCutoffs):
            raise SelectivePredictionError("cutoffs must be ResearchCutoffs")
        if model_cutoff != self.cutoffs.train_end_utc:
            raise SelectivePredictionError("model_cutoff_utc must equal cutoffs.train_end_utc")
        if not model_cutoff < calibration_end <= self.cutoffs.validation_end_utc:
            raise SelectivePredictionError(
                "calibration_end_utc must be after train and no later than validation"
            )
        _validate_sha256(self.data_sha256, "data_sha256")
        _validate_sha256(self.calibration_sha256, "calibration_sha256")
        if self.schema != SELECTIVE_SCHEMA:
            raise SelectivePredictionError(f"unsupported selective schema: {self.schema!r}")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise SelectivePredictionError(f"{name} must remain false for offline selective prediction")
        object.__setattr__(self, "uncertainty_threshold", threshold)
        object.__setattr__(self, "target_coverage", target)
        object.__setattr__(self, "max_risk", max_risk)
        object.__setattr__(self, "calibration_count", count)
        object.__setattr__(self, "calibration_accepted_count", accepted_count)
        object.__setattr__(self, "calibration_coverage", coverage)
        object.__setattr__(self, "calibration_risk", risk)
        object.__setattr__(self, "calibration_end_utc", calibration_end)
        object.__setattr__(self, "model_cutoff_utc", model_cutoff)
        expected = _sha256_json(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise SelectivePredictionError("receipt_sha256 does not match canonical calibration")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method": "uncertainty_threshold_selective_regression",
            "uncertainty_threshold": self.uncertainty_threshold,
            "target_coverage": self.target_coverage,
            "max_risk": self.max_risk,
            "calibration_count": self.calibration_count,
            "calibration_accepted_count": self.calibration_accepted_count,
            "calibration_coverage": self.calibration_coverage,
            "calibration_risk": self.calibration_risk,
            "calibration_end_utc": self.calibration_end_utc,
            "model_cutoff_utc": self.model_cutoff_utc,
            "data_sha256": self.data_sha256,
            "calibration_sha256": self.calibration_sha256,
            "cutoffs": self.cutoffs.as_dict(),
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["status"] = "PREP_ONLY"
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def decide(
        self,
        point_predictions: Sequence[float] | np.ndarray,
        uncertainty_scores: Sequence[float] | np.ndarray,
        timestamps_utc: Sequence[int] | np.ndarray,
    ) -> tuple[SelectiveDecision, ...]:
        """Apply the fixed gate to future, strictly ordered predictions."""

        points = _numeric_vector(point_predictions, "point_predictions")
        uncertainties = _uncertainty_vector(uncertainty_scores, "uncertainty_scores")
        timestamps = _timestamps(timestamps_utc, "prediction_timestamps_utc")
        if not len(points) == len(uncertainties) == len(timestamps):
            raise SelectivePredictionError(
                "predictions, uncertainty scores and timestamps must have the same length"
            )
        if any(timestamp <= self.cutoffs.validation_end_utc for timestamp in timestamps):
            raise SelectivePredictionError("prediction timestamps must be after validation cutoff")
        if any(timestamp > self.cutoffs.test_end_utc for timestamp in timestamps):
            raise SelectivePredictionError("prediction timestamps must not exceed test cutoff")
        return tuple(
            SelectiveDecision(
                timestamp_utc=timestamp,
                point=point,
                uncertainty=uncertainty,
                accepted=uncertainty <= self.uncertainty_threshold,
            )
            for timestamp, point, uncertainty in zip(timestamps, points, uncertainties, strict=True)
        )


@dataclass(frozen=True)
class SelectiveEvaluation:
    """Out-of-sample coverage/risk receipt for one fixed selective policy."""

    sample_count: int
    accepted_count: int
    coverage: float
    abstention_rate: float
    selective_risk: float | None
    cutoff_timestamp: int
    policy_sha256: str
    data_sha256: str
    evaluation_sha256: str
    schema: str = SELECTIVE_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        count = _strict_int(self.sample_count, "sample_count", minimum=1)
        accepted = _strict_int(self.accepted_count, "accepted_count", minimum=0)
        if accepted > count:
            raise SelectivePredictionError("accepted_count cannot exceed sample_count")
        coverage = _finite(self.coverage, "coverage")
        abstention = _finite(self.abstention_rate, "abstention_rate")
        if not 0.0 <= coverage <= 1.0 or not math.isclose(
            coverage, accepted / count, rel_tol=0.0, abs_tol=1e-15
        ):
            raise SelectivePredictionError("coverage must match accepted count")
        if not 0.0 <= abstention <= 1.0 or not math.isclose(
            abstention, 1.0 - accepted / count, rel_tol=0.0, abs_tol=1e-15
        ):
            raise SelectivePredictionError("abstention_rate must match accepted count")
        if self.selective_risk is not None:
            _nonnegative(self.selective_risk, "selective_risk")
            if accepted == 0:
                raise SelectivePredictionError("selective_risk must be None when no rows are accepted")
        elif accepted != 0:
            raise SelectivePredictionError("selective_risk is required when rows are accepted")
        _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        _validate_sha256(self.policy_sha256, "policy_sha256")
        _validate_sha256(self.data_sha256, "data_sha256")
        _validate_sha256(self.evaluation_sha256, "evaluation_sha256")
        if self.schema != SELECTIVE_SCHEMA:
            raise SelectivePredictionError(f"unsupported selective schema: {self.schema!r}")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise SelectivePredictionError(f"{name} must remain false for offline selective evaluation")
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "accepted_count", accepted)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "abstention_rate", abstention)
        if self.selective_risk is not None:
            object.__setattr__(self, "selective_risk", _nonnegative(self.selective_risk, "selective_risk"))
        expected = _sha256_json(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise SelectivePredictionError("receipt_sha256 does not match canonical evaluation")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sample_count": self.sample_count,
            "accepted_count": self.accepted_count,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "selective_risk": self.selective_risk,
            "cutoff_timestamp": self.cutoff_timestamp,
            "policy_sha256": self.policy_sha256,
            "data_sha256": self.data_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["status"] = "PREP_ONLY"
        result["receipt_sha256"] = self.receipt_sha256
        return result


def fit_selective_threshold(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    uncertainty_scores: Sequence[float] | np.ndarray,
    timestamps_utc: Sequence[int] | np.ndarray,
    *,
    target_coverage: float,
    max_risk: float,
    data_sha256: str,
    cutoffs: ResearchCutoffs,
) -> SelectiveCalibration:
    """Fit the largest safe uncertainty threshold on calibration data.

    Calibration rows must be strictly after the model cutoff and no later than
    the validation cutoff.  The selected threshold must retain at least
    ``target_coverage`` rows and have mean absolute error no greater than
    ``max_risk``.  If no observed threshold satisfies both conditions, fitting
    fails closed instead of silently accepting an all-abstain policy.
    """

    if not isinstance(cutoffs, ResearchCutoffs):
        raise SelectivePredictionError("cutoffs must be ResearchCutoffs")
    actual = _numeric_vector(y_true, "y_true")
    predicted = _numeric_vector(y_pred, "y_pred")
    uncertainties = _uncertainty_vector(uncertainty_scores, "uncertainty_scores")
    timestamps = _timestamps(timestamps_utc, "calibration_timestamps_utc")
    if not len(actual) == len(predicted) == len(uncertainties) == len(timestamps):
        raise SelectivePredictionError("calibration values and timestamps must have the same length")
    if any(timestamp <= cutoffs.train_end_utc for timestamp in timestamps):
        raise SelectivePredictionError("calibration timestamps must be after train cutoff")
    if any(timestamp > cutoffs.validation_end_utc for timestamp in timestamps):
        raise SelectivePredictionError("calibration timestamps must not exceed validation cutoff")
    target = _finite(target_coverage, "target_coverage")
    if not 0.0 < target <= 1.0:
        raise SelectivePredictionError("target_coverage must be in (0, 1]")
    risk_limit = _nonnegative(max_risk, "max_risk")
    data_sha256 = _validate_sha256(data_sha256, "data_sha256")
    errors = _absolute_errors(actual, predicted)
    minimum_accepted = max(1, math.ceil(target * len(actual)))
    best: tuple[int, float, float] | None = None
    for threshold in sorted(set(uncertainties)):
        accepted = [index for index, value in enumerate(uncertainties) if value <= threshold]
        if len(accepted) < minimum_accepted:
            continue
        risk = _mean([errors[index] for index in accepted], "calibration risk")
        if risk <= risk_limit + 1e-12:
            candidate = (len(accepted), threshold, risk)
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
    if best is None:
        raise SelectivePredictionError("no uncertainty threshold satisfies target_coverage and max_risk")
    accepted_count, threshold, risk = best
    calibration_payload = {
        "schema": "selective-calibration-data-v1",
        "timestamps_utc": list(timestamps),
        "y_true": list(actual),
        "y_pred": list(predicted),
        "uncertainty_scores": list(uncertainties),
    }
    return SelectiveCalibration(
        uncertainty_threshold=threshold,
        target_coverage=target,
        max_risk=risk_limit,
        calibration_count=len(actual),
        calibration_accepted_count=accepted_count,
        calibration_coverage=accepted_count / len(actual),
        calibration_risk=risk,
        calibration_end_utc=timestamps[-1],
        model_cutoff_utc=cutoffs.train_end_utc,
        data_sha256=data_sha256,
        calibration_sha256=_sha256_json(calibration_payload),
        cutoffs=cutoffs,
    )


def evaluate_selective(
    calibration: SelectiveCalibration,
    actual: Sequence[float] | np.ndarray,
    decisions: Sequence[SelectiveDecision],
    *,
    data_sha256: str,
) -> SelectiveEvaluation:
    """Measure future coverage and risk without changing the fixed gate."""

    if not isinstance(calibration, SelectiveCalibration):
        raise SelectivePredictionError("calibration must be SelectiveCalibration")
    actual_values = _numeric_vector(actual, "actual")
    decision_values = tuple(decisions)
    if not decision_values or len(actual_values) != len(decision_values):
        raise SelectivePredictionError("actual and decisions must have the same non-empty length")
    if any(not isinstance(item, SelectiveDecision) for item in decision_values):
        raise SelectivePredictionError("decisions must contain SelectiveDecision values")
    timestamps = tuple(item.timestamp_utc for item in decision_values)
    if any(left >= right for left, right in zip(timestamps, timestamps[1:], strict=False)):
        raise SelectivePredictionError("decision timestamps must be strictly increasing")
    if any(timestamp <= calibration.cutoffs.validation_end_utc for timestamp in timestamps):
        raise SelectivePredictionError("decision timestamps must be after validation cutoff")
    if any(timestamp > calibration.cutoffs.test_end_utc for timestamp in timestamps):
        raise SelectivePredictionError("decision timestamps must not exceed test cutoff")
    data_sha256 = _validate_sha256(data_sha256, "data_sha256")
    accepted = [index for index, item in enumerate(decision_values) if item.accepted]
    risk: float | None
    if accepted:
        errors = _absolute_errors(
            tuple(actual_values[index] for index in accepted),
            tuple(decision_values[index].point for index in accepted),
        )
        risk = _mean(errors, "selective risk")
    else:
        risk = None
    evaluation_payload = {
        "schema": "selective-evaluation-data-v1",
        "timestamps_utc": list(timestamps),
        "actual": list(actual_values),
        "decisions": [item.as_dict() for item in decision_values],
    }
    return SelectiveEvaluation(
        sample_count=len(decision_values),
        accepted_count=len(accepted),
        coverage=len(accepted) / len(decision_values),
        abstention_rate=1.0 - len(accepted) / len(decision_values),
        selective_risk=risk,
        cutoff_timestamp=timestamps[-1],
        policy_sha256=calibration.receipt_sha256 or "",
        data_sha256=data_sha256,
        evaluation_sha256=_sha256_json(evaluation_payload),
    )


__all__ = [
    "SELECTIVE_SCHEMA",
    "SelectiveCalibration",
    "SelectiveDecision",
    "SelectiveEvaluation",
    "SelectivePredictionError",
    "evaluate_selective",
    "fit_selective_threshold",
]
