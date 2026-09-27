"""Provider-free temporal causal-effect diagnostics for offline research.

This module is intentionally smaller than a general causal-ML framework.  It
accepts point-in-time treatment/outcome rows, fits nuisance models only on the
oldest temporal split, and reports an augmented inverse-propensity (AIPW)
estimate on later splits.  It never fetches data, calls a provider/broker, or
grants execution authority.  A result is research evidence only; it is not a
trading signal or a causal claim without the stated identification assumptions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

CAUSAL_EFFECT_SCHEMA = "causal-effect-pilot-v1"
ESTIMATOR_VERSION = "temporal-aipw-train-fit-v1"
MAX_COVARIATES = 16
MAX_ABS_VALUE = 1e12
_SHA256_LENGTH = 64
ResultStatus = Literal["accepted", "inconclusive", "rejected"]


class CausalEffectError(ValueError):
    """Raised when a causal-effect input violates the offline contract."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CausalEffectError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CausalEffectError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CausalEffectError(f"{name} must be finite") from exc
    if not math.isfinite(parsed) or abs(parsed) > MAX_ABS_VALUE:
        raise CausalEffectError(f"{name} must be finite and have absolute value <= {MAX_ABS_VALUE:g}")
    return parsed


def _text(value: Any, name: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalEffectError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise CausalEffectError(f"{name} must be at most {maximum} characters")
    return normalized


def _sha256_json(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CausalEffectError("causal-effect values must be finite JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise CausalEffectError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CausalEffectError(f"{name} must be a hexadecimal SHA-256") from exc
    return value.lower()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze_covariates(covariates: Mapping[str, Any]) -> Mapping[str, float]:
    if not isinstance(covariates, Mapping):
        raise CausalEffectError("covariates must be a mapping")
    if not covariates:
        raise CausalEffectError("covariates must contain at least one pre-treatment variable")
    if len(covariates) > MAX_COVARIATES:
        raise CausalEffectError(f"covariates cannot contain more than {MAX_COVARIATES} variables")
    normalized: dict[str, float] = {}
    for key, value in covariates.items():
        name = _text(key, "covariate name", maximum=64)
        if name in normalized:
            raise CausalEffectError(f"duplicate covariate name: {name!r}")
        normalized[name] = _finite(value, f"covariate {name!r}")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class TreatmentObservation:
    """One point-in-time binary intervention and delayed numeric outcome.

    ``covariates`` are required to be pre-treatment values.  Their availability
    is represented by ``known_at``; the outcome is evaluation-only and may not
    be available until ``outcome_known_at``.
    """

    unit_id: str
    event_time: int
    known_at: int
    outcome_known_at: int
    treatment: int
    outcome: float
    covariates: Mapping[str, Any]
    sequence: int = 0

    def __post_init__(self) -> None:
        unit_id = _text(self.unit_id, "unit_id")
        event_time = _strict_int(self.event_time, "event_time")
        known_at = _strict_int(self.known_at, "known_at")
        outcome_known_at = _strict_int(self.outcome_known_at, "outcome_known_at")
        if known_at < event_time:
            raise CausalEffectError("known_at cannot precede event_time")
        if outcome_known_at <= known_at:
            raise CausalEffectError("outcome_known_at must be after treatment known_at")
        if type(self.treatment) is not int or self.treatment not in (0, 1):
            raise CausalEffectError("treatment must be the integer 0 or 1")
        outcome = _finite(self.outcome, "outcome")
        sequence = _strict_int(self.sequence, "sequence")
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "outcome_known_at", outcome_known_at)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "covariates", _freeze_covariates(self.covariates))
        object.__setattr__(self, "sequence", sequence)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TreatmentObservation:
        if not isinstance(payload, Mapping):
            raise CausalEffectError("treatment observation must be a mapping")
        try:
            return cls(
                unit_id=payload.get("unit_id", "default"),
                event_time=payload["event_time"],
                known_at=payload["known_at"],
                outcome_known_at=payload["outcome_known_at"],
                treatment=payload["treatment"],
                outcome=payload["outcome"],
                covariates=payload["covariates"],
                sequence=payload.get("sequence", 0),
            )
        except KeyError as exc:
            raise CausalEffectError(f"treatment observation is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "event_time": self.event_time,
            "known_at": self.known_at,
            "outcome_known_at": self.outcome_known_at,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "covariates": _thaw(self.covariates),
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class TemporalCausalSplit:
    """Half-open temporal boundaries expressed as inclusive timestamps."""

    train_end: int
    calibration_end: int
    holdout_end: int | None = None

    def __post_init__(self) -> None:
        train_end = _strict_int(self.train_end, "train_end")
        calibration_end = _strict_int(self.calibration_end, "calibration_end")
        if calibration_end <= train_end:
            raise CausalEffectError("calibration_end must be after train_end")
        holdout_end = self.holdout_end
        if holdout_end is not None:
            holdout_end = _strict_int(holdout_end, "holdout_end")
            if holdout_end <= calibration_end:
                raise CausalEffectError("holdout_end must be after calibration_end")
        object.__setattr__(self, "train_end", train_end)
        object.__setattr__(self, "calibration_end", calibration_end)
        object.__setattr__(self, "holdout_end", holdout_end)

    def as_dict(self) -> dict[str, int | None]:
        return {
            "train_end": self.train_end,
            "calibration_end": self.calibration_end,
            "holdout_end": self.holdout_end,
        }


@dataclass(frozen=True)
class CausalEffectConfig:
    """Pre-registered numerical and overlap guards for the bounded pilot."""

    ridge_lambda: float = 1e-3
    propensity_epsilon: float = 0.05
    min_overlap: float = 0.80
    min_effective_sample_size: float = 5.0
    min_arm_count: int = 4
    max_iterations: int = 80
    tolerance: float = 1e-8
    placebo_shift: int = 1

    def __post_init__(self) -> None:
        ridge_lambda = _finite(self.ridge_lambda, "ridge_lambda")
        if ridge_lambda <= 0:
            raise CausalEffectError("ridge_lambda must be > 0")
        propensity_epsilon = _finite(self.propensity_epsilon, "propensity_epsilon")
        if not 0 < propensity_epsilon < 0.5:
            raise CausalEffectError("propensity_epsilon must be in (0, 0.5)")
        min_overlap = _finite(self.min_overlap, "min_overlap")
        if not 0 < min_overlap <= 1:
            raise CausalEffectError("min_overlap must be in (0, 1]")
        ess = _finite(self.min_effective_sample_size, "min_effective_sample_size")
        if ess < 1:
            raise CausalEffectError("min_effective_sample_size must be >= 1")
        min_arm_count = _strict_int(self.min_arm_count, "min_arm_count", minimum=2)
        max_iterations = _strict_int(self.max_iterations, "max_iterations", minimum=1)
        tolerance = _finite(self.tolerance, "tolerance")
        if tolerance <= 0:
            raise CausalEffectError("tolerance must be > 0")
        placebo_shift = _strict_int(self.placebo_shift, "placebo_shift", minimum=1)
        object.__setattr__(self, "ridge_lambda", ridge_lambda)
        object.__setattr__(self, "propensity_epsilon", propensity_epsilon)
        object.__setattr__(self, "min_overlap", min_overlap)
        object.__setattr__(self, "min_effective_sample_size", ess)
        object.__setattr__(self, "min_arm_count", min_arm_count)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "placebo_shift", placebo_shift
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ridge_lambda": self.ridge_lambda,
            "propensity_epsilon": self.propensity_epsilon,
            "min_overlap": self.min_overlap,
            "min_effective_sample_size": self.min_effective_sample_size,
            "min_arm_count": self.min_arm_count,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "placebo_shift": self.placebo_shift,
        }


@dataclass(frozen=True)
class CausalEffectResult:
    """Immutable offline result with deny-only capability flags."""

    status: ResultStatus
    reason: str | None
    estimator: str
    data_sha256: str
    model_sha256: str
    split: TemporalCausalSplit
    train_count: int
    calibration_count: int
    holdout_count: int
    ate_calibration: float | None
    ate_holdout: float | None
    baseline_difference_in_means: float | None
    placebo_difference_in_means: float | None
    overlap_fraction: float | None
    effective_sample_size_treated: float | None
    effective_sample_size_control: float | None
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    schema: str = CAUSAL_EFFECT_SCHEMA
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != CAUSAL_EFFECT_SCHEMA:
            raise CausalEffectError(f"unsupported causal-effect schema: {self.schema!r}")
        if self.status not in {"accepted", "inconclusive", "rejected"}:
            raise CausalEffectError("unsupported causal-effect status")
        if not isinstance(self.estimator, str) or not self.estimator.strip():
            raise CausalEffectError("estimator must be a non-empty string")
        _validate_sha256(self.data_sha256, "data_sha256")
        _validate_sha256(self.model_sha256, "model_sha256")
        for name in ("train_count", "calibration_count", "holdout_count"):
            _strict_int(getattr(self, name), name)
        if self.reason is not None:
            _text(self.reason, "reason", maximum=512)
        for name in (
            "ate_calibration",
            "ate_holdout",
            "baseline_difference_in_means",
            "placebo_difference_in_means",
            "overlap_fraction",
            "effective_sample_size_treated",
            "effective_sample_size_control",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.overlap_fraction is not None and not 0 <= self.overlap_fraction <= 1:
            raise CausalEffectError("overlap_fraction must be in [0, 1]")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise CausalEffectError(f"{name} must remain false for offline research")
        expected = _sha256_json(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise CausalEffectError("receipt_sha256 does not match canonical result")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "reason": self.reason,
            "estimator": self.estimator,
            "data_sha256": self.data_sha256,
            "model_sha256": self.model_sha256,
            "split": self.split.as_dict(),
            "train_count": self.train_count,
            "calibration_count": self.calibration_count,
            "holdout_count": self.holdout_count,
            "ate_calibration": self.ate_calibration,
            "ate_holdout": self.ate_holdout,
            "baseline_difference_in_means": self.baseline_difference_in_means,
            "placebo_difference_in_means": self.placebo_difference_in_means,
            "overlap_fraction": self.overlap_fraction,
            "effective_sample_size_treated": self.effective_sample_size_treated,
            "effective_sample_size_control": self.effective_sample_size_control,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["receipt_sha256"] = self.receipt_sha256
        return result


def _normalize_observations(
    observations: Iterable[TreatmentObservation | Mapping[str, Any]],
) -> tuple[TreatmentObservation, ...]:
    normalized = tuple(
        item if isinstance(item, TreatmentObservation) else TreatmentObservation.from_mapping(item)
        for item in observations
    )
    if not normalized:
        raise CausalEffectError("at least one treatment observation is required")
    covariate_names = tuple(normalized[0].covariates.keys())
    for item in normalized[1:]:
        if tuple(item.covariates.keys()) != covariate_names:
            raise CausalEffectError("all observations must use the same covariate names")
    ordered = tuple(
        sorted(
            normalized,
            key=lambda item: (
                item.outcome_known_at,
                item.known_at,
                item.sequence,
                item.unit_id,
            ),
        )
    )
    event_keys: set[tuple[str, int, int]] = set()
    known_keys: set[tuple[str, int, int]] = set()
    for item in ordered:
        event_key = (item.unit_id, item.event_time, item.sequence)
        known_key = (item.unit_id, item.known_at, item.sequence)
        if event_key in event_keys:
            raise CausalEffectError("duplicate unit/event_time/sequence key")
        if known_key in known_keys:
            raise CausalEffectError("duplicate unit/known_at/sequence key")
        event_keys.add(event_key)
        known_keys.add(known_key)
    return ordered


def split_temporal_observations(
    observations: Iterable[TreatmentObservation | Mapping[str, Any]],
    split: TemporalCausalSplit,
) -> tuple[
    tuple[TreatmentObservation, ...],
    tuple[TreatmentObservation, ...],
    tuple[TreatmentObservation, ...],
]:
    """Assign rows by outcome availability, preserving a strict time order."""

    normalized = _normalize_observations(observations)
    train = tuple(item for item in normalized if item.outcome_known_at <= split.train_end)
    calibration = tuple(
        item
        for item in normalized
        if split.train_end < item.outcome_known_at <= split.calibration_end
    )
    holdout = tuple(
        item
        for item in normalized
        if item.outcome_known_at > split.calibration_end
        and (split.holdout_end is None or item.outcome_known_at <= split.holdout_end)
    )
    return train, calibration, holdout


def _design_matrix(rows: tuple[TreatmentObservation, ...], names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([[item.covariates[name] for name in names] for item in rows], dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise CausalEffectError("covariates produced a non-finite design matrix")
    return np.column_stack((np.ones(len(values), dtype=np.float64), values))


def _robust_scale(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(train_x, axis=0)
    centered = train_x - center
    if not np.isfinite(center).all() or not np.isfinite(centered).all():
        raise CausalEffectError("covariate centering became non-finite")
    mad = np.median(np.abs(centered), axis=0)
    scale = np.maximum(1.4826 * mad, 1.0)
    transformed = (x - center) / scale
    if not np.isfinite(transformed).all() or np.max(np.abs(transformed)) > MAX_ABS_VALUE:
        raise CausalEffectError("scaled covariates are outside the finite numerical bound")
    return transformed, center, scale


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    if len(x) == 0:
        raise CausalEffectError("cannot fit an empty nuisance model")
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    with np.errstate(over="raise", invalid="raise"):
        try:
            gram = design.T @ design + penalty
            rhs = design.T @ y
            coefficients = np.linalg.solve(gram, rhs)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise CausalEffectError("outcome nuisance fit failed numerically") from exc
    if not np.isfinite(coefficients).all():
        raise CausalEffectError("outcome nuisance fit produced non-finite coefficients")
    return coefficients


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    result = np.empty_like(clipped)
    positive = clipped >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-clipped[positive]))
    negative = ~positive
    exp_value = np.exp(clipped[negative])
    result[negative] = exp_value / (1.0 + exp_value)
    return result


def _logistic_fit(x: np.ndarray, treatment: np.ndarray, config: CausalEffectConfig) -> np.ndarray:
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * config.ridge_lambda
    penalty[0, 0] = 0.0
    for _ in range(config.max_iterations):
        eta = design @ coefficients
        probabilities = _sigmoid(eta)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        with np.errstate(over="raise", invalid="raise"):
            try:
                hessian = design.T @ (weights[:, None] * design) + penalty
                gradient = design.T @ (probabilities - treatment) + penalty @ coefficients
                step = np.linalg.solve(hessian, gradient)
            except (FloatingPointError, np.linalg.LinAlgError) as exc:
                raise CausalEffectError("propensity nuisance fit failed numerically") from exc
        coefficients -= step
        if not np.isfinite(coefficients).all():
            raise CausalEffectError("propensity nuisance fit produced non-finite coefficients")
        if float(np.max(np.abs(step))) <= config.tolerance:
            break
    return coefficients


def _predict_outcome(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    values = np.column_stack((np.ones(len(x), dtype=np.float64), x)) @ coefficients
    if not np.isfinite(values).all():
        raise CausalEffectError("outcome nuisance prediction became non-finite")
    return values


def _predict_propensity(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    logits = np.column_stack((np.ones(len(x), dtype=np.float64), x)) @ coefficients
    values = _sigmoid(logits)
    if not np.isfinite(values).all():
        raise CausalEffectError("propensity prediction became non-finite")
    return values


def _mean_or_none(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    value = float(np.mean(values, dtype=np.float64))
    return value if math.isfinite(value) else None


def _difference_in_means(outcomes: np.ndarray, treatment: np.ndarray) -> float | None:
    treated = outcomes[treatment == 1]
    control = outcomes[treatment == 0]
    if len(treated) == 0 or len(control) == 0:
        return None
    return _mean_or_none(treated) - _mean_or_none(control)


def _effective_sample_size(
    propensity: np.ndarray, treatment: np.ndarray
) -> tuple[float | None, float | None]:
    treated_weights = 1.0 / propensity[treatment == 1]
    control_weights = 1.0 / (1.0 - propensity[treatment == 0])

    def ess(weights: np.ndarray) -> float | None:
        if len(weights) == 0:
            return None
        denominator = float(np.sum(weights * weights, dtype=np.float64))
        numerator = float(np.sum(weights, dtype=np.float64)) ** 2
        if denominator <= 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
            return None
        value = numerator / denominator
        return value if math.isfinite(value) else None

    return ess(treated_weights), ess(control_weights)


def _aipw(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    propensity: np.ndarray,
) -> float:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            pseudo = m1 - m0 + treatment * (outcomes - m1) / propensity - (1.0 - treatment) * (
                outcomes - m0
            ) / (1.0 - propensity)
            value = float(np.mean(pseudo, dtype=np.float64))
        except (FloatingPointError, ZeroDivisionError) as exc:
            raise CausalEffectError("AIPW pseudo-outcome became non-finite") from exc
    if not math.isfinite(value):
        raise CausalEffectError("AIPW estimate became non-finite")
    return value


def _inconclusive(
    *,
    reason: str,
    data_sha256: str,
    model_sha256: str,
    split: TemporalCausalSplit,
    train_count: int,
    calibration_count: int,
    holdout_count: int,
    overlap_fraction: float | None = None,
    effective_sample_size_treated: float | None = None,
    effective_sample_size_control: float | None = None,
) -> CausalEffectResult:
    return CausalEffectResult(
        status="inconclusive",
        reason=reason,
        estimator=ESTIMATOR_VERSION,
        data_sha256=data_sha256,
        model_sha256=model_sha256,
        split=split,
        train_count=train_count,
        calibration_count=calibration_count,
        holdout_count=holdout_count,
        ate_calibration=None,
        ate_holdout=None,
        baseline_difference_in_means=None,
        placebo_difference_in_means=None,
        overlap_fraction=overlap_fraction,
        effective_sample_size_treated=effective_sample_size_treated,
        effective_sample_size_control=effective_sample_size_control,
    )


def run_causal_effect_pilot(
    observations: Iterable[TreatmentObservation | Mapping[str, Any]],
    *,
    split: TemporalCausalSplit,
    config: CausalEffectConfig | None = None,
) -> CausalEffectResult:
    """Fit nuisances on train and evaluate a temporal AIPW diagnostic.

    The calibration and holdout outcomes are never used to fit nuisances.  A
    result is marked ``inconclusive`` when either treatment arm is absent,
    propensity overlap is weak, or weighted effective sample size is below the
    pre-registered threshold.
    """

    if config is None:
        config = CausalEffectConfig()
    normalized = _normalize_observations(observations)
    data_sha256 = _sha256_json([item.as_dict() for item in normalized])
    model_sha256 = _sha256_json(
        {"estimator": ESTIMATOR_VERSION, "config": config.as_dict(), "split": split.as_dict()}
    )
    train, calibration, holdout = split_temporal_observations(normalized, split)
    names = tuple(normalized[0].covariates.keys())
    if len(train) < config.min_arm_count * 2:
        return _inconclusive(
            reason="train split is too small",
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            split=split,
            train_count=len(train),
            calibration_count=len(calibration),
            holdout_count=len(holdout),
        )
    train_treatment = np.asarray([item.treatment for item in train], dtype=np.int64)
    train_control_count = np.count_nonzero(train_treatment == 0)
    train_treated_count = np.count_nonzero(train_treatment == 1)
    if train_control_count < config.min_arm_count or train_treated_count < config.min_arm_count:
        return _inconclusive(
            reason="train split lacks both treatment arms",
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            split=split,
            train_count=len(train),
            calibration_count=len(calibration),
            holdout_count=len(holdout),
        )
    if not calibration or not holdout:
        return _inconclusive(
            reason="calibration and holdout splits are required",
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            split=split,
            train_count=len(train),
            calibration_count=len(calibration),
            holdout_count=len(holdout),
        )
    train_x_raw = _design_matrix(train, names)[:, 1:]
    calibration_x_raw = _design_matrix(calibration, names)[:, 1:]
    holdout_x_raw = _design_matrix(holdout, names)[:, 1:]
    train_x, _, _ = _robust_scale(train_x_raw, train_x_raw)
    calibration_x, _, _ = _robust_scale(train_x_raw, calibration_x_raw)
    holdout_x, _, _ = _robust_scale(train_x_raw, holdout_x_raw)
    train_y = np.asarray([item.outcome for item in train], dtype=np.float64)
    train_y0 = train_y[train_treatment == 0]
    train_y1 = train_y[train_treatment == 1]
    fit_x0 = train_x[train_treatment == 0]
    fit_x1 = train_x[train_treatment == 1]
    m0_coefficients = _ridge_fit(fit_x0, train_y0, config.ridge_lambda)
    m1_coefficients = _ridge_fit(fit_x1, train_y1, config.ridge_lambda)
    propensity_coefficients = _logistic_fit(train_x, train_treatment.astype(np.float64), config)

    calibration_treatment = np.asarray([item.treatment for item in calibration], dtype=np.int64)
    holdout_treatment = np.asarray([item.treatment for item in holdout], dtype=np.int64)
    calibration_y = np.asarray([item.outcome for item in calibration], dtype=np.float64)
    holdout_y = np.asarray([item.outcome for item in holdout], dtype=np.float64)
    calibration_propensity_raw = _predict_propensity(propensity_coefficients, calibration_x)
    holdout_propensity_raw = _predict_propensity(propensity_coefficients, holdout_x)
    overlap_mask = (holdout_propensity_raw >= config.propensity_epsilon) & (
        holdout_propensity_raw <= 1.0 - config.propensity_epsilon
    )
    calibration_propensity = np.clip(
        calibration_propensity_raw, config.propensity_epsilon, 1.0 - config.propensity_epsilon
    )
    holdout_propensity = np.clip(
        holdout_propensity_raw, config.propensity_epsilon, 1.0 - config.propensity_epsilon
    )
    overlap_fraction = float(np.mean(overlap_mask, dtype=np.float64))
    holdout_ess_treated, holdout_ess_control = _effective_sample_size(holdout_propensity, holdout_treatment)
    holdout_control_count = np.count_nonzero(holdout_treatment == 0)
    holdout_treated_count = np.count_nonzero(holdout_treatment == 1)
    if holdout_control_count < config.min_arm_count or holdout_treated_count < config.min_arm_count:
        return _inconclusive(
            reason="holdout split lacks both treatment arms",
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            split=split,
            train_count=len(train),
            calibration_count=len(calibration),
            holdout_count=len(holdout),
            overlap_fraction=overlap_fraction,
            effective_sample_size_treated=holdout_ess_treated,
            effective_sample_size_control=holdout_ess_control,
        )
    if overlap_fraction < config.min_overlap:
        return _inconclusive(
            reason="propensity overlap is below the pre-registered threshold",
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            split=split,
            train_count=len(train),
            calibration_count=len(calibration),
            holdout_count=len(holdout),
            overlap_fraction=overlap_fraction,
            effective_sample_size_treated=holdout_ess_treated,
            effective_sample_size_control=holdout_ess_control,
        )
    weak_ess = (
        holdout_ess_treated is None
        or holdout_ess_control is None
        or min(holdout_ess_treated, holdout_ess_control) < config.min_effective_sample_size
    )
    if weak_ess:
        return _inconclusive(
            reason="holdout effective sample size is below the pre-registered threshold",
            data_sha256=data_sha256,
            model_sha256=model_sha256,
            split=split,
            train_count=len(train),
            calibration_count=len(calibration),
            holdout_count=len(holdout),
            overlap_fraction=overlap_fraction,
            effective_sample_size_treated=holdout_ess_treated,
            effective_sample_size_control=holdout_ess_control,
        )
    calibration_m0 = _predict_outcome(m0_coefficients, calibration_x)
    calibration_m1 = _predict_outcome(m1_coefficients, calibration_x)
    holdout_m0 = _predict_outcome(m0_coefficients, holdout_x)
    holdout_m1 = _predict_outcome(m1_coefficients, holdout_x)
    ate_calibration = _aipw(
        calibration_y,
        calibration_treatment,
        calibration_m0,
        calibration_m1,
        calibration_propensity,
    )
    ate_holdout = _aipw(holdout_y, holdout_treatment, holdout_m0, holdout_m1, holdout_propensity)
    baseline = _difference_in_means(holdout_y, holdout_treatment)
    placebo_treatment = np.roll(holdout_treatment, config.placebo_shift)
    placebo = _difference_in_means(holdout_y, placebo_treatment)
    return CausalEffectResult(
        status="accepted",
        reason=None,
        estimator=ESTIMATOR_VERSION,
        data_sha256=data_sha256,
        model_sha256=model_sha256,
        split=split,
        train_count=len(train),
        calibration_count=len(calibration),
        holdout_count=len(holdout),
        ate_calibration=ate_calibration,
        ate_holdout=ate_holdout,
        baseline_difference_in_means=baseline,
        placebo_difference_in_means=placebo,
        overlap_fraction=overlap_fraction,
        effective_sample_size_treated=holdout_ess_treated,
        effective_sample_size_control=holdout_ess_control,
    )


__all__ = [
    "CAUSAL_EFFECT_SCHEMA",
    "ESTIMATOR_VERSION",
    "CausalEffectConfig",
    "CausalEffectError",
    "CausalEffectResult",
    "TemporalCausalSplit",
    "TreatmentObservation",
    "run_causal_effect_pilot",
    "split_temporal_observations",
]
