"""Deterministic, provider-free causal state primitives for Quant Lab.

The primitives in this module are deliberately small and conservative.  They
turn already captured observations into typed research artifacts while making
the point-in-time boundary explicit.  An observation whose ``known_at`` is
after a supplied cutoff is rejected instead of being silently dropped; this
keeps accidental holdout access visible in research receipts.

No primitive in this module fetches data, calls a model provider, or grants
execution authority.  The Kalman update uses a Joseph-form covariance update,
and the probability/entropy helpers reject non-finite input before doing any
math.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

KALMAN_STATE_SCHEMA = "kalman-1d-state-v1"
ENTROPY_FEATURE_SCHEMA = "shannon-entropy-feature-v1"
BAYESIAN_POSTERIOR_SCHEMA = "beta-bernoulli-posterior-v1"


class StatePrimitiveError(ValueError):
    """Raised when a state primitive receives invalid or non-causal input."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StatePrimitiveError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise StatePrimitiveError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StatePrimitiveError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise StatePrimitiveError(f"{name} must be finite")
    return parsed


def _nonnegative(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0:
        raise StatePrimitiveError(f"{name} must be >= 0")
    return parsed


def _positive(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0:
        raise StatePrimitiveError(f"{name} must be > 0")
    return parsed


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StatePrimitiveError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise StatePrimitiveError(f"{name} must be at most {maximum} characters")
    return normalized


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StatePrimitiveError("value must be finite JSON data") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CausalObservation:
    """One numeric observation with explicit event and availability times."""

    event_time: int
    known_at: int
    value: float
    sequence: int = 0

    def __post_init__(self) -> None:
        event_time = _strict_int(self.event_time, "event_time")
        known_at = _strict_int(self.known_at, "known_at")
        if known_at < event_time:
            raise StatePrimitiveError("known_at cannot precede event_time")
        value = _finite(self.value, "value")
        sequence = _strict_int(self.sequence, "sequence")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "sequence", sequence)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CausalObservation:
        if not isinstance(payload, Mapping):
            raise StatePrimitiveError("observation must be a mapping")
        try:
            return cls(
                event_time=payload["event_time"],
                known_at=payload["known_at"],
                value=payload["value"],
                sequence=payload.get("sequence", 0),
            )
        except KeyError as exc:
            raise StatePrimitiveError(f"observation is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "value": self.value,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class CategoricalObservation:
    """One categorical observation used by the entropy feature."""

    event_time: int
    known_at: int
    category: str
    sequence: int = 0

    def __post_init__(self) -> None:
        event_time = _strict_int(self.event_time, "event_time")
        known_at = _strict_int(self.known_at, "known_at")
        if known_at < event_time:
            raise StatePrimitiveError("known_at cannot precede event_time")
        category = _text(self.category, "category", maximum=128)
        sequence = _strict_int(self.sequence, "sequence")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "sequence", sequence)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CategoricalObservation:
        if not isinstance(payload, Mapping):
            raise StatePrimitiveError("categorical observation must be a mapping")
        try:
            return cls(
                event_time=payload["event_time"],
                known_at=payload["known_at"],
                category=payload["category"],
                sequence=payload.get("sequence", 0),
            )
        except KeyError as exc:
            raise StatePrimitiveError(f"categorical observation is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "category": self.category,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class BernoulliObservation:
    """One causal binary outcome for sequential Beta-Bernoulli updating."""

    event_time: int
    known_at: int
    outcome: int
    sequence: int = 0

    def __post_init__(self) -> None:
        event_time = _strict_int(self.event_time, "event_time")
        known_at = _strict_int(self.known_at, "known_at")
        if known_at < event_time:
            raise StatePrimitiveError("known_at cannot precede event_time")
        if type(self.outcome) is not int or self.outcome not in (0, 1):
            raise StatePrimitiveError("outcome must be the integer 0 or 1")
        sequence = _strict_int(self.sequence, "sequence")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "sequence", sequence)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BernoulliObservation:
        if not isinstance(payload, Mapping):
            raise StatePrimitiveError("Bernoulli observation must be a mapping")
        try:
            return cls(
                event_time=payload["event_time"],
                known_at=payload["known_at"],
                outcome=payload["outcome"],
                sequence=payload.get("sequence", 0),
            )
        except KeyError as exc:
            raise StatePrimitiveError(f"Bernoulli observation is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "outcome": self.outcome,
            "sequence": self.sequence,
        }


def _normalize_observations(
    observations: Iterable[CausalObservation | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None,
) -> tuple[CausalObservation, ...]:
    cutoff = None if cutoff_timestamp is None else _strict_int(cutoff_timestamp, "cutoff_timestamp")
    normalized = tuple(
        item if isinstance(item, CausalObservation) else CausalObservation.from_mapping(item)
        for item in observations
    )
    ordered = tuple(sorted(normalized, key=lambda item: (item.known_at, item.sequence, item.event_time)))
    previous_key: tuple[int, int] | None = None
    for item in ordered:
        if cutoff is not None and item.known_at > cutoff:
            raise StatePrimitiveError("observation exceeds research cutoff")
        key = (item.known_at, item.sequence)
        if previous_key == key:
            raise StatePrimitiveError("observations must have unique known_at/sequence keys")
        previous_key = key
    return ordered


def _normalize_categories(
    observations: Iterable[CategoricalObservation | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None,
) -> tuple[CategoricalObservation, ...]:
    cutoff = None if cutoff_timestamp is None else _strict_int(cutoff_timestamp, "cutoff_timestamp")
    normalized = tuple(
        item if isinstance(item, CategoricalObservation) else CategoricalObservation.from_mapping(item)
        for item in observations
    )
    ordered = tuple(sorted(normalized, key=lambda item: (item.known_at, item.sequence, item.event_time)))
    previous_key: tuple[int, int] | None = None
    for item in ordered:
        if cutoff is not None and item.known_at > cutoff:
            raise StatePrimitiveError("categorical observation exceeds research cutoff")
        key = (item.known_at, item.sequence)
        if previous_key == key:
            raise StatePrimitiveError("categorical observations must have unique known_at/sequence keys")
        previous_key = key
    return ordered


def _normalize_bernoulli(
    observations: Iterable[BernoulliObservation | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None,
) -> tuple[BernoulliObservation, ...]:
    cutoff = None if cutoff_timestamp is None else _strict_int(cutoff_timestamp, "cutoff_timestamp")
    normalized = tuple(
        item if isinstance(item, BernoulliObservation) else BernoulliObservation.from_mapping(item)
        for item in observations
    )
    ordered = tuple(sorted(normalized, key=lambda item: (item.known_at, item.sequence, item.event_time)))
    previous_key: tuple[int, int] | None = None
    for item in ordered:
        if cutoff is not None and item.known_at > cutoff:
            raise StatePrimitiveError("Bernoulli observation exceeds research cutoff")
        key = (item.known_at, item.sequence)
        if previous_key == key:
            raise StatePrimitiveError("Bernoulli observations must have unique known_at/sequence keys")
        previous_key = key
    return ordered


@dataclass(frozen=True)
class Kalman1DConfig:
    """Scalar local-level Kalman model parameters."""

    transition: float = 1.0
    process_variance: float = 1e-4
    measurement_variance: float = 1e-2
    initial_mean: float = 0.0
    initial_variance: float = 1.0

    def __post_init__(self) -> None:
        transition = _finite(self.transition, "transition")
        process_variance = _nonnegative(self.process_variance, "process_variance")
        measurement_variance = _positive(self.measurement_variance, "measurement_variance")
        initial_mean = _finite(self.initial_mean, "initial_mean")
        initial_variance = _nonnegative(self.initial_variance, "initial_variance")
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "process_variance", process_variance)
        object.__setattr__(self, "measurement_variance", measurement_variance)
        object.__setattr__(self, "initial_mean", initial_mean)
        object.__setattr__(self, "initial_variance", initial_variance)

    def as_dict(self) -> dict[str, float]:
        return {
            "transition": self.transition,
            "process_variance": self.process_variance,
            "measurement_variance": self.measurement_variance,
            "initial_mean": self.initial_mean,
            "initial_variance": self.initial_variance,
        }


@dataclass(frozen=True)
class KalmanEstimate:
    event_time: int
    known_at: int
    observed_value: float
    predicted_mean: float
    predicted_variance: float
    innovation: float
    innovation_variance: float
    gain: float
    filtered_mean: float
    filtered_variance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "observed_value": self.observed_value,
            "predicted_mean": self.predicted_mean,
            "predicted_variance": self.predicted_variance,
            "innovation": self.innovation,
            "innovation_variance": self.innovation_variance,
            "gain": self.gain,
            "filtered_mean": self.filtered_mean,
            "filtered_variance": self.filtered_variance,
        }


@dataclass(frozen=True)
class Kalman1DResult:
    observations: tuple[KalmanEstimate, ...]
    config: Kalman1DConfig
    cutoff_timestamp: int | None
    observation_sha256: str
    schema: str = KALMAN_STATE_SCHEMA
    execution_capability: bool = False

    def __post_init__(self) -> None:
        if self.schema != KALMAN_STATE_SCHEMA:
            raise StatePrimitiveError(f"unsupported Kalman schema: {self.schema!r}")
        if self.execution_capability is not False:
            raise StatePrimitiveError("state primitives cannot grant execution capability")
        if self.cutoff_timestamp is not None:
            _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        if not isinstance(self.config, Kalman1DConfig):
            raise StatePrimitiveError("config must be Kalman1DConfig")
        if not isinstance(self.observations, tuple):
            raise StatePrimitiveError("observations must be a tuple")
        if not isinstance(self.observation_sha256, str) or len(self.observation_sha256) != 64:
            raise StatePrimitiveError("observation_sha256 must be a SHA-256 hex digest")

    @property
    def final_mean(self) -> float:
        return self.config.initial_mean if not self.observations else self.observations[-1].filtered_mean

    @property
    def final_variance(self) -> float:
        return (
            self.config.initial_variance
            if not self.observations
            else self.observations[-1].filtered_variance
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_capability": self.execution_capability,
            "cutoff_timestamp": self.cutoff_timestamp,
            "observation_sha256": self.observation_sha256,
            "config": self.config.as_dict(),
            "observations": [item.as_dict() for item in self.observations],
            "final_mean": self.final_mean,
            "final_variance": self.final_variance,
        }

    def canonical_sha256(self) -> str:
        return _sha256(self.as_dict())


def run_kalman_1d(
    observations: Iterable[CausalObservation | Mapping[str, Any]],
    *,
    config: Kalman1DConfig | None = None,
    cutoff_timestamp: int | None = None,
) -> Kalman1DResult:
    """Run a scalar forward-only Kalman filter over causal observations.

    The covariance update is the scalar Joseph form ``(1-K)^2 P + K^2 R``;
    it is algebraically equivalent to the usual update while preserving a
    non-negative covariance under floating-point round-off.
    """

    model = config or Kalman1DConfig()
    normalized = _normalize_observations(observations, cutoff_timestamp=cutoff_timestamp)
    mean = model.initial_mean
    variance = model.initial_variance
    estimates: list[KalmanEstimate] = []
    for observation in normalized:
        predicted_mean = model.transition * mean
        predicted_variance = model.transition * model.transition * variance + model.process_variance
        if not math.isfinite(predicted_mean) or not math.isfinite(predicted_variance):
            raise StatePrimitiveError("Kalman prediction became non-finite")
        predicted_variance = max(0.0, predicted_variance)
        innovation = observation.value - predicted_mean
        innovation_variance = predicted_variance + model.measurement_variance
        if not math.isfinite(innovation_variance) or innovation_variance <= 0:
            raise StatePrimitiveError("Kalman innovation variance must be finite and > 0")
        gain = predicted_variance / innovation_variance
        filtered_mean = predicted_mean + gain * innovation
        filtered_variance = (1.0 - gain) * (1.0 - gain) * predicted_variance
        filtered_variance += gain * gain * model.measurement_variance
        if not all(
            math.isfinite(value)
            for value in (innovation, gain, filtered_mean, filtered_variance)
        ):
            raise StatePrimitiveError("Kalman update became non-finite")
        filtered_variance = max(0.0, filtered_variance)
        estimates.append(
            KalmanEstimate(
                event_time=observation.event_time,
                known_at=observation.known_at,
                observed_value=observation.value,
                predicted_mean=predicted_mean,
                predicted_variance=predicted_variance,
                innovation=innovation,
                innovation_variance=innovation_variance,
                gain=gain,
                filtered_mean=filtered_mean,
                filtered_variance=filtered_variance,
            )
        )
        mean, variance = filtered_mean, filtered_variance
    return Kalman1DResult(
        observations=tuple(estimates),
        config=model,
        cutoff_timestamp=cutoff_timestamp,
        observation_sha256=_sha256([item.as_dict() for item in normalized]),
    )


@dataclass(frozen=True)
class EntropyFeature:
    counts: Mapping[str, int]
    probabilities: Mapping[str, float]
    sample_count: int
    entropy_nats: float
    miller_madow_correction: float
    cutoff_timestamp: int | None
    observation_sha256: str
    schema: str = ENTROPY_FEATURE_SCHEMA
    execution_capability: bool = False

    def __post_init__(self) -> None:
        if self.schema != ENTROPY_FEATURE_SCHEMA:
            raise StatePrimitiveError(f"unsupported entropy schema: {self.schema!r}")
        if self.execution_capability is not False:
            raise StatePrimitiveError("state primitives cannot grant execution capability")
        _strict_int(self.sample_count, "sample_count", minimum=1)
        if not math.isfinite(self.entropy_nats) or self.entropy_nats < 0:
            raise StatePrimitiveError("entropy_nats must be finite and >= 0")
        if not math.isfinite(self.miller_madow_correction) or self.miller_madow_correction < 0:
            raise StatePrimitiveError("miller_madow_correction must be finite and >= 0")
        if not isinstance(self.counts, Mapping) or not isinstance(self.probabilities, Mapping):
            raise StatePrimitiveError("counts and probabilities must be mappings")
        if set(self.counts) != set(self.probabilities):
            raise StatePrimitiveError("counts and probabilities must have identical categories")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_capability": self.execution_capability,
            "cutoff_timestamp": self.cutoff_timestamp,
            "sample_count": self.sample_count,
            "counts": dict(self.counts),
            "probabilities": dict(self.probabilities),
            "entropy_nats": self.entropy_nats,
            "miller_madow_correction": self.miller_madow_correction,
            "observation_sha256": self.observation_sha256,
        }

    def canonical_sha256(self) -> str:
        return _sha256(self.as_dict())


def compute_shannon_entropy(
    observations: Iterable[CategoricalObservation | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None = None,
    miller_madow: bool = True,
) -> EntropyFeature:
    """Compute categorical Shannon entropy from a point-in-time sample.

    Entropy is returned in nats.  Miller–Madow is an optional small-sample
    bias correction, kept explicit in the receipt rather than silently mixed
    into the raw statistic.
    """

    if type(miller_madow) is not bool:
        raise StatePrimitiveError("miller_madow must be a boolean")
    normalized = _normalize_categories(observations, cutoff_timestamp=cutoff_timestamp)
    if not normalized:
        raise StatePrimitiveError("entropy requires at least one observation")
    counts = Counter(item.category for item in normalized)
    sample_count = len(normalized)
    probabilities = {category: count / sample_count for category, count in sorted(counts.items())}
    entropy = -sum(probability * math.log(probability) for probability in probabilities.values())
    correction = (len(probabilities) - 1) / (2.0 * sample_count) if miller_madow else 0.0
    corrected = entropy + correction
    if not math.isfinite(corrected):
        raise StatePrimitiveError("entropy became non-finite")
    return EntropyFeature(
        counts=MappingProxyType(dict(sorted(counts.items()))),
        probabilities=MappingProxyType(probabilities),
        sample_count=sample_count,
        entropy_nats=corrected,
        miller_madow_correction=correction,
        cutoff_timestamp=cutoff_timestamp,
        observation_sha256=_sha256([item.as_dict() for item in normalized]),
    )


@dataclass(frozen=True)
class BetaPrior:
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        alpha = _positive(self.alpha, "alpha")
        beta = _positive(self.beta, "beta")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    def as_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta}


@dataclass(frozen=True)
class BayesianEstimate:
    event_time: int
    known_at: int
    outcome: int
    alpha: float
    beta: float
    mean: float
    variance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "outcome": self.outcome,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "variance": self.variance,
        }


@dataclass(frozen=True)
class BayesianPosteriorResult:
    observations: tuple[BayesianEstimate, ...]
    prior: BetaPrior
    cutoff_timestamp: int | None
    observation_sha256: str
    schema: str = BAYESIAN_POSTERIOR_SCHEMA
    execution_capability: bool = False

    def __post_init__(self) -> None:
        if self.schema != BAYESIAN_POSTERIOR_SCHEMA:
            raise StatePrimitiveError(f"unsupported Bayesian schema: {self.schema!r}")
        if self.execution_capability is not False:
            raise StatePrimitiveError("state primitives cannot grant execution capability")
        if self.cutoff_timestamp is not None:
            _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        if not isinstance(self.prior, BetaPrior):
            raise StatePrimitiveError("prior must be BetaPrior")

    @property
    def final_alpha(self) -> float:
        return self.prior.alpha if not self.observations else self.observations[-1].alpha

    @property
    def final_beta(self) -> float:
        return self.prior.beta if not self.observations else self.observations[-1].beta

    @property
    def final_mean(self) -> float:
        alpha, beta = self.final_alpha, self.final_beta
        return alpha / (alpha + beta)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_capability": self.execution_capability,
            "cutoff_timestamp": self.cutoff_timestamp,
            "prior": self.prior.as_dict(),
            "observation_sha256": self.observation_sha256,
            "observations": [item.as_dict() for item in self.observations],
            "final_alpha": self.final_alpha,
            "final_beta": self.final_beta,
            "final_mean": self.final_mean,
        }

    def canonical_sha256(self) -> str:
        return _sha256(self.as_dict())


def update_beta_bernoulli(
    observations: Iterable[BernoulliObservation | Mapping[str, Any]],
    *,
    prior: BetaPrior | None = None,
    cutoff_timestamp: int | None = None,
) -> BayesianPosteriorResult:
    """Apply sequential conjugate Beta-Bernoulli updates in known-time order."""

    model = prior or BetaPrior()
    normalized = _normalize_bernoulli(observations, cutoff_timestamp=cutoff_timestamp)
    alpha, beta = model.alpha, model.beta
    estimates: list[BayesianEstimate] = []
    for observation in normalized:
        alpha += observation.outcome
        beta += 1 - observation.outcome
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (total * total * (total + 1.0))
        if not all(math.isfinite(value) for value in (alpha, beta, mean, variance)):
            raise StatePrimitiveError("Bayesian posterior became non-finite")
        estimates.append(
            BayesianEstimate(
                event_time=observation.event_time,
                known_at=observation.known_at,
                outcome=observation.outcome,
                alpha=alpha,
                beta=beta,
                mean=mean,
                variance=variance,
            )
        )
    return BayesianPosteriorResult(
        observations=tuple(estimates),
        prior=model,
        cutoff_timestamp=cutoff_timestamp,
        observation_sha256=_sha256([item.as_dict() for item in normalized]),
    )


__all__ = [
    "BAYESIAN_POSTERIOR_SCHEMA",
    "ENTROPY_FEATURE_SCHEMA",
    "KALMAN_STATE_SCHEMA",
    "BayesianEstimate",
    "BayesianPosteriorResult",
    "BetaPrior",
    "BernoulliObservation",
    "CategoricalObservation",
    "CausalObservation",
    "EntropyFeature",
    "Kalman1DConfig",
    "Kalman1DResult",
    "KalmanEstimate",
    "StatePrimitiveError",
    "compute_shannon_entropy",
    "run_kalman_1d",
    "update_beta_bernoulli",
]
