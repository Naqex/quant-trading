"""Causal Bayesian online change-point diagnostics for offline research.

This module implements a deliberately narrow BOCPD model for a scalar stream:
each segment uses a Normal-Inverse-Gamma prior, which gives a Student-t
posterior predictive distribution.  The run-length posterior is updated in
known-time order using log-space recursion.  It is a diagnostic for
distributional shifts, not a trading signal, forecast, or execution
permission.

The optional ``max_run_length`` is a safety bound, not a truncation policy. If
the bound would be exceeded, the detector raises instead of dropping posterior
mass or silently changing the model.  This keeps stress tests and research
receipts honest.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .state_primitives import CausalObservation, StatePrimitiveError

BAYESIAN_CHANGE_SCHEMA = "bayesian-change-point-nig-v1"
_MAX_FLOAT = sys.float_info.max


class BayesianChangePointError(ValueError):
    """Raised when the causal BOCPD contract is violated."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BayesianChangePointError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise BayesianChangePointError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BayesianChangePointError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise BayesianChangePointError(f"{name} must be finite")
    return parsed


def _positive(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0:
        raise BayesianChangePointError(f"{name} must be > 0")
    return parsed


def _probability(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if not 0.0 < parsed <= 1.0:
        raise BayesianChangePointError(f"{name} must be in (0, 1]")
    return parsed


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BayesianChangePointError("value must be finite JSON data") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _logsumexp(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        raise BayesianChangePointError("logsumexp requires at least one value")
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    if not math.isfinite(maximum):
        raise BayesianChangePointError("logsumexp received a non-finite value")
    total = math.fsum(math.exp(value - maximum) for value in values if value != -math.inf)
    if not math.isfinite(total) or total <= 0:
        raise BayesianChangePointError("logsumexp became non-finite")
    result = maximum + math.log(total)
    if not math.isfinite(result):
        raise BayesianChangePointError("logsumexp result became non-finite")
    return result


def _weighted_mean(left: float, right: float, right_weight: float) -> float:
    """Stable convex interpolation without subtracting opposite extremes."""

    left_weight = 1.0 - right_weight
    result = math.fsum((left * left_weight, right * right_weight))
    if not math.isfinite(result):
        raise BayesianChangePointError("posterior mean became non-finite")
    return result


def _log_abs_difference(left: float, right: float) -> float:
    """Compute ``log(abs(left-right))`` without overflowing subtraction."""

    left_abs, right_abs = abs(left), abs(right)
    larger = max(left_abs, right_abs)
    smaller = min(left_abs, right_abs)
    if larger == 0.0:
        return -math.inf
    # Same-sign equal values cancel exactly. Opposite-sign values are summed;
    # the ratio form keeps this finite even when the raw sum would overflow.
    if left * right >= 0.0 and left_abs == right_abs:
        return -math.inf
    return math.log(larger) + math.log1p(smaller / larger)


def _log1pexp(value: float) -> float:
    if value > 40.0:
        return value + math.log1p(math.exp(-value))
    if value < -40.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _student_t_logpdf(value: float, mean: float, kappa: float, alpha: float, beta: float) -> float:
    """Finite-clamped Student-t predictive log density for NIG parameters."""

    if not all(math.isfinite(item) and item > 0 for item in (kappa, alpha, beta)):
        raise BayesianChangePointError("predictive parameters must be finite and > 0")
    degrees_of_freedom = 2.0 * alpha
    if not math.isfinite(degrees_of_freedom):
        raise BayesianChangePointError("predictive degrees of freedom became non-finite")
    log_scale_squared = math.log(beta) + math.log1p(1.0 / kappa) - math.log(alpha)
    log_delta = _log_abs_difference(value, mean)
    log_ratio = (
        -math.inf
        if log_delta == -math.inf
        else 2.0 * log_delta - math.log(degrees_of_freedom) - log_scale_squared
    )
    tail = 0.0 if log_ratio == -math.inf else _log1pexp(log_ratio)
    score = (
        math.lgamma((degrees_of_freedom + 1.0) / 2.0)
        - math.lgamma(degrees_of_freedom / 2.0)
        - 0.5 * (math.log(degrees_of_freedom) + math.log(math.pi) + log_scale_squared)
        - ((degrees_of_freedom + 1.0) / 2.0) * tail
    )
    if not math.isfinite(score):
        return -_MAX_FLOAT if score < 0 else 0.0
    return max(-_MAX_FLOAT, min(0.0, score))


@dataclass(frozen=True)
class BOCPDConfig:
    """Explicit Normal-Inverse-Gamma BOCPD assumptions.

    ``hazard`` is a constant prior probability of a change at each step.
    ``prior_kappa`` controls prior mean strength; ``prior_alpha`` and
    ``prior_beta`` parameterize the inverse-gamma variance prior.
    """

    prior_mean: float = 0.0
    prior_kappa: float = 1.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    hazard: float = 0.05
    max_run_length: int | None = None

    def __post_init__(self) -> None:
        prior_mean = _finite(self.prior_mean, "prior_mean")
        prior_kappa = _positive(self.prior_kappa, "prior_kappa")
        prior_alpha = _positive(self.prior_alpha, "prior_alpha")
        prior_beta = _positive(self.prior_beta, "prior_beta")
        hazard = _probability(self.hazard, "hazard")
        if self.max_run_length is not None:
            _strict_int(self.max_run_length, "max_run_length", minimum=1)
        object.__setattr__(self, "prior_mean", prior_mean)
        object.__setattr__(self, "prior_kappa", prior_kappa)
        object.__setattr__(self, "prior_alpha", prior_alpha)
        object.__setattr__(self, "prior_beta", prior_beta)
        object.__setattr__(self, "hazard", hazard)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prior_mean": self.prior_mean,
            "prior_kappa": self.prior_kappa,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "hazard": self.hazard,
            "max_run_length": self.max_run_length,
        }


@dataclass(frozen=True)
class RunLengthProbability:
    run_length: int
    probability: float

    def __post_init__(self) -> None:
        _strict_int(self.run_length, "run_length")
        probability = _finite(self.probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise BayesianChangePointError("probability must be in [0, 1]")
        object.__setattr__(self, "probability", probability)

    def as_dict(self) -> dict[str, Any]:
        return {"run_length": self.run_length, "probability": self.probability}


@dataclass(frozen=True)
class BOCPDStep:
    """One causal posterior update after observing one scalar value."""

    event_time: int
    known_at: int
    sequence: int
    observed_value: float
    predictive_log_score: float
    change_probability: float
    map_run_length: int
    run_length_posterior: tuple[RunLengthProbability, ...]

    def __post_init__(self) -> None:
        _strict_int(self.event_time, "event_time")
        known_at = _strict_int(self.known_at, "known_at")
        sequence = _strict_int(self.sequence, "sequence")
        if known_at < self.event_time:
            raise BayesianChangePointError("known_at cannot precede event_time")
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "observed_value", _finite(self.observed_value, "observed_value"))
        object.__setattr__(
            self,
            "predictive_log_score",
            _finite(self.predictive_log_score, "predictive_log_score"),
        )
        change_probability = _finite(self.change_probability, "change_probability")
        if not 0.0 <= change_probability <= 1.0:
            raise BayesianChangePointError("change_probability must be in [0, 1]")
        object.__setattr__(self, "change_probability", change_probability)
        map_run_length = _strict_int(self.map_run_length, "map_run_length")
        object.__setattr__(self, "map_run_length", map_run_length)
        if not isinstance(self.run_length_posterior, tuple) or not self.run_length_posterior:
            raise BayesianChangePointError("run_length_posterior must be a non-empty tuple")
        previous = -1
        total = 0.0
        posterior_zero: float | None = None
        map_probability = -1.0
        run_lengths: set[int] = set()
        for item in self.run_length_posterior:
            if not isinstance(item, RunLengthProbability):
                raise BayesianChangePointError("run_length_posterior has an invalid item")
            if item.run_length <= previous:
                raise BayesianChangePointError("run_length_posterior must be strictly ordered")
            previous = item.run_length
            run_lengths.add(item.run_length)
            total = math.fsum((total, item.probability))
            if item.run_length == 0:
                posterior_zero = item.probability
            if item.probability > map_probability:
                map_probability = item.probability
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise BayesianChangePointError("run_length_posterior must sum to one")
        if posterior_zero is None or not math.isclose(
            posterior_zero, change_probability, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise BayesianChangePointError("change_probability must match run_length zero posterior")
        if self.map_run_length not in run_lengths:
            raise BayesianChangePointError("map_run_length is absent from posterior")

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "sequence": self.sequence,
            "observed_value": self.observed_value,
            "predictive_log_score": self.predictive_log_score,
            "change_probability": self.change_probability,
            "map_run_length": self.map_run_length,
            "run_length_posterior": [item.as_dict() for item in self.run_length_posterior],
        }


@dataclass(frozen=True)
class BOCPDResult:
    steps: tuple[BOCPDStep, ...]
    config: BOCPDConfig
    cutoff_timestamp: int | None
    observation_sha256: str
    schema: str = BAYESIAN_CHANGE_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != BAYESIAN_CHANGE_SCHEMA:
            raise BayesianChangePointError(f"unsupported BOCPD schema: {self.schema!r}")
        if not isinstance(self.steps, tuple):
            raise BayesianChangePointError("steps must be a tuple")
        if not isinstance(self.config, BOCPDConfig):
            raise BayesianChangePointError("config must be BOCPDConfig")
        if self.cutoff_timestamp is not None:
            _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise BayesianChangePointError(f"{name} must remain false")
        if (
            not isinstance(self.observation_sha256, str)
            or len(self.observation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.observation_sha256)
        ):
            raise BayesianChangePointError("observation_sha256 must be a SHA-256 hex digest")
        previous_key: tuple[int, int, int] | None = None
        for step in self.steps:
            if not isinstance(step, BOCPDStep):
                raise BayesianChangePointError("steps must contain BOCPDStep values")
            if self.cutoff_timestamp is not None and step.known_at > self.cutoff_timestamp:
                raise BayesianChangePointError("step known_at exceeds cutoff_timestamp")
            key = (step.known_at, step.sequence, step.event_time)
            if previous_key is not None and key < previous_key:
                raise BayesianChangePointError("steps must be in known-time order")
            previous_key = key

    @property
    def change_probabilities(self) -> tuple[float, ...]:
        return tuple(step.change_probability for step in self.steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
            "cutoff_timestamp": self.cutoff_timestamp,
            "config": self.config.as_dict(),
            "observation_sha256": self.observation_sha256,
            "steps": [step.as_dict() for step in self.steps],
        }

    def canonical_sha256(self) -> str:
        return _sha256(self.as_dict())


def _normalize_observations(
    observations: Iterable[CausalObservation | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None,
) -> tuple[CausalObservation, ...]:
    if cutoff_timestamp is not None:
        _strict_int(cutoff_timestamp, "cutoff_timestamp")
    try:
        iterator = iter(observations)
    except TypeError as exc:
        raise BayesianChangePointError("observations must be iterable") from exc
    normalized: list[CausalObservation] = []
    for item in iterator:
        try:
            observation = (
                item if isinstance(item, CausalObservation) else CausalObservation.from_mapping(item)
            )
        except (StatePrimitiveError, TypeError) as exc:
            raise BayesianChangePointError(str(exc)) from exc
        if cutoff_timestamp is not None and observation.known_at > cutoff_timestamp:
            raise BayesianChangePointError(
                "observation known_at exceeds cutoff_timestamp; future data is rejected"
            )
        normalized.append(observation)
    ordered = tuple(sorted(normalized, key=lambda item: (item.known_at, item.sequence, item.event_time)))
    seen: set[tuple[int, int]] = set()
    for item in ordered:
        key = (item.known_at, item.sequence)
        if key in seen:
            raise BayesianChangePointError("observations must have unique known_at/sequence keys")
        seen.add(key)
    return ordered


def _posterior_update(
    mean: float,
    kappa: float,
    alpha: float,
    beta: float,
    value: float,
) -> tuple[float, float, float, float]:
    new_kappa = kappa + 1.0
    if not math.isfinite(new_kappa):
        raise BayesianChangePointError("posterior kappa became non-finite")
    posterior_mean = _weighted_mean(mean, value, 1.0 / new_kappa)
    delta_log = _log_abs_difference(value, mean)
    weight = 0.5 * (kappa / new_kappa)
    if delta_log == -math.inf:
        increment = 0.0
    else:
        log_increment = math.log(weight) + 2.0 * delta_log
        increment = _MAX_FLOAT if log_increment >= math.log(_MAX_FLOAT) else math.exp(log_increment)
    new_beta = beta + increment
    if not math.isfinite(new_beta):
        new_beta = _MAX_FLOAT
    new_alpha = alpha + 0.5
    if not math.isfinite(new_alpha):
        raise BayesianChangePointError("posterior alpha became non-finite")
    return posterior_mean, new_kappa, new_alpha, new_beta


@dataclass(frozen=True)
class _RunState:
    run_length: int
    log_probability: float
    mean: float
    kappa: float
    alpha: float
    beta: float


def run_bocpd(
    observations: Iterable[CausalObservation | Mapping[str, Any]],
    *,
    config: BOCPDConfig | None = None,
    cutoff_timestamp: int | None = None,
) -> BOCPDResult:
    """Run Normal-Inverse-Gamma BOCPD in causal known-time order.

    The current observation is scored against the previous posterior before it
    updates any run state. A row after the cutoff is rejected rather than
    silently discarded. No network, provider, broker, or execution path is
    reachable from this function.
    """

    model = config or BOCPDConfig()
    normalized = _normalize_observations(observations, cutoff_timestamp=cutoff_timestamp)
    states: tuple[_RunState, ...] = (
        _RunState(
            run_length=0,
            log_probability=0.0,
            mean=model.prior_mean,
            kappa=model.prior_kappa,
            alpha=model.prior_alpha,
            beta=model.prior_beta,
        ),
    )
    log_hazard = math.log(model.hazard)
    log_survival = -math.inf if model.hazard == 1.0 else math.log1p(-model.hazard)
    steps: list[BOCPDStep] = []
    for observation in normalized:
        value = observation.value
        prior_predictive = _student_t_logpdf(
            value,
            model.prior_mean,
            model.prior_kappa,
            model.prior_alpha,
            model.prior_beta,
        )
        growth_masses: list[tuple[_RunState, float]] = []
        cp_terms: list[float] = []
        predictive_terms: list[float] = []
        for state in states:
            predictive = _student_t_logpdf(value, state.mean, state.kappa, state.alpha, state.beta)
            growth_log_mass = state.log_probability + log_survival + predictive
            cp_log_mass = state.log_probability + log_hazard + prior_predictive
            predictive_terms.extend((growth_log_mass, cp_log_mass))
            if growth_log_mass != -math.inf:
                if model.max_run_length is not None and state.run_length >= model.max_run_length:
                    raise BayesianChangePointError(
                        "max_run_length would be exceeded; increase the explicit safety bound"
                    )
                updated_mean, updated_kappa, updated_alpha, updated_beta = _posterior_update(
                    state.mean, state.kappa, state.alpha, state.beta, value
                )
                growth_masses.append(
                    (
                        _RunState(
                            run_length=state.run_length + 1,
                            log_probability=growth_log_mass,
                            mean=updated_mean,
                            kappa=updated_kappa,
                            alpha=updated_alpha,
                            beta=updated_beta,
                        ),
                        growth_log_mass,
                    )
                )
            cp_terms.append(cp_log_mass)
        cp_log_mass = _logsumexp(cp_terms)
        cp_mean, cp_kappa, cp_alpha, cp_beta = _posterior_update(
            model.prior_mean, model.prior_kappa, model.prior_alpha, model.prior_beta, value
        )
        candidates = [
            _RunState(
                run_length=0,
                log_probability=cp_log_mass,
                mean=cp_mean,
                kappa=cp_kappa,
                alpha=cp_alpha,
                beta=cp_beta,
            ),
            *(state for state, _ in growth_masses),
        ]
        normalization = _logsumexp(state.log_probability for state in candidates)
        posterior_states: list[_RunState] = []
        for state in candidates:
            posterior_probability = math.exp(state.log_probability - normalization)
            if not math.isfinite(posterior_probability):
                raise BayesianChangePointError("posterior probability became non-finite")
            posterior_states.append(
                _RunState(
                    run_length=state.run_length,
                    log_probability=(
                        math.log(posterior_probability) if posterior_probability > 0 else -math.inf
                    ),
                    mean=state.mean,
                    kappa=state.kappa,
                    alpha=state.alpha,
                    beta=state.beta,
                )
            )
        posterior_states.sort(key=lambda state: state.run_length)
        probabilities = [math.exp(state.log_probability) for state in posterior_states]
        total = math.fsum(probabilities)
        if not math.isfinite(total) or total <= 0:
            raise BayesianChangePointError("posterior normalization became non-finite")
        probabilities = [probability / total for probability in probabilities]
        posterior = tuple(
            RunLengthProbability(run_length=state.run_length, probability=probability)
            for state, probability in zip(posterior_states, probabilities, strict=True)
        )
        change_probability = posterior[0].probability
        map_item = max(posterior, key=lambda item: (item.probability, -item.run_length))
        predictive_log_score = _logsumexp(predictive_terms)
        steps.append(
            BOCPDStep(
                event_time=observation.event_time,
                known_at=observation.known_at,
                sequence=observation.sequence,
                observed_value=observation.value,
                predictive_log_score=predictive_log_score,
                change_probability=change_probability,
                map_run_length=map_item.run_length,
                run_length_posterior=posterior,
            )
        )
        states = tuple(posterior_states)
    return BOCPDResult(
        steps=tuple(steps),
        config=model,
        cutoff_timestamp=cutoff_timestamp,
        observation_sha256=_sha256([item.as_dict() for item in normalized]),
    )


__all__ = [
    "BAYESIAN_CHANGE_SCHEMA",
    "BOCPDConfig",
    "BOCPDResult",
    "BOCPDStep",
    "BayesianChangePointError",
    "RunLengthProbability",
    "run_bocpd",
]
