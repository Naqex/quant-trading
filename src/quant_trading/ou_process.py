"""Offline Ornstein-Uhlenbeck diagnostics for captured spread/residual series.

This module intentionally models a *stationary spread or residual*, never a
raw price.  It fits the exact irregular-time OU transition with a deterministic
profile likelihood and can produce a forward-only conditional diagnostic.  No
provider, broker, wallet, model service or execution path is reachable here.

The implementation is a research primitive, not a trading signal.  A fit is
rejected when the profile likelihood lands on the near-random-walk boundary or
when the estimated stationary variance is not numerically meaningful.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

OU_SCHEMA = "ou-spread-residual-v1"
OU_FILTER_SCHEMA = "ou-forward-filter-v1"
OU_SERIES_KINDS = frozenset({"spread", "residual"})


class OUProcessError(ValueError):
    """Raised when an OU input, fit or causal filter is invalid."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OUProcessError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise OUProcessError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OUProcessError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise OUProcessError(f"{name} must be finite")
    return parsed


def _positive(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0:
        raise OUProcessError(f"{name} must be > 0")
    return parsed


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OUProcessError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise OUProcessError(f"{name} must be at most {maximum} characters")
    return normalized


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OUProcessError("value must be finite JSON data") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _time_deltas(observations: tuple[OUObservation, ...]) -> list[float]:
    deltas: list[float] = []
    for left, right in pairwise(observations):
        try:
            delta = float(right.event_time - left.event_time)
        except OverflowError as exc:
            raise OUProcessError("event_time gap must be representable as a finite float") from exc
        if not math.isfinite(delta) or delta <= 0:
            raise OUProcessError("event_time gaps must be finite and > 0")
        deltas.append(delta)
    return deltas


@dataclass(frozen=True)
class OUObservation:
    """One causal observation from a stationary spread or residual series."""

    event_time: int
    known_at: int
    value: float
    series_kind: str
    sequence: int = 0

    def __post_init__(self) -> None:
        event_time = _strict_int(self.event_time, "event_time")
        known_at = _strict_int(self.known_at, "known_at")
        if known_at < event_time:
            raise OUProcessError("known_at cannot precede event_time")
        value = _finite(self.value, "value")
        series_kind = _text(self.series_kind, "series_kind", maximum=32).lower()
        if series_kind not in OU_SERIES_KINDS:
            raise OUProcessError("series_kind must be 'spread' or 'residual'; raw prices are unsupported")
        sequence = _strict_int(self.sequence, "sequence")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "series_kind", series_kind)
        object.__setattr__(self, "sequence", sequence)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> OUObservation:
        if not isinstance(payload, Mapping):
            raise OUProcessError("OU observation must be a mapping")
        try:
            return cls(
                event_time=payload["event_time"],
                known_at=payload["known_at"],
                value=payload["value"],
                series_kind=payload["series_kind"],
                sequence=payload.get("sequence", 0),
            )
        except KeyError as exc:
            raise OUProcessError(f"OU observation is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "value": self.value,
            "series_kind": self.series_kind,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class OUConfig:
    """Deterministic fitting and stationarity guard parameters."""

    min_observations: int = 12
    min_kappa: float = 1e-6
    max_kappa: float = 10.0
    stationarity_tolerance: float = 1e-6
    min_stationary_variance: float = 1e-14
    grid_points: int = 96
    refinement_steps: int = 32

    def __post_init__(self) -> None:
        min_observations = _strict_int(self.min_observations, "min_observations", minimum=3)
        min_kappa = _positive(self.min_kappa, "min_kappa")
        max_kappa = _positive(self.max_kappa, "max_kappa")
        if max_kappa <= min_kappa:
            raise OUProcessError("max_kappa must be greater than min_kappa")
        stationarity_tolerance = _finite(self.stationarity_tolerance, "stationarity_tolerance")
        if not 0 < stationarity_tolerance < 1:
            raise OUProcessError("stationarity_tolerance must be between 0 and 1")
        min_stationary_variance = _positive(self.min_stationary_variance, "min_stationary_variance")
        grid_points = _strict_int(self.grid_points, "grid_points", minimum=16)
        refinement_steps = _strict_int(self.refinement_steps, "refinement_steps", minimum=1)
        object.__setattr__(self, "min_observations", min_observations)
        object.__setattr__(self, "min_kappa", min_kappa)
        object.__setattr__(self, "max_kappa", max_kappa)
        object.__setattr__(self, "stationarity_tolerance", stationarity_tolerance)
        object.__setattr__(self, "min_stationary_variance", min_stationary_variance)
        object.__setattr__(self, "grid_points", grid_points)
        object.__setattr__(self, "refinement_steps", refinement_steps)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "min_observations": self.min_observations,
            "min_kappa": self.min_kappa,
            "max_kappa": self.max_kappa,
            "stationarity_tolerance": self.stationarity_tolerance,
            "min_stationary_variance": self.min_stationary_variance,
            "grid_points": self.grid_points,
            "refinement_steps": self.refinement_steps,
        }


@dataclass(frozen=True)
class OUFit:
    """A stationary, provider-free OU fit with point-in-time provenance."""

    series_kind: str
    mean_level: float
    kappa: float
    diffusion_variance_rate: float
    stationary_variance: float
    half_life: float
    log_likelihood: float
    observation_count: int
    min_dt: float
    max_dt: float
    cutoff_timestamp: int | None
    observation_sha256: str
    config: OUConfig
    schema: str = OU_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != OU_SCHEMA:
            raise OUProcessError(f"unsupported OU schema: {self.schema!r}")
        if self.series_kind not in OU_SERIES_KINDS:
            raise OUProcessError("OU fit must identify a spread or residual series")
        for name in (
            "mean_level",
            "kappa",
            "diffusion_variance_rate",
            "stationary_variance",
            "half_life",
            "log_likelihood",
            "min_dt",
            "max_dt",
        ):
            _finite(getattr(self, name), name)
        if self.kappa <= 0 or self.diffusion_variance_rate <= 0 or self.stationary_variance <= 0:
            raise OUProcessError("OU fit parameters must be strictly positive")
        if self.half_life <= 0 or self.min_dt <= 0 or self.max_dt < self.min_dt:
            raise OUProcessError("OU fit time parameters are invalid")
        _strict_int(self.observation_count, "observation_count", minimum=3)
        if self.cutoff_timestamp is not None:
            _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        if not isinstance(self.config, OUConfig):
            raise OUProcessError("config must be OUConfig")
        if not isinstance(self.observation_sha256, str) or len(self.observation_sha256) != 64:
            raise OUProcessError("observation_sha256 must be a SHA-256 hex digest")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise OUProcessError(f"{name} must remain false for offline OU research")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
            "series_kind": self.series_kind,
            "mean_level": self.mean_level,
            "kappa": self.kappa,
            "diffusion_variance_rate": self.diffusion_variance_rate,
            "stationary_variance": self.stationary_variance,
            "half_life": self.half_life,
            "log_likelihood": self.log_likelihood,
            "observation_count": self.observation_count,
            "min_dt": self.min_dt,
            "max_dt": self.max_dt,
            "cutoff_timestamp": self.cutoff_timestamp,
            "observation_sha256": self.observation_sha256,
            "config": self.config.as_dict(),
        }

    def canonical_sha256(self) -> str:
        return _sha256(self.as_dict())


@dataclass(frozen=True)
class OUEstimate:
    """One forward-only conditional OU diagnostic."""

    event_time: int
    known_at: int
    dt: float
    observed_value: float
    conditional_mean: float
    conditional_variance: float
    innovation: float
    standardized_innovation: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "dt": self.dt,
            "observed_value": self.observed_value,
            "conditional_mean": self.conditional_mean,
            "conditional_variance": self.conditional_variance,
            "innovation": self.innovation,
            "standardized_innovation": self.standardized_innovation,
        }


@dataclass(frozen=True)
class OUFilterResult:
    estimates: tuple[OUEstimate, ...]
    fit_sha256: str
    cutoff_timestamp: int | None
    observation_sha256: str
    schema: str = OU_FILTER_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != OU_FILTER_SCHEMA:
            raise OUProcessError(f"unsupported OU filter schema: {self.schema!r}")
        if not isinstance(self.estimates, tuple):
            raise OUProcessError("estimates must be a tuple")
        for name in ("fit_sha256", "observation_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise OUProcessError(f"{name} must be a SHA-256 hex digest")
        if self.cutoff_timestamp is not None:
            _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise OUProcessError(f"{name} must remain false for offline OU research")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
            "fit_sha256": self.fit_sha256,
            "cutoff_timestamp": self.cutoff_timestamp,
            "observation_sha256": self.observation_sha256,
            "estimates": [item.as_dict() for item in self.estimates],
        }

    def canonical_sha256(self) -> str:
        return _sha256(self.as_dict())


def _normalize_observations(
    observations: Iterable[OUObservation | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None,
) -> tuple[OUObservation, ...]:
    cutoff = None if cutoff_timestamp is None else _strict_int(cutoff_timestamp, "cutoff_timestamp")
    normalized = tuple(
        item if isinstance(item, OUObservation) else OUObservation.from_mapping(item)
        for item in observations
    )
    if not normalized:
        raise OUProcessError("at least one OU observation is required")
    keys = [(item.known_at, item.sequence) for item in normalized]
    if len(keys) != len(set(keys)):
        raise OUProcessError("OU observations must have unique known_at/sequence keys")
    if cutoff is not None and any(item.known_at > cutoff for item in normalized):
        raise OUProcessError("OU observation exceeds research cutoff")
    series_kinds = {item.series_kind for item in normalized}
    if len(series_kinds) != 1:
        raise OUProcessError("OU observations must belong to one series_kind")
    ordered = tuple(sorted(normalized, key=lambda item: (item.event_time, item.known_at, item.sequence)))
    if any(
        left.event_time >= right.event_time
        for left, right in pairwise(ordered)
    ):
        raise OUProcessError("OU event_time values must be strictly increasing")
    return ordered


def _profile_at_kappa(
    kappa: float, values: list[float], dts: list[float]
) -> tuple[float, float, float] | None:
    """Return (mean, diffusion variance rate, log likelihood) at one kappa."""

    weighted_c2 = 0.0
    weighted_cy = 0.0
    phis: list[float] = []
    q_bases: list[float] = []
    ys: list[float] = []
    for (previous, current), dt in zip(pairwise(values), dts, strict=True):
        exponent = -2.0 * kappa * dt
        q_base = (-math.expm1(exponent)) / (2.0 * kappa) if math.isfinite(exponent) else 1.0 / (2.0 * kappa)
        if not math.isfinite(q_base) or q_base <= 0:
            return None
        phi = math.exp(-kappa * dt) if kappa * dt < 745.0 else 0.0
        c = 1.0 - phi
        y = current - phi * previous
        weight = 1.0 / q_base
        weighted_c2 += weight * c * c
        weighted_cy += weight * c * y
        phis.append(phi)
        q_bases.append(q_base)
        ys.append(y)
    if not math.isfinite(weighted_c2) or weighted_c2 <= 1e-15:
        return None
    mean_level = weighted_cy / weighted_c2
    if not math.isfinite(mean_level):
        return None
    weighted_sse = 0.0
    residuals: list[float] = []
    for y, phi, q_base in zip(ys, phis, q_bases, strict=True):
        residual = y - mean_level * (1.0 - phi)
        residuals.append(residual)
        weighted_sse += residual * residual / q_base
    count = len(residuals)
    diffusion = weighted_sse / count
    if not math.isfinite(diffusion) or diffusion <= 0:
        return None
    log_likelihood = -0.5 * sum(
        math.log(2.0 * math.pi * diffusion * q_base) + residual * residual / (diffusion * q_base)
        for residual, q_base in zip(residuals, q_bases, strict=True)
    )
    if not math.isfinite(log_likelihood):
        return None
    return mean_level, diffusion, log_likelihood


def fit_ou_process(
    observations: Iterable[OUObservation | Mapping[str, Any]],
    *,
    config: OUConfig | None = None,
    cutoff_timestamp: int | None = None,
) -> OUFit:
    """Fit an exact irregular-time OU model to a causal spread/residual series.

    The one-dimensional profile search is deterministic: a log-spaced grid is
    followed by bounded golden-section refinement.  The near-zero kappa edge is
    rejected so a random walk cannot be silently labeled mean reverting.
    """

    model = config or OUConfig()
    normalized = _normalize_observations(observations, cutoff_timestamp=cutoff_timestamp)
    if len(normalized) < model.min_observations:
        raise OUProcessError(f"at least {model.min_observations} observations are required for an OU fit")
    raw_values = [item.value for item in normalized]
    scale = max(1.0, max(abs(value) for value in raw_values))
    values = [value / scale for value in raw_values]
    dts = _time_deltas(normalized)
    min_dt, max_dt = min(dts), max(dts)
    log_min = math.log(model.min_kappa)
    log_max = math.log(model.max_kappa)
    grid_logs = [
        log_min + (log_max - log_min) * i / (model.grid_points - 1)
        for i in range(model.grid_points)
    ]
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for log_kappa in grid_logs:
        profile = _profile_at_kappa(math.exp(log_kappa), values, dts)
        if profile is not None:
            candidates.append((log_kappa, profile))
    if not candidates:
        raise OUProcessError("OU profile likelihood is not finite for this series")
    best_index = max(range(len(candidates)), key=lambda index: candidates[index][1][2])
    best_log, best_profile = candidates[best_index]
    if best_index == 0 or best_index == len(candidates) - 1:
        raise OUProcessError("OU fit is on the kappa boundary; stationarity is not established")
    left_log = candidates[best_index - 1][0]
    right_log = candidates[best_index + 1][0]
    phi_golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right_log - phi_golden * (right_log - left_log)
    x2 = left_log + phi_golden * (right_log - left_log)
    p1 = _profile_at_kappa(math.exp(x1), values, dts)
    p2 = _profile_at_kappa(math.exp(x2), values, dts)
    for _ in range(model.refinement_steps):
        if p1 is None or (p2 is not None and p1[2] < p2[2]):
            left_log, x1, p1 = x1, x2, p2
            x2 = left_log + phi_golden * (right_log - left_log)
            p2 = _profile_at_kappa(math.exp(x2), values, dts)
        else:
            right_log, x2, p2 = x2, x1, p1
            x1 = right_log - phi_golden * (right_log - left_log)
            p1 = _profile_at_kappa(math.exp(x1), values, dts)
    refined = [
        candidate
        for candidate in ((x1, p1), (x2, p2), (best_log, best_profile))
        if candidate[1] is not None
    ]
    best_log, best_profile = max(refined, key=lambda item: item[1][2])
    kappa = math.exp(best_log)
    mean_level_norm, diffusion_norm, log_likelihood = best_profile
    mean_level = mean_level_norm * scale
    diffusion = diffusion_norm * scale * scale
    if not all(math.isfinite(value) for value in (mean_level, diffusion, log_likelihood)) or diffusion <= 0:
        raise OUProcessError("OU fit became non-finite after restoring data scale")
    stationary_variance = diffusion / (2.0 * kappa)
    half_life = math.log(2.0) / kappa
    if not math.isfinite(stationary_variance) or stationary_variance < model.min_stationary_variance:
        raise OUProcessError("OU stationary variance is too small or non-finite")
    # At the smallest observed interval, phi must be materially below one.
    max_phi = math.exp(-kappa * min_dt) if kappa * min_dt < 745.0 else 0.0
    if max_phi >= 1.0 - model.stationarity_tolerance:
        raise OUProcessError("OU stationarity guard rejected near-unit transition")
    return OUFit(
        series_kind=normalized[0].series_kind,
        mean_level=mean_level,
        kappa=kappa,
        diffusion_variance_rate=diffusion,
        stationary_variance=stationary_variance,
        half_life=half_life,
        log_likelihood=log_likelihood,
        observation_count=len(normalized),
        min_dt=min_dt,
        max_dt=max_dt,
        cutoff_timestamp=cutoff_timestamp,
        observation_sha256=_sha256([item.as_dict() for item in normalized]),
        config=model,
    )


def filter_ou_process(
    observations: Iterable[OUObservation | Mapping[str, Any]],
    *,
    fit: OUFit,
    cutoff_timestamp: int | None = None,
) -> OUFilterResult:
    """Produce forward-only conditional means, innovations and z-scores."""

    normalized = _normalize_observations(observations, cutoff_timestamp=cutoff_timestamp)
    if any(item.series_kind != fit.series_kind for item in normalized):
        raise OUProcessError("filter observations must match the OU fit series_kind")
    if fit.cutoff_timestamp is not None and any(item.known_at > fit.cutoff_timestamp for item in normalized):
        raise OUProcessError("filter observation exceeds fit cutoff")
    estimates: list[OUEstimate] = []
    previous: OUObservation | None = None
    for observation in normalized:
        if previous is None:
            dt = 0.0
            conditional_mean = fit.mean_level
            conditional_variance = fit.stationary_variance
        else:
            try:
                dt = float(observation.event_time - previous.event_time)
            except OverflowError as exc:
                raise OUProcessError("event_time gap must be representable as a finite float") from exc
            if not math.isfinite(dt) or dt <= 0:
                raise OUProcessError("event_time gaps must be finite and > 0")
            phi = math.exp(-fit.kappa * dt) if fit.kappa * dt < 745.0 else 0.0
            conditional_mean = fit.mean_level + phi * (previous.value - fit.mean_level)
            conditional_variance = (
                fit.diffusion_variance_rate
                * (-math.expm1(-2.0 * fit.kappa * dt))
                / (2.0 * fit.kappa)
            )
        if (
            not math.isfinite(conditional_mean)
            or not math.isfinite(conditional_variance)
            or conditional_variance <= 0
        ):
            raise OUProcessError("OU conditional state became non-finite or non-positive")
        innovation = observation.value - conditional_mean
        standardized = innovation / math.sqrt(conditional_variance)
        if not math.isfinite(innovation) or not math.isfinite(standardized):
            raise OUProcessError("OU innovation became non-finite")
        estimates.append(
            OUEstimate(
                event_time=observation.event_time,
                known_at=observation.known_at,
                dt=dt,
                observed_value=observation.value,
                conditional_mean=conditional_mean,
                conditional_variance=conditional_variance,
                innovation=innovation,
                standardized_innovation=standardized,
            )
        )
        previous = observation
    return OUFilterResult(
        estimates=tuple(estimates),
        fit_sha256=fit.canonical_sha256(),
        cutoff_timestamp=cutoff_timestamp,
        observation_sha256=_sha256([item.as_dict() for item in normalized]),
    )


__all__ = [
    "OUConfig",
    "OUEstimate",
    "OUFilterResult",
    "OUFit",
    "OUObservation",
    "OUProcessError",
    "OU_FILTER_SCHEMA",
    "OU_SCHEMA",
    "OU_SERIES_KINDS",
    "filter_ou_process",
    "fit_ou_process",
]
