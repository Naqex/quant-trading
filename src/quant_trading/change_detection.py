"""Deterministic, causal CUSUM change detection for offline research.

This module is intentionally narrower than a trading signal.  It evaluates an
already captured numeric stream in known-time order and reports threshold
crossings relative to an explicit baseline.  It does not fetch data, infer a
baseline, fit a model, call a provider, or grant execution authority.

The detector uses a two-sided Page-style cumulative sum (CUSUM):

``S+_t = max(0, S+_(t-1) + (x_t - baseline) - drift)``
``S-_t = min(0, S-_(t-1) + (x_t - baseline) + drift)``

An observation is flagged when either statistic reaches ``threshold`` in
magnitude.  A flag is diagnostic evidence of a distributional shift under the
configured assumptions; it is not a buy/sell recommendation.  By default the
accumulators reset after a crossing so one persistent shift does not produce a
flag on every subsequent row.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .state_primitives import CausalObservation, StatePrimitiveError

CUSUM_CHANGE_SCHEMA = "cusum-change-detection-v1"
ChangeDirection = Literal["none", "positive", "negative", "both"]


class ChangeDetectionError(ValueError):
    """Raised when the causal CUSUM contract is violated."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ChangeDetectionError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ChangeDetectionError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ChangeDetectionError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ChangeDetectionError(f"{name} must be finite")
    return parsed


def _nonnegative(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0:
        raise ChangeDetectionError(f"{name} must be >= 0")
    return parsed


def _positive(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0:
        raise ChangeDetectionError(f"{name} must be > 0")
    return parsed


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ChangeDetectionError("value must be finite JSON data") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CUSUMConfig:
    """Explicit CUSUM assumptions.

    ``baseline`` and ``threshold`` are required to make the detector
    reproducible.  The implementation never estimates either value from the
    observations, which prevents an accidental look-ahead through an inferred
    baseline.  ``drift`` is a non-negative allowance for small deviations.
    """

    baseline: float
    threshold: float
    drift: float = 0.0
    reset_on_detection: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline", _finite(self.baseline, "baseline"))
        object.__setattr__(self, "threshold", _positive(self.threshold, "threshold"))
        object.__setattr__(self, "drift", _nonnegative(self.drift, "drift"))
        if type(self.reset_on_detection) is not bool:
            raise ChangeDetectionError("reset_on_detection must be a boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "threshold": self.threshold,
            "drift": self.drift,
            "reset_on_detection": self.reset_on_detection,
        }


@dataclass(frozen=True)
class CUSUMStep:
    """One forward-only diagnostic update.

    ``positive_stat`` and ``negative_stat`` are the values immediately before
    an optional reset.  This preserves the evidence that crossed the threshold
    even when the next step starts from zero.
    """

    event_time: int
    known_at: int
    sequence: int
    value: float
    deviation: float
    positive_stat: float
    negative_stat: float
    detected: bool
    direction: ChangeDirection

    def __post_init__(self) -> None:
        _strict_int(self.event_time, "event_time")
        _strict_int(self.known_at, "known_at")
        _strict_int(self.sequence, "sequence")
        if self.known_at < self.event_time:
            raise ChangeDetectionError("known_at cannot precede event_time")
        for name in ("value", "deviation", "positive_stat", "negative_stat"):
            parsed = _finite(getattr(self, name), name)
            if name == "positive_stat" and parsed < 0:
                raise ChangeDetectionError("positive_stat must be >= 0")
            if name == "negative_stat" and parsed > 0:
                raise ChangeDetectionError("negative_stat must be <= 0")
            object.__setattr__(self, name, parsed)
        if type(self.detected) is not bool:
            raise ChangeDetectionError("detected must be a boolean")
        if self.direction not in {"none", "positive", "negative", "both"}:
            raise ChangeDetectionError("unsupported change direction")
        if self.detected is False and self.direction != "none":
            raise ChangeDetectionError("undetected step must have direction none")
        if self.detected is True and self.direction == "none":
            raise ChangeDetectionError("detected step must have a direction")

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "known_at": self.known_at,
            "sequence": self.sequence,
            "value": self.value,
            "deviation": self.deviation,
            "positive_stat": self.positive_stat,
            "negative_stat": self.negative_stat,
            "detected": self.detected,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class CUSUMResult:
    """Immutable offline CUSUM output with provenance and deny-only flags."""

    steps: tuple[CUSUMStep, ...]
    config: CUSUMConfig
    cutoff_timestamp: int | None
    observation_sha256: str
    schema: str = CUSUM_CHANGE_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != CUSUM_CHANGE_SCHEMA:
            raise ChangeDetectionError(f"unsupported CUSUM schema: {self.schema!r}")
        if not isinstance(self.steps, tuple):
            raise ChangeDetectionError("steps must be a tuple")
        if not isinstance(self.config, CUSUMConfig):
            raise ChangeDetectionError("config must be CUSUMConfig")
        if self.cutoff_timestamp is not None:
            _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        if type(self.execution_capability) is not bool or self.execution_capability:
            raise ChangeDetectionError("execution_capability must remain false")
        if type(self.provider_access) is not bool or self.provider_access:
            raise ChangeDetectionError("provider_access must remain false")
        if type(self.broker_access) is not bool or self.broker_access:
            raise ChangeDetectionError("broker_access must remain false")
        if (
            not isinstance(self.observation_sha256, str)
            or len(self.observation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.observation_sha256)
        ):
            raise ChangeDetectionError("observation_sha256 must be a SHA-256 hex digest")
        previous_key: tuple[int, int, int] | None = None
        for step in self.steps:
            if not isinstance(step, CUSUMStep):
                raise ChangeDetectionError("steps must contain CUSUMStep values")
            if self.cutoff_timestamp is not None and step.known_at > self.cutoff_timestamp:
                raise ChangeDetectionError("step known_at exceeds cutoff_timestamp")
            key = (step.known_at, step.sequence, step.event_time)
            if previous_key is not None and key < previous_key:
                raise ChangeDetectionError("steps must be in known-time order")
            previous_key = key

    @property
    def detections(self) -> tuple[CUSUMStep, ...]:
        return tuple(step for step in self.steps if step.detected)

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
            "detection_count": len(self.detections),
        }

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


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
        raise ChangeDetectionError("observations must be iterable") from exc
    normalized: list[CausalObservation] = []
    for item in iterator:
        try:
            observation = (
                item
                if isinstance(item, CausalObservation)
                else CausalObservation.from_mapping(item)
            )
        except (StatePrimitiveError, TypeError) as exc:
            raise ChangeDetectionError(str(exc)) from exc
        if cutoff_timestamp is not None and observation.known_at > cutoff_timestamp:
            raise ChangeDetectionError(
                "observation known_at exceeds cutoff_timestamp; future data is rejected"
            )
        normalized.append(observation)
    ordered = tuple(sorted(normalized, key=lambda item: (item.known_at, item.sequence, item.event_time)))
    seen: set[tuple[int, int]] = set()
    for item in ordered:
        key = (item.known_at, item.sequence)
        if key in seen:
            raise ChangeDetectionError("observations must have unique known_at/sequence keys")
        seen.add(key)
    return ordered


def run_cusum(
    observations: Iterable[CausalObservation | Mapping[str, Any]],
    *,
    config: CUSUMConfig,
    cutoff_timestamp: int | None = None,
) -> CUSUMResult:
    """Run a deterministic two-sided CUSUM over causal observations.

    Rows are processed in ``(known_at, sequence, event_time)`` order.  A row
    outside ``cutoff_timestamp`` is rejected, rather than silently removed, so
    holdout leakage cannot hide in a report.  Empty input is valid and returns
    an empty diagnostic result with a stable hash.
    """

    if not isinstance(config, CUSUMConfig):
        raise ChangeDetectionError("config must be CUSUMConfig")
    normalized = _normalize_observations(observations, cutoff_timestamp=cutoff_timestamp)
    positive = 0.0
    negative = 0.0
    steps: list[CUSUMStep] = []
    for observation in normalized:
        deviation = observation.value - config.baseline
        positive = max(0.0, positive + deviation - config.drift)
        negative = min(0.0, negative + deviation + config.drift)
        positive_detected = positive >= config.threshold
        negative_detected = -negative >= config.threshold
        detected = positive_detected or negative_detected
        if positive_detected and negative_detected:
            direction: ChangeDirection = "both"
        elif positive_detected:
            direction = "positive"
        elif negative_detected:
            direction = "negative"
        else:
            direction = "none"
        if not all(math.isfinite(value) for value in (deviation, positive, negative)):
            raise ChangeDetectionError("CUSUM state became non-finite")
        steps.append(
            CUSUMStep(
                event_time=observation.event_time,
                known_at=observation.known_at,
                sequence=observation.sequence,
                value=observation.value,
                deviation=deviation,
                positive_stat=positive,
                negative_stat=negative,
                detected=detected,
                direction=direction,
            )
        )
        if detected and config.reset_on_detection:
            positive = 0.0
            negative = 0.0
    return CUSUMResult(
        steps=tuple(steps),
        config=config,
        cutoff_timestamp=cutoff_timestamp,
        observation_sha256=_canonical_sha256([item.as_dict() for item in normalized]),
    )


__all__ = [
    "CUSUM_CHANGE_SCHEMA",
    "CUSUMConfig",
    "CUSUMResult",
    "CUSUMStep",
    "ChangeDetectionError",
    "run_cusum",
]
