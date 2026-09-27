"""Causal population-stability diagnostics for offline research.

Population Stability Index (PSI) is useful as a small, auditable drift/OOD
screen for Adaptive Markets and model governance.  This module intentionally
requires bin edges fixed before the comparison: deriving bins from the current
sample would leak information into the diagnostic.  It does not label a
regime, make a forecast, or grant execution authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .state_primitives import CausalObservation, StatePrimitiveError

PSI_SCHEMA = "population-stability-index-v1"


class DistributionDriftError(ValueError):
    """Raised when a PSI contract is malformed or non-causal."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DistributionDriftError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DistributionDriftError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DistributionDriftError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise DistributionDriftError(f"{name} must be finite")
    return parsed


def _positive(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0.0:
        raise DistributionDriftError(f"{name} must be > 0")
    return parsed


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DistributionDriftError("value must be finite JSON data") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PSIConfig:
    """Fixed binning and smoothing assumptions for one PSI comparison.

    ``bin_edges`` must be strictly increasing.  Values outside the declared
    support are rejected instead of silently clipped into a tail bucket.
    ``epsilon`` is applied to every bin before normalization so empty bins do
    not produce non-finite logarithms.
    """

    bin_edges: Sequence[float]
    epsilon: float = 1e-6
    alert_threshold: float = 0.2

    def __post_init__(self) -> None:
        try:
            edges = tuple(_finite(value, "bin_edges") for value in self.bin_edges)
        except TypeError as exc:
            raise DistributionDriftError("bin_edges must be a finite sequence") from exc
        if len(edges) < 2:
            raise DistributionDriftError("bin_edges must contain at least two values")
        if any(right <= left for left, right in zip(edges[:-1], edges[1:], strict=True)):
            raise DistributionDriftError("bin_edges must be strictly increasing")
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "epsilon", _positive(self.epsilon, "epsilon"))
        object.__setattr__(self, "alert_threshold", _positive(self.alert_threshold, "alert_threshold"))

    @property
    def bin_count(self) -> int:
        return len(self.bin_edges) - 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "bin_edges": list(self.bin_edges),
            "epsilon": self.epsilon,
            "alert_threshold": self.alert_threshold,
        }


@dataclass(frozen=True)
class PSIBin:
    """One bin's explicit counts, smoothed probabilities and contribution."""

    lower: float
    upper: float
    reference_count: int
    current_count: int
    reference_probability: float
    current_probability: float
    contribution: float

    def __post_init__(self) -> None:
        lower = _finite(self.lower, "lower")
        upper = _finite(self.upper, "upper")
        if upper <= lower:
            raise DistributionDriftError("PSI bin upper must be greater than lower")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        _strict_int(self.reference_count, "reference_count", minimum=0)
        _strict_int(self.current_count, "current_count", minimum=0)
        for name in ("reference_probability", "current_probability"):
            probability = _finite(getattr(self, name), name)
            if not 0.0 <= probability <= 1.0:
                raise DistributionDriftError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, probability)
        contribution = _finite(self.contribution, "contribution")
        if contribution < 0.0:
            raise DistributionDriftError("contribution must be >= 0")
        object.__setattr__(self, "contribution", contribution)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "reference_probability": self.reference_probability,
            "current_probability": self.current_probability,
            "contribution": self.contribution,
        }


@dataclass(frozen=True)
class PSIResult:
    """Immutable PSI receipt with deny-only capability flags."""

    bins: tuple[PSIBin, ...]
    config: PSIConfig
    reference_sample_count: int
    current_sample_count: int
    psi: float
    drift_flag: bool
    cutoff_timestamp: int | None
    reference_sha256: str
    current_sha256: str
    schema: str = PSI_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != PSI_SCHEMA:
            raise DistributionDriftError(f"unsupported PSI schema: {self.schema!r}")
        if not isinstance(self.config, PSIConfig):
            raise DistributionDriftError("config must be PSIConfig")
        if not isinstance(self.bins, tuple) or len(self.bins) != self.config.bin_count:
            raise DistributionDriftError("bins must match config.bin_edges")
        if any(not isinstance(item, PSIBin) for item in self.bins):
            raise DistributionDriftError("bins must contain PSIBin values")
        _strict_int(self.reference_sample_count, "reference_sample_count", minimum=1)
        _strict_int(self.current_sample_count, "current_sample_count", minimum=1)
        if not math.isfinite(self.psi) or self.psi < 0.0:
            raise DistributionDriftError("psi must be finite and >= 0")
        if type(self.drift_flag) is not bool:
            raise DistributionDriftError("drift_flag must be a boolean")
        if self.drift_flag != (self.psi >= self.config.alert_threshold):
            raise DistributionDriftError("drift_flag does not match psi and alert_threshold")
        if sum(item.reference_count for item in self.bins) != self.reference_sample_count:
            raise DistributionDriftError("reference bin counts do not match sample count")
        if sum(item.current_count for item in self.bins) != self.current_sample_count:
            raise DistributionDriftError("current bin counts do not match sample count")
        if not math.isclose(
            sum(item.reference_probability for item in self.bins), 1.0, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise DistributionDriftError("reference probabilities must sum to one")
        if not math.isclose(
            sum(item.current_probability for item in self.bins), 1.0, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise DistributionDriftError("current probabilities must sum to one")
        if not math.isclose(
            sum(item.contribution for item in self.bins), self.psi, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise DistributionDriftError("bin contributions do not sum to psi")
        if self.cutoff_timestamp is not None:
            _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        for name in ("reference_sha256", "current_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise DistributionDriftError(f"{name} must be a SHA-256 hex digest")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise DistributionDriftError(f"{name} must remain false for offline diagnostics")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
            "cutoff_timestamp": self.cutoff_timestamp,
            "config": self.config.as_dict(),
            "reference_sample_count": self.reference_sample_count,
            "current_sample_count": self.current_sample_count,
            "reference_sha256": self.reference_sha256,
            "current_sha256": self.current_sha256,
            "bins": [item.as_dict() for item in self.bins],
            "psi": self.psi,
            "drift_flag": self.drift_flag,
        }

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


def _normalize(
    observations: Iterable[CausalObservation | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None,
    name: str,
) -> tuple[CausalObservation, ...]:
    if cutoff_timestamp is not None:
        _strict_int(cutoff_timestamp, "cutoff_timestamp")
    try:
        normalized = tuple(
            item if isinstance(item, CausalObservation) else CausalObservation.from_mapping(item)
            for item in observations
        )
    except (StatePrimitiveError, TypeError) as exc:
        raise DistributionDriftError(f"{name} contains an invalid observation") from exc
    ordered = tuple(sorted(normalized, key=lambda item: (item.known_at, item.sequence, item.event_time)))
    seen: set[tuple[int, int]] = set()
    for item in ordered:
        if cutoff_timestamp is not None and item.known_at > cutoff_timestamp:
            raise DistributionDriftError(f"{name} observation exceeds cutoff_timestamp")
        key = (item.known_at, item.sequence)
        if key in seen:
            raise DistributionDriftError(f"{name} observations must have unique known_at/sequence keys")
        seen.add(key)
    if not ordered:
        raise DistributionDriftError(f"{name} requires at least one observation")
    return ordered


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    if value < edges[0] or value > edges[-1]:
        raise DistributionDriftError("observation value falls outside declared bin support")
    if value == edges[-1]:
        return len(edges) - 2
    for index, upper in enumerate(edges[1:]):
        if value < upper:
            return index
    raise DistributionDriftError("observation could not be assigned to a bin")


def _counts(observations: tuple[CausalObservation, ...], edges: tuple[float, ...]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for observation in observations:
        counts[_bin_index(observation.value, edges)] += 1
    return counts


def compute_psi(
    reference: Iterable[CausalObservation | Mapping[str, Any]],
    current: Iterable[CausalObservation | Mapping[str, Any]],
    *,
    config: PSIConfig,
    cutoff_timestamp: int | None = None,
) -> PSIResult:
    """Compare two point-in-time samples using fixed-bin PSI.

    Inputs are sorted by known time for stable hashes, and rows beyond the
    cutoff are rejected.  The reference and current bins are smoothed with
    the configured epsilon; no thresholds or edges are inferred from either
    sample.
    """

    if not isinstance(config, PSIConfig):
        raise DistributionDriftError("config must be PSIConfig")
    reference_rows = _normalize(reference, cutoff_timestamp=cutoff_timestamp, name="reference")
    current_rows = _normalize(current, cutoff_timestamp=cutoff_timestamp, name="current")
    reference_counts = _counts(reference_rows, config.bin_edges)
    current_counts = _counts(current_rows, config.bin_edges)
    bin_count = config.bin_count
    epsilon = config.epsilon
    ref_total = float(sum(reference_counts)) + epsilon * bin_count
    cur_total = float(sum(current_counts)) + epsilon * bin_count
    reference_probabilities = tuple((count + epsilon) / ref_total for count in reference_counts)
    current_probabilities = tuple((count + epsilon) / cur_total for count in current_counts)
    contributions = tuple(
        (current_probability - reference_probability)
        * math.log(current_probability / reference_probability)
        for reference_probability, current_probability in zip(
            reference_probabilities, current_probabilities, strict=True
        )
    )
    psi = sum(contributions)
    if not math.isfinite(psi) or psi < 0.0:
        raise DistributionDriftError("PSI became non-finite or negative")
    bins = tuple(
        PSIBin(
            lower=config.bin_edges[index],
            upper=config.bin_edges[index + 1],
            reference_count=reference_counts[index],
            current_count=current_counts[index],
            reference_probability=reference_probabilities[index],
            current_probability=current_probabilities[index],
            contribution=contributions[index],
        )
        for index in range(bin_count)
    )
    return PSIResult(
        bins=bins,
        config=config,
        reference_sample_count=len(reference_rows),
        current_sample_count=len(current_rows),
        psi=psi,
        drift_flag=psi >= config.alert_threshold,
        cutoff_timestamp=cutoff_timestamp,
        reference_sha256=_canonical_sha256([item.as_dict() for item in reference_rows]),
        current_sha256=_canonical_sha256([item.as_dict() for item in current_rows]),
    )


__all__ = ["PSI_SCHEMA", "PSIConfig", "PSIBin", "PSIResult", "DistributionDriftError", "compute_psi"]
