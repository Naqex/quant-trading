"""Finite, deterministic risk metrics for offline spot research.

The functions in this module evaluate an already captured sequence of simple
returns.  They do not infer positions, apply leverage, fetch data or authorize
execution.  ``compute_risk_metrics`` is deliberately scoped to a chronological
test window described by :class:`~quant_trading.discrete_optimizer.ResearchCutoffs`.

The tail metrics are expressed in return space and loss space together:

* ``var_return`` is the empirical lower-tail return quantile;
* ``expected_shortfall_return`` (CVaR) is the deterministic average of the
  worst ``alpha`` probability mass, using a fractional final order statistic;
* ``var_loss`` and ``expected_shortfall_loss`` are the corresponding positive
  loss magnitudes, floored at zero.

The fractional-tail calculation avoids a hidden dependence on ``floor(n
* alpha)`` for short samples.  All numeric outputs are finite.  A zero
downside denominator makes Sortino undefined; this is represented explicitly by
``sortino_defined=False`` and a finite ``sortino_zero_target=0.0`` rather than
an infinity or NaN.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from quant_trading.discrete_optimizer import ResearchCutoffs

RISK_METRICS_SCHEMA = "offline-risk-metrics-v1"
RiskWindow = Literal["train", "validation", "test"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RiskMetricsError(ValueError):
    """Raised when an offline risk-metrics contract is malformed."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RiskMetricsError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise RiskMetricsError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RiskMetricsError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise RiskMetricsError(f"{name} must be finite")
    return parsed


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RiskMetricsError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RiskMetricsError("risk receipt values must be finite JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _numeric_vector(values: Sequence[float] | np.ndarray, name: str) -> tuple[float, ...]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RiskMetricsError(f"{name} must be a non-empty one-dimensional numeric sequence") from exc
    if array.ndim != 1 or array.size == 0:
        raise RiskMetricsError(f"{name} must be a non-empty one-dimensional numeric sequence")
    if not np.all(np.isfinite(array)):
        raise RiskMetricsError(f"{name} must contain only finite values")
    return tuple(float(item) for item in array.tolist())


def _timestamps(values: Sequence[int] | np.ndarray, name: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise RiskMetricsError(f"{name} must be a non-empty timestamp sequence") from exc
    if not raw:
        raise RiskMetricsError(f"{name} must be a non-empty timestamp sequence")
    normalized = tuple(_strict_int(item, f"{name}[{index}]") for index, item in enumerate(raw))
    if any(left >= right for left, right in zip(normalized, normalized[1:], strict=False)):
        raise RiskMetricsError(f"{name} must be strictly increasing")
    return normalized


def _scaled_mean(values: Sequence[float], name: str) -> float:
    """Average finite values without overflowing an intermediate sum."""

    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = math.fsum(value / scale for value in values) / len(values)
    result = scale * normalized
    if not math.isfinite(result):
        raise RiskMetricsError(f"{name} became non-finite")
    return result


def _scaled_weighted_mean(values: Sequence[float], weights: Sequence[float], name: str) -> float:
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    denominator = math.fsum(weights)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise RiskMetricsError(f"{name} has invalid tail mass")
    normalized = math.fsum((value / scale) * weight for value, weight in zip(values, weights, strict=True))
    result = scale * (normalized / denominator)
    if not math.isfinite(result):
        raise RiskMetricsError(f"{name} became non-finite")
    return result


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return NumPy-linear quantile semantics with overflow-safe interpolation."""

    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower_index = int(math.floor(position))
    fraction = position - lower_index
    if fraction == 0.0:
        return float(sorted_values[lower_index])
    lower = sorted_values[lower_index]
    upper = sorted_values[lower_index + 1]
    scale = max(abs(lower), abs(upper))
    if scale == 0.0:
        return 0.0
    result = scale * ((lower / scale) * (1.0 - fraction) + (upper / scale) * fraction)
    if not math.isfinite(result):
        raise RiskMetricsError("value-at-risk quantile became non-finite")
    return float(result)


def expected_shortfall(returns: Sequence[float] | np.ndarray, *, alpha: float) -> float:
    """Return lower-tail CVaR in return space for a finite sample.

    ``alpha`` is the explicit probability mass of the worst tail and must be
    strictly between zero and one.  The final order statistic receives a
    fractional weight when ``len(returns) * alpha`` is not an integer.
    """

    values = _numeric_vector(returns, "returns")
    alpha = _finite_float(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise RiskMetricsError("alpha must be strictly between 0 and 1")
    ordered = tuple(sorted(values))
    mass = alpha * len(ordered)
    full_count = int(math.floor(mass))
    remainder = mass - full_count
    if full_count == 0:
        return ordered[0]
    tail_values = list(ordered[:full_count])
    weights = [1.0] * full_count
    if remainder > 0.0:
        tail_values.append(ordered[full_count])
        weights.append(remainder)
    return _scaled_weighted_mean(tail_values, weights, "expected shortfall")


def _downside_rms(values: Sequence[float]) -> float:
    negatives = tuple(max(0.0, -value) for value in values)
    scale = max(negatives)
    if scale == 0.0:
        return 0.0
    normalized = math.fsum((value / scale) ** 2 for value in negatives) / len(negatives)
    result = scale * math.sqrt(normalized)
    if not math.isfinite(result):
        raise RiskMetricsError("downside deviation became non-finite")
    return result


def downside_deviation(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: int = 365,
) -> float:
    """Return annualized zero-target downside deviation."""

    values = _numeric_vector(returns, "returns")
    periods = _strict_int(periods_per_year, "periods_per_year", minimum=1)
    result = _downside_rms(values) * math.sqrt(periods)
    if not math.isfinite(result):
        raise RiskMetricsError("annualized downside deviation became non-finite")
    return float(result)


def _window_bounds(cutoffs: ResearchCutoffs, window: RiskWindow) -> tuple[int, int]:
    if not isinstance(cutoffs, ResearchCutoffs):
        raise RiskMetricsError("cutoffs must be ResearchCutoffs")
    if not isinstance(window, str):
        raise RiskMetricsError("window must be one of: train, validation, test")
    if window == "train":
        return 0, cutoffs.train_end_utc
    if window == "validation":
        return cutoffs.train_end_utc, cutoffs.validation_end_utc
    if window == "test":
        return cutoffs.validation_end_utc, cutoffs.test_end_utc
    raise RiskMetricsError("window must be one of: train, validation, test")


@dataclass(frozen=True)
class RiskMetricsResult:
    """Immutable offline risk receipt with no execution authority."""

    alpha: float
    periods_per_year: int
    window: RiskWindow
    sample_count: int
    tail_mass: float
    tail_full_count: int
    var_return: float
    expected_shortfall_return: float
    var_loss: float
    expected_shortfall_loss: float
    mean_return: float
    worst_return: float
    downside_deviation: float
    sortino_zero_target: float
    sortino_defined: bool
    data_sha256: str
    returns_sha256: str
    timestamps_sha256: str
    evaluation_start_utc: int
    evaluation_end_utc: int
    cutoffs: ResearchCutoffs
    schema: str = RISK_METRICS_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != RISK_METRICS_SCHEMA:
            raise RiskMetricsError(f"unsupported risk metrics schema: {self.schema!r}")
        alpha = _finite_float(self.alpha, "alpha")
        if not 0.0 < alpha < 1.0:
            raise RiskMetricsError("alpha must be strictly between 0 and 1")
        periods = _strict_int(self.periods_per_year, "periods_per_year", minimum=1)
        if not isinstance(self.window, str) or self.window not in {"train", "validation", "test"}:
            raise RiskMetricsError("window must be one of: train, validation, test")
        count = _strict_int(self.sample_count, "sample_count", minimum=1)
        _strict_int(self.tail_full_count, "tail_full_count", minimum=0)
        for name in (
            "tail_mass",
            "var_return",
            "expected_shortfall_return",
            "var_loss",
            "expected_shortfall_loss",
            "mean_return",
            "worst_return",
            "downside_deviation",
            "sortino_zero_target",
        ):
            _finite_float(getattr(self, name), name)
        if self.tail_mass <= 0.0 or self.tail_mass >= count:
            raise RiskMetricsError("tail_mass must be strictly between zero and sample_count")
        if self.var_loss < 0.0 or self.expected_shortfall_loss < 0.0:
            raise RiskMetricsError("loss metrics must be >= 0")
        if type(self.sortino_defined) is not bool:
            raise RiskMetricsError("sortino_defined must be a boolean")
        _sha256(self.data_sha256, "data_sha256")
        _sha256(self.returns_sha256, "returns_sha256")
        _sha256(self.timestamps_sha256, "timestamps_sha256")
        if not isinstance(self.cutoffs, ResearchCutoffs):
            raise RiskMetricsError("cutoffs must be ResearchCutoffs")
        start = _strict_int(self.evaluation_start_utc, "evaluation_start_utc")
        end = _strict_int(self.evaluation_end_utc, "evaluation_end_utc")
        if start > end:
            raise RiskMetricsError("evaluation_start_utc must be <= evaluation_end_utc")
        if self.execution_capability is not False:
            raise RiskMetricsError("risk metrics cannot grant execution capability")
        if self.provider_access is not False:
            raise RiskMetricsError("risk metrics cannot grant provider access")
        if self.broker_access is not False:
            raise RiskMetricsError("risk metrics cannot grant broker access")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "periods_per_year", periods)
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "evaluation_start_utc", start)
        object.__setattr__(self, "evaluation_end_utc", end)

    @property
    def var(self) -> float:
        """Alias for the lower-tail return quantile."""

        return self.var_return

    @property
    def cvar_return(self) -> float:
        """Alias for expected shortfall in return space."""

        return self.expected_shortfall_return

    @property
    def cvar_loss(self) -> float:
        """Alias for expected shortfall as a non-negative loss magnitude."""

        return self.expected_shortfall_loss

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": "PREP_ONLY",
            "scope": "crypto_spot_research",
            "method": "empirical_lower_tail_expected_shortfall",
            "alpha": self.alpha,
            "periods_per_year": self.periods_per_year,
            "window": self.window,
            "sample_count": self.sample_count,
            "tail_mass": self.tail_mass,
            "tail_full_count": self.tail_full_count,
            "var_return": self.var_return,
            "expected_shortfall_return": self.expected_shortfall_return,
            "var_loss": self.var_loss,
            "expected_shortfall_loss": self.expected_shortfall_loss,
            "cvar_return": self.cvar_return,
            "cvar_loss": self.cvar_loss,
            "mean_return": self.mean_return,
            "worst_return": self.worst_return,
            "downside_deviation": self.downside_deviation,
            "sortino_zero_target": self.sortino_zero_target,
            "sortino_defined": self.sortino_defined,
            "data_sha256": self.data_sha256,
            "returns_sha256": self.returns_sha256,
            "timestamps_sha256": self.timestamps_sha256,
            "evaluation_start_utc": self.evaluation_start_utc,
            "evaluation_end_utc": self.evaluation_end_utc,
            "cutoffs": self.cutoffs.as_dict(),
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


def compute_risk_metrics(
    returns: Sequence[float] | np.ndarray,
    timestamps_utc: Sequence[int] | np.ndarray,
    *,
    alpha: float,
    data_sha256: str,
    cutoffs: ResearchCutoffs,
    periods_per_year: int = 365,
    window: RiskWindow = "test",
) -> RiskMetricsResult:
    """Compute deterministic finite risk metrics for one research window.

    Rows must be strictly ordered and belong to the selected chronological
    window.  The default is the locked test window ``validation_end < t <=
    test_end``; callers may explicitly choose train or validation for an
    evaluation artifact.  No position sizing, leverage, costs or execution is
    inferred here.
    """

    values = _numeric_vector(returns, "returns")
    timestamps = _timestamps(timestamps_utc, "timestamps_utc")
    if len(values) != len(timestamps):
        raise RiskMetricsError("returns and timestamps_utc must have the same length")
    alpha = _finite_float(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise RiskMetricsError("alpha must be strictly between 0 and 1")
    periods = _strict_int(periods_per_year, "periods_per_year", minimum=1)
    data_sha256 = _sha256(data_sha256, "data_sha256")
    lower_bound, upper_bound = _window_bounds(cutoffs, window)
    if any(timestamp <= lower_bound or timestamp > upper_bound for timestamp in timestamps):
        raise RiskMetricsError(
            f"timestamps must satisfy {lower_bound} < timestamp <= {upper_bound} for {window} window"
        )

    ordered = tuple(sorted(values))
    sample_count = len(values)
    tail_mass = alpha * sample_count
    full_count = int(math.floor(tail_mass))
    cvar = expected_shortfall(values, alpha=alpha)
    var_return = _linear_quantile(ordered, alpha)
    mean_return = _scaled_mean(values, "mean return")
    downside_raw = _downside_rms(values)
    annualized_downside = downside_raw * math.sqrt(periods)
    if not math.isfinite(annualized_downside):
        raise RiskMetricsError("annualized downside deviation became non-finite")
    if downside_raw > 0.0:
        sortino = (mean_return / downside_raw) * math.sqrt(periods)
        sortino_defined = True
    else:
        # A ratio with no negative observations is undefined.  Keep the
        # numeric field finite and expose the semantic state separately.
        sortino = 0.0
        sortino_defined = False
    if not math.isfinite(sortino):
        raise RiskMetricsError("Sortino ratio became non-finite")

    sample_payload = {
        "schema": "risk-return-sample-v1",
        "timestamps_utc": list(timestamps),
        "returns": list(values),
    }
    return RiskMetricsResult(
        alpha=alpha,
        periods_per_year=periods,
        window=window,
        sample_count=sample_count,
        tail_mass=tail_mass,
        tail_full_count=full_count,
        var_return=var_return,
        expected_shortfall_return=cvar,
        var_loss=max(0.0, -var_return),
        expected_shortfall_loss=max(0.0, -cvar),
        mean_return=mean_return,
        worst_return=ordered[0],
        downside_deviation=annualized_downside,
        sortino_zero_target=sortino,
        sortino_defined=sortino_defined,
        data_sha256=data_sha256,
        returns_sha256=_canonical_sha256(sample_payload),
        timestamps_sha256=_canonical_sha256(
            {"schema": "risk-timestamps-v1", "timestamps_utc": list(timestamps)}
        ),
        evaluation_start_utc=timestamps[0],
        evaluation_end_utc=timestamps[-1],
        cutoffs=cutoffs,
    )


__all__ = [
    "RISK_METRICS_SCHEMA",
    "RiskMetricsError",
    "RiskMetricsResult",
    "compute_risk_metrics",
    "downside_deviation",
    "expected_shortfall",
]
