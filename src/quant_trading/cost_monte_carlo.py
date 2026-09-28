"""Deterministic Monte Carlo stress for modeled execution costs.

The runner perturbs fee and slippage assumptions while keeping the supplied
causal signal and price fixture fixed.  It is deliberately a research
diagnostic: it does not model fills, fetch data, or grant paper/live
execution capability.  For a causal out-of-sample result, pass a signal that
was produced by :mod:`quant_trading.walk_forward` or an equivalent point-in-
time pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.backtest import run_backtest
from quant_lab.metrics import performance_metrics

COST_MONTE_CARLO_SCHEMA = "cost-monte-carlo-receipt-v1"


class CostMonteCarloError(ValueError):
    """Raised when the offline cost-stress contract is malformed or unsafe."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CostMonteCarloError("cost Monte Carlo receipt must contain finite JSON data") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CostMonteCarloError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CostMonteCarloError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise CostMonteCarloError(f"{name} must be finite")
    return parsed


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CostMonteCarloError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise CostMonteCarloError(f"{name} must be at most {maximum} characters")
    return normalized


def _strict_int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bound = f" and <= {maximum}" if maximum is not None else ""
        raise CostMonteCarloError(f"{name} must be an integer >= {minimum}{bound}")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CostMonteCarloError(f"{name} must be a lowercase SHA-256")
    return value


def _label(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    return str(value)


def _metrics(result: pd.DataFrame) -> dict[str, float | int | None]:
    normalized: dict[str, float | int | None] = {}
    for key, value in performance_metrics(result).items():
        normalized[key] = None if not math.isfinite(float(value)) else value
    return normalized


def _summary(values: list[float | int | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "min": None,
            "q05": None,
            "median": None,
            "q95": None,
            "max": None,
            "draw_count": 0,
        }
    quantiles = np.quantile(finite, (0.05, 0.5, 0.95), method="linear")
    return {
        "min": float(np.min(finite)),
        "q05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "max": float(np.max(finite)),
        "draw_count": int(finite.size),
    }


def _unit_interval(seed: int, draw_index: int, component: str) -> float:
    """Return a reproducible value in [0, 1) without RNG-version drift."""

    payload = f"cost-monte-carlo-v1|{seed}|{draw_index}|{component}".encode("ascii")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return integer / 2**64


def _multiplier(seed: int, draw_index: int, component: str, low: float, high: float) -> float:
    return low + (high - low) * _unit_interval(seed, draw_index, component)


@dataclass(frozen=True)
class CostMonteCarloConfig:
    """Bounded, pre-registered cost multiplier ranges."""

    draws: int = 256
    seed: int = 20260928
    fee_multiplier_low: float = 0.5
    fee_multiplier_high: float = 3.0
    slippage_multiplier_low: float = 0.5
    slippage_multiplier_high: float = 3.0

    def __post_init__(self) -> None:
        draws = _strict_int(self.draws, "draws", minimum=1, maximum=5_000)
        seed = _strict_int(self.seed, "seed", minimum=0)
        fee_low = _finite(self.fee_multiplier_low, "fee_multiplier_low")
        fee_high = _finite(self.fee_multiplier_high, "fee_multiplier_high")
        slip_low = _finite(self.slippage_multiplier_low, "slippage_multiplier_low")
        slip_high = _finite(self.slippage_multiplier_high, "slippage_multiplier_high")
        for name, value in (
            ("fee_multiplier_low", fee_low),
            ("fee_multiplier_high", fee_high),
            ("slippage_multiplier_low", slip_low),
            ("slippage_multiplier_high", slip_high),
        ):
            if value <= 0.0:
                raise CostMonteCarloError(f"{name} must be greater than zero")
        if fee_high < fee_low or slip_high < slip_low:
            raise CostMonteCarloError("cost multiplier high bounds must be >= low bounds")
        object.__setattr__(self, "draws", draws)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "fee_multiplier_low", fee_low)
        object.__setattr__(self, "fee_multiplier_high", fee_high)
        object.__setattr__(self, "slippage_multiplier_low", slip_low)
        object.__setattr__(self, "slippage_multiplier_high", slip_high)

    def as_dict(self) -> dict[str, Any]:
        return {
            "draws": self.draws,
            "seed": self.seed,
            "fee_multiplier_low": self.fee_multiplier_low,
            "fee_multiplier_high": self.fee_multiplier_high,
            "slippage_multiplier_low": self.slippage_multiplier_low,
            "slippage_multiplier_high": self.slippage_multiplier_high,
        }


@dataclass(frozen=True)
class CostMonteCarloDraw:
    """One deterministic cost perturbation and its offline metrics."""

    draw_index: int
    fee_multiplier: float
    slippage_multiplier: float
    fee_bps: float
    slippage_bps: float
    metrics: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        index = _strict_int(self.draw_index, "draw_index", minimum=0)
        fee_multiplier = _finite(self.fee_multiplier, "fee_multiplier")
        slip_multiplier = _finite(self.slippage_multiplier, "slippage_multiplier")
        fee = _finite(self.fee_bps, "fee_bps")
        slippage = _finite(self.slippage_bps, "slippage_bps")
        if fee_multiplier <= 0.0 or slip_multiplier <= 0.0 or fee < 0.0 or slippage < 0.0:
            raise CostMonteCarloError("draw cost values must be positive multipliers and non-negative bps")
        if fee + slippage >= 10_000.0:
            raise CostMonteCarloError("draw costs must remain below 100% per turnover")
        normalized = dict(self.metrics)
        for key, value in normalized.items():
            _text(key, "metric name", maximum=128)
            if value is not None and not math.isfinite(float(value)):
                raise CostMonteCarloError("draw metrics must be finite numbers or None")
        object.__setattr__(self, "draw_index", index)
        object.__setattr__(self, "fee_multiplier", fee_multiplier)
        object.__setattr__(self, "slippage_multiplier", slip_multiplier)
        object.__setattr__(self, "fee_bps", fee)
        object.__setattr__(self, "slippage_bps", slippage)
        object.__setattr__(self, "metrics", normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "draw_index": self.draw_index,
            "fee_multiplier": self.fee_multiplier,
            "slippage_multiplier": self.slippage_multiplier,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class CostMonteCarloReceipt:
    """Immutable stress evidence; it can never become execution authority."""

    strategy_id: str
    input_sha256: str
    base_fee_bps: float
    base_slippage_bps: float
    baseline_metrics: Mapping[str, float | int | None]
    config: CostMonteCarloConfig
    draws: tuple[CostMonteCarloDraw, ...]
    metric_summary: Mapping[str, Mapping[str, float | int | None]]
    schema: str = COST_MONTE_CARLO_SCHEMA
    status: str = "PREP_ONLY"
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        strategy_id = _text(self.strategy_id, "strategy_id")
        input_sha256 = _sha256(self.input_sha256, "input_sha256")
        fee = _finite(self.base_fee_bps, "base_fee_bps")
        slippage = _finite(self.base_slippage_bps, "base_slippage_bps")
        if fee < 0.0 or slippage < 0.0 or fee + slippage >= 10_000.0:
            raise CostMonteCarloError("base costs must be non-negative and below 100% per turnover")
        if self.schema != COST_MONTE_CARLO_SCHEMA:
            raise CostMonteCarloError(f"unsupported cost Monte Carlo schema: {self.schema!r}")
        if self.status != "PREP_ONLY":
            raise CostMonteCarloError("cost Monte Carlo status must remain PREP_ONLY")
        if not isinstance(self.config, CostMonteCarloConfig):
            raise CostMonteCarloError("config must be CostMonteCarloConfig")
        draws = tuple(self.draws)
        if len(draws) != self.config.draws:
            raise CostMonteCarloError("draw count must match config.draws")
        if any(not isinstance(draw, CostMonteCarloDraw) for draw in draws):
            raise CostMonteCarloError("draws must contain CostMonteCarloDraw values")
        if tuple(draw.draw_index for draw in draws) != tuple(range(self.config.draws)):
            raise CostMonteCarloError("draw indexes must be contiguous and sorted")
        baseline = dict(self.baseline_metrics)
        for metrics in (baseline,):
            for key, value in metrics.items():
                _text(key, "metric name", maximum=128)
                if value is not None and not math.isfinite(float(value)):
                    raise CostMonteCarloError("baseline metrics must be finite numbers or None")
        expected_summary: dict[str, dict[str, float | int | None]] = {}
        metric_names = sorted({key for draw in draws for key in draw.metrics})
        for metric in metric_names:
            expected_summary[metric] = _summary([draw.metrics.get(metric) for draw in draws])
        supplied_summary = {
            str(metric): dict(values) for metric, values in self.metric_summary.items()
        }
        if supplied_summary != expected_summary:
            raise CostMonteCarloError("metric_summary does not match draw metrics")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise CostMonteCarloError(f"{name} must remain false for offline research")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "input_sha256", input_sha256)
        object.__setattr__(self, "base_fee_bps", fee)
        object.__setattr__(self, "base_slippage_bps", slippage)
        object.__setattr__(self, "baseline_metrics", baseline)
        object.__setattr__(self, "draws", draws)
        object.__setattr__(self, "metric_summary", expected_summary)
        expected_hash = _sha256_json(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected_hash:
            raise CostMonteCarloError("receipt_sha256 does not match canonical receipt")
        object.__setattr__(self, "receipt_sha256", expected_hash)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "strategy_id": self.strategy_id,
            "input_sha256": self.input_sha256,
            "base_fee_bps": self.base_fee_bps,
            "base_slippage_bps": self.base_slippage_bps,
            "baseline_metrics": dict(self.baseline_metrics),
            "config": self.config.as_dict(),
            "draws": [draw.as_dict() for draw in self.draws],
            "metric_summary": {key: dict(value) for key, value in self.metric_summary.items()},
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["receipt_sha256"] = self.receipt_sha256
        return result


def _input_hash(close: pd.Series, signal: pd.Series) -> str:
    return _sha256_json(
        {
            "index": [_label(item) for item in close.index],
            "close": [float(value) for value in close.to_numpy(dtype=np.float64)],
            "signal": [None if pd.isna(value) else float(value) for value in signal.tolist()],
        }
    )


def run_cost_monte_carlo(
    close: pd.Series,
    signal: pd.Series,
    *,
    strategy_id: str,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    config: CostMonteCarloConfig | None = None,
) -> CostMonteCarloReceipt:
    """Run bounded, deterministic fee/slippage perturbations on one signal fixture."""

    config = CostMonteCarloConfig() if config is None else config
    if not isinstance(config, CostMonteCarloConfig):
        raise CostMonteCarloError("config must be CostMonteCarloConfig")
    base_fee = _finite(fee_bps, "fee_bps")
    base_slippage = _finite(slippage_bps, "slippage_bps")
    if base_fee < 0.0 or base_slippage < 0.0 or base_fee + base_slippage >= 10_000.0:
        raise CostMonteCarloError("base costs must be non-negative and below 100% per turnover")
    maximum_fee = base_fee * config.fee_multiplier_high
    maximum_slippage = base_slippage * config.slippage_multiplier_high
    if maximum_fee + maximum_slippage >= 10_000.0:
        raise CostMonteCarloError("configured multiplier range can exceed 100% per turnover")
    try:
        baseline_result = run_backtest(close, signal, base_fee, base_slippage)
    except (TypeError, ValueError) as exc:
        raise CostMonteCarloError("close and signal must satisfy the offline backtest contract") from exc
    baseline_metrics = _metrics(baseline_result)
    draws: list[CostMonteCarloDraw] = []
    for draw_index in range(config.draws):
        fee_multiplier = _multiplier(
            config.seed,
            draw_index,
            "fee",
            config.fee_multiplier_low,
            config.fee_multiplier_high,
        )
        slippage_multiplier = _multiplier(
            config.seed,
            draw_index,
            "slippage",
            config.slippage_multiplier_low,
            config.slippage_multiplier_high,
        )
        draw_fee = base_fee * fee_multiplier
        draw_slippage = base_slippage * slippage_multiplier
        result = run_backtest(close, signal, draw_fee, draw_slippage)
        draws.append(
            CostMonteCarloDraw(
                draw_index=draw_index,
                fee_multiplier=fee_multiplier,
                slippage_multiplier=slippage_multiplier,
                fee_bps=draw_fee,
                slippage_bps=draw_slippage,
                metrics=_metrics(result),
            )
        )
    metric_names = sorted({key for draw in draws for key in draw.metrics})
    summary = {
        metric: _summary([draw.metrics.get(metric) for draw in draws]) for metric in metric_names
    }
    return CostMonteCarloReceipt(
        strategy_id=strategy_id,
        input_sha256=_input_hash(close, signal),
        base_fee_bps=base_fee,
        base_slippage_bps=base_slippage,
        baseline_metrics=baseline_metrics,
        config=config,
        draws=tuple(draws),
        metric_summary=summary,
    )


__all__ = [
    "COST_MONTE_CARLO_SCHEMA",
    "CostMonteCarloConfig",
    "CostMonteCarloDraw",
    "CostMonteCarloError",
    "CostMonteCarloReceipt",
    "run_cost_monte_carlo",
]
