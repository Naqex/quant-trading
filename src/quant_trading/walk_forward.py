"""Deterministic chronological walk-forward and cost-stress diagnostics.

This module evaluates a causal signal factory on expanding-history windows. A
factory is replayed on each chronological prefix and only its last value is
accepted for that bar, so the runner cannot hand it a future OOS price. The
result is a review receipt, not a model promotion or trading instruction:
there is no provider, broker, wallet or execution path here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.backtest import run_backtest
from quant_lab.metrics import performance_metrics

WALK_FORWARD_SCHEMA = "walk-forward-cost-stress-receipt-v1"
SignalFactory = Callable[[pd.Series], pd.Series]


class WalkForwardError(ValueError):
    """Raised when a walk-forward contract is malformed or unsafe."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WalkForwardError("walk-forward receipt must contain finite JSON data") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise WalkForwardError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise WalkForwardError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise WalkForwardError(f"{name} must be finite")
    return parsed


def _positive(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0.0:
        raise WalkForwardError(f"{name} must be greater than zero")
    return parsed


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WalkForwardError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WalkForwardError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise WalkForwardError(f"{name} must be at most {maximum} characters")
    return normalized


def _label(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    return str(value)


def _validate_close(close: pd.Series) -> pd.Series:
    if not isinstance(close, pd.Series) or close.empty:
        raise WalkForwardError("close must be a non-empty pandas Series")
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise WalkForwardError("close index must be unique and sorted ascending")
    try:
        normalized = close.astype(float)
    except (TypeError, ValueError) as exc:
        raise WalkForwardError("close must contain numeric prices") from exc
    values = normalized.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise WalkForwardError("close prices must be finite and strictly positive")
    return normalized


def _metrics_with_boundary(
    result: pd.DataFrame, test_start: int, test_end: int
) -> dict[str, float | int | None]:
    period = result.iloc[test_start : test_end + 1].copy()
    period["equity"] = (1.0 + period["strategy_return"]).cumprod()
    period["drawdown"] = period["equity"] / period["equity"].cummax() - 1.0
    metrics = dict(performance_metrics(period))
    # The first test row can be the entry from a signal calculated on the last
    # training close.  performance_metrics cannot see the preceding row, so
    # restore that single boundary entry without changing test returns.
    boundary_entry = (
        test_start > 0
        and result["position"].iloc[test_start] > 0.0
        and result["position"].iloc[test_start - 1] <= 0.0
    )
    if boundary_entry:
        metrics["entries"] = int(metrics["entries"]) + 1
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in metrics.items()
    }


def _call_signal_factory(
    signal_factory: SignalFactory,
    history: pd.Series,
    fold_index: int,
) -> pd.Series:
    try:
        signal = signal_factory(history.copy())
    except Exception as exc:
        raise WalkForwardError(f"signal_factory failed in fold {fold_index}") from exc
    if not isinstance(signal, pd.Series) or not signal.index.equals(history.index):
        raise WalkForwardError(
            f"signal_factory must return a Series with the exact history index in fold {fold_index}"
        )
    return signal


def _build_point_in_time_signal(
    close: pd.Series,
    test_start: int,
    test_end: int,
    signal_factory: SignalFactory,
    fold_index: int,
) -> tuple[pd.Series, pd.Series]:
    """Build train plus test signals without exposing a future test close.

    The factory is a pure function contract: it receives a prefix and returns
    a signal series for that prefix.  Replaying each OOS prefix is deliberate;
    it prevents a vectorized factory from using a later test row through a
    centered window or a negative shift.  The last returned value is the only
    value accepted for that OOS bar.
    """

    train_history = close.iloc[:test_start]
    train_signal = _call_signal_factory(signal_factory, train_history, fold_index)
    signal_values: list[Any] = list(train_signal.tolist())
    for end in range(test_start, test_end + 1):
        prefix = close.iloc[: end + 1]
        prefix_signal = _call_signal_factory(signal_factory, prefix, fold_index)
        signal_values.append(prefix_signal.iloc[-1])
    combined = pd.Series(signal_values, index=close.index[: test_end + 1], name="signal")
    return close.iloc[: test_end + 1].copy(), combined


def _aggregate_metrics(folds: tuple[FoldEvaluation, ...]) -> dict[str, float | int | None]:
    if not folds:
        raise WalkForwardError("cannot aggregate an empty fold sequence")
    keys = sorted({key for fold in folds for key in fold.metrics})
    aggregate: dict[str, float | int | None] = {}
    for key in keys:
        values = [fold.metrics[key] for fold in folds]
        numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not numeric:
            aggregate[key] = None
        elif key in {"max_drawdown", "worst_period_return"}:
            aggregate[key] = min(numeric)
        else:
            aggregate[key] = float(np.median(np.asarray(numeric, dtype=np.float64)))
    cagr_values = [fold.metrics.get("cagr") for fold in folds]
    valid_cagr = [float(value) for value in cagr_values if value is not None and math.isfinite(float(value))]
    aggregate["positive_cagr_fraction"] = (
        sum(value > 0.0 for value in valid_cagr) / len(valid_cagr) if valid_cagr else None
    )
    aggregate["fold_count"] = len(folds)
    return aggregate


@dataclass(frozen=True)
class WalkForwardConfig:
    """Chronological expanding-window layout.

    ``train_bars`` rows precede each ``test_bars`` block.  ``step_bars``
    defaults to ``test_bars`` so test windows do not overlap.  A smaller step
    is allowed for diagnostic overlap, but each fold still has an isolated
    test interval and receives no rows after that interval.
    """

    train_bars: int = 180
    test_bars: int = 30
    step_bars: int | None = None
    minimum_folds: int = 2
    stress_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)

    def __post_init__(self) -> None:
        train = _strict_int(self.train_bars, "train_bars", minimum=1)
        test = _strict_int(self.test_bars, "test_bars", minimum=1)
        step = test if self.step_bars is None else _strict_int(self.step_bars, "step_bars", minimum=1)
        minimum = _strict_int(self.minimum_folds, "minimum_folds", minimum=1)
        if not isinstance(self.stress_multipliers, tuple) or not self.stress_multipliers:
            raise WalkForwardError("stress_multipliers must be a non-empty tuple")
        multipliers = tuple(_positive(value, "stress multiplier") for value in self.stress_multipliers)
        if len(set(multipliers)) != len(multipliers):
            raise WalkForwardError("stress_multipliers must be unique")
        if 1.0 not in multipliers:
            raise WalkForwardError("stress_multipliers must include the 1.0 baseline")
        object.__setattr__(self, "train_bars", train)
        object.__setattr__(self, "test_bars", test)
        object.__setattr__(self, "step_bars", step)
        object.__setattr__(self, "minimum_folds", minimum)
        object.__setattr__(self, "stress_multipliers", tuple(sorted(multipliers)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "step_bars": self.step_bars,
            "minimum_folds": self.minimum_folds,
            "stress_multipliers": list(self.stress_multipliers),
        }


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_count: int
    test_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_count": self.train_count,
            "test_count": self.test_count,
        }


@dataclass(frozen=True)
class FoldEvaluation:
    fold: WalkForwardFold
    metrics: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        if not isinstance(self.fold, WalkForwardFold):
            raise WalkForwardError("fold must be WalkForwardFold")
        normalized = dict(self.metrics)
        for key, value in normalized.items():
            _text(key, "metric name", maximum=128)
            if value is not None:
                if isinstance(value, bool) or not math.isfinite(float(value)):
                    raise WalkForwardError("fold metrics must be finite numbers or None")
        object.__setattr__(self, "metrics", normalized)

    def as_dict(self) -> dict[str, Any]:
        return {"fold": self.fold.as_dict(), "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class StressCase:
    multiplier: float
    fee_bps: float
    slippage_bps: float
    folds: tuple[FoldEvaluation, ...]
    aggregate_metrics: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        multiplier = _positive(self.multiplier, "multiplier")
        fee = _finite(self.fee_bps, "fee_bps")
        slippage = _finite(self.slippage_bps, "slippage_bps")
        if fee < 0.0 or slippage < 0.0:
            raise WalkForwardError("fee_bps and slippage_bps must be non-negative")
        if not self.folds:
            raise WalkForwardError("stress case must contain at least one fold")
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "fee_bps", fee)
        object.__setattr__(self, "slippage_bps", slippage)
        object.__setattr__(self, "folds", tuple(self.folds))
        object.__setattr__(self, "aggregate_metrics", dict(self.aggregate_metrics))

    def as_dict(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "folds": [fold.as_dict() for fold in self.folds],
            "aggregate_metrics": dict(self.aggregate_metrics),
        }


@dataclass(frozen=True)
class WalkForwardReceipt:
    strategy_id: str
    input_sha256: str
    base_fee_bps: float
    base_slippage_bps: float
    config: WalkForwardConfig
    folds: tuple[WalkForwardFold, ...]
    cases: tuple[StressCase, ...]
    schema: str = WALK_FORWARD_SCHEMA
    status: str = "PREP_ONLY"
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", _text(self.strategy_id, "strategy_id"))
        if len(self.input_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.input_sha256):
            raise WalkForwardError("input_sha256 must be a lowercase SHA-256")
        fee = _finite(self.base_fee_bps, "base_fee_bps")
        slippage = _finite(self.base_slippage_bps, "base_slippage_bps")
        if fee < 0.0 or slippage < 0.0:
            raise WalkForwardError("base costs must be non-negative")
        if self.schema != WALK_FORWARD_SCHEMA:
            raise WalkForwardError(f"unsupported walk-forward schema: {self.schema!r}")
        if self.status != "PREP_ONLY":
            raise WalkForwardError("walk-forward status must remain PREP_ONLY")
        if not isinstance(self.config, WalkForwardConfig) or not self.folds or not self.cases:
            raise WalkForwardError("receipt config, folds and cases are required")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise WalkForwardError(f"{name} must remain false for offline research")
        object.__setattr__(self, "base_fee_bps", fee)
        object.__setattr__(self, "base_slippage_bps", slippage)
        expected = _sha256_json(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise WalkForwardError("receipt_sha256 does not match canonical receipt")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "strategy_id": self.strategy_id,
            "input_sha256": self.input_sha256,
            "base_fee_bps": self.base_fee_bps,
            "base_slippage_bps": self.base_slippage_bps,
            "config": self.config.as_dict(),
            "folds": [fold.as_dict() for fold in self.folds],
            "cases": [case.as_dict() for case in self.cases],
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["receipt_sha256"] = self.receipt_sha256
        return result


def build_walk_forward_folds(index: pd.Index, config: WalkForwardConfig) -> tuple[WalkForwardFold, ...]:
    """Build non-leaking chronological folds from an ordered index."""

    if not isinstance(index, pd.Index) or len(index) == 0:
        raise WalkForwardError("index must be a non-empty pandas Index")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise WalkForwardError("index must be unique and sorted ascending")
    folds: list[WalkForwardFold] = []
    test_start = config.train_bars
    fold_index = 0
    while test_start + config.test_bars <= len(index):
        test_end = test_start + config.test_bars - 1
        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_start=_label(index[0]),
                train_end=_label(index[test_start - 1]),
                test_start=_label(index[test_start]),
                test_end=_label(index[test_end]),
                train_count=test_start,
                test_count=config.test_bars,
            )
        )
        fold_index += 1
        test_start += config.step_bars
    if len(folds) < config.minimum_folds:
        raise WalkForwardError(
            f"insufficient complete folds: got {len(folds)}, need {config.minimum_folds}"
        )
    return tuple(folds)


def run_walk_forward_stress(
    close: pd.Series,
    signal_factory: SignalFactory,
    *,
    strategy_id: str,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    config: WalkForwardConfig | None = None,
) -> WalkForwardReceipt:
    """Evaluate a causal signal across OOS folds and cost stress cases.

    The signal is calculated once per fold on ``close.iloc[:test_end + 1]``.
    It therefore sees the full training history and the current test history,
    but never a row after that fold's test end.  Costs are then replayed at
    deterministic multipliers without changing the signal.
    """

    if not callable(signal_factory):
        raise WalkForwardError("signal_factory must be callable")
    normalized_close = _validate_close(close)
    if config is None:
        config = WalkForwardConfig()
    if not isinstance(config, WalkForwardConfig):
        raise WalkForwardError("config must be WalkForwardConfig")
    base_fee = _finite(fee_bps, "fee_bps")
    base_slippage = _finite(slippage_bps, "slippage_bps")
    if base_fee < 0.0 or base_slippage < 0.0:
        raise WalkForwardError("fee_bps and slippage_bps must be non-negative")
    maximum_multiplier = max(config.stress_multipliers)
    if (base_fee + base_slippage) * maximum_multiplier >= 10_000.0:
        raise WalkForwardError("stress costs must remain below 100% per unit turnover")
    folds = build_walk_forward_folds(normalized_close.index, config)
    input_sha256 = _sha256_json(
        {
            "index": [_label(item) for item in normalized_close.index],
            "close": [float(value) for value in normalized_close.to_numpy(dtype=np.float64)],
        }
    )

    histories: list[tuple[int, int, pd.Series, pd.Series]] = []
    for fold in folds:
        test_start = config.train_bars + fold.fold_index * config.step_bars
        test_end = test_start + config.test_bars - 1
        history, signal = _build_point_in_time_signal(
            normalized_close,
            test_start,
            test_end,
            signal_factory,
            fold.fold_index,
        )
        # run_backtest validates range and finiteness after all prefixes are assembled.
        histories.append((test_start, test_end, history, signal))

    cases: list[StressCase] = []
    for multiplier in config.stress_multipliers:
        case_folds: list[FoldEvaluation] = []
        stressed_fee = base_fee * multiplier
        stressed_slippage = base_slippage * multiplier
        for (test_start, test_end, history, signal), fold in zip(histories, folds, strict=True):
            result = run_backtest(history, signal, stressed_fee, stressed_slippage)
            metrics = _metrics_with_boundary(result, test_start, test_end)
            case_folds.append(FoldEvaluation(fold=fold, metrics=metrics))
        frozen_folds = tuple(case_folds)
        cases.append(
            StressCase(
                multiplier=multiplier,
                fee_bps=stressed_fee,
                slippage_bps=stressed_slippage,
                folds=frozen_folds,
                aggregate_metrics=_aggregate_metrics(frozen_folds),
            )
        )
    return WalkForwardReceipt(
        strategy_id=strategy_id,
        input_sha256=input_sha256,
        base_fee_bps=base_fee,
        base_slippage_bps=base_slippage,
        config=config,
        folds=folds,
        cases=tuple(cases),
    )


__all__ = [
    "FoldEvaluation",
    "SignalFactory",
    "StressCase",
    "WALK_FORWARD_SCHEMA",
    "WalkForwardConfig",
    "WalkForwardError",
    "WalkForwardFold",
    "WalkForwardReceipt",
    "build_walk_forward_folds",
    "run_walk_forward_stress",
]
