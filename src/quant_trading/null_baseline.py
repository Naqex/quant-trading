"""Deterministic offline reference baselines for chronological research.

The runner evaluates two deliberately boring references through the same
walk-forward and fee/slippage contract as a candidate strategy:

``cash``
    Never holds the asset.  This is a no-signal/null reference.
``buy_hold``
    Holds the asset continuously.  This is a passive market benchmark.

These references make a strategy receipt comparable without silently
reimplementing accounting.  They do not provide a statistical p-value, model
selection, promotion decision, provider access or execution authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_trading.walk_forward import (
    WalkForwardConfig,
    WalkForwardError,
    WalkForwardReceipt,
    run_walk_forward_stress,
)

NULL_BASELINE_SCHEMA = "null-baseline-receipt-v1"
_BASELINE_FACTORIES = {
    "buy_hold": lambda history: pd.Series(1.0, index=history.index, name="signal"),
    "cash": lambda history: pd.Series(0.0, index=history.index, name="signal"),
}


class NullBaselineError(ValueError):
    """Raised when a null-baseline contract is malformed or unsafe."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise NullBaselineError("null-baseline receipt must contain finite JSON data") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NullBaselineError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise NullBaselineError(f"{name} must be at most {maximum} characters")
    return normalized


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise NullBaselineError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise NullBaselineError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise NullBaselineError(f"{name} must be finite")
    return parsed


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise NullBaselineError(f"{name} must be a lowercase SHA-256")
    return value


def cash_signal_factory(history: pd.Series) -> pd.Series:
    """Return the all-cash null signal for a supplied history index."""

    return _BASELINE_FACTORIES["cash"](history)


def buy_and_hold_signal_factory(history: pd.Series) -> pd.Series:
    """Return the continuous long-only buy-and-hold signal."""

    return _BASELINE_FACTORIES["buy_hold"](history)


@dataclass(frozen=True)
class NullBaselineEvaluation:
    """One reference baseline evaluated with the shared WFO accounting."""

    baseline_id: str
    receipt: WalkForwardReceipt
    baseline_metrics: Mapping[str, float | int | None]
    stress_metrics: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        baseline_id = _text(self.baseline_id, "baseline_id")
        if baseline_id not in _BASELINE_FACTORIES:
            raise NullBaselineError(f"unsupported baseline_id: {baseline_id!r}")
        if not isinstance(self.receipt, WalkForwardReceipt):
            raise NullBaselineError("baseline receipt must be WalkForwardReceipt")
        normalized: list[dict[str, float | int | None]] = []
        for metrics in (self.baseline_metrics, self.stress_metrics):
            current = dict(metrics)
            for key, value in current.items():
                _text(key, "metric name")
                if value is not None:
                    try:
                        finite = math.isfinite(float(value))
                    except (TypeError, ValueError):
                        finite = False
                    if not finite:
                        raise NullBaselineError("baseline metrics must be finite numbers or None")
            normalized.append(current)
        object.__setattr__(self, "baseline_id", baseline_id)
        object.__setattr__(self, "baseline_metrics", normalized[0])
        object.__setattr__(self, "stress_metrics", normalized[1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "receipt": self.receipt.as_dict(),
            "baseline_metrics": dict(self.baseline_metrics),
            "stress_metrics": dict(self.stress_metrics),
        }


@dataclass(frozen=True)
class NullBaselineReceipt:
    """Immutable, PREP_ONLY evidence for cash and passive references."""

    suite_id: str
    input_sha256: str
    base_fee_bps: float
    base_slippage_bps: float
    config: WalkForwardConfig
    evaluations: tuple[NullBaselineEvaluation, ...]
    schema: str = NULL_BASELINE_SCHEMA
    status: str = "PREP_ONLY"
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        suite_id = _text(self.suite_id, "suite_id")
        input_sha256 = _sha256(self.input_sha256, "input_sha256")
        fee = _finite(self.base_fee_bps, "base_fee_bps")
        slippage = _finite(self.base_slippage_bps, "base_slippage_bps")
        if fee < 0.0 or slippage < 0.0:
            raise NullBaselineError("base costs must be non-negative")
        if self.schema != NULL_BASELINE_SCHEMA:
            raise NullBaselineError(f"unsupported null-baseline schema: {self.schema!r}")
        if self.status != "PREP_ONLY":
            raise NullBaselineError("null-baseline status must remain PREP_ONLY")
        if not isinstance(self.config, WalkForwardConfig) or not self.evaluations:
            raise NullBaselineError("config and at least one baseline evaluation are required")
        ids = [evaluation.baseline_id for evaluation in self.evaluations]
        if len(set(ids)) != len(ids):
            raise NullBaselineError("baseline_id values must be unique")
        for evaluation in self.evaluations:
            nested = evaluation.receipt
            if nested.input_sha256 != input_sha256:
                raise NullBaselineError("all baselines must use the same input fixture")
            if nested.config.as_dict() != self.config.as_dict():
                raise NullBaselineError("all baselines must use the same walk-forward config")
            if nested.base_fee_bps != fee or nested.base_slippage_bps != slippage:
                raise NullBaselineError("all baselines must use the receipt cost assumptions")
            baseline_case = next((case for case in nested.cases if case.multiplier == 1.0), None)
            stressed_case = max(nested.cases, key=lambda case: case.multiplier)
            if baseline_case is None:
                raise NullBaselineError("each baseline must contain the 1.0 cost case")
            if dict(evaluation.baseline_metrics) != dict(baseline_case.aggregate_metrics):
                raise NullBaselineError("baseline metrics do not match nested receipt")
            if dict(evaluation.stress_metrics) != dict(stressed_case.aggregate_metrics):
                raise NullBaselineError("stress metrics do not match nested receipt")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise NullBaselineError(f"{name} must remain false for offline research")
        object.__setattr__(self, "suite_id", suite_id)
        object.__setattr__(self, "input_sha256", input_sha256)
        object.__setattr__(self, "base_fee_bps", fee)
        object.__setattr__(self, "base_slippage_bps", slippage)
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        expected = _sha256_json(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise NullBaselineError("receipt_sha256 does not match canonical receipt")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "suite_id": self.suite_id,
            "input_sha256": self.input_sha256,
            "base_fee_bps": self.base_fee_bps,
            "base_slippage_bps": self.base_slippage_bps,
            "config": self.config.as_dict(),
            "evaluations": [evaluation.as_dict() for evaluation in self.evaluations],
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["receipt_sha256"] = self.receipt_sha256
        return result


def run_null_baselines(
    close: pd.Series,
    *,
    suite_id: str = "null-baseline-v1",
    baseline_ids: tuple[str, ...] = ("cash", "buy_hold"),
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    config: WalkForwardConfig | None = None,
) -> NullBaselineReceipt:
    """Evaluate registered cash and buy-and-hold references offline.

    ``baseline_ids`` is sorted before evaluation, so the receipt is invariant
    to mapping/order choices.  The same walk-forward folds, next-bar timing,
    fee and slippage assumptions are used for every baseline.
    """

    suite_id = _text(suite_id, "suite_id")
    if not isinstance(baseline_ids, tuple) or not baseline_ids:
        raise NullBaselineError("baseline_ids must be a non-empty tuple")
    if any(not isinstance(item, str) for item in baseline_ids):
        raise NullBaselineError("baseline_ids must contain strings")
    normalized_ids = tuple(sorted(item.strip() for item in baseline_ids))
    if any(not item for item in normalized_ids):
        raise NullBaselineError("baseline_ids must contain non-empty strings")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise NullBaselineError("baseline_ids must be unique")
    unsupported = [item for item in normalized_ids if item not in _BASELINE_FACTORIES]
    if unsupported:
        raise NullBaselineError(f"unsupported baseline_id: {unsupported[0]!r}")
    config = WalkForwardConfig() if config is None else config
    if not isinstance(config, WalkForwardConfig):
        raise NullBaselineError("config must be WalkForwardConfig")
    evaluations: list[NullBaselineEvaluation] = []
    expected_input: str | None = None
    for baseline_id in normalized_ids:
        try:
            receipt = run_walk_forward_stress(
                close,
                _BASELINE_FACTORIES[baseline_id],
                strategy_id=f"{suite_id}:{baseline_id}",
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                config=config,
            )
        except WalkForwardError as exc:
            raise NullBaselineError(f"baseline {baseline_id!r} failed walk-forward contract") from exc
        if expected_input is None:
            expected_input = receipt.input_sha256
        elif receipt.input_sha256 != expected_input:
            raise NullBaselineError("all baselines must use the same input fixture")
        baseline_case = next((case for case in receipt.cases if case.multiplier == 1.0), None)
        if baseline_case is None:
            raise NullBaselineError("each baseline must contain the 1.0 cost case")
        stressed_case = max(receipt.cases, key=lambda case: case.multiplier)
        evaluations.append(
            NullBaselineEvaluation(
                baseline_id=baseline_id,
                receipt=receipt,
                baseline_metrics=baseline_case.aggregate_metrics,
                stress_metrics=stressed_case.aggregate_metrics,
            )
        )
    assert expected_input is not None
    return NullBaselineReceipt(
        suite_id=suite_id,
        input_sha256=expected_input,
        base_fee_bps=fee_bps,
        base_slippage_bps=slippage_bps,
        config=config,
        evaluations=tuple(evaluations),
    )


__all__ = [
    "NULL_BASELINE_SCHEMA",
    "NullBaselineError",
    "NullBaselineEvaluation",
    "NullBaselineReceipt",
    "buy_and_hold_signal_factory",
    "cash_signal_factory",
    "run_null_baselines",
]
