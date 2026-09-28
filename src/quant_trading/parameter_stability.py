"""Offline parameter-neighbourhood stability diagnostics.

This module compares a pre-registered family of signal variants by delegating
each variant to the causal walk-forward runner. It reports dispersion of the
same out-of-sample metrics across variants; it deliberately does not choose a
winner or promote a model. There is no provider, broker, wallet or execution
path here.
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

from quant_trading.walk_forward import (
    SignalFactory,
    WalkForwardConfig,
    WalkForwardError,
    WalkForwardReceipt,
    run_walk_forward_stress,
)

PARAMETER_STABILITY_SCHEMA = "parameter-stability-receipt-v1"


class ParameterStabilityError(ValueError):
    """Raised when a parameter-stability contract is malformed or unsafe."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ParameterStabilityError("parameter-stability receipt must contain finite JSON data") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParameterStabilityError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ParameterStabilityError(f"{name} must be at most {maximum} characters")
    return normalized


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ParameterStabilityError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ParameterStabilityError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ParameterStabilityError(f"{name} must be finite")
    return parsed


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ParameterStabilityError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class VariantEvaluation:
    """The immutable evidence for one pre-registered parameter variant."""

    variant_id: str
    receipt: WalkForwardReceipt
    baseline_metrics: Mapping[str, float | int | None]
    stress_metrics: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        variant_id = _text(self.variant_id, "variant_id")
        if not isinstance(self.receipt, WalkForwardReceipt):
            raise ParameterStabilityError("variant receipt must be WalkForwardReceipt")
        baseline = dict(self.baseline_metrics)
        stress = dict(self.stress_metrics)
        for metrics in (baseline, stress):
            for key, value in metrics.items():
                _text(key, "metric name")
                if value is not None and not math.isfinite(float(value)):
                    raise ParameterStabilityError("variant metrics must be finite numbers or None")
        object.__setattr__(self, "variant_id", variant_id)
        object.__setattr__(self, "baseline_metrics", baseline)
        object.__setattr__(self, "stress_metrics", stress)

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "receipt": self.receipt.as_dict(),
            "baseline_metrics": dict(self.baseline_metrics),
            "stress_metrics": dict(self.stress_metrics),
        }


@dataclass(frozen=True)
class ParameterStabilityReceipt:
    """A diagnostic receipt; it contains dispersion, never a promotion decision."""

    family_id: str
    input_sha256: str
    config: WalkForwardConfig
    variants: tuple[VariantEvaluation, ...]
    metric_summary: Mapping[str, Mapping[str, float | int | None]]
    worst_stress_multiplier: float
    schema: str = PARAMETER_STABILITY_SCHEMA
    status: str = "PREP_ONLY"
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        family_id = _text(self.family_id, "family_id")
        input_sha256 = _sha256(self.input_sha256, "input_sha256")
        if self.schema != PARAMETER_STABILITY_SCHEMA:
            raise ParameterStabilityError(f"unsupported parameter-stability schema: {self.schema!r}")
        if self.status != "PREP_ONLY":
            raise ParameterStabilityError("parameter-stability status must remain PREP_ONLY")
        if not isinstance(self.config, WalkForwardConfig) or len(self.variants) < 2:
            raise ParameterStabilityError("config and at least two variants are required")
        if len({variant.variant_id for variant in self.variants}) != len(self.variants):
            raise ParameterStabilityError("variant_id values must be unique")
        for variant in self.variants:
            nested = variant.receipt
            if nested.input_sha256 != input_sha256:
                raise ParameterStabilityError("all variants must use the receipt input fixture")
            if nested.config.as_dict() != self.config.as_dict():
                raise ParameterStabilityError("all variants must use the receipt walk-forward config")
            baseline_case = next((case for case in nested.cases if case.multiplier == 1.0), None)
            stressed_case = max(nested.cases, key=lambda case: case.multiplier)
            if baseline_case is None:
                raise ParameterStabilityError("each variant must contain the 1.0 baseline case")
            if dict(variant.baseline_metrics) != dict(baseline_case.aggregate_metrics):
                raise ParameterStabilityError("variant baseline metrics do not match nested receipt")
            if dict(variant.stress_metrics) != dict(stressed_case.aggregate_metrics):
                raise ParameterStabilityError("variant stress metrics do not match nested receipt")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise ParameterStabilityError(f"{name} must remain false for offline research")
        multiplier = _finite(self.worst_stress_multiplier, "worst_stress_multiplier")
        if multiplier <= 0.0 or multiplier not in self.config.stress_multipliers:
            raise ParameterStabilityError("worst_stress_multiplier must be one configured multiplier")
        summary: dict[str, dict[str, float | int | None]] = {}
        for metric_name, values in self.metric_summary.items():
            _text(metric_name, "metric name")
            normalized = dict(values)
            expected = {"min", "median", "max", "range", "variant_count"}
            if set(normalized) != expected:
                raise ParameterStabilityError(f"metric summary for {metric_name!r} has invalid fields")
            for key, value in normalized.items():
                if key == "variant_count":
                    if type(value) is not int or value < 0:
                        raise ParameterStabilityError(
                            "metric summary variant_count must be a non-negative integer"
                        )
                elif value is not None and not math.isfinite(float(value)):
                    raise ParameterStabilityError("metric summary values must be finite or None")
            summary[metric_name] = normalized
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "input_sha256", input_sha256)
        object.__setattr__(self, "variants", tuple(self.variants))
        object.__setattr__(self, "metric_summary", summary)
        object.__setattr__(self, "worst_stress_multiplier", multiplier)
        expected_hash = _sha256_json(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected_hash:
            raise ParameterStabilityError("receipt_sha256 does not match canonical receipt")
        object.__setattr__(self, "receipt_sha256", expected_hash)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "family_id": self.family_id,
            "input_sha256": self.input_sha256,
            "config": self.config.as_dict(),
            "variants": [variant.as_dict() for variant in self.variants],
            "metric_summary": {key: dict(value) for key, value in self.metric_summary.items()},
            "worst_stress_multiplier": self.worst_stress_multiplier,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["receipt_sha256"] = self.receipt_sha256
        return result


def _summarize_metric(values: list[float | int | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"min": None, "median": None, "max": None, "range": None, "variant_count": 0}
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    return {
        "min": minimum,
        "median": float(np.median(finite)),
        "max": maximum,
        "range": maximum - minimum,
        "variant_count": int(finite.size),
    }


def run_parameter_stability(
    close: pd.Series,
    variants: Mapping[str, SignalFactory],
    *,
    family_id: str,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    config: WalkForwardConfig | None = None,
) -> ParameterStabilityReceipt:
    """Compare a pre-registered parameter neighbourhood via causal WFO.

    Variants are sorted by id before evaluation, making the evidence and hash
    deterministic regardless of mapping insertion order. The runner rejects
    malformed signals and enforces point-in-time prefixes for every variant.
    """

    if not isinstance(variants, Mapping) or len(variants) < 2:
        raise ParameterStabilityError("variants must contain at least two named factories")
    if any(not isinstance(name, str) for name in variants):
        raise ParameterStabilityError("variant ids must be strings")
    config = WalkForwardConfig() if config is None else config
    if not isinstance(config, WalkForwardConfig):
        raise ParameterStabilityError("config must be WalkForwardConfig")
    evaluations: list[VariantEvaluation] = []
    expected_input: str | None = None
    expected_config = config.as_dict()
    for variant_id in sorted(variants):
        _text(variant_id, "variant_id")
        factory = variants[variant_id]
        if not callable(factory):
            raise ParameterStabilityError(f"variant {variant_id!r} factory must be callable")
        try:
            receipt = run_walk_forward_stress(
                close,
                factory,
                strategy_id=f"{family_id}:{variant_id}",
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                config=config,
            )
        except WalkForwardError as exc:
            raise ParameterStabilityError(f"variant {variant_id!r} failed walk-forward contract") from exc
        if expected_input is None:
            expected_input = receipt.input_sha256
        elif receipt.input_sha256 != expected_input:
            raise ParameterStabilityError("all variants must use the same input fixture")
        if receipt.config.as_dict() != expected_config:
            raise ParameterStabilityError("all variants must use the same walk-forward config")
        baseline = next((case for case in receipt.cases if case.multiplier == 1.0), None)
        if baseline is None:
            raise ParameterStabilityError("each variant must contain the 1.0 baseline case")
        stressed = max(receipt.cases, key=lambda case: case.multiplier)
        evaluations.append(
            VariantEvaluation(
                variant_id=variant_id,
                receipt=receipt,
                baseline_metrics=baseline.aggregate_metrics,
                stress_metrics=stressed.aggregate_metrics,
            )
        )
    assert expected_input is not None
    metric_names = sorted(
        {metric for evaluation in evaluations for metric in evaluation.baseline_metrics}
    )
    summary: dict[str, dict[str, float | int | None]] = {}
    for metric in metric_names:
        summary[f"baseline.{metric}"] = _summarize_metric(
            [evaluation.baseline_metrics.get(metric) for evaluation in evaluations]
        )
        summary[f"stress.{metric}"] = _summarize_metric(
            [evaluation.stress_metrics.get(metric) for evaluation in evaluations]
        )
    return ParameterStabilityReceipt(
        family_id=family_id,
        input_sha256=expected_input,
        config=config,
        variants=tuple(evaluations),
        metric_summary=summary,
        worst_stress_multiplier=max(config.stress_multipliers),
    )


__all__ = [
    "PARAMETER_STABILITY_SCHEMA",
    "ParameterStabilityError",
    "ParameterStabilityReceipt",
    "VariantEvaluation",
    "run_parameter_stability",
]
