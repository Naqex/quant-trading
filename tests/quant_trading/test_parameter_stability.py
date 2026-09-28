from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_trading.parameter_stability import (
    ParameterStabilityError,
    run_parameter_stability,
)
from quant_trading.walk_forward import WalkForwardConfig


def _close(rows: int = 80) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    # An oscillating fixture gives the variants distinct, deterministic returns.
    values = 100.0 + np.arange(rows, dtype=float) + 2.0 * np.sin(np.arange(rows))
    return pd.Series(values, index=index, name="close")


def _config() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_bars=20,
        test_bars=10,
        step_bars=10,
        minimum_folds=3,
        stress_multipliers=(1.0, 2.0),
    )


def _always_on(history: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=history.index)


def _alternating(history: pd.Series) -> pd.Series:
    return pd.Series(np.arange(len(history)) % 2, index=history.index, dtype=float)


def test_parameter_stability_is_sorted_deterministic_and_offline() -> None:
    variants = {"z-always": _always_on, "a-alternating": _alternating}
    first = run_parameter_stability(
        _close(), variants, family_id="ma-neighbourhood", config=_config()
    )
    second = run_parameter_stability(
        _close(), dict(reversed(list(variants.items()))), family_id="ma-neighbourhood", config=_config()
    )

    assert [variant.variant_id for variant in first.variants] == ["a-alternating", "z-always"]
    assert first.as_dict() == second.as_dict()
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False
    assert first.status == "PREP_ONLY"
    assert first.metric_summary["baseline.total_return"]["variant_count"] == 2
    assert first.metric_summary["stress.total_return"]["range"] is not None


def test_parameter_stability_reuses_same_fixture_and_cost_cases() -> None:
    receipt = run_parameter_stability(
        _close(), {"fast": _alternating, "slow": _always_on}, family_id="family", config=_config()
    )
    assert receipt.worst_stress_multiplier == 2.0
    assert receipt.input_sha256 == receipt.variants[0].receipt.input_sha256
    for variant in receipt.variants:
        assert variant.receipt.base_fee_bps == 10.0
        assert variant.receipt.base_slippage_bps == 5.0
        assert [case.multiplier for case in variant.receipt.cases] == [1.0, 2.0]
        assert variant.receipt.strategy_id.startswith("family:")


@pytest.mark.parametrize(
    "variants, error",
    [
        ({}, "at least two"),
        ({"only": _always_on}, "at least two"),
        ({"bad": None, "ok": _always_on}, "callable"),
    ],
)
def test_parameter_stability_rejects_malformed_variant_sets(variants: dict, error: str) -> None:
    with pytest.raises(ParameterStabilityError, match=error):
        run_parameter_stability(_close(), variants, family_id="bad", config=_config())


def test_parameter_stability_rejects_tampered_receipt_hash() -> None:
    receipt = run_parameter_stability(
        _close(), {"a": _always_on, "b": _alternating}, family_id="family", config=_config()
    )
    with pytest.raises(ParameterStabilityError, match="does not match"):
        replace(receipt, receipt_sha256="0" * 64)
