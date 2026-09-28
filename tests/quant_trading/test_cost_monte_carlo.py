from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_trading.cost_monte_carlo import (
    CostMonteCarloConfig,
    CostMonteCarloError,
    run_cost_monte_carlo,
)


def _fixture(rows: int = 64) -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    close = pd.Series(100.0 + np.arange(rows) + 2.0 * np.sin(np.arange(rows)), index=index)
    signal = pd.Series((np.arange(rows) % 3 != 0).astype(float), index=index)
    return close, signal


def _config() -> CostMonteCarloConfig:
    return CostMonteCarloConfig(
        draws=24,
        seed=17,
        fee_multiplier_low=0.75,
        fee_multiplier_high=2.0,
        slippage_multiplier_low=0.5,
        slippage_multiplier_high=3.0,
    )


def test_cost_monte_carlo_is_deterministic_and_offline() -> None:
    close, signal = _fixture()
    first = run_cost_monte_carlo(
        close,
        signal,
        strategy_id="trend-costs",
        config=_config(),
    )
    second = run_cost_monte_carlo(
        close.copy(),
        signal.copy(),
        strategy_id="trend-costs",
        config=_config(),
    )

    assert first.as_dict() == second.as_dict()
    assert first.receipt_sha256 == second.receipt_sha256
    assert len(first.draws) == 24
    assert first.draws[0].draw_index == 0
    assert first.draws[-1].draw_index == 23
    assert first.metric_summary["total_return"]["draw_count"] == 24
    assert first.status == "PREP_ONLY"
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False


def test_cost_monte_carlo_changes_costs_but_keeps_signal_fixture_hash() -> None:
    close, signal = _fixture()
    low = run_cost_monte_carlo(
        close, signal, strategy_id="s", config=CostMonteCarloConfig(draws=8, seed=1)
    )
    high = run_cost_monte_carlo(
        close, signal, strategy_id="s", config=CostMonteCarloConfig(draws=8, seed=2)
    )

    assert low.input_sha256 == high.input_sha256
    assert low.draws != high.draws
    assert low.baseline_metrics == high.baseline_metrics
    assert all(draw.fee_bps + draw.slippage_bps < 10_000 for draw in low.draws)


def test_cost_monte_carlo_rejects_invalid_config_values() -> None:
    with pytest.raises(CostMonteCarloError, match="draws"):
        CostMonteCarloConfig(draws=0)
    with pytest.raises(CostMonteCarloError, match="high bounds"):
        CostMonteCarloConfig(fee_multiplier_low=3.0, fee_multiplier_high=1.0)
    with pytest.raises(CostMonteCarloError, match="greater than zero"):
        CostMonteCarloConfig(slippage_multiplier_low=0.0)


def test_cost_monte_carlo_rejects_cost_range_that_can_exceed_equity_floor() -> None:
    close, signal = _fixture()
    with pytest.raises(CostMonteCarloError, match="exceed 100%"):
        run_cost_monte_carlo(
            close,
            signal,
            strategy_id="unsafe",
            fee_bps=6_000,
            slippage_bps=1_000,
            config=CostMonteCarloConfig(
                draws=2,
                fee_multiplier_low=1.0,
                fee_multiplier_high=2.0,
                slippage_multiplier_low=1.0,
                slippage_multiplier_high=1.0,
            ),
        )


def test_cost_monte_carlo_rejects_tampered_receipt_hash() -> None:
    close, signal = _fixture()
    receipt = run_cost_monte_carlo(close, signal, strategy_id="s", config=_config())
    with pytest.raises(CostMonteCarloError, match="does not match"):
        replace(receipt, receipt_sha256="0" * 64)

