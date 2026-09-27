import numpy as np
import pandas as pd
import pytest

from quant_lab.backtest import apply_volatility_target, run_backtest
from quant_lab.metrics import performance_metrics


def test_signal_is_executed_on_next_bar() -> None:
    close = pd.Series([100.0, 110.0, 121.0])
    signal = pd.Series([0.0, 1.0, 1.0])

    result = run_backtest(close, signal, fee_bps=0, slippage_bps=0)

    assert result["position"].tolist() == [0.0, 0.0, 1.0]
    assert result["strategy_return"].tolist() == pytest.approx([0.0, 0.0, 0.1])


def test_cost_is_charged_when_position_changes() -> None:
    close = pd.Series([100.0, 100.0, 100.0, 100.0])
    signal = pd.Series([1.0, 1.0, 0.0, 0.0])

    result = run_backtest(close, signal, fee_bps=10, slippage_bps=5)

    assert result["turnover"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result["strategy_return"].sum() == pytest.approx(-0.003)


def test_rejects_leverage_in_spot_lab() -> None:
    close = pd.Series([100.0, 101.0])
    signal = pd.Series([0.0, 1.2])

    with pytest.raises(ValueError, match="between 0 and 1"):
        run_backtest(close, signal)


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_trading_costs(cost: float) -> None:
    close = pd.Series([100.0, 101.0])
    signal = pd.Series([0.0, 1.0])

    with pytest.raises(ValueError, match="finite"):
        run_backtest(close, signal, fee_bps=cost)


def test_rejects_costs_that_can_make_equity_negative() -> None:
    close = pd.Series([100.0, 101.0])
    signal = pd.Series([0.0, 1.0])

    with pytest.raises(ValueError, match="less than 100%"):
        run_backtest(close, signal, fee_bps=9_000, slippage_bps=1_000)


@pytest.mark.parametrize("bad_close", [pd.Series([100.0, float("nan")]), pd.Series([100.0, 0.0])])
def test_rejects_invalid_close_prices(bad_close: pd.Series) -> None:
    signal = pd.Series([0.0, 1.0])

    with pytest.raises(ValueError, match="strictly positive"):
        run_backtest(bad_close, signal)


def test_rejects_unsorted_or_duplicate_dates() -> None:
    signal = pd.Series([0.0, 1.0], index=[1, 0])
    close = pd.Series([100.0, 101.0], index=[1, 0])

    with pytest.raises(ValueError, match="unique and sorted"):
        run_backtest(close, signal)


def test_volatility_target_never_adds_leverage() -> None:
    close = pd.Series([100.0, 120.0, 90.0, 130.0, 80.0])
    signal = pd.Series(1.0, index=close.index)

    sized = apply_volatility_target(close, signal, target_annual_volatility=0.2, lookback=2)

    assert sized.between(0, 1).all()
    assert sized.iloc[:2].tolist() == [0.0, 0.0]


def test_performance_metrics_expose_downside_and_tail_risk() -> None:
    close = pd.Series([100.0, 102.0, 100.0, 99.0, 101.0, 100.0])
    signal = pd.Series([1.0] * len(close))
    result = run_backtest(close, signal, fee_bps=0, slippage_bps=0)

    metrics = performance_metrics(result, periods_per_year=1)

    assert metrics["downside_deviation"] > 0
    assert np.isfinite(metrics["sortino_zero_rate"])
    assert metrics["expected_shortfall_95"] <= 0
    assert metrics["worst_period_return"] < 0
    assert metrics["max_loss_streak"] == 2
