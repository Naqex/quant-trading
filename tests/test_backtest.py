import pandas as pd
import pytest

from quant_lab.backtest import apply_volatility_target, run_backtest


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


def test_volatility_target_never_adds_leverage() -> None:
    close = pd.Series([100.0, 120.0, 90.0, 130.0, 80.0])
    signal = pd.Series(1.0, index=close.index)

    sized = apply_volatility_target(close, signal, target_annual_volatility=0.2, lookback=2)

    assert sized.between(0, 1).all()
    assert sized.iloc[:2].tolist() == [0.0, 0.0]
