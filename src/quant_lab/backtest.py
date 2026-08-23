from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    """Backtest a long/cash spot strategy with next-bar execution."""
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("Trading costs cannot be negative.")
    if not close.index.equals(signal.index):
        raise ValueError("Close and signal indexes must match.")
    if not signal.fillna(0).between(0, 1).all():
        raise ValueError("Spot strategy signal must stay between 0 and 1.")

    asset_return = close.astype(float).pct_change().fillna(0.0)
    # A signal calculated using today's close can only be held from the next bar.
    position = signal.fillna(0.0).shift(1, fill_value=0.0)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    cost_rate = (fee_bps + slippage_bps) / 10_000
    strategy_return = position * asset_return - turnover * cost_rate
    equity = (1.0 + strategy_return).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    return pd.DataFrame(
        {
            "close": close.astype(float),
            "signal": signal.astype(float),
            "position": position,
            "asset_return": asset_return,
            "turnover": turnover,
            "strategy_return": strategy_return,
            "equity": equity,
            "drawdown": drawdown,
        }
    ).replace([np.inf, -np.inf], np.nan)


def apply_volatility_target(
    close: pd.Series,
    signal: pd.Series,
    target_annual_volatility: float = 0.30,
    lookback: int = 30,
) -> pd.Series:
    """Scale a spot signal down when recent volatility exceeds the target."""
    if target_annual_volatility <= 0 or lookback <= 1:
        raise ValueError("Volatility target must be positive and lookback greater than one.")
    if not close.index.equals(signal.index):
        raise ValueError("Close and signal indexes must match.")

    realized_volatility = close.pct_change(fill_method=None).rolling(
        lookback,
        min_periods=lookback,
    ).std(ddof=0) * np.sqrt(365)
    scale = (target_annual_volatility / realized_volatility).clip(lower=0.0, upper=1.0)
    return (signal * scale.fillna(0.0)).rename("signal")
