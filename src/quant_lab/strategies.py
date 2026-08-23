from __future__ import annotations

import pandas as pd


def buy_and_hold_signal(close: pd.Series) -> pd.Series:
    """Stay invested after the first observable close."""
    return pd.Series(1.0, index=close.index, name="signal")


def moving_average_trend_signal(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long when the fast moving average is above the slow one."""
    if fast <= 0 or slow <= 0:
        raise ValueError("Moving-average windows must be positive.")
    if fast >= slow:
        raise ValueError("The fast window must be smaller than the slow window.")

    fast_average = close.rolling(fast, min_periods=fast).mean()
    slow_average = close.rolling(slow, min_periods=slow).mean()
    signal = (fast_average > slow_average).astype(float)
    signal[slow_average.isna()] = 0.0
    return signal.rename("signal")


def price_above_moving_average_signal(close: pd.Series, window: int = 200) -> pd.Series:
    """Long when price is above its long-term moving average."""
    if window <= 0:
        raise ValueError("Moving-average window must be positive.")

    average = close.rolling(window, min_periods=window).mean()
    signal = (close > average).astype(float)
    signal[average.isna()] = 0.0
    return signal.rename("signal")


def time_series_momentum_signal(close: pd.Series, lookback: int = 180) -> pd.Series:
    """Long after a positive trailing return; otherwise hold cash."""
    if lookback <= 0:
        raise ValueError("Momentum lookback must be positive.")

    trailing_return = close.pct_change(lookback, fill_method=None)
    signal = (trailing_return > 0).astype(float)
    signal[trailing_return.isna()] = 0.0
    return signal.rename("signal")


def donchian_breakout_signal(
    close: pd.Series,
    entry_window: int = 100,
    exit_window: int = 50,
) -> pd.Series:
    """Enter on a prior-channel high and exit on a prior-channel low."""
    if entry_window <= 0 or exit_window <= 0:
        raise ValueError("Donchian windows must be positive.")
    if exit_window >= entry_window:
        raise ValueError("The exit window must be smaller than the entry window.")

    prior_high = close.shift(1).rolling(entry_window, min_periods=entry_window).max()
    prior_low = close.shift(1).rolling(exit_window, min_periods=exit_window).min()
    signal = pd.Series(0.0, index=close.index, name="signal")
    invested = False

    for index in close.index:
        price = close.loc[index]
        if not invested and pd.notna(prior_high.loc[index]) and price > prior_high.loc[index]:
            invested = True
        elif invested and pd.notna(prior_low.loc[index]) and price < prior_low.loc[index]:
            invested = False
        signal.loc[index] = float(invested)

    return signal


def rsi_mean_reversion_signal(
    close: pd.Series,
    window: int = 14,
    entry_level: float = 30.0,
    exit_level: float = 55.0,
    regime_window: int = 200,
) -> pd.Series:
    """Buy oversold pullbacks only while price remains in a long-term uptrend."""
    if window <= 0 or regime_window <= 0:
        raise ValueError("RSI and regime windows must be positive.")
    if not 0 <= entry_level < exit_level <= 100:
        raise ValueError("RSI levels must satisfy 0 <= entry < exit <= 100.")

    change = close.diff()
    average_gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = -change.clip(upper=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - 100 / (1 + relative_strength)
    regime_average = close.rolling(regime_window, min_periods=regime_window).mean()
    uptrend = close > regime_average

    signal = pd.Series(0.0, index=close.index, name="signal")
    invested = False
    for index in close.index:
        if invested and (not uptrend.loc[index] or rsi.loc[index] > exit_level):
            invested = False
        elif not invested and uptrend.loc[index] and rsi.loc[index] < entry_level:
            invested = True
        signal.loc[index] = float(invested)

    return signal
