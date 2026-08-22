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
