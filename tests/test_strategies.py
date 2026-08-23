import pandas as pd
import pytest

from quant_lab.strategies import (
    donchian_breakout_signal,
    moving_average_trend_signal,
    price_above_moving_average_signal,
    rsi_mean_reversion_signal,
    time_series_momentum_signal,
)


def test_trend_signal_waits_for_full_slow_window() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    signal = moving_average_trend_signal(close, fast=2, slow=4)

    assert signal.iloc[:3].tolist() == [0.0, 0.0, 0.0]
    assert signal.iloc[3:].tolist() == [1.0, 1.0]


def test_fast_window_must_be_smaller() -> None:
    with pytest.raises(ValueError, match="smaller"):
        moving_average_trend_signal(pd.Series([1.0, 2.0]), fast=2, slow=2)


def test_price_above_average_waits_for_full_window() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0])

    signal = price_above_moving_average_signal(close, window=3)

    assert signal.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_momentum_uses_trailing_return() -> None:
    close = pd.Series([100.0, 90.0, 110.0, 80.0])

    signal = time_series_momentum_signal(close, lookback=2)

    assert signal.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_donchian_channel_excludes_current_close() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 3.0, 2.0])

    signal = donchian_breakout_signal(close, entry_window=3, exit_window=2)

    assert signal.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]


def test_mean_reversion_signal_stays_in_spot_range() -> None:
    close = pd.Series([100.0] * 20 + [90.0, 80.0, 85.0, 95.0, 105.0])

    signal = rsi_mean_reversion_signal(
        close,
        window=3,
        entry_level=40,
        exit_level=60,
        regime_window=5,
    )

    assert signal.between(0, 1).all()
