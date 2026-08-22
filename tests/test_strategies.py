import pandas as pd
import pytest

from quant_lab.strategies import moving_average_trend_signal


def test_trend_signal_waits_for_full_slow_window() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    signal = moving_average_trend_signal(close, fast=2, slow=4)

    assert signal.iloc[:3].tolist() == [0.0, 0.0, 0.0]
    assert signal.iloc[3:].tolist() == [1.0, 1.0]


def test_fast_window_must_be_smaller() -> None:
    with pytest.raises(ValueError, match="smaller"):
        moving_average_trend_signal(pd.Series([1.0, 2.0]), fast=2, slow=2)
