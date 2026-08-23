from __future__ import annotations

import math

import numpy as np
import pandas as pd


def performance_metrics(backtest: pd.DataFrame, periods_per_year: int = 365) -> dict[str, float | int]:
    returns = backtest["strategy_return"].dropna()
    if returns.empty:
        raise ValueError("Backtest has no returns.")

    total_return = float(backtest["equity"].iloc[-1] - 1.0)
    years = len(returns) / periods_per_year
    cagr = float(backtest["equity"].iloc[-1] ** (1 / years) - 1) if years > 0 else math.nan
    volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = (
        float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))
        if returns.std(ddof=0) > 0
        else math.nan
    )
    max_drawdown = float(backtest["drawdown"].min())
    entries = int(((backtest["position"] > 0) & (backtest["position"].shift(1, fill_value=0) == 0)).sum())
    time_in_market = float(backtest["position"].mean())
    annual_turnover = float(backtest["turnover"].sum() / years) if years > 0 else math.nan
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else math.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe_zero_rate": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "annual_turnover": annual_turnover,
        "entries": entries,
        "time_in_market": time_in_market,
    }
