from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _longest_loss_streak(returns: pd.Series) -> int:
    longest = current = 0
    for value in returns:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def performance_metrics(backtest: pd.DataFrame, periods_per_year: int = 365) -> dict[str, float | int]:
    returns = backtest["strategy_return"].dropna()
    if returns.empty:
        raise ValueError("Backtest has no returns.")

    total_return = float(backtest["equity"].iloc[-1] - 1.0)
    years = len(returns) / periods_per_year
    cagr = float(backtest["equity"].iloc[-1] ** (1 / years) - 1) if years > 0 else math.nan
    volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    downside = returns.clip(upper=0.0)
    downside_root_mean_square = float(np.sqrt(np.mean(np.square(downside))))
    downside_deviation = float(downside_root_mean_square * np.sqrt(periods_per_year))
    sharpe = (
        float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))
        if returns.std(ddof=0) > 0
        else math.nan
    )
    sortino = (
        float(returns.mean() / downside_root_mean_square * np.sqrt(periods_per_year))
        if downside_root_mean_square > 0
        else math.nan
    )
    max_drawdown = float(backtest["drawdown"].min())
    entries = int(((backtest["position"] > 0) & (backtest["position"].shift(1, fill_value=0) == 0)).sum())
    time_in_market = float(backtest["position"].mean())
    annual_turnover = float(backtest["turnover"].sum() / years) if years > 0 else math.nan
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else math.nan
    gains = float(returns[returns > 0].sum())
    losses = float(returns[returns < 0].sum())
    profit_factor = float(gains / abs(losses)) if losses < 0 else math.nan
    tail_cutoff = float(returns.quantile(0.05))
    expected_shortfall_95 = float(returns[returns <= tail_cutoff].mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": volatility,
        "downside_deviation": downside_deviation,
        "sharpe_zero_rate": sharpe,
        "sortino_zero_rate": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "expected_shortfall_95": expected_shortfall_95,
        "worst_period_return": float(returns.min()),
        "max_loss_streak": _longest_loss_streak(returns),
        "annual_turnover": annual_turnover,
        "entries": entries,
        "time_in_market": time_in_market,
    }
