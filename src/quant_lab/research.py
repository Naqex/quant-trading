from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from quant_lab.backtest import apply_volatility_target, run_backtest
from quant_lab.data import load_candles
from quant_lab.metrics import performance_metrics
from quant_lab.strategies import (
    buy_and_hold_signal,
    donchian_breakout_signal,
    moving_average_trend_signal,
    price_above_moving_average_signal,
    rsi_mean_reversion_signal,
    time_series_momentum_signal,
)

SignalFactory = Callable[[pd.Series], pd.Series]

DATA_DIRECTORY = Path("data/raw")
REPORT_DIRECTORY = Path("reports/research")

STRATEGIES: dict[str, SignalFactory] = {
    "buy_and_hold": buy_and_hold_signal,
    "price_sma_200": lambda close: price_above_moving_average_signal(close, 200),
    "ma_50_200": lambda close: moving_average_trend_signal(close, 50, 200),
    "momentum_180": lambda close: time_series_momentum_signal(close, 180),
    "donchian_100_50": lambda close: donchian_breakout_signal(close, 100, 50),
    "rsi_reversion": lambda close: rsi_mean_reversion_signal(close, 14, 30, 55, 200),
}

ROBUSTNESS_STRATEGIES: dict[str, SignalFactory] = {
    "price_sma_150": lambda close: price_above_moving_average_signal(close, 150),
    "price_sma_200": STRATEGIES["price_sma_200"],
    "price_sma_250": lambda close: price_above_moving_average_signal(close, 250),
    "ma_30_150": lambda close: moving_average_trend_signal(close, 30, 150),
    "ma_50_200": STRATEGIES["ma_50_200"],
    "ma_75_250": lambda close: moving_average_trend_signal(close, 75, 250),
    "momentum_90": lambda close: time_series_momentum_signal(close, 90),
    "momentum_180": STRATEGIES["momentum_180"],
    "momentum_270": lambda close: time_series_momentum_signal(close, 270),
    "donchian_60_30": lambda close: donchian_breakout_signal(close, 60, 30),
    "donchian_100_50": STRATEGIES["donchian_100_50"],
    "donchian_150_75": lambda close: donchian_breakout_signal(close, 150, 75),
}


def _close_series(symbol: str) -> pd.Series:
    path = DATA_DIRECTORY / f"{symbol.upper()}_1d.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Download it first.")
    return load_candles(path).set_index("open_time")["close"]


def _rebase_period(backtest: pd.DataFrame) -> pd.DataFrame:
    period = backtest.copy()
    period["equity"] = (1.0 + period["strategy_return"]).cumprod()
    period["drawdown"] = period["equity"] / period["equity"].cummax() - 1.0
    return period


def evaluate_strategy(
    symbol: str,
    name: str,
    factory: SignalFactory,
    test_start: str,
    fee_bps: float,
    slippage_bps: float,
    volatility_target: float | None = None,
) -> list[dict[str, object]]:
    close = _close_series(symbol)
    signal = factory(close)
    if volatility_target is not None:
        signal = apply_volatility_target(close, signal, volatility_target, lookback=30)
        name = f"{name}_vol{int(volatility_target * 100)}"

    result = run_backtest(close, signal, fee_bps, slippage_bps)
    rows: list[dict[str, object]] = []
    masks = {
        "train": close.index < pd.Timestamp(test_start, tz="UTC"),
        "test": close.index >= pd.Timestamp(test_start, tz="UTC"),
    }
    for period_name, mask in masks.items():
        period = _rebase_period(result.loc[mask])
        if period.empty:
            continue
        rows.append(
            {
                "symbol": symbol.upper(),
                "strategy": name,
                "period": period_name,
                "start": period.index.min().date().isoformat(),
                "end": period.index.max().date().isoformat(),
                **performance_metrics(period),
            }
        )
    return rows


def evaluate_portfolio(
    symbols: list[str],
    name: str,
    factory: SignalFactory,
    test_start: str,
    fee_bps: float,
    slippage_bps: float,
    volatility_target: float | None = None,
) -> dict[str, object]:
    sleeve_results: list[pd.DataFrame] = []
    for symbol in symbols:
        close = _close_series(symbol)
        signal = factory(close)
        if volatility_target is not None:
            signal = apply_volatility_target(close, signal, volatility_target, lookback=30)
        result = run_backtest(close, signal, fee_bps, slippage_bps)
        sleeve_results.append(result.loc[result.index >= pd.Timestamp(test_start, tz="UTC")])

    common_index = sleeve_results[0].index
    for result in sleeve_results[1:]:
        common_index = common_index.intersection(result.index)
    if common_index.empty:
        raise ValueError("Portfolio sleeves have no common dates.")

    sleeve_equities = pd.concat(
        [(1 + result.loc[common_index, "strategy_return"]).cumprod() for result in sleeve_results],
        axis=1,
    )
    portfolio_equity = sleeve_equities.mean(axis=1)
    portfolio_return = portfolio_equity.pct_change().fillna(portfolio_equity.iloc[0] - 1)
    portfolio = pd.DataFrame(
        {
            "strategy_return": portfolio_return,
            "equity": portfolio_equity,
            "drawdown": portfolio_equity / portfolio_equity.cummax() - 1,
            "position": pd.concat(
                [result.loc[common_index, "position"] for result in sleeve_results], axis=1
            ).mean(axis=1),
            "turnover": pd.concat(
                [result.loc[common_index, "turnover"] for result in sleeve_results], axis=1
            ).mean(axis=1),
        }
    )
    if volatility_target is not None:
        name = f"{name}_vol{int(volatility_target * 100)}"
    return {
        "strategy": name,
        "start": common_index.min().date().isoformat(),
        "end": common_index.max().date().isoformat(),
        **performance_metrics(portfolio),
    }


def yearly_returns(
    symbol: str,
    name: str,
    factory: SignalFactory,
    fee_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    close = _close_series(symbol)
    result = run_backtest(close, factory(close), fee_bps, slippage_bps)
    annual = (1 + result["strategy_return"]).groupby(result.index.year).prod() - 1
    return annual.rename(name).to_frame()


def run_research(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for symbol in args.symbols:
        for name, factory in STRATEGIES.items():
            rows.extend(
                evaluate_strategy(
                    symbol,
                    name,
                    factory,
                    args.test_start,
                    args.fee_bps,
                    args.slippage_bps,
                )
            )
        rows.extend(
            evaluate_strategy(
                symbol,
                "ma_50_200",
                STRATEGIES["ma_50_200"],
                args.test_start,
                args.fee_bps,
                args.slippage_bps,
                volatility_target=0.30,
            )
        )

    portfolio_specs: list[tuple[str, SignalFactory, float | None]] = [
        (name, factory, None) for name, factory in STRATEGIES.items()
    ] + [
        ("ma_50_200", STRATEGIES["ma_50_200"], 0.30),
        ("donchian_100_50", STRATEGIES["donchian_100_50"], 0.30),
    ]
    portfolio = pd.DataFrame(
        [
            evaluate_portfolio(
                args.symbols,
                name,
                factory,
                args.test_start,
                args.fee_bps,
                args.slippage_bps,
                volatility_target,
            )
            for name, factory, volatility_target in portfolio_specs
        ]
    ).sort_values(["calmar", "sharpe_zero_rate"], ascending=False)

    summary = pd.DataFrame(rows)
    robustness_rows: list[dict[str, object]] = []
    for symbol in args.symbols:
        for name, factory in ROBUSTNESS_STRATEGIES.items():
            robustness_rows.extend(
                evaluate_strategy(
                    symbol,
                    name,
                    factory,
                    args.test_start,
                    args.fee_bps,
                    args.slippage_bps,
                )
            )
    robustness = pd.DataFrame(robustness_rows)

    annual_frames: list[pd.DataFrame] = []
    for symbol in args.symbols:
        for name in ["buy_and_hold", "ma_50_200", "momentum_180", "donchian_100_50"]:
            frame = yearly_returns(symbol, name, STRATEGIES[name], args.fee_bps, args.slippage_bps)
            frame.columns = pd.MultiIndex.from_tuples([(symbol.upper(), name)])
            annual_frames.append(frame)
    annual = pd.concat(annual_frames, axis=1).sort_index()

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIRECTORY / "strategy_summary.csv", index=False)
    robustness.to_csv(REPORT_DIRECTORY / "parameter_robustness.csv", index=False)
    annual.to_csv(REPORT_DIRECTORY / "yearly_returns.csv")
    portfolio.to_csv(REPORT_DIRECTORY / "portfolio_test.csv", index=False)

    test = summary[summary["period"] == "test"].copy()
    aggregate = (
        test.groupby("strategy")
        .agg(
            median_cagr=("cagr", "median"),
            median_sharpe=("sharpe_zero_rate", "median"),
            median_max_drawdown=("max_drawdown", "median"),
            worst_max_drawdown=("max_drawdown", "min"),
            median_calmar=("calmar", "median"),
            assets=("symbol", "count"),
        )
        .sort_values(["median_calmar", "median_sharpe"], ascending=False)
    )
    aggregate.to_csv(REPORT_DIRECTORY / "test_aggregate.csv")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(aggregate.round(4).to_string())
        print("\nEqual-capital portfolio | test")
        print(
            portfolio[
                ["strategy", "cagr", "sharpe_zero_rate", "max_drawdown", "calmar", "time_in_market"]
            ].round(4).to_string(index=False)
        )
    print(f"\nResearch artifacts saved to {REPORT_DIRECTORY}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the predefined strategy comparison")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    return parser


def main() -> None:
    run_research(build_parser().parse_args())


if __name__ == "__main__":
    main()
