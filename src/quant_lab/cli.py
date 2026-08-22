from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_lab.backtest import run_backtest
from quant_lab.data import download_binance_klines, load_candles, save_candles
from quant_lab.metrics import performance_metrics
from quant_lab.strategies import buy_and_hold_signal, moving_average_trend_signal

DATA_DIRECTORY = Path("data/raw")
REPORT_DIRECTORY = Path("reports")


def _data_path(symbol: str, interval: str = "1d") -> Path:
    return DATA_DIRECTORY / f"{symbol.upper()}_{interval}.csv"


def _print_metrics(name: str, metrics: dict[str, float | int]) -> None:
    print(f"\n{name}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:22s}: {value:,.4f}")
        else:
            print(f"  {key:22s}: {value}")


def download_command(args: argparse.Namespace) -> None:
    frame = download_binance_klines(args.symbol, args.start, args.end)
    destination = save_candles(frame, _data_path(args.symbol))
    print(f"Saved {len(frame):,} candles to {destination}")


def _period_metrics(result: pd.DataFrame, mask: pd.Series) -> dict[str, float | int]:
    period = result.loc[mask].copy()
    # Rebase equity so each period is independently interpretable.
    period["equity"] = (1.0 + period["strategy_return"]).cumprod()
    period["drawdown"] = period["equity"] / period["equity"].cummax() - 1.0
    return performance_metrics(period)


def backtest_command(args: argparse.Namespace) -> None:
    source = _data_path(args.symbol)
    if not source.exists():
        raise SystemExit(f"Missing {source}. Run the download command first.")

    candles = load_candles(source).set_index("open_time")
    close = candles["close"]
    split_index = max(1, min(len(close) - 1, int(len(close) * args.train_ratio)))
    train_mask = pd.Series(False, index=close.index)
    train_mask.iloc[:split_index] = True
    test_mask = ~train_mask

    strategies = {
        "buy_and_hold": buy_and_hold_signal(close),
        f"ma_{args.fast}_{args.slow}": moving_average_trend_signal(close, args.fast, args.slow),
    }
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for name, signal in strategies.items():
        result = run_backtest(close, signal, args.fee_bps, args.slippage_bps)
        result.to_csv(REPORT_DIRECTORY / f"{args.symbol.upper()}_{name}.csv")
        _print_metrics(f"{name} | train", _period_metrics(result, train_mask))
        _print_metrics(f"{name} | test (out-of-sample)", _period_metrics(result, test_mask))

    print(f"\nTrain ends: {close.index[split_index - 1].date()}")
    print(f"Test starts: {close.index[split_index].date()}")
    print(f"Reports saved to {REPORT_DIRECTORY}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transparent crypto quant research lab")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download public daily Binance candles")
    download.add_argument("--symbol", default="BTCUSDT")
    download.add_argument("--start", default="2018-01-01")
    download.add_argument("--end")
    download.set_defaults(handler=download_command)

    backtest = commands.add_parser("backtest", help="Compare a trend strategy with buy-and-hold")
    backtest.add_argument("--symbol", default="BTCUSDT")
    backtest.add_argument("--fast", type=int, default=50)
    backtest.add_argument("--slow", type=int, default=200)
    backtest.add_argument("--fee-bps", type=float, default=10.0)
    backtest.add_argument("--slippage-bps", type=float, default=5.0)
    backtest.add_argument("--train-ratio", type=float, default=0.7)
    backtest.set_defaults(handler=backtest_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "train_ratio", 0.7) <= 0 or getattr(args, "train_ratio", 0.7) >= 1:
        parser.error("--train-ratio must be between 0 and 1.")
    args.handler(args)


if __name__ == "__main__":
    main()
