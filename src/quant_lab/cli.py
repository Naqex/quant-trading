from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.backtest import run_backtest
from quant_lab.causal_universe import run_sleeve2_offline_smoke
from quant_lab.data import audit_ohlcv_directory, download_binance_klines, load_candles, save_candles
from quant_lab.metrics import performance_metrics
from quant_lab.strategies import buy_and_hold_signal, moving_average_trend_signal
from quant_lab.stress_report import (
    StressReportConfig,
    StressReportError,
    build_smoke_ohlcv,
    load_close_from_ohlcv,
    render_stress_report_markdown,
    run_stress_report,
)
from quant_trading.paper_soak import (
    PaperSoakError,
    PaperSoakHealthReport,
    PaperSoakReceipt,
    audit_paper_soak_receipt,
)

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


def _print_soak_health(report: PaperSoakHealthReport) -> None:
    print("\nPaper-soak health & reconciliation report (PREP_ONLY_OFFLINE)")
    print(f"  verdict                : {report.verdict}")
    print(f"  receipt_status         : {report.receipt_status}")
    print(f"  evidence_level         : {report.evidence_level}")
    print(f"  observed_days          : {report.observed_days} (gaps: {report.calendar_gap_days})")
    print(
        f"  healthy_reconciled     : {report.healthy_reconciled_days}/{report.observed_days}"
        f" (unhealthy: {report.unhealthy_days}, mismatch: {report.mismatch_days})"
    )
    print(f"  intents / fills        : {report.total_paper_intents} / {report.total_fills}")
    print(
        f"  restarts_passed        : {report.restart_checks_passed}/{report.restart_checks_total}"
        f" (duplicate_intents: {report.total_duplicate_intents})"
    )
    print(f"  realized_pnl_vnd       : {report.total_realized_pnl_vnd:,.2f}")
    print(f"  fees_vnd               : {report.total_fees_vnd:,.2f}")
    if report.max_observed_slippage_bps is not None:
        print(
            f"  slippage_bps (avg/max) : {report.mean_slippage_bps:.2f}"
            f" / {report.max_observed_slippage_bps:.2f}"
        )
    print(
        f"  capabilities           : execution={report.execution_capability},"
        f" promotion={report.promotion_allowed}, broker={report.broker_access}"
    )
    if report.blockers:
        print(f"  blockers               : {', '.join(report.blockers)}")
    if report.warnings:
        print(f"  warnings               : {', '.join(report.warnings)}")
    print(f"  receipt_sha256         : {report.receipt_sha256}")
    print(f"  report_sha256          : {report.report_sha256}")


def soak_check_command(args: argparse.Namespace) -> None:
    if bool(args.receipt) == bool(args.smoke):
        raise SystemExit("Specify exactly one of --receipt PATH or --smoke.")

    try:
        if args.smoke:
            smoke = run_sleeve2_offline_smoke(
                strategy=args.strategy,
                soak_days=args.days,
                seed=args.seed,
            )
            receipt: PaperSoakReceipt = smoke["receipt"]
            if args.write_receipt is not None:
                destination = Path(args.write_receipt)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(receipt.as_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            report = audit_paper_soak_receipt(
                receipt,
                max_slippage_bps=args.max_slippage_bps,
                max_calendar_gap_days=args.max_gap_days,
            )
            if args.json:
                payload = {
                    "mode": "offline_smoke",
                    "strategy": smoke["strategy"],
                    "health_report": report.as_dict(),
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_metrics(f"{smoke['strategy']} | offline synthetic smoke", smoke["backtest_metrics"])
                _print_soak_health(report)
        else:
            source = Path(args.receipt)
            if not source.exists():
                raise SystemExit(f"Missing paper-soak receipt file: {source}")
            raw = json.loads(source.read_text(encoding="utf-8"))
            receipt = PaperSoakReceipt.from_mapping(raw)
            report = audit_paper_soak_receipt(
                receipt,
                max_slippage_bps=args.max_slippage_bps,
                max_calendar_gap_days=args.max_gap_days,
            )
            if args.json:
                print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
            else:
                _print_soak_health(report)
    except (json.JSONDecodeError, PaperSoakError, ValueError) as exc:
        raise SystemExit(f"Paper-soak check failed: {exc}") from exc

    if report.verdict == "REJECTED":
        raise SystemExit(f"Paper-soak receipt REJECTED: {', '.join(report.blockers)}")


def _print_data_audit(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"\nOHLCV Data Audit ({report['scope']}) -> {report['status']}")
    print(
        f"  files_scanned : {summary['files_scanned']}"
        f" (passed={summary['passed_symbols']}, failed={summary['failed_symbols']})"
    )
    print(
        f"  total_rows    : {summary['total_rows']:,}"
        f" | total_gaps={summary['total_gaps']}"
        f" | zero_volume_bars={summary['total_zero_volume_bars']}"
    )
    if report["errors"]:
        print(f"  errors        : {', '.join(report['errors'])}")
    for item in report["symbols"]:
        err_suffix = f" | errors={','.join(item['errors'])}" if item["errors"] else ""
        print(
            f"  [{item['status']:4s}] {item['symbol']:10s} ({item['timeframe']}) "
            f"rows={item['rows']} first={item['first_utc']} last={item['last_utc']} "
            f"dup={item['duplicate_timestamps']} order={item['monotonic_violations']} "
            f"gaps={item['missing_candle_gaps']} nonpos={item['non_positive_prices']} "
            f"nonfin={item['non_finite_prices']} ohlc={item['ohlc_consistency_violations']} "
            f"zvol={item['zero_volume_bars']}{err_suffix}"
        )
    print(f"  report_sha256 : {report['report_sha256']}")


def data_audit_command(args: argparse.Namespace) -> None:
    try:
        report = audit_ohlcv_directory(args.dir, timeframe=args.timeframe)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Data audit error: {exc}") from exc

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        if report["status"] != "PASS":
            raise SystemExit(1)
    else:
        _print_data_audit(report)
        if report["status"] != "PASS":
            raise SystemExit(
                f"OHLCV data audit FAILED ({report['summary']['failed_symbols']} failed symbol(s))."
            )


def _print_stress_report(report: dict[str, Any]) -> None:
    wfo = report["purged_walk_forward"]
    strat = wfo["stitched_strategy_metrics"]
    fee = report["fee_sensitivity"]
    plat = report["parameter_plateau"]
    boot = report["bootstrap"]
    dd = boot["tail_drawdown"]
    print(f"\nQuant Lab Offline Stress Report ({report['scope']}) -> {report['status']}")
    print(
        f"  strategy         : {plat['center_variant_id']}"
        f" | folds={wfo['fold_count']} (purge={wfo['purge_bars']})"
    )
    print(
        f"  stitched OOS     : cagr={strat['cagr']} sharpe={strat['sharpe_zero_rate']}"
        f" max_dd={strat['max_drawdown']} turnover={strat['annual_turnover']}"
    )
    print(
        f"  fee breakeven    : {fee['cost_breakeven_bps']:.2f} bps"
        f" (threshold >= {fee['min_cost_breakeven_bps']:.2f} bps)"
    )
    print(
        f"  plateau (+-20%)  : {plat['positive_variant_count']}/{plat['variant_count']} positive"
        f" (med_sharpe={plat['sharpe_median']}, retention={plat['retention_ratio']})"
    )
    print(
        f"  bootstrap 95% CI : sharpe=[{boot['sharpe_ci_95']['lo']}, {boot['sharpe_ci_95']['hi']}]"
        f" cagr=[{boot['cagr_ci_95']['lo']}, {boot['cagr_ci_95']['hi']}]"
    )
    print(
        f"  tail drawdown    : obs_dd={dd['observed_max_drawdown']}"
        f" worst5_var={dd['worst_5pct_drawdown_var']} worst5_cvar={dd['worst_5pct_drawdown_cvar']}"
    )
    if report["blockers"]:
        print(f"  blockers         : {', '.join(report['blockers'])}")
    print(f"  report_sha256    : {report['report_sha256']}")


def stress_report_command(args: argparse.Namespace) -> None:
    if bool(args.input) == bool(args.smoke):
        raise SystemExit("Specify exactly one of --input PATH or --smoke.")

    try:
        config = StressReportConfig(
            strategy=args.strategy,
            fast=args.fast,
            slow=args.slow,
            entry_window=args.entry,
            exit_window=args.exit,
            lookback=args.lookback,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            purge_bars=args.purge_bars,
            slippage_bps=args.slippage_bps,
            bootstrap_resamples=args.bootstrap_resamples,
            block_length=args.block_length,
            seed=args.seed,
            min_cost_breakeven_bps=args.min_breakeven_bps,
            min_plateau_pass_rate=args.min_plateau_pass_rate,
            min_plateau_median_sharpe=args.min_plateau_sharpe,
        )
        if args.smoke:
            smoke_df = build_smoke_ohlcv(seed=args.seed)
            close = pd.Series(
                smoke_df["close"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(smoke_df["open_time"]),
                name="close",
            )
            report = run_stress_report(close, config=config, input_source=f"offline_smoke:seed={args.seed}")
        else:
            close, data_sha = load_close_from_ohlcv(args.input)
            report = run_stress_report(
                close,
                config=config,
                input_source=f"file:{Path(args.input).name}",
                input_sha256=data_sha,
            )
    except (FileNotFoundError, StressReportError, ValueError) as exc:
        raise SystemExit(f"Stress report error: {exc}") from exc

    if args.out_md is not None:
        out_path = Path(args.out_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_stress_report_markdown(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        if report["status"] != "PASS":
            raise SystemExit(1)
    else:
        _print_stress_report(report)
        if report["status"] != "PASS":
            raise SystemExit(f"Stress report FAILED: {', '.join(report['blockers'])}")


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

    soak_check = commands.add_parser(
        "soak-check",
        help="Audit a paper-soak receipt or run an offline Sleeve #2 smoke check (zero API keys)",
    )
    soak_check.add_argument("--receipt", help="Path to a PaperSoakReceipt JSON file to audit")
    soak_check.add_argument(
        "--smoke",
        action="store_true",
        help="Run deterministic offline synthetic Sleeve #2 smoke pipeline",
    )
    soak_check.add_argument("--strategy", choices=("f4", "f3"), default="f4")
    soak_check.add_argument("--days", type=int, default=5)
    soak_check.add_argument("--seed", type=int, default=2026)
    soak_check.add_argument("--max-slippage-bps", type=float, default=None)
    soak_check.add_argument("--max-gap-days", type=int, default=0)
    soak_check.add_argument("--write-receipt", help="Optional JSON path to save generated smoke receipt")
    soak_check.add_argument("--json", action="store_true", help="Emit canonical JSON output")
    soak_check.set_defaults(handler=soak_check_command)

    data_audit = commands.add_parser(
        "data-audit",
        help="Audit offline CSV/Parquet OHLCV directories for gaps and data quality anomalies",
    )
    data_audit.add_argument("--dir", required=True, help="Directory containing CSV/Parquet OHLCV files")
    data_audit.add_argument(
        "--timeframe",
        choices=("1d", "4h", "1h"),
        default=None,
        help="Expected candle timeframe (defaults to filename suffix or 1d)",
    )
    data_audit.add_argument("--json", action="store_true", help="Emit canonical JSON report")
    data_audit.set_defaults(handler=data_audit_command)

    stress = commands.add_parser(
        "stress-report",
        help="Run purged WFO, 10/20/35 bps fee ladder, +-20% plateau, and block-bootstrap CI offline",
    )
    stress.add_argument("--input", help="Path to local CSV/Parquet OHLCV file")
    stress.add_argument(
        "--smoke",
        action="store_true",
        help="Run on deterministic offline synthetic fixture (zero network/API keys)",
    )
    stress.add_argument("--strategy", choices=("ma", "donchian", "momentum"), default="ma")
    stress.add_argument("--fast", type=int, default=20)
    stress.add_argument("--slow", type=int, default=60)
    stress.add_argument("--entry", type=int, default=40)
    stress.add_argument("--exit", type=int, default=20)
    stress.add_argument("--lookback", type=int, default=60)
    stress.add_argument("--train-bars", type=int, default=180)
    stress.add_argument("--test-bars", type=int, default=60)
    stress.add_argument("--purge-bars", type=int, default=5)
    stress.add_argument("--slippage-bps", type=float, default=5.0)
    stress.add_argument("--min-breakeven-bps", type=float, default=35.0)
    stress.add_argument("--min-plateau-pass-rate", type=float, default=0.60)
    stress.add_argument("--min-plateau-sharpe", type=float, default=0.0)
    stress.add_argument("--bootstrap-resamples", type=int, default=1_000)
    stress.add_argument("--block-length", type=int, default=7)
    stress.add_argument("--seed", type=int, default=20261125)
    stress.add_argument("--out-md", help="Optional Markdown summary destination path")
    stress.add_argument("--json", action="store_true", help="Emit canonical JSON report")
    stress.set_defaults(handler=stress_report_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "train_ratio", 0.7) <= 0 or getattr(args, "train_ratio", 0.7) >= 1:
        parser.error("--train-ratio must be between 0 and 1.")
    args.handler(args)


if __name__ == "__main__":
    main()
