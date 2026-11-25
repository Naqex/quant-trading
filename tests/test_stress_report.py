from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_lab.cli import build_parser
from quant_lab.stress_report import (
    StressReportConfig,
    StressReportError,
    build_smoke_ohlcv,
    render_stress_report_markdown,
    run_stress_report,
)


def _smoke_close(periods: int = 540, seed: int = 20261125) -> pd.Series:
    df = build_smoke_ohlcv(periods=periods, seed=seed)
    return pd.Series(
        df["close"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["open_time"]),
        name="close",
    )


def _mean_reverting_chop_close(periods: int = 420, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=periods, freq="1D", tz="UTC")
    # High-frequency alternating chop forces trend followers into whipsaw losses
    sign = np.where(np.arange(periods) % 4 < 2, 1.0, -1.0)
    close = 100.0 * np.exp(0.03 * sign + rng.normal(0.0, 0.005, size=periods))
    return pd.Series(close, index=idx, name="close")


@pytest.mark.parametrize("strategy", ["ma", "donchian", "momentum"])
def test_stress_report_passes_deterministically_across_strategies_and_locks_capabilities(
    strategy: str,
) -> None:
    close = _smoke_close()
    cfg = StressReportConfig(strategy=strategy, bootstrap_resamples=300, seed=20261125)
    first = run_stress_report(close, config=cfg, input_source="smoke")
    second = run_stress_report(close, config=cfg, input_source="smoke")

    assert first["status"] == "PASS"
    assert first["report_sha256"] == second["report_sha256"]
    assert len(first["report_sha256"]) == 64
    assert first["execution_capability"] is False
    assert first["promotion_allowed"] is False
    assert first["provider_access"] is False
    assert first["broker_access"] is False
    assert first["oauth_access"] is False
    assert first["blockers"] == []

    # Purged walk-forward invariants
    wfo = first["purged_walk_forward"]
    assert wfo["fold_count"] >= 2
    assert wfo["purge_bars"] == 5
    for fold in wfo["folds"]:
        assert fold["train_end"] < fold["test_start"]

    # Fee ladder (10 / 20 / 35 bps) monotonic degradation + breakeven >= 35 bps
    tiers = first["fee_sensitivity"]["tiers"]
    assert [t["fee_bps"] for t in tiers] == [10.0, 20.0, 35.0]
    assert tiers[0]["stitched_total_return"] >= tiers[1]["stitched_total_return"] - 1e-12
    assert tiers[1]["stitched_total_return"] >= tiers[2]["stitched_total_return"] - 1e-12
    assert first["fee_sensitivity"]["cost_breakeven_bps"] >= 35.0

    # Bounded +-20% parameter plateau
    plat = first["parameter_plateau"]
    assert plat["variant_count"] >= 3
    assert plat["positive_variant_fraction"] >= 0.60

    # Block-bootstrap 95% CIs & worst-5% tail drawdown
    boot = first["bootstrap"]
    assert boot["sharpe_ci_95"]["lo"] <= boot["sharpe_ci_95"]["median"] <= boot["sharpe_ci_95"]["hi"]
    assert boot["cagr_ci_95"]["lo"] <= boot["cagr_ci_95"]["median"] <= boot["cagr_ci_95"]["hi"]
    dd = boot["tail_drawdown"]
    assert dd["worst_5pct_drawdown_cvar"] <= dd["worst_5pct_drawdown_var"] + 1e-12
    assert dd["worst_5pct_drawdown_var"] <= 0.0

    # Strictly finite JSON & clean Markdown without bare dollar signs
    json.dumps(first, allow_nan=False)
    md = render_stress_report_markdown(first)
    assert "$" not in md
    assert first["report_sha256"] in md


def test_purged_walk_forward_has_zero_future_lookahead() -> None:
    close1 = _smoke_close(periods=540, seed=20261125)
    close2 = close1.copy()
    # Fold 0: train_bars=180, purge_bars=5 -> test_start=185, test_end=244.
    # Wildly mutate all bars strictly after fold 0's test_end (index >= 245).
    close2.iloc[245:] = close2.iloc[245:] * 25.0

    cfg = StressReportConfig(
        strategy="ma", train_bars=180, test_bars=60, purge_bars=5, bootstrap_resamples=200
    )
    r1 = run_stress_report(close1, config=cfg)
    r2 = run_stress_report(close2, config=cfg)

    assert r1["purged_walk_forward"]["folds"][0] == r2["purged_walk_forward"]["folds"][0]
    assert r1["purged_walk_forward"]["folds"][1] != r2["purged_walk_forward"]["folds"][1]


def test_stress_report_fails_closed_on_plateau_collapse_and_cost_breakeven() -> None:
    chop = _mean_reverting_chop_close()
    cfg = StressReportConfig(
        strategy="ma",
        fast=10,
        slow=30,
        train_bars=120,
        test_bars=60,
        purge_bars=5,
        bootstrap_resamples=200,
    )
    failed = run_stress_report(chop, config=cfg, input_source="chop_fixture")
    assert failed["status"] == "FAIL"
    assert any(b.startswith("cost_breakeven_below_threshold:") for b in failed["blockers"])
    assert any(b.startswith("plateau_collapse_pass_rate:") for b in failed["blockers"])

    # Tight cost breakeven gate also fails closed on smoke series
    tight = StressReportConfig(
        strategy="ma",
        bootstrap_resamples=200,
        min_cost_breakeven_bps=1_900.0,
    )
    failed_cost = run_stress_report(_smoke_close(), config=tight)
    assert failed_cost["status"] == "FAIL"
    assert any(b.startswith("cost_breakeven_below_threshold:") for b in failed_cost["blockers"])


def test_stress_report_config_and_input_validation_fail_closed() -> None:
    with pytest.raises(StressReportError, match="fast must be strictly smaller"):
        StressReportConfig(fast=50, slow=50)
    with pytest.raises(StressReportError, match="exit_window"):
        StressReportConfig(entry_window=20, exit_window=30)
    with pytest.raises(StressReportError, match="Insufficient rows"):
        run_stress_report(_smoke_close(periods=150), config=StressReportConfig(train_bars=120, test_bars=60))


def test_stress_report_cli_smoke_csv_parquet_markdown_and_fail_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    assert "stress-report" in parser.format_help()
    md_path = tmp_path / "stress_summary.md"

    # 1) --smoke + --json + --out-md -> PASS (exit 0)
    smoke_args = parser.parse_args(
        [
            "stress-report",
            "--smoke",
            "--strategy",
            "ma",
            "--bootstrap-resamples",
            "200",
            "--out-md",
            str(md_path),
            "--json",
        ]
    )
    smoke_args.handler(smoke_args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["execution_capability"] is False
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert payload["report_sha256"] in md_text
    assert "$" not in md_text

    # 2) CSV file input -> PASS (text mode)
    csv_path = tmp_path / "BTCUSDT_1d.csv"
    build_smoke_ohlcv().to_csv(csv_path, index=False)
    csv_args = parser.parse_args(
        ["stress-report", "--input", str(csv_path), "--bootstrap-resamples", "200"]
    )
    csv_args.handler(csv_args)
    assert "-> PASS" in capsys.readouterr().out

    # 3) Parquet file input (via monkeypatch, no pyarrow dep required in CI)
    pq_path = tmp_path / "ETHUSDT_1d.parquet"
    pq_path.write_bytes(b"placeholder")
    smoke_df = build_smoke_ohlcv()
    monkeypatch.setattr(pd, "read_parquet", lambda _path: smoke_df.copy())
    pq_args = parser.parse_args(
        ["stress-report", "--input", str(pq_path), "--bootstrap-resamples", "200", "--json"]
    )
    pq_args.handler(pq_args)
    pq_payload = json.loads(capsys.readouterr().out)
    assert pq_payload["status"] == "PASS"
    assert pq_payload["input_source"] == "file:ETHUSDT_1d.parquet"

    # 4) Threshold breach -> emits valid FAIL JSON AND exits non-zero (code 1)
    fail_args = parser.parse_args(
        [
            "stress-report",
            "--smoke",
            "--min-breakeven-bps",
            "1900",
            "--bootstrap-resamples",
            "200",
            "--json",
        ]
    )
    with pytest.raises(SystemExit) as exc_info:
        fail_args.handler(fail_args)
    assert exc_info.value.code == 1
    fail_payload = json.loads(capsys.readouterr().out)
    assert fail_payload["status"] == "FAIL"

    # 5) Malformed OHLCV CSV fails closed cleanly
    bad_df = build_smoke_ohlcv()
    bad_df.loc[5, "close"] = -1.0
    bad_csv = tmp_path / "BROKEN_1d.csv"
    bad_df.to_csv(bad_csv, index=False)
    bad_args = parser.parse_args(["stress-report", "--input", str(bad_csv)])
    with pytest.raises(SystemExit, match="quality preflight failed"):
        bad_args.handler(bad_args)
