from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_lab.causal_universe import (
    build_sleeve2_paper_soak_day,
    build_sleeve2_paper_soak_receipt,
    causal_rolling_top_n_mask,
    f3_donchian_breakout_weights,
    f4_ema_trend_btc_weights,
    run_multi_asset_spot_backtest,
    run_sleeve2_offline_smoke,
)
from quant_lab.cli import build_parser
from quant_lab.metrics import performance_metrics
from quant_trading.paper_soak import PaperSoakError, PaperSoakReceipt, PaperSoakRestartCheck


def _synthetic_4h_panel(n_bars: int = 420, seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="4h", tz="UTC")
    cols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "WAVESUSDT"]
    rets = rng.normal(0.0005, 0.015, size=(n_bars, len(cols)))
    # Give BTC an upward drift so EMA60 > EMA300 activates cleanly
    rets[:, 0] += 0.0012
    close_arr = 100.0 * np.cumprod(1.0 + rets, axis=0)
    high_arr = close_arr * (1.0 + rng.uniform(0.002, 0.02, size=close_arr.shape))
    low_arr = close_arr * (1.0 - rng.uniform(0.002, 0.02, size=close_arr.shape))
    qv_arr = rng.uniform(5e6, 2e8, size=close_arr.shape)
    # Late listing for SOLUSDT (bar 60) and delisting for WAVESUSDT (bar 350)
    for arr in (close_arr, high_arr, low_arr, qv_arr):
        arr[:60, 2] = np.nan
        arr[350:, 5] = np.nan
    return {
        "close": pd.DataFrame(close_arr, index=idx, columns=cols),
        "high": pd.DataFrame(high_arr, index=idx, columns=cols),
        "low": pd.DataFrame(low_arr, index=idx, columns=cols),
        "quote_volume": pd.DataFrame(qv_arr, index=idx, columns=cols),
    }


def test_causal_top_n_and_sleeve2_strategies_zero_lookahead() -> None:
    p1 = _synthetic_4h_panel(n_bars=420, seed=11)
    t_cut = 320

    m1 = causal_rolling_top_n_mask(p1["quote_volume"], p1["close"], lookback_bars=30, top_n=4)
    w_f4_1 = f4_ema_trend_btc_weights(p1["close"], m1, btc_fast=30, btc_slow=120, vol_window=60)
    w_f3_1 = f3_donchian_breakout_weights(
        p1["close"], p1["high"], p1["low"], m1, entry_window=60, exit_window=30, vol_window=60
    )

    # Wildly mutate all bars strictly after t_cut
    p2 = {k: v.copy() for k, v in p1.items()}
    p2["close"].iloc[t_cut + 1 :] *= 50.0
    p2["high"].iloc[t_cut + 1 :] *= 55.0
    p2["low"].iloc[t_cut + 1 :] *= 0.2
    p2["quote_volume"].iloc[t_cut + 1 :, 0] = 1.0
    p2["quote_volume"].iloc[t_cut + 1 :, 4] = 1e14

    m2 = causal_rolling_top_n_mask(p2["quote_volume"], p2["close"], lookback_bars=30, top_n=4)
    w_f4_2 = f4_ema_trend_btc_weights(p2["close"], m2, btc_fast=30, btc_slow=120, vol_window=60)
    w_f3_2 = f3_donchian_breakout_weights(
        p2["close"], p2["high"], p2["low"], m2, entry_window=60, exit_window=30, vol_window=60
    )

    assert m1.iloc[: t_cut + 1].equals(m2.iloc[: t_cut + 1])
    assert np.allclose(w_f4_1.iloc[: t_cut + 1].to_numpy(), w_f4_2.iloc[: t_cut + 1].to_numpy())
    assert np.allclose(w_f3_1.iloc[: t_cut + 1].to_numpy(), w_f3_2.iloc[: t_cut + 1].to_numpy())
    # Pre-listing coin SOLUSDT (listed bar 60) must never be selected before bar 85
    assert not m1.iloc[:85]["SOLUSDT"].any()
    # Spot invariants: [0, 1] weights, row sums <= 1.0
    for w in (w_f4_1, w_f3_1):
        assert (w.to_numpy() >= 0.0).all()
        assert (w.sum(axis=1).to_numpy() <= 1.0 + 1e-9).all()


def test_multi_asset_spot_backtest_next_bar_and_delisting_haircut() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    close = pd.DataFrame(
        {"BTCUSDT": [100.0, 110.0, 121.0, 121.0], "WAVESUSDT": [50.0, 55.0, np.nan, np.nan]},
        index=idx,
    )
    tw = pd.DataFrame(
        {"BTCUSDT": [0.5, 0.5, 0.5, 0.5], "WAVESUSDT": [0.5, 0.5, 0.0, 0.0]},
        index=idx,
    )
    bt1 = run_multi_asset_spot_backtest(close, tw, fee_bps=10.0, slippage_bps=5.0, delist_haircut_bps=100.0)
    bt2 = run_multi_asset_spot_backtest(close, tw, fee_bps=20.0, slippage_bps=10.0, delist_haircut_bps=100.0)

    assert bt1["gross_exposure"].iloc[0] == 0.0
    assert bt1["strategy_return"].iloc[1] == pytest.approx(0.10 - 0.0015, abs=1e-9)
    assert bt1["delist_exit_weight"].iloc[2] == pytest.approx(0.5)
    assert np.isfinite(bt1["equity"].to_numpy()).all()
    assert (bt2["strategy_return"].to_numpy() <= bt1["strategy_return"].to_numpy() + 1e-12).all()


def test_sleeve2_paper_soak_receipt_fail_closed() -> None:
    empty_receipt = build_sleeve2_paper_soak_receipt([])
    assert empty_receipt.status == "NOT_RUN"
    assert empty_receipt.execution_capability is False
    assert empty_receipt.promotion_allowed is False

    days = [
        build_sleeve2_paper_soak_day(
            date_utc=f"2026-10-{i + 1:02d}",
            input_cutoff_utc=1_790_812_800 + i * 86_400,
            target_weights={"BTCUSDT": 0.12, "ETHUSDT": 0.08},
            realized_pnl_vnd=150_000.0,
            unrealized_pnl_vnd=25_000.0,
            fees_vnd=12_000.0,
            slippage_bps=1.8,
            equity_vnd=250_000_000.0 + i * 150_000.0,
        )
        for i in range(30)
    ]
    restart = PaperSoakRestartCheck(
        observed_at_utc=1_790_812_800 + 15 * 86_400,
        attempt_no=1,
        checkpoint="day-15-restart",
        duplicate_intents=0,
        status="passed",
        evidence_hash="a" * 64,
    )
    obs = build_sleeve2_paper_soak_receipt(days[:5], [restart], owner_reviewed=False)
    assert obs.status == "OBSERVED"
    assert obs.execution_capability is False

    acc = build_sleeve2_paper_soak_receipt(days, [restart], owner_reviewed=True)
    assert acc.status == "ACCEPTED"
    assert acc.observed_days == 30
    assert acc.execution_capability is False
    assert acc.promotion_allowed is False
    assert acc.broker_access is False
    assert acc.receipt_sha256 is not None and len(acc.receipt_sha256) == 64
    assert PaperSoakReceipt.from_mapping(acc.as_dict()).receipt_sha256 == acc.receipt_sha256


def test_causal_top_n_tie_breaking_is_column_order_invariant() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    cols_a = ["ETHUSDT", "ADAUSDT", "BTCUSDT"]
    cols_b = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    qv_a = pd.DataFrame(1e7, index=idx, columns=cols_a)
    cl_a = pd.DataFrame(100.0, index=idx, columns=cols_a)

    mask_a = causal_rolling_top_n_mask(qv_a, cl_a, lookback_bars=5, top_n=2)
    mask_b = causal_rolling_top_n_mask(qv_a[cols_b], cl_a[cols_b], lookback_bars=5, top_n=2)

    assert mask_a[cols_b].equals(mask_b)
    assert mask_a.iloc[-1]["ADAUSDT"] and mask_a.iloc[-1]["BTCUSDT"]
    assert not mask_a.iloc[-1]["ETHUSDT"]


def test_causal_universe_and_backtest_fail_closed_on_malformed_inputs() -> None:
    p = _synthetic_4h_panel(n_bars=140, seed=7)
    m = causal_rolling_top_n_mask(p["quote_volume"], p["close"], lookback_bars=10, top_n=3)

    dup_close = p["close"].copy()
    dup_close.columns = ["BTCUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "WAVESUSDT"]
    with pytest.raises(ValueError, match="columns must be unique"):
        causal_rolling_top_n_mask(dup_close, dup_close, lookback_bars=5, top_n=2)

    inf_close = p["close"].copy()
    inf_close.iloc[10, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        causal_rolling_top_n_mask(p["quote_volume"], inf_close, lookback_bars=5, top_n=2)

    neg_close = p["close"].copy()
    neg_close.iloc[10, 0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        f4_ema_trend_btc_weights(
            neg_close, m, fast_span=5, slow_span=20, btc_fast=5, btc_slow=20, vol_window=10
        )

    bad_high = p["high"].copy()
    bad_high.iloc[10, 0] = p["low"].iloc[10, 0] - 1.0
    with pytest.raises(ValueError, match="below low"):
        f3_donchian_breakout_weights(
            p["close"], bad_high, p["low"], m, entry_window=20, exit_window=10, vol_window=10
        )

    with pytest.raises(ValueError, match="vol_window"):
        f4_ema_trend_btc_weights(
            p["close"], m, fast_span=5, slow_span=20, btc_fast=5, btc_slow=20, vol_window=1
        )
    with pytest.raises(ValueError, match="deadband"):
        f4_ema_trend_btc_weights(
            p["close"],
            m,
            fast_span=5,
            slow_span=20,
            btc_fast=5,
            btc_slow=20,
            vol_window=10,
            deadband=float("nan"),
        )

    w = f4_ema_trend_btc_weights(
        p["close"], m, fast_span=5, slow_span=20, btc_fast=5, btc_slow=20, vol_window=10
    )
    with pytest.raises(ValueError, match="finite"):
        run_multi_asset_spot_backtest(p["close"], w, fee_bps=float("nan"))
    with pytest.raises(ValueError, match="less than 100%"):
        run_multi_asset_spot_backtest(p["close"], w, fee_bps=9_000.0, slippage_bps=1_000.0)

    unlisted_w = w.copy()
    unlisted_w.iloc[0, 2] = 0.25  # SOLUSDT is NaN before bar 60
    with pytest.raises(ValueError, match="close price is missing"):
        run_multi_asset_spot_backtest(p["close"], unlisted_w)

    bt = run_multi_asset_spot_backtest(p["close"], w)
    metrics = performance_metrics(bt, periods_per_year=365 * 6)
    assert "cagr" in metrics and "time_in_market" in metrics and "entries" in metrics


def test_sleeve2_paper_soak_helpers_enforce_spot_and_rejection_invariants() -> None:
    with pytest.raises(ValueError, match=">= 0.0"):
        build_sleeve2_paper_soak_day(
            date_utc="2026-10-01",
            input_cutoff_utc=1_790_812_800,
            target_weights={"BTCUSDT": -0.1},
            realized_pnl_vnd=0.0,
            unrealized_pnl_vnd=0.0,
            fees_vnd=0.0,
            slippage_bps=1.0,
            equity_vnd=100_000_000.0,
        )

    with pytest.raises(ValueError, match="zero leverage"):
        build_sleeve2_paper_soak_day(
            date_utc="2026-10-01",
            input_cutoff_utc=1_790_812_800,
            target_weights={"BTCUSDT": 0.6, "ETHUSDT": 0.6},
            realized_pnl_vnd=0.0,
            unrealized_pnl_vnd=0.0,
            fees_vnd=0.0,
            slippage_bps=1.0,
            equity_vnd=100_000_000.0,
        )

    mismatch_day = build_sleeve2_paper_soak_day(
        date_utc="2026-10-01",
        input_cutoff_utc=1_790_812_800,
        target_weights={"BTCUSDT": 0.2},
        realized_pnl_vnd=-10_000.0,
        unrealized_pnl_vnd=0.0,
        fees_vnd=5_000.0,
        slippage_bps=2.0,
        equity_vnd=249_990_000.0,
        reconciled=False,
        pause_reason="ledger balance delta",
    )
    rejected = build_sleeve2_paper_soak_receipt([mismatch_day])
    assert rejected.status == "REJECTED"
    assert rejected.execution_capability is False

    failed_restart = PaperSoakRestartCheck(
        observed_at_utc=1_790_812_900,
        attempt_no=1,
        checkpoint="day-1-restart",
        duplicate_intents=1,
        status="failed",
        evidence_hash="b" * 64,
        reason="duplicate intent after restart",
    )
    with pytest.raises(PaperSoakError, match="without observed soak days"):
        build_sleeve2_paper_soak_receipt([], [failed_restart])


@pytest.mark.parametrize("strategy", ["f4", "f3"])
def test_sleeve2_offline_smoke_and_cli_roundtrip(strategy: str, tmp_path, capsys) -> None:
    smoke = run_sleeve2_offline_smoke(strategy=strategy, soak_days=5, seed=2026)
    receipt = smoke["receipt"]
    health = smoke["health_report"]

    assert receipt.status == "OBSERVED"
    assert receipt.evidence_level == "offline_smoke_fixture"
    assert receipt.execution_capability is False
    assert receipt.promotion_allowed is False
    assert health.verdict == "IN_PROGRESS"
    assert "non_live_evidence_level:offline_smoke_fixture" in health.blockers
    assert health.calendar_gap_days == 0

    receipt_path = tmp_path / f"{strategy}_smoke_receipt.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "soak-check",
            "--smoke",
            "--strategy",
            strategy,
            "--days",
            "5",
            "--write-receipt",
            str(receipt_path),
            "--json",
        ]
    )
    args.handler(args)
    smoke_out = json.loads(capsys.readouterr().out)
    assert smoke_out["health_report"]["report_sha256"] == health.report_sha256
    assert receipt_path.exists()

    audit_args = parser.parse_args(["soak-check", "--receipt", str(receipt_path), "--json"])
    audit_args.handler(audit_args)
    audit_out = json.loads(capsys.readouterr().out)
    assert audit_out["receipt_sha256"] == receipt.receipt_sha256
    assert audit_out["verdict"] == "IN_PROGRESS"
    assert audit_out["execution_capability"] is False

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["execution_capability"] = True
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    bad_args = parser.parse_args(["soak-check", "--receipt", str(tampered_path)])
    with pytest.raises(SystemExit, match="execution_capability"):
        bad_args.handler(bad_args)
