"""Unit & Parity Tests for UnifiedSpotEngine, FundingCarryV2, and Shadow Parity (pure numpy+pandas)."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from quant_lab import (
    DEFAULT_SHADOW_4H_KLINE_LIMIT,
    MIN_RECOMMENDED_4H_KLINE_LIMIT,
    F4EmaConfig,
    FundingCarryConfig,
    UnifiedSpotEngine,
    apply_deadband_step,
    carry_btc_eth_cons_weights,
    carry_top_k_hysteresis_weights,
    carry_v2_blend_weights,
    causal_ema_1d,
    causal_rolling_top_n_mask,
    compute_carry_btc_eth_live_step,
    enforce_pre_holdout_dates,
    f4_ema_trend_btc_weights,
    run_funding_carry_backtest,
    update_shadow_spot_accounting,
)


def _synthetic_spot_panel(n_bars: int = 420, seed: int = 20261127) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="4h", tz="UTC")
    cols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "WAVESUSDT"]
    rets = rng.normal(0.0005, 0.014, size=(n_bars, len(cols)))
    rets[:, 0] += 0.0014
    close_arr = 100.0 * np.cumprod(1.0 + rets, axis=0)
    high_arr = close_arr * (1.0 + rng.uniform(0.002, 0.018, size=close_arr.shape))
    low_arr = close_arr * (1.0 - rng.uniform(0.002, 0.018, size=close_arr.shape))
    qv_arr = rng.uniform(5e6, 2e8, size=close_arr.shape)
    for arr in (close_arr, high_arr, low_arr, qv_arr):
        arr[:45, 2] = np.nan
        arr[360:, 5] = np.nan
    return {
        "close": pd.DataFrame(close_arr, index=idx, columns=cols),
        "high": pd.DataFrame(high_arr, index=idx, columns=cols),
        "low": pd.DataFrame(low_arr, index=idx, columns=cols),
        "quote_volume": pd.DataFrame(qv_arr, index=idx, columns=cols),
    }


def _synthetic_funding_panel(n_days: int = 220, seed: int = 2026) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="1D", tz="UTC")
    cols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]
    n_k = len(cols)
    fund = rng.normal(0.00040, 0.00020, size=(n_days, n_k))
    fund[120:130, :] = -0.00035
    spot = np.zeros((n_days, n_k), dtype=np.float64)
    perp = np.zeros((n_days, n_k), dtype=np.float64)
    spot[0] = 100.0
    perp[0] = 100.05
    for t in range(1, n_days):
        r = rng.normal(0.0004, 0.025, size=n_k)
        spot[t] = spot[t - 1] * np.exp(r)
        perp[t] = spot[t] * (1.0 + rng.normal(0.0, 0.0007, size=n_k))
    qv = rng.uniform(5e7, 5e8, size=(n_days, n_k))
    # Late-listed coin on last column
    for arr in (fund, spot, perp, qv):
        arr[:75, -1] = np.nan
    return {
        "funding": pd.DataFrame(fund, index=idx, columns=cols),
        "spot_close": pd.DataFrame(spot, index=idx, columns=cols),
        "perp_close": pd.DataFrame(perp, index=idx, columns=cols),
        "spot_high": pd.DataFrame(spot * 1.03, index=idx, columns=cols),
        "perp_high": pd.DataFrame(perp * 1.035, index=idx, columns=cols),
        "perp_qv": pd.DataFrame(qv, index=idx, columns=cols),
    }


def test_enforce_pre_holdout_dates_strict_and_python_dash_o_safe() -> None:
    assert enforce_pre_holdout_dates(["2024-01-01", "2024-12-31"]) == 0
    assert enforce_pre_holdout_dates(pd.date_range("2024-12-01", "2024-12-31", freq="D", tz="UTC")) == 0

    for bad in (
        "2025-01-01",
        ["2024-12-31", "2025-01-01"],
        pd.date_range("2024-12-30", periods=3, freq="D", tz="UTC"),
        "2025-01-01T00:00:00Z",
        "2025-01-01T01:00:00+00:00",
        "not-a-date",
        None,
        pd.NaT,
        [pd.Timestamp("2024-01-01"), pd.NaT],
        pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.NaT]),
        pd.Series([pd.Timestamp("2024-01-01"), pd.NaT]),
        np.datetime64("NaT", "D"),
    ):
        with pytest.raises(ValueError, match="HOLDOUT_LOCK_VIOLATION"):
            enforce_pre_holdout_dates(bad)  # type: ignore[arg-type]

    # Verify enforcement holds under `python -O` (assert-stripping mode!) for both 2025+ and pd.NaT
    code = (
        "import pandas as pd\n"
        "from quant_lab.unified_engine import enforce_pre_holdout_dates\n"
        "for bad in (['2025-01-01'], pd.NaT, [pd.Timestamp('2024-01-01'), pd.NaT]):\n"
        "    try:\n"
        "        enforce_pre_holdout_dates(bad)\n"
        "    except ValueError:\n"
        "        continue\n"
        "    raise SystemExit(99)\n"
        "raise SystemExit(0)\n"
    )
    proc = subprocess.run([sys.executable, "-O", "-c", code], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, f"Failed under python -O: {proc.stderr}"


def test_unified_engine_matches_causal_universe_and_shadow_step_bit_for_bit() -> None:
    panel = _synthetic_spot_panel(n_bars=260, seed=77)
    mask = causal_rolling_top_n_mask(panel["quote_volume"], panel["close"], lookback_bars=20, top_n=4)
    cfg = F4EmaConfig(
        fast_span=12,
        slow_span=40,
        btc_fast=20,
        btc_slow=80,
        vol_window=40,
        deadband=0.015,
    )
    engine = UnifiedSpotEngine(cfg)

    w_unified = engine.build_f4_weights(panel["close"], mask)
    w_legacy = f4_ema_trend_btc_weights(
        panel["close"],
        mask,
        fast_span=cfg.fast_span,
        slow_span=cfg.slow_span,
        btc_fast=cfg.btc_fast,
        btc_slow=cfg.btc_slow,
        vol_window=cfg.vol_window,
        deadband=cfg.deadband,
    )
    assert np.allclose(w_unified.to_numpy(), w_legacy.to_numpy(), atol=1e-12, rtol=1e-12)

    # Now replay Shadow step-by-step from bar 0..259 and assert bit-for-bit parity with Backtest!
    cols = list(panel["close"].columns)
    prev_w: dict[str, float] = {}
    for t in range(len(panel["close"])):
        closed_closes = {c: panel["close"][c].iloc[: t + 1].to_numpy(dtype=np.float64) for c in cols}
        elig_syms = [c for c in cols if bool(mask[c].iloc[t])]
        step = engine.compute_f4_step(closed_closes, elig_syms, prev_weights=prev_w)
        w_row = w_unified.iloc[t]
        for c in cols:
            expected = float(w_row[c])
            got = float(step["weights"].get(c, 0.0))
            assert got == pytest.approx(expected, abs=1e-12), (
                f"Parity mismatch at bar {t} {c}: {got} vs {expected}"
            )
        prev_w = step["weights"]


def test_ema_warmup_gap_reset_and_kline_limit_drift_reduction() -> None:
    assert DEFAULT_SHADOW_4H_KLINE_LIMIT >= MIN_RECOMMENDED_4H_KLINE_LIMIT >= 600

    # 1) Warmup parity: EMA(300) requires 300 consecutive valid bars and resets after NaN!
    prices = np.linspace(100.0, 200.0, 400, dtype=np.float64)
    ema = causal_ema_1d(prices, 300)
    assert np.isnan(ema[:299]).all()
    assert np.isfinite(ema[299:]).all()

    prices_gap = prices.copy()
    prices_gap[310] = np.nan
    ema_gap = causal_ema_1d(prices_gap, 300)
    assert np.isnan(ema_gap[310:]).all()

    # 2) Quantify EMA(300) initialization drift: 1000 bars vs 325 bars against 3000-bar reference
    rng = np.random.default_rng(123)
    long_px = 50_000.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=3000))
    ref = causal_ema_1d(long_px, 300)[-1]
    short_325 = causal_ema_1d(long_px[-324:], 300)[-1]
    long_1000 = causal_ema_1d(long_px[-999:], 300)[-1]
    err_325 = abs(short_325 - ref) / ref
    err_1000 = abs(long_1000 - ref) / ref
    assert err_1000 < err_325 * 0.05
    assert err_1000 < 0.002


def test_deadband_step_and_shadow_accounting_idempotency_and_exit_mtm() -> None:
    # 1) Deadband retains tiny weight wobble, executes flips & >= deadband moves
    w0 = {"BTCUSDT": 0.30, "ETHUSDT": 0.20}
    w_small = apply_deadband_step({"BTCUSDT": 0.305, "ETHUSDT": 0.195}, w0, deadband=0.015)
    assert w_small == w0
    w_flip = apply_deadband_step({"BTCUSDT": 0.305, "SOLUSDT": 0.15}, w0, deadband=0.015)
    assert "ETHUSDT" not in w_flip
    assert w_flip["BTCUSDT"] == pytest.approx(0.30)
    assert w_flip["SOLUSDT"] == pytest.approx(0.15)

    # 2) Shadow idempotency lock: repeated 5m polls on SAME 4H bar deduct ZERO extra fees!
    bar1_ms = 1_704_081_599_999
    bar2_ms = bar1_ms + 4 * 3_600_000
    depth_t1 = {
        "BTCUSDT": {"mid": 50_000.0, "slip_10k_bps": 2.0},
        "OLDCOINUSDT": {"mid": 110.0, "slip_10k_bps": 4.0},
    }
    # Initial bar transition: OLDCOINUSDT dropped out of Top-15 (+10% exit MTM from 100 -> 110!)
    s1 = update_shadow_spot_accounting(
        prev_weights={"OLDCOINUSDT": 0.50},
        prev_mids={"OLDCOINUSDT": 100.0},
        prev_equity_usd=10_000.0,
        prev_bar_close_ms=bar1_ms - 4 * 3_600_000,
        current_bar_close_ms=bar1_ms,
        candidate_weights={"BTCUSDT": 0.40},
        depth_metrics=depth_t1,
    )
    assert s1["bar_rebalanced"] is True
    assert s1["mtm_return"] == pytest.approx(0.05, abs=1e-12)  # 0.50 * (110/100 - 1) = +5%!
    assert s1["turnover"] == pytest.approx(0.90, abs=1e-12)
    assert s1["cost_return"] > 0.0
    assert s1["missing_exit_mids"] == []

    # Repeated 5-minute poll on the SAME 4H bar (`bar1_ms`): even if candidate_weights wobbles, ZERO fees!
    depth_t1_poll2 = {
        "BTCUSDT": {"mid": 50_500.0, "slip_10k_bps": 2.0},
        "ETHUSDT": {"mid": 3_000.0, "slip_10k_bps": 2.0},
    }
    s1_dup = update_shadow_spot_accounting(
        prev_weights=s1["weights"],
        prev_mids={"BTCUSDT": 50_000.0},
        prev_equity_usd=s1["equity_usd"],
        prev_bar_close_ms=s1["last_4h_close_ms"],
        current_bar_close_ms=bar1_ms,
        candidate_weights={"BTCUSDT": 0.20, "ETHUSDT": 0.30},
        depth_metrics=depth_t1_poll2,
    )
    assert s1_dup["bar_rebalanced"] is False
    assert s1_dup["turnover"] == 0.0
    assert s1_dup["cost_return"] == 0.0
    assert s1_dup["weights"] == s1["weights"]
    assert s1_dup["mtm_return"] == pytest.approx(0.40 * (50_500.0 / 50_000.0 - 1.0), abs=1e-12)

    # Next closed 4H bar (`bar2_ms`): rebalance unlocks and applies fees ONCE
    s2 = update_shadow_spot_accounting(
        prev_weights=s1_dup["weights"],
        prev_mids={"BTCUSDT": 50_500.0},
        prev_equity_usd=s1_dup["equity_usd"],
        prev_bar_close_ms=s1_dup["last_4h_close_ms"],
        current_bar_close_ms=bar2_ms,
        candidate_weights={"BTCUSDT": 0.20, "ETHUSDT": 0.30},
        depth_metrics=depth_t1_poll2,
    )
    assert s2["bar_rebalanced"] is True
    assert s2["turnover"] == pytest.approx(0.50, abs=1e-12)
    assert s2["weights"] == {"BTCUSDT": 0.20, "ETHUSDT": 0.30}


def test_shadow_accounting_rejects_poisoned_prices_slippage_and_bar_timestamps() -> None:
    bar_ms = 1_704_081_599_999

    # 1) Reject bool / non-positive bar timestamps fail-closed
    for bad_cur in (0, -100, True, False):
        with pytest.raises(ValueError, match="current_bar_close_ms"):
            update_shadow_spot_accounting(
                prev_weights={},
                prev_mids={},
                prev_equity_usd=10_000.0,
                prev_bar_close_ms=None,
                current_bar_close_ms=bad_cur,  # type: ignore[arg-type]
                candidate_weights={"BTCUSDT": 0.25},
                depth_metrics={"BTCUSDT": {"mid": 50_000.0, "slip_10k_bps": 2.0}},
            )
    for bad_prev in (0, -1, True):
        with pytest.raises(ValueError, match="prev_bar_close_ms"):
            update_shadow_spot_accounting(
                prev_weights={},
                prev_mids={},
                prev_equity_usd=10_000.0,
                prev_bar_close_ms=bad_prev,  # type: ignore[arg-type]
                current_bar_close_ms=bar_ms,
                candidate_weights={"BTCUSDT": 0.25},
                depth_metrics={"BTCUSDT": {"mid": 50_000.0, "slip_10k_bps": 2.0}},
            )

    # 2) Poisoned Inf mid + negative slippage must NOT inflate equity or produce Inf/NaN
    res_inf = update_shadow_spot_accounting(
        prev_weights={"BTCUSDT": 0.50},
        prev_mids={"BTCUSDT": 50_000.0},
        prev_equity_usd=10_000.0,
        prev_bar_close_ms=bar_ms - 4 * 3_600_000,
        current_bar_close_ms=bar_ms,
        candidate_weights={"BTCUSDT": 0.20},
        depth_metrics={"BTCUSDT": {"mid": float("inf"), "slip_10k_bps": -50.0}},
        taker_fee_bps=10.0,
        fallback_slip_bps=5.0,
    )
    assert res_inf["mtm_return"] == 0.0
    assert res_inf["missing_exit_mids"] == ["BTCUSDT"]
    assert res_inf["live_tca_bps"] == pytest.approx(15.0, abs=1e-12)
    assert math.isfinite(res_inf["equity_usd"])
    assert res_inf["equity_usd"] < 10_000.0

    # 3) Poisoned Inf prev_mid & bool mid/slippage must also fail-safe to fallback TCA & finite equity
    res_bool = update_shadow_spot_accounting(
        prev_weights={"BTCUSDT": 0.50},
        prev_mids={"BTCUSDT": float("inf")},
        prev_equity_usd=10_000.0,
        prev_bar_close_ms=bar_ms - 4 * 3_600_000,
        current_bar_close_ms=bar_ms,
        candidate_weights={"BTCUSDT": 0.0},
        depth_metrics={"BTCUSDT": {"mid": True, "slip_10k_bps": True}},
        taker_fee_bps=10.0,
        fallback_slip_bps=5.0,
    )
    assert res_bool["mtm_return"] == 0.0
    assert res_bool["missing_exit_mids"] == ["BTCUSDT"]
    assert res_bool["live_tca_bps"] == pytest.approx(15.0, abs=1e-12)
    assert math.isfinite(res_bool["equity_usd"])
    assert res_bool["equity_usd"] == pytest.approx(10_000.0 * (1.0 - 0.50 * 0.0015), abs=1e-9)


def test_unified_engine_rejects_holdout_2025_dates() -> None:
    idx = pd.date_range("2024-12-20", periods=40, freq="1D", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": np.linspace(50_000.0, 55_000.0, 40)}, index=idx)
    mask = pd.DataFrame({"BTCUSDT": [True] * 40}, index=idx)
    engine = UnifiedSpotEngine(F4EmaConfig(fast_span=3, slow_span=10, btc_fast=3, btc_slow=10, vol_window=5))
    with pytest.raises(ValueError, match="HOLDOUT_LOCK_VIOLATION"):
        engine.build_f4_weights(close, mask)
    with pytest.raises(ValueError, match="HOLDOUT_LOCK_VIOLATION"):
        engine.run_backtest(close, pd.DataFrame({"BTCUSDT": [0.0] * 40}, index=idx))


def test_funding_carry_v2_causality_margin_caps_and_shadow_step() -> None:
    p = _synthetic_funding_panel(n_days=220, seed=42)
    cut = 135

    w_core_1 = carry_btc_eth_cons_weights(p["funding"], p["spot_close"], p["perp_close"])
    w_top5_1 = carry_top_k_hysteresis_weights(p["funding"], p["spot_close"], p["perp_close"], p["perp_qv"])
    w_blend_1 = carry_v2_blend_weights(w_core_1, w_top5_1, core_share=0.70, satellite_share=0.30)

    # Mutate all future data after `cut`
    p2 = {k: v.copy() for k, v in p.items()}
    p2["funding"].iloc[cut + 1 :] = -0.02
    p2["spot_close"].iloc[cut + 1 :] *= 3.5
    p2["perp_close"].iloc[cut + 1 :] *= 0.4
    p2["perp_qv"].iloc[cut + 1 :] = 1.0

    w_core_2 = carry_btc_eth_cons_weights(p2["funding"], p2["spot_close"], p2["perp_close"])
    w_top5_2 = carry_top_k_hysteresis_weights(
        p2["funding"], p2["spot_close"], p2["perp_close"], p2["perp_qv"]
    )
    assert np.array_equal(w_core_1.iloc[: cut + 1].to_numpy(), w_core_2.iloc[: cut + 1].to_numpy())
    assert np.array_equal(w_top5_1.iloc[: cut + 1].to_numpy(), w_top5_2.iloc[: cut + 1].to_numpy())

    # Zero pre-listing allocation on late-listed coin (LINKUSDT unlisted days 0..74)
    assert float(w_top5_1.iloc[:75]["LINKUSDT"].abs().max()) == 0.0

    # 2x equity buffer -> gross equity notional <= 0.50; 2x cost <= 1x cost monotonicity
    res1 = run_funding_carry_backtest(
        p["funding"],
        p["spot_close"],
        p["perp_close"],
        w_blend_1,
        perp_high=p["perp_high"],
        spot_high=p["spot_high"],
        config=FundingCarryConfig(
            equity_buffer_mult=2.0,
            oneway_pair_cost_bps=20.0,
            basis_cross_slip_bps=3.0,
        ),
    )
    res2 = run_funding_carry_backtest(
        p["funding"],
        p["spot_close"],
        p["perp_close"],
        w_blend_1,
        perp_high=p["perp_high"],
        spot_high=p["spot_high"],
        config=FundingCarryConfig(
            equity_buffer_mult=2.0,
            oneway_pair_cost_bps=40.0,
            basis_cross_slip_bps=6.0,
            exit_shock_mult=2.0,
        ),
    )
    assert float(res1["backtest"]["gross_exposure"].max()) <= 0.50 + 1e-12
    assert res1["holdout_rows_analysed"] == 0
    assert float(res2["backtest"]["equity"].iloc[-1]) <= float(res1["backtest"]["equity"].iloc[-1]) + 1e-12

    # Live carry step hysteresis parity
    hi = [0.0001] * 21
    mid = [0.000015] * 21
    neg = [-0.00005] * 21
    st1 = compute_carry_btc_eth_live_step(
        {"BTCUSDT": hi, "ETHUSDT": mid},
        prev_state={"BTCUSDT": False, "ETHUSDT": False},
    )
    assert st1["state"] == {"BTCUSDT": True, "ETHUSDT": False}
    assert st1["notional_weights_of_equity"] == {"BTCUSDT": 0.25}
    st2 = compute_carry_btc_eth_live_step({"BTCUSDT": mid, "ETHUSDT": mid}, prev_state=st1["state"])
    assert st2["state"] == {"BTCUSDT": True, "ETHUSDT": False}
    st3 = compute_carry_btc_eth_live_step({"BTCUSDT": neg, "ETHUSDT": neg}, prev_state=st2["state"])
    assert st3["state"] == {"BTCUSDT": False, "ETHUSDT": False}
    assert st3["notional_weights_of_equity"] == {}

    # Holdout 2025+ rejection in funding_carry_v2
    bad_idx = pd.date_range("2024-12-25", periods=15, freq="1D", tz="UTC")
    bad_f = pd.DataFrame({"BTCUSDT": 0.0003, "ETHUSDT": 0.0003}, index=bad_idx)
    bad_p = pd.DataFrame({"BTCUSDT": 100.0, "ETHUSDT": 100.0}, index=bad_idx)
    with pytest.raises(ValueError, match="HOLDOUT_LOCK_VIOLATION"):
        carry_btc_eth_cons_weights(bad_f, bad_p, bad_p)
