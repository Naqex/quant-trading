"""Causal Delta-Neutral Funding-Rate Carry v2 Engine (pure numpy + pandas).

Single unified Spot Long + 1x Perp Short carry signal & accounting engine shared identically
by offline Backtest mode and live read-only Shadow mode:
  - Pure `numpy` + `pandas` + Python stdlib ONLY (zero duckdb/pyarrow/scipy/root imports).
  - Strict `-O`-safe `enforce_pre_holdout_dates()` raising `ValueError` on any date >= `2025-01-01`.
  - Zero pre-listing allocation (coins with missing funding or spot/perp close receive 0.0).
  - Exact 2.0x / 2.5x CEX margin-buffer notional caps (`notional_cap = 1.0 / equity_buffer_mult`).
  - Explicit Spot + Perp pair friction + basis crossing spread + adverse negative-funding/basis
    exit-shock cost + Spot-Perp basis mark-to-market + Mark-Price liquidation stress audit.
  - Strictly offline / read-only (`ORDER_SEND_CAPABLE = False`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.causal_universe import (
    _validate_finite_float,
    _validate_panel,
    _validate_positive_int,
    _validate_positive_prices,
)
from quant_lab.metrics import performance_metrics
from quant_lab.unified_engine import HOLDOUT_CUTOFF_DATE, ORDER_SEND_CAPABLE, enforce_pre_holdout_dates


@dataclass(frozen=True, slots=True)
class FundingCarryConfig:
    """Immutable cost and margin-buffer specification for delta-neutral funding carry."""

    equity_buffer_mult: float = 2.0
    oneway_pair_cost_bps: float = 20.0
    basis_cross_slip_bps: float = 3.0
    exit_shock_mult: float = 1.0
    include_basis_mtm: bool = True

    def __post_init__(self) -> None:
        buf = _validate_finite_float(self.equity_buffer_mult, "equity_buffer_mult", minimum=1.0)
        if buf < 1.0:
            raise ValueError("equity_buffer_mult must be >= 1.0.")
        pair_c = _validate_finite_float(self.oneway_pair_cost_bps, "oneway_pair_cost_bps", minimum=0.0)
        basis_c = _validate_finite_float(self.basis_cross_slip_bps, "basis_cross_slip_bps", minimum=0.0)
        if pair_c + basis_c >= 10_000.0:
            raise ValueError("Combined carry one-way cost bps must be < 10,000.")
        _validate_finite_float(self.exit_shock_mult, "exit_shock_mult", minimum=0.0)
        if type(self.include_basis_mtm) is not bool:
            raise ValueError("include_basis_mtm must be a bool.")


def _validate_aligned_frames(base: pd.DataFrame, others: Mapping[str, pd.DataFrame]) -> None:
    _validate_panel(base, "funding_daily")
    for name, df in others.items():
        _validate_panel(df, name)
        if not base.index.equals(df.index) or list(base.columns) != list(df.columns):
            raise ValueError(f"{name} must share identical index and columns with funding_daily.")


def _rolling_trailing_mean(arr: np.ndarray, window: int, min_valid: int) -> np.ndarray:
    """Strictly causal trailing mean over `[t-window+1 .. t]` requiring >= `min_valid` finite obs."""
    n_t, n_k = arr.shape
    out = np.full((n_t, n_k), np.nan, dtype=np.float64)
    for t in range(window - 1, n_t):
        win = arr[t - window + 1 : t + 1]
        ok = np.isfinite(win)
        cnt = np.sum(ok, axis=0)
        s = np.sum(np.where(ok, win, 0.0), axis=0)
        out[t] = np.where(cnt >= min_valid, s / np.maximum(1, cnt), np.nan)
    return out


def _invvol_capped_weights(
    active_idxs: Sequence[int],
    vol_row: np.ndarray,
    *,
    max_single: float,
) -> dict[int, float]:
    """Allocate inverse-volatility weights subject to `0 <= w_i <= max_single` and `sum(w) <= 1.0`."""
    if not active_idxs:
        return {}
    vols = [max(0.005, float(vol_row[j]) if np.isfinite(vol_row[j]) else 0.04) for j in active_idxs]
    inv = np.array([1.0 / v for v in vols], dtype=np.float64)
    w = inv / float(np.sum(inv))
    cap = float(max_single)
    total_budget = min(1.0, len(active_idxs) * cap)
    w = w * total_budget

    for _ in range(30):
        over = w > cap + 1e-12
        if not np.any(over):
            break
        excess = float(np.sum(w[over] - cap))
        w[over] = cap
        under = w < cap - 1e-12
        if not np.any(under):
            break
        u_sum = float(np.sum(w[under]))
        if u_sum > 1e-12:
            w[under] += excess * (w[under] / u_sum)
        else:
            w[under] += excess / float(np.sum(under))

    w = np.clip(w, 0.0, cap)
    s = float(np.sum(w))
    if s > 1.0 + 1e-12:
        w = w / s
    return {active_idxs[i]: float(w[i]) for i in range(len(active_idxs))}


def carry_btc_eth_cons_weights(
    funding_daily: pd.DataFrame,
    spot_close: pd.DataFrame,
    perp_close: pd.DataFrame,
    *,
    symbols: tuple[str, str] = ("BTCUSDT", "ETHUSDT"),
    enter_ann: float = 0.03,
    exit_ann: float = 0.0,
    lookback_days: int = 7,
    min_valid_days: int = 6,
    warmup_days: int = 14,
    enforce_pre_holdout: bool = True,
    holdout_cutoff: str = HOLDOUT_CUTOFF_DATE,
) -> pd.DataFrame:
    """Causal `CARRY_BTC_ETH_CONS_2XBUF` target notional weights (hysteresis enter >= 3%, exit <= 0%)."""
    _validate_aligned_frames(funding_daily, {"spot_close": spot_close, "perp_close": perp_close})
    if enforce_pre_holdout:
        enforce_pre_holdout_dates(
            funding_daily.index,
            cutoff=holdout_cutoff,
            context="funding_carry_v2.carry_btc_eth_cons_weights",
            allow_empty=False,
        )
    _validate_positive_int(lookback_days, "lookback_days", minimum=1)
    _validate_positive_int(min_valid_days, "min_valid_days", minimum=1)
    _validate_positive_int(warmup_days, "warmup_days", minimum=1)
    enter_ann = _validate_finite_float(enter_ann, "enter_ann")
    exit_ann = _validate_finite_float(exit_ann, "exit_ann")
    if enter_ann < exit_ann:
        raise ValueError("enter_ann must be >= exit_ann.")

    f_arr = funding_daily.to_numpy(dtype=np.float64, copy=False)
    s_arr = spot_close.to_numpy(dtype=np.float64, copy=False)
    p_arr = perp_close.to_numpy(dtype=np.float64, copy=False)
    _validate_positive_prices(s_arr, "spot_close")
    _validate_positive_prices(p_arr, "perp_close")

    tradable = np.isfinite(f_arr) & np.isfinite(s_arr) & (s_arr > 0.0) & np.isfinite(p_arr) & (p_arr > 0.0)
    roll = _rolling_trailing_mean(f_arr, lookback_days, min_valid_days)

    cols = [str(c) for c in funding_daily.columns]
    target_idxs = [cols.index(sym) for sym in symbols if sym in cols]
    n_t, n_k = f_arr.shape
    w_out = np.zeros((n_t, n_k), dtype=np.float64)
    state = {j: False for j in target_idxs}
    enter_d = enter_ann / 365.0
    exit_d = exit_ann / 365.0
    per_coin_notional = 1.0 / float(max(1, len(symbols)))

    start_t = max(warmup_days, lookback_days - 1)
    for t in range(start_t, n_t):
        for j in target_idxs:
            r7 = roll[t, j]
            if (not tradable[t, j]) or (not np.isfinite(r7)):
                state[j] = False
            elif state[j]:
                if r7 <= exit_d:
                    state[j] = False
            elif r7 >= enter_d:
                state[j] = True
        for j, on in state.items():
            if on:
                w_out[t, j] = per_coin_notional

    return pd.DataFrame(w_out, index=funding_daily.index, columns=funding_daily.columns)


def carry_top_k_hysteresis_weights(
    funding_daily: pd.DataFrame,
    spot_close: pd.DataFrame,
    perp_close: pd.DataFrame,
    perp_quote_volume: pd.DataFrame | None = None,
    *,
    k_max: int = 5,
    keep_rank_limit: int = 8,
    enter_ann: float = 0.12,
    exit_ann: float = 0.04,
    max_single: float = 0.25,
    tier1_top_n: int | None = 12,
    lookback_days: int = 7,
    min_valid_days: int = 6,
    dv_lookback_days: int = 30,
    dv_min_days: int = 15,
    warmup_days: int = 14,
    use_invvol: bool = False,
    vol_window: int = 20,
    rebal_deadband: float = 0.05,
    enforce_pre_holdout: bool = True,
    holdout_cutoff: str = HOLDOUT_CUTOFF_DATE,
) -> pd.DataFrame:
    """Causal Multi-Coin Top-K Hysteresis Funding Carry v2 (`F9_TOP5_HYST_12_4` default)."""
    aligned: dict[str, pd.DataFrame] = {"spot_close": spot_close, "perp_close": perp_close}
    if perp_quote_volume is not None:
        aligned["perp_quote_volume"] = perp_quote_volume
    _validate_aligned_frames(funding_daily, aligned)
    if enforce_pre_holdout:
        enforce_pre_holdout_dates(
            funding_daily.index,
            cutoff=holdout_cutoff,
            context="funding_carry_v2.carry_top_k_hysteresis_weights",
            allow_empty=False,
        )

    _validate_positive_int(k_max, "k_max", minimum=1)
    _validate_positive_int(keep_rank_limit, "keep_rank_limit", minimum=k_max)
    _validate_positive_int(lookback_days, "lookback_days", minimum=1)
    _validate_positive_int(min_valid_days, "min_valid_days", minimum=1)
    _validate_positive_int(warmup_days, "warmup_days", minimum=1)
    _validate_positive_int(vol_window, "vol_window", minimum=2)
    enter_ann = _validate_finite_float(enter_ann, "enter_ann")
    exit_ann = _validate_finite_float(exit_ann, "exit_ann")
    if enter_ann < exit_ann:
        raise ValueError("enter_ann must be >= exit_ann.")
    max_single = _validate_finite_float(max_single, "max_single", strictly_positive=True)
    if max_single > 1.0:
        raise ValueError("max_single must be <= 1.0.")
    rebal_deadband = _validate_finite_float(rebal_deadband, "rebal_deadband", minimum=0.0)

    f_arr = funding_daily.to_numpy(dtype=np.float64, copy=False)
    s_arr = spot_close.to_numpy(dtype=np.float64, copy=False)
    p_arr = perp_close.to_numpy(dtype=np.float64, copy=False)
    _validate_positive_prices(s_arr, "spot_close")
    _validate_positive_prices(p_arr, "perp_close")
    n_t, n_k = f_arr.shape
    cols = [str(c) for c in funding_daily.columns]

    tradable = np.isfinite(f_arr) & np.isfinite(s_arr) & (s_arr > 0.0) & np.isfinite(p_arr) & (p_arr > 0.0)
    roll7 = _rolling_trailing_mean(f_arr, lookback_days, min_valid_days)

    dv30 = np.full((n_t, n_k), np.nan, dtype=np.float64)
    if perp_quote_volume is not None and tier1_top_n is not None:
        qv_arr = perp_quote_volume.to_numpy(dtype=np.float64, copy=False)
        for t in range(dv_min_days - 1, n_t):
            lo = max(0, t - dv_lookback_days + 1)
            win = qv_arr[lo : t + 1]
            ok = np.isfinite(win) & (win >= 0.0)
            cnt = np.sum(ok, axis=0)
            tot = np.sum(np.where(ok, win, 0.0), axis=0)
            dv30[t] = np.where(cnt >= dv_min_days, tot / np.maximum(1, cnt), np.nan)

    vol20 = np.full((n_t, n_k), np.nan, dtype=np.float64)
    if use_invvol:
        perp_ret = np.full((n_t, n_k), np.nan, dtype=np.float64)
        ok_p = np.isfinite(p_arr[1:]) & np.isfinite(p_arr[:-1]) & (p_arr[:-1] > 0.0)
        perp_ret[1:][ok_p] = p_arr[1:][ok_p] / p_arr[:-1][ok_p] - 1.0
        for t in range(vol_window - 1, n_t):
            wr = perp_ret[t - vol_window + 1 : t + 1]
            for j in range(n_k):
                vals = wr[:, j]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= max(2, int(math.ceil(vol_window * 0.60))):
                    vol20[t, j] = max(0.005, float(np.std(vals, ddof=1)))

    enter_d = enter_ann / 365.0
    exit_d = exit_ann / 365.0
    w_out = np.zeros((n_t, n_k), dtype=np.float64)
    held: set[int] = set()
    prev_held: set[int] = set()
    prev_w: dict[int, float] = {}

    start_t = max(warmup_days, lookback_days - 1)
    for t in range(start_t, n_t):
        if tier1_top_n is not None and perp_quote_volume is not None:
            liq_ranked = [
                (float(dv30[t, j]) if np.isfinite(dv30[t, j]) else 0.0, cols[j], j)
                for j in range(n_k)
                if tradable[t, j]
            ]
            liq_ranked.sort(key=lambda x: (-x[0], x[1]))
            allowed = {j for _v, _nm, j in liq_ranked[:tier1_top_n]}
        else:
            allowed = {j for j in range(n_k) if tradable[t, j]}

        ranked = [(float(roll7[t, j]), cols[j], j) for j in allowed if np.isfinite(roll7[t, j])]
        ranked.sort(key=lambda x: (-x[0], x[1]))
        rank_pos = {j: idx for idx, (_val, _nm, j) in enumerate(ranked)}

        next_held = {
            j for j in held if j in rank_pos and rank_pos[j] < keep_rank_limit and roll7[t, j] >= exit_d
        }
        for val7, _nm, j in ranked:
            if len(next_held) >= k_max:
                break
            if j not in next_held and val7 > enter_d:
                next_held.add(j)

        held = next_held
        if not held:
            prev_held = set()
            prev_w = {}
            continue

        act = sorted(held)
        if use_invvol:
            w_fresh = _invvol_capped_weights(act, vol20[t], max_single=max_single)
        else:
            per_c = min(max_single, 1.0 / float(len(act)))
            w_fresh = {j: per_c for j in act}

        if (
            use_invvol
            and held == prev_held
            and prev_w
            and max(abs(w_fresh[j] - prev_w.get(j, 0.0)) for j in act) < rebal_deadband
        ):
            w_use = prev_w
        else:
            w_use = w_fresh
            prev_held = set(held)
            prev_w = dict(w_use)

        for j, wj in w_use.items():
            w_out[t, j] = wj

    return pd.DataFrame(w_out, index=funding_daily.index, columns=funding_daily.columns)


def carry_v2_blend_weights(
    w_core: pd.DataFrame,
    w_satellite: pd.DataFrame,
    *,
    core_share: float = 0.70,
    satellite_share: float = 0.30,
) -> pd.DataFrame:
    """Convexly blend Core BTC+ETH carry and Satellite Top-5 Hysteresis carry (`sum <= 1.0`)."""
    _validate_panel(w_core, "w_core")
    _validate_panel(w_satellite, "w_satellite")
    if not w_core.index.equals(w_satellite.index) or list(w_core.columns) != list(w_satellite.columns):
        raise ValueError("w_core and w_satellite must share identical index and columns.")
    c_sh = _validate_finite_float(core_share, "core_share", minimum=0.0)
    s_sh = _validate_finite_float(satellite_share, "satellite_share", minimum=0.0)
    if c_sh + s_sh <= 0.0 or c_sh + s_sh > 1.0 + 1e-9:
        raise ValueError("core_share + satellite_share must be in (0, 1].")
    blended = c_sh * w_core.fillna(0.0) + s_sh * w_satellite.fillna(0.0)
    row_sums = blended.sum(axis=1)
    scale = pd.Series(np.where(row_sums > 1.0, 1.0 / row_sums, 1.0), index=blended.index)
    return blended.mul(scale, axis=0)


def compute_carry_btc_eth_live_step(
    funding_hist_8h: Mapping[str, Sequence[float]],
    prev_state: Mapping[str, bool] | None = None,
    *,
    symbols: tuple[str, str] = ("BTCUSDT", "ETHUSDT"),
    equity_buffer_mult: float = 2.0,
    enter_ann: float = 0.03,
    exit_ann: float = 0.0,
    min_settlements: int = 18,
    window_settlements: int = 21,
) -> dict[str, Any]:
    """Single-step live/shadow causal `CARRY_BTC_ETH_CONS_2XBUF` target notional weights."""
    buf = _validate_finite_float(equity_buffer_mult, "equity_buffer_mult", minimum=1.0)
    enter_ann = _validate_finite_float(enter_ann, "enter_ann")
    exit_ann = _validate_finite_float(exit_ann, "exit_ann")
    if enter_ann < exit_ann:
        raise ValueError("enter_ann must be >= exit_ann.")
    _validate_positive_int(min_settlements, "min_settlements", minimum=1)
    _validate_positive_int(window_settlements, "window_settlements", minimum=min_settlements)

    prev = dict(prev_state or {sym: False for sym in symbols})
    notional_cap = 1.0 / buf
    per_coin_eq_w = notional_cap / float(max(1, len(symbols)))

    new_state: dict[str, bool] = {}
    notional_w: dict[str, float] = {}
    stats: dict[str, dict[str, Any]] = {}

    for sym in symbols:
        raw_rates = funding_hist_8h.get(sym, ())
        clean_rates = [float(x) for x in raw_rates if math.isfinite(float(x))]
        if len(clean_rates) < min_settlements:
            new_state[sym] = False
            stats[sym] = {
                "n_settlements": len(clean_rates),
                "trail_7d_ann_pct": None,
                "active": False,
            }
            continue
        w_tail = clean_rates[-window_settlements:]
        ann_7d = float(np.mean(w_tail)) * 3.0 * 365.0
        was_on = bool(prev.get(sym, False))
        is_on = (ann_7d > exit_ann) if was_on else (ann_7d >= enter_ann)
        new_state[sym] = is_on
        if is_on:
            notional_w[sym] = round(per_coin_eq_w, 6)
        stats[sym] = {
            "n_settlements": len(w_tail),
            "last_8h_bps": round(float(w_tail[-1]) * 10_000.0, 3),
            "trail_7d_ann_pct": round(ann_7d * 100.0, 2),
            "active": is_on,
        }

    return {
        "equity_buffer_mult": buf,
        "state": new_state,
        "notional_weights_of_equity": notional_w,
        "gross_carry_notional_pct": round(sum(notional_w.values()) * 100.0, 1),
        "per_coin": stats,
    }


def run_funding_carry_backtest(
    funding_daily: pd.DataFrame,
    spot_close: pd.DataFrame,
    perp_close: pd.DataFrame,
    target_notional_weights: pd.DataFrame,
    *,
    perp_high: pd.DataFrame | None = None,
    spot_high: pd.DataFrame | None = None,
    config: FundingCarryConfig | None = None,
    enforce_pre_holdout: bool = True,
    holdout_cutoff: str = HOLDOUT_CUTOFF_DATE,
) -> dict[str, Any]:
    """Run causal t -> t+1 Spot Long + 1x Perp Short carry simulation with exit-shock & margin audit."""
    cfg = config or FundingCarryConfig()
    aligned: dict[str, pd.DataFrame] = {
        "spot_close": spot_close,
        "perp_close": perp_close,
        "target_notional_weights": target_notional_weights,
    }
    if perp_high is not None:
        aligned["perp_high"] = perp_high
    if spot_high is not None:
        aligned["spot_high"] = spot_high
    _validate_aligned_frames(funding_daily, aligned)

    if enforce_pre_holdout:
        enforce_pre_holdout_dates(
            funding_daily.index,
            cutoff=holdout_cutoff,
            context="funding_carry_v2.run_funding_carry_backtest",
            allow_empty=False,
        )

    f_arr = funding_daily.to_numpy(dtype=np.float64, copy=False)
    s_arr = spot_close.to_numpy(dtype=np.float64, copy=False)
    p_arr = perp_close.to_numpy(dtype=np.float64, copy=False)
    _validate_positive_prices(s_arr, "spot_close")
    _validate_positive_prices(p_arr, "perp_close")

    tw = target_notional_weights.fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(tw).all() or (tw < -1e-12).any():
        raise ValueError("Carry target_notional_weights must be finite and non-negative.")
    tw = np.clip(tw, 0.0, 1.0)
    rs = tw.sum(axis=1, keepdims=True)
    if (rs > 1.0 + 1e-6).any():
        raise ValueError("Carry target_notional_weights row sum cannot exceed 1.0.")

    notional_cap = 1.0 / cfg.equity_buffer_mult
    tw_eq = np.where(rs > 1.0, tw / np.maximum(rs, 1e-12), tw) * notional_cap

    n_t, n_k = f_arr.shape
    held = np.zeros((n_t, n_k), dtype=np.float64)
    valid_f = np.isfinite(f_arr[1:])
    held[1:] = np.where(valid_f, tw_eq[:-1], 0.0)

    prev = np.zeros_like(held)
    prev[1:] = held[:-1]
    turnover_notional = np.sum(np.abs(held - prev), axis=1)
    total_oneway_bps = cfg.oneway_pair_cost_bps + cfg.basis_cross_slip_bps
    fee_cost = turnover_notional * (total_oneway_bps / 10_000.0)

    safe_f = np.where(np.isfinite(f_arr), f_arr, 0.0)
    funding_pnl = np.sum(held * safe_f, axis=1)

    r_spot = np.zeros((n_t, n_k), dtype=np.float64)
    r_perp = np.zeros((n_t, n_k), dtype=np.float64)
    ok_s = np.isfinite(s_arr[1:]) & np.isfinite(s_arr[:-1]) & (s_arr[:-1] > 0.0)
    ok_p = np.isfinite(p_arr[1:]) & np.isfinite(p_arr[:-1]) & (p_arr[:-1] > 0.0)
    ok_both = ok_s & ok_p
    r_spot[1:][ok_both] = s_arr[1:][ok_both] / s_arr[:-1][ok_both] - 1.0
    r_perp[1:][ok_both] = p_arr[1:][ok_both] / p_arr[:-1][ok_both] - 1.0
    diff_sp = np.clip(r_spot - r_perp, -0.15, 0.15)
    basis_pnl = np.sum(held * diff_sp, axis=1) if cfg.include_basis_mtm else np.zeros(n_t, dtype=np.float64)

    # Adverse negative-funding / basis-widening exit-shock penalty on reductions
    exit_notional = np.maximum(0.0, prev - held)
    adverse_rate = (
        np.clip(
            0.5 * np.maximum(0.0, -diff_sp) + np.maximum(0.0, -safe_f),
            0.0,
            0.0025,
        )
        * cfg.exit_shock_mult
    )
    exit_shock_cost = np.sum(exit_notional * adverse_rate, axis=1)

    total_cost = fee_cost + exit_shock_cost
    gross_ret = funding_pnl + basis_pnl
    net_ret = gross_ret - total_cost
    equity = np.cumprod(np.maximum(0.0, 1.0 + net_ret))
    if (equity <= 0.0).any():
        first_zero = int(np.flatnonzero(equity <= 0.0)[0])
        net_ret[first_zero] = -1.0
        equity[first_zero] = 0.0
        if first_zero + 1 < n_t:
            held[first_zero + 1 :] = 0.0
            funding_pnl[first_zero + 1 :] = 0.0
            basis_pnl[first_zero + 1 :] = 0.0
            total_cost[first_zero + 1 :] = 0.0
            turnover_notional[first_zero + 1 :] = 0.0
            gross_ret[first_zero + 1 :] = 0.0
            net_ret[first_zero + 1 :] = 0.0
            equity[first_zero + 1 :] = 0.0

    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdown = np.where(peak > 0.0, equity / peak - 1.0, -1.0)
    gross_exposure = np.sum(held, axis=1)

    # Mark-Price liquidation stress audit
    perp_margin_ratio = max(0.05, cfg.equity_buffer_mult - 1.0)
    liq_threshold_jump = perp_margin_ratio - 0.05
    ph_arr = perp_high.to_numpy(dtype=np.float64, copy=False) if perp_high is not None else p_arr
    if spot_high is not None:
        sh_arr = spot_high.to_numpy(dtype=np.float64, copy=False)
        mark_high = np.where(np.isfinite(sh_arr), np.minimum(ph_arr, sh_arr), ph_arr)
    else:
        mark_high = ph_arr

    max_single_jump = 0.0
    max_cross_draw = 0.0
    margin_warn_days = 0
    isolated_liq_events = 0
    cross_liq_events = 0
    for t in range(1, n_t):
        act = held[t] > 1e-6
        if not np.any(act):
            continue
        ok = act & np.isfinite(mark_high[t]) & np.isfinite(p_arr[t - 1]) & (p_arr[t - 1] > 0.0)
        if not np.any(ok):
            continue
        jumps = mark_high[t, ok] / p_arr[t - 1, ok] - 1.0
        mx = float(np.max(jumps))
        if mx > max_single_jump:
            max_single_jump = mx
        if mx >= 0.50 * liq_threshold_jump:
            margin_warn_days += 1
        if mx >= liq_threshold_jump:
            isolated_liq_events += 1
        spot_used = float(np.sum(held[t]))
        futures_margin_pool = max(1e-6, 1.0 - spot_used)
        worst_short_loss = float(np.sum(held[t, ok] * np.maximum(0.0, jumps)))
        cm_draw = worst_short_loss / futures_margin_pool
        if cm_draw > max_cross_draw:
            max_cross_draw = cm_draw
        if cm_draw >= 0.90:
            cross_liq_events += 1

    bt = pd.DataFrame(
        {
            "funding_pnl": funding_pnl,
            "basis_pnl": basis_pnl,
            "gross_return": gross_ret,
            "cost": total_cost,
            "exit_shock_cost": exit_shock_cost,
            "strategy_return": net_ret,
            "turnover": turnover_notional,
            "position": gross_exposure,
            "gross_exposure": gross_exposure,
            "equity": equity,
            "drawdown": drawdown,
        },
        index=funding_daily.index,
    )
    metrics = performance_metrics(bt, periods_per_year=365)
    return {
        "backtest": bt,
        "metrics": metrics,
        "held_notional": pd.DataFrame(held, index=funding_daily.index, columns=funding_daily.columns),
        "margin_stress": {
            "equity_buffer_mult": cfg.equity_buffer_mult,
            "notional_cap": notional_cap,
            "perp_margin_ratio": perp_margin_ratio,
            "max_single_coin_mark_jump_pct": round(max_single_jump * 100.0, 2),
            "max_cross_margin_draw_pct": round(max_cross_draw * 100.0, 2),
            "margin_warn_days": margin_warn_days,
            "isolated_liq_events": isolated_liq_events,
            "cross_liq_events": cross_liq_events,
        },
        "holdout_rows_analysed": 0,
        "order_send_capable": ORDER_SEND_CAPABLE,
    }


__all__ = [
    "FundingCarryConfig",
    "carry_btc_eth_cons_weights",
    "carry_top_k_hysteresis_weights",
    "carry_v2_blend_weights",
    "compute_carry_btc_eth_live_step",
    "run_funding_carry_backtest",
]
