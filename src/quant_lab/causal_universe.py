"""Survivorship-bias-free causal Top-N crypto spot universe & Sleeve #2 strategies.

Promoted per CRYPTO-PROMOTION-SPEC-20261124.md:
  - Strictly causal rolling dollar-volume Top-N selector using data <= bar t.
  - Spot long/cash invariants only (0 <= w_j, sum(w) <= 1.0, zero leverage, zero shorting).
  - Strictly next-bar execution (t -> t+1) with explicit fee, slippage, and delisting haircut.
  - Fail-closed offline PaperSoakReceipt integration (`PREP_ONLY_OFFLINE`, OrderSend = 0).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from quant_trading.paper_soak import (
    PaperSoakDay,
    PaperSoakReceipt,
    PaperSoakRestartCheck,
)


def _validate_panel(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} panel cannot be empty.")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and sorted ascending.")


def _canonical_sha256(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def causal_rolling_top_n_mask(
    quote_volume: pd.DataFrame,
    close: pd.DataFrame,
    *,
    lookback_bars: int = 30,
    top_n: int = 15,
    min_valid_frac: float = 0.90,
) -> pd.DataFrame:
    """Select Top-N assets at bar t strictly from trailing quote_volume[t-lookback+1 .. t].

    Never forward-fills pre-listing NaNs; requires >= min_valid_frac positive finite volume bars
    in the trailing window and a finite positive close at bar t.
    """
    _validate_panel(quote_volume, "quote_volume")
    _validate_panel(close, "close")
    if not quote_volume.index.equals(close.index) or list(quote_volume.columns) != list(close.columns):
        raise ValueError("quote_volume and close must share identical index and columns.")
    if lookback_bars <= 0 or top_n <= 0:
        raise ValueError("lookback_bars and top_n must be positive.")
    if not 0.0 < min_valid_frac <= 1.0:
        raise ValueError("min_valid_frac must be in (0, 1].")

    qv = quote_volume.to_numpy(dtype=np.float64, copy=False)
    cl = close.to_numpy(dtype=np.float64, copy=False)
    n_t, n_k = qv.shape
    mask = np.zeros((n_t, n_k), dtype=bool)
    min_req = max(1, int(math.ceil(lookback_bars * min_valid_frac)))

    for t in range(lookback_bars - 1, n_t):
        win = qv[t - lookback_bars + 1 : t + 1]
        valid_cnt = np.sum(np.isfinite(win) & (win > 0.0), axis=0)
        cur_ok = np.isfinite(cl[t]) & (cl[t] > 0.0) & (valid_cnt >= min_req)
        if not np.any(cur_ok):
            continue
        safe = np.where(np.isfinite(win) & (win > 0.0), win, 0.0)
        mean_qv = np.where(cur_ok, np.sum(safe, axis=0) / np.maximum(1, valid_cnt), -1.0)
        elig_idx = np.flatnonzero(cur_ok)
        if len(elig_idx) <= top_n:
            mask[t, elig_idx] = True
        else:
            order = sorted(elig_idx.tolist(), key=lambda j: (-float(mean_qv[j]), j))
            mask[t, order[:top_n]] = True

    return pd.DataFrame(mask, index=close.index, columns=close.columns)


def _causal_ema_matrix(closes: np.ndarray, span: int) -> np.ndarray:
    n_t, n_k = closes.shape
    out = np.full((n_t, n_k), np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    for j in range(n_k):
        prev = np.nan
        for t in range(n_t):
            c = closes[t, j]
            if not np.isfinite(c):
                prev = np.nan
                continue
            prev = c if not np.isfinite(prev) else (alpha * c + (1.0 - alpha) * prev)
            out[t, j] = prev
    return out


def _causal_rolling_vol(closes: np.ndarray, window: int) -> np.ndarray:
    n_t, n_k = closes.shape
    rets = np.full((n_t, n_k), np.nan, dtype=np.float64)
    valid = np.isfinite(closes[1:]) & np.isfinite(closes[:-1]) & (closes[:-1] > 0.0)
    rets[1:][valid] = closes[1:][valid] / closes[:-1][valid] - 1.0

    ok = np.isfinite(rets)
    rz = np.where(ok, rets, 0.0)
    cs_cnt = np.cumsum(ok.astype(np.int64), axis=0)
    cs_s1 = np.cumsum(rz, axis=0)
    cs_s2 = np.cumsum(rz * rz, axis=0)

    out = np.full((n_t, n_k), np.nan, dtype=np.float64)
    if n_t > window:
        cnt = cs_cnt[window:] - cs_cnt[:-window]
        s1 = cs_s1[window:] - cs_s1[:-window]
        s2 = cs_s2[window:] - cs_s2[:-window]
        full = cnt == window
        mean = s1 / window
        var = np.maximum(0.0, (s2 - window * mean * mean) / max(1, window - 1))
        out[window:] = np.where(full, np.sqrt(var), np.nan)
    return out


def _apply_rebalance_deadband(raw_w: np.ndarray, deadband: float) -> np.ndarray:
    if deadband <= 0.0:
        return raw_w
    out = np.zeros_like(raw_w)
    prev = np.zeros(raw_w.shape[1], dtype=np.float64)
    for t in range(len(raw_w)):
        cur = raw_w[t]
        flip = (cur == 0.0) != (prev == 0.0)
        upd = np.where(flip | (np.abs(cur - prev) >= deadband), cur, prev)
        total = float(upd.sum())
        if total > 1.0:
            upd = upd / total
        out[t] = upd
        prev = upd
    return out


def f4_ema_trend_btc_weights(
    close: pd.DataFrame,
    eligible_mask: pd.DataFrame,
    *,
    btc_symbol: str = "BTCUSDT",
    fast_span: int = 24,
    slow_span: int = 120,
    btc_fast: int = 60,
    btc_slow: int = 300,
    vol_window: int = 120,
    target_ann_vol: float = 0.15,
    bars_per_year: float = 365.0 * 6.0,
    deadband: float = 0.015,
) -> pd.DataFrame:
    """Sleeve #2 Primary (`F4_4H_EMA_24_120_BTC`): Causal EMA trend + BTC macro filter + vol targeting."""
    _validate_panel(close, "close")
    if not close.index.equals(eligible_mask.index) or list(close.columns) != list(eligible_mask.columns):
        raise ValueError("close and eligible_mask must share identical index and columns.")
    if btc_symbol not in close.columns:
        raise ValueError(f"btc_symbol '{btc_symbol}' must exist in close columns.")
    if fast_span <= 0 or slow_span <= fast_span or btc_fast <= 0 or btc_slow <= btc_fast:
        raise ValueError("EMA spans must satisfy 0 < fast < slow.")

    cl = close.to_numpy(dtype=np.float64, copy=False)
    elig = eligible_mask.to_numpy(dtype=bool, copy=False)
    btc_idx = list(close.columns).index(btc_symbol)

    ema_f = _causal_ema_matrix(cl, fast_span)
    ema_s = _causal_ema_matrix(cl, slow_span)
    btc_col = cl[:, btc_idx : btc_idx + 1]
    btc_ef = _causal_ema_matrix(btc_col, btc_fast)[:, 0]
    btc_es = _causal_ema_matrix(btc_col, btc_slow)[:, 0]
    vol = _causal_rolling_vol(cl, vol_window)

    vt_bar = target_ann_vol / math.sqrt(bars_per_year)
    inv_vol = np.where(np.isfinite(vol) & (vol > 1e-6), vt_bar / vol, 0.0)
    n_elig = np.maximum(1, elig.sum(axis=1, keepdims=True))
    btc_bull = (np.isfinite(btc_ef) & np.isfinite(btc_es) & (btc_ef > btc_es))[:, None]

    sig = elig & btc_bull & np.isfinite(ema_f) & np.isfinite(ema_s) & (ema_f > ema_s)
    raw = np.where(sig, inv_vol, 0.0) / n_elig
    row_sum = raw.sum(axis=1, keepdims=True)
    capped = np.where(row_sum > 1.0, raw / np.maximum(row_sum, 1e-12), raw)
    weights = _apply_rebalance_deadband(capped, deadband)
    return pd.DataFrame(weights, index=close.index, columns=close.columns)


def f3_donchian_breakout_weights(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    eligible_mask: pd.DataFrame,
    *,
    entry_window: int = 120,
    exit_window: int = 60,
    vol_window: int = 120,
    target_ann_vol: float = 0.15,
    bars_per_year: float = 365.0 * 6.0,
    deadband: float = 0.015,
) -> pd.DataFrame:
    """Sleeve #2 Low-Turnover Backup (`F3_DONCHIAN_4H_120_60`): Causal Donchian breakout + vol targeting."""
    _validate_panel(close, "close")
    for name, df in (("high", high), ("low", low), ("eligible_mask", eligible_mask)):
        if not close.index.equals(df.index) or list(close.columns) != list(df.columns):
            raise ValueError(f"{name} must align exactly with close.")
    if entry_window <= 0 or exit_window <= 0 or exit_window >= entry_window:
        raise ValueError("Donchian windows must satisfy 0 < exit_window < entry_window.")

    cl = close.to_numpy(dtype=np.float64, copy=False)
    hi = high.to_numpy(dtype=np.float64, copy=False)
    lo = low.to_numpy(dtype=np.float64, copy=False)
    elig = eligible_mask.to_numpy(dtype=bool, copy=False)
    n_t, n_k = cl.shape

    vol = _causal_rolling_vol(cl, vol_window)
    vt_bar = target_ann_vol / math.sqrt(bars_per_year)
    inv_vol = np.where(np.isfinite(vol) & (vol > 1e-6), vt_bar / vol, 0.0)

    state = np.zeros(n_k, dtype=bool)
    raw = np.zeros((n_t, n_k), dtype=np.float64)
    start_t = max(entry_window, exit_window)

    for t in range(start_t, n_t):
        prior_hi = np.full(n_k, np.nan, dtype=np.float64)
        prior_lo = np.full(n_k, np.nan, dtype=np.float64)
        h_win = hi[t - entry_window : t]
        l_win = lo[t - exit_window : t]
        h_ok = np.all(np.isfinite(h_win), axis=0)
        l_ok = np.all(np.isfinite(l_win), axis=0)
        if np.any(h_ok):
            prior_hi[h_ok] = np.max(h_win[:, h_ok], axis=0)
        if np.any(l_ok):
            prior_lo[l_ok] = np.min(l_win[:, l_ok], axis=0)

        c = cl[t]
        el = elig[t]
        exit_cond = (~np.isfinite(c)) | (~el) | (np.isfinite(prior_lo) & (c < prior_lo))
        enter_cond = el & np.isfinite(c) & np.isfinite(prior_hi) & (c > prior_hi)
        state = np.where(state, ~exit_cond, enter_cond)
        if np.any(state):
            row = np.where(state, inv_vol[t], 0.0) / max(1, int(el.sum()))
            total = float(row.sum())
            raw[t] = row / total if total > 1.0 else row

    weights = _apply_rebalance_deadband(raw, deadband)
    return pd.DataFrame(weights, index=close.index, columns=close.columns)


def run_multi_asset_spot_backtest(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    delist_haircut_bps: float = 100.0,
) -> pd.DataFrame:
    """Strictly causal t -> t+1 multi-asset spot long/cash backtest with delisting exit haircut."""
    _validate_panel(close, "close")
    if not close.index.equals(target_weights.index) or list(close.columns) != list(target_weights.columns):
        raise ValueError("close and target_weights must share identical index and columns.")
    if fee_bps < 0 or slippage_bps < 0 or delist_haircut_bps < 0:
        raise ValueError("Costs and haircuts cannot be negative.")

    cl = close.to_numpy(dtype=np.float64, copy=False)
    tw = target_weights.fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(tw).all() or (tw < -1e-12).any():
        raise ValueError("Spot target_weights must be finite and non-negative.")
    tw = np.clip(tw, 0.0, 1.0)
    row_sums = tw.sum(axis=1, keepdims=True)
    if (row_sums > 1.0 + 1e-6).any():
        raise ValueError("Spot target_weights row sum cannot exceed 1.0 (zero leverage).")
    tw = np.where(row_sums > 1.0, tw / np.maximum(row_sums, 1e-12), tw)

    n_t, n_k = cl.shape
    rets = np.zeros((n_t, n_k), dtype=np.float64)
    valid_ret = np.isfinite(cl[1:]) & np.isfinite(cl[:-1]) & (cl[:-1] > 0.0)
    rets[1:][valid_ret] = cl[1:][valid_ret] / cl[:-1][valid_ret] - 1.0

    # Strict causal shift: target_weights[t] becomes held[t+1]
    held = np.zeros_like(tw)
    held[1:] = tw[:-1]
    delist_w = np.zeros_like(tw)
    missing_next = (~valid_ret) & (tw[:-1] > 1e-12)
    delist_w[1:][missing_next] = tw[:-1][missing_next]
    held[1:][~valid_ret] = 0.0

    prev = np.zeros_like(held)
    prev[1:] = held[:-1]
    turnover = np.sum(np.abs(held - prev), axis=1)
    cost = turnover * ((fee_bps + slippage_bps) / 10_000.0) + np.sum(delist_w, axis=1) * (
        delist_haircut_bps / 10_000.0
    )
    gross = np.sum(held * rets, axis=1)
    net = gross - cost
    equity = np.cumprod(1.0 + net)
    drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1.0

    return pd.DataFrame(
        {
            "gross_return": gross,
            "cost": cost,
            "strategy_return": net,
            "turnover": turnover,
            "gross_exposure": np.sum(held, axis=1),
            "delist_exit_weight": np.sum(delist_w, axis=1),
            "equity": equity,
            "drawdown": drawdown,
        },
        index=close.index,
    )


def build_sleeve2_paper_soak_day(
    *,
    date_utc: str,
    input_cutoff_utc: int,
    target_weights: dict[str, float],
    realized_pnl_vnd: float,
    unrealized_pnl_vnd: float,
    fees_vnd: float,
    slippage_bps: float,
    equity_vnd: float,
    reconciled: bool = True,
    healthy: bool = True,
    pause_reason: str | None = None,
) -> PaperSoakDay:
    """Create a canonical-hashed PaperSoakDay observation for Sleeve #2 offline soak accounting."""
    active_intents = sum(1 for v in target_weights.values() if abs(float(v)) > 1e-9)
    equity_hash = _canonical_sha256({"date_utc": date_utc, "equity_vnd": round(float(equity_vnd), 2)})
    ledger_hash = _canonical_sha256(
        {"date_utc": date_utc, "weights": {k: round(float(v), 6) for k, v in sorted(target_weights.items())}}
    )
    return PaperSoakDay(
        date_utc=date_utc,
        input_cutoff_utc=input_cutoff_utc,
        candidate_count=len(target_weights),
        paper_intents=active_intents,
        fills=active_intents if reconciled else 0,
        realized_pnl_vnd=realized_pnl_vnd,
        unrealized_pnl_vnd=unrealized_pnl_vnd,
        fees_vnd=fees_vnd,
        slippage_bps=slippage_bps,
        equity_hash=equity_hash,
        ledger_hash=ledger_hash,
        reconciliation_status="reconciled" if reconciled else "mismatch",
        health_status="healthy" if healthy else "paused",
        pause_or_stop_reason=pause_reason,
    )


def build_sleeve2_paper_soak_receipt(
    days: list[PaperSoakDay] | tuple[PaperSoakDay, ...],
    restart_checks: list[PaperSoakRestartCheck] | tuple[PaperSoakRestartCheck, ...] = (),
    *,
    run_id: str = "sleeve2-causal-top15-soak-v1",
    attempt_no: int = 1,
    fence_token: str = "offline-prep-only",
    strategy_name: str = "F4_4H_EMA_24_120_BTC",
    owner_reviewed: bool = False,
) -> PaperSoakReceipt:
    """Assemble a fail-closed PREP_ONLY_OFFLINE PaperSoakReceipt (execution_capability always False)."""
    if not days:
        return PaperSoakReceipt()
    cfg_sha = _canonical_sha256({"universe": "causal_top15_30d_qv", "fee_bps": 10.0, "slippage_bps": 5.0})
    strat_sha = _canonical_sha256({"strategy": strategy_name, "spot_only": True, "leverage": 1.0})
    risk_sha = _canonical_sha256({"max_gross": 1.0, "max_coin": 0.25, "dd_kill_pct": 12.0})

    all_ok = all(d.reconciliation_status == "reconciled" and d.health_status == "healthy" for d in days)
    restarts_ok = bool(restart_checks) and all(
        c.status == "passed" and c.duplicate_intents == 0 for c in restart_checks
    )
    status = "ACCEPTED" if (len(days) >= 30 and all_ok and restarts_ok and owner_reviewed) else "OBSERVED"
    return PaperSoakReceipt(
        daily_reconciliation=tuple(days),
        restart_checks=tuple(restart_checks),
        run_id=run_id,
        attempt_no=attempt_no,
        fence_token=fence_token,
        config_sha256=cfg_sha,
        strategy_sha256=strat_sha,
        risk_budget_sha256=risk_sha,
        owner_reviewed=owner_reviewed,
        status=status,
        evidence_level="offline_soak_evidence",
    )
