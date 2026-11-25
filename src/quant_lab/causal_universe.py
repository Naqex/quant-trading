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
from collections.abc import Mapping
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant_lab.metrics import performance_metrics
from quant_trading.paper_soak import (
    MAXIMUM_SOAK_DAYS,
    MINIMUM_SOAK_DAYS,
    PaperSoakDay,
    PaperSoakError,
    PaperSoakReceipt,
    PaperSoakRestartCheck,
    audit_paper_soak_receipt,
)


def _validate_panel(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} panel cannot be empty.")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and sorted ascending.")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} columns must be unique.")


def _validate_positive_prices(arr: np.ndarray, name: str) -> None:
    if np.isinf(arr).any():
        raise ValueError(f"{name} prices cannot contain non-finite +/-inf values.")
    finite = np.isfinite(arr)
    if (arr[finite] <= 0.0).any():
        raise ValueError(f"{name} prices must be strictly positive where present.")


def _validate_nonnegative_panel(arr: np.ndarray, name: str) -> None:
    if np.isinf(arr).any():
        raise ValueError(f"{name} cannot contain non-finite +/-inf values.")
    finite = np.isfinite(arr)
    if (arr[finite] < 0.0).any():
        raise ValueError(f"{name} cannot contain negative values.")


def _validate_positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _validate_finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    if strictly_positive and parsed <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return parsed


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
    _validate_positive_int(lookback_bars, "lookback_bars", minimum=1)
    _validate_positive_int(top_n, "top_n", minimum=1)
    min_valid_frac = _validate_finite_float(min_valid_frac, "min_valid_frac")
    if not 0.0 < min_valid_frac <= 1.0:
        raise ValueError("min_valid_frac must be in (0, 1].")

    qv = quote_volume.to_numpy(dtype=np.float64, copy=False)
    cl = close.to_numpy(dtype=np.float64, copy=False)
    _validate_nonnegative_panel(qv, "quote_volume")
    _validate_positive_prices(cl, "close")

    n_t, n_k = qv.shape
    cols = [str(col) for col in close.columns]
    mask = np.zeros((n_t, n_k), dtype=bool)
    min_req = max(1, int(math.ceil(lookback_bars * min_valid_frac)))

    for t in range(lookback_bars - 1, n_t):
        win = qv[t - lookback_bars + 1 : t + 1]
        valid_cnt = np.sum(np.isfinite(win) & (win > 0.0), axis=0)
        cur_ok = (
            np.isfinite(cl[t])
            & (cl[t] > 0.0)
            & np.isfinite(qv[t])
            & (qv[t] > 0.0)
            & (valid_cnt >= min_req)
        )
        if not np.any(cur_ok):
            continue
        safe = np.where(np.isfinite(win) & (win > 0.0), win, 0.0)
        mean_qv = np.where(cur_ok, np.sum(safe, axis=0) / float(lookback_bars), -1.0)
        elig_idx = np.flatnonzero(cur_ok)
        if len(elig_idx) <= top_n:
            mask[t, elig_idx] = True
        else:
            order = sorted(elig_idx.tolist(), key=lambda j: (-float(mean_qv[j]), cols[j]))
            mask[t, order[:top_n]] = True

    return pd.DataFrame(mask, index=close.index, columns=close.columns)


def _causal_ema_matrix(closes: np.ndarray, span: int) -> np.ndarray:
    n_t, n_k = closes.shape
    out = np.full((n_t, n_k), np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    for j in range(n_k):
        prev = np.nan
        valid_streak = 0
        for t in range(n_t):
            c = closes[t, j]
            if not np.isfinite(c) or c <= 0.0:
                prev = np.nan
                valid_streak = 0
                continue
            prev = c if not np.isfinite(prev) else (alpha * c + (1.0 - alpha) * prev)
            valid_streak += 1
            if valid_streak >= span:
                out[t, j] = prev
    return out


def _causal_rolling_vol(closes: np.ndarray, window: int) -> np.ndarray:
    _validate_positive_int(window, "vol_window", minimum=2)
    n_t, n_k = closes.shape
    rets = np.full((n_t, n_k), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(closes[1:])
        & np.isfinite(closes[:-1])
        & (closes[:-1] > 0.0)
        & (closes[1:] > 0.0)
    )
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
    deadband = _validate_finite_float(deadband, "deadband", minimum=0.0)
    if deadband == 0.0:
        return raw_w.copy()
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
    _validate_panel(eligible_mask, "eligible_mask")
    if not close.index.equals(eligible_mask.index) or list(close.columns) != list(eligible_mask.columns):
        raise ValueError("close and eligible_mask must share identical index and columns.")
    if btc_symbol not in close.columns:
        raise ValueError(f"btc_symbol '{btc_symbol}' must exist in close columns.")
    _validate_positive_int(fast_span, "fast_span", minimum=1)
    _validate_positive_int(slow_span, "slow_span", minimum=1)
    _validate_positive_int(btc_fast, "btc_fast", minimum=1)
    _validate_positive_int(btc_slow, "btc_slow", minimum=1)
    if slow_span <= fast_span or btc_slow <= btc_fast:
        raise ValueError("EMA spans must satisfy 0 < fast < slow.")
    _validate_positive_int(vol_window, "vol_window", minimum=2)
    target_ann_vol = _validate_finite_float(target_ann_vol, "target_ann_vol", strictly_positive=True)
    bars_per_year = _validate_finite_float(bars_per_year, "bars_per_year", strictly_positive=True)
    deadband = _validate_finite_float(deadband, "deadband", minimum=0.0)

    cl = close.to_numpy(dtype=np.float64, copy=False)
    _validate_positive_prices(cl, "close")
    elig = eligible_mask.to_numpy(dtype=bool, copy=False)
    btc_idx = list(close.columns).index(btc_symbol)

    ema_f = _causal_ema_matrix(cl, fast_span)
    ema_s = _causal_ema_matrix(cl, slow_span)
    btc_col = cl[:, btc_idx : btc_idx + 1]
    btc_ef = _causal_ema_matrix(btc_col, btc_fast)[:, 0]
    btc_es = _causal_ema_matrix(btc_col, btc_slow)[:, 0]
    vol = _causal_rolling_vol(cl, vol_window)

    vt_bar = target_ann_vol / math.sqrt(bars_per_year)
    valid_vol = np.isfinite(vol) & (vol > 1e-6)
    safe_vol = np.where(valid_vol, vol, 1.0)
    inv_vol = np.where(valid_vol, vt_bar / safe_vol, 0.0)
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
        _validate_panel(df, name)
        if not close.index.equals(df.index) or list(close.columns) != list(df.columns):
            raise ValueError(f"{name} must align exactly with close.")
    _validate_positive_int(entry_window, "entry_window", minimum=1)
    _validate_positive_int(exit_window, "exit_window", minimum=1)
    if exit_window >= entry_window:
        raise ValueError("Donchian windows must satisfy 0 < exit_window < entry_window.")
    _validate_positive_int(vol_window, "vol_window", minimum=2)
    target_ann_vol = _validate_finite_float(target_ann_vol, "target_ann_vol", strictly_positive=True)
    bars_per_year = _validate_finite_float(bars_per_year, "bars_per_year", strictly_positive=True)
    deadband = _validate_finite_float(deadband, "deadband", minimum=0.0)

    cl = close.to_numpy(dtype=np.float64, copy=False)
    hi = high.to_numpy(dtype=np.float64, copy=False)
    lo = low.to_numpy(dtype=np.float64, copy=False)
    _validate_positive_prices(cl, "close")
    _validate_positive_prices(hi, "high")
    _validate_positive_prices(lo, "low")
    hl_both = np.isfinite(hi) & np.isfinite(lo)
    if (hi[hl_both] < lo[hl_both]).any():
        raise ValueError("high prices cannot be below low prices.")

    elig = eligible_mask.to_numpy(dtype=bool, copy=False)
    n_t, n_k = cl.shape

    vol = _causal_rolling_vol(cl, vol_window)
    vt_bar = target_ann_vol / math.sqrt(bars_per_year)
    valid_vol = np.isfinite(vol) & (vol > 1e-6)
    safe_vol = np.where(valid_vol, vol, 1.0)
    inv_vol = np.where(valid_vol, vt_bar / safe_vol, 0.0)

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
    _validate_panel(target_weights, "target_weights")
    if not close.index.equals(target_weights.index) or list(close.columns) != list(target_weights.columns):
        raise ValueError("close and target_weights must share identical index and columns.")
    fee_bps = _validate_finite_float(fee_bps, "fee_bps", minimum=0.0)
    slippage_bps = _validate_finite_float(slippage_bps, "slippage_bps", minimum=0.0)
    delist_haircut_bps = _validate_finite_float(delist_haircut_bps, "delist_haircut_bps", minimum=0.0)
    if fee_bps + slippage_bps >= 10_000.0:
        raise ValueError("Combined trading costs must be less than 100% per unit turnover.")
    if delist_haircut_bps > 10_000.0:
        raise ValueError("delist_haircut_bps cannot exceed 100%.")

    cl = close.to_numpy(dtype=np.float64, copy=False)
    _validate_positive_prices(cl, "close")

    tw = target_weights.fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(tw).all() or (tw < -1e-12).any():
        raise ValueError("Spot target_weights must be finite and non-negative.")
    tw = np.clip(tw, 0.0, 1.0)
    row_sums = tw.sum(axis=1, keepdims=True)
    if (row_sums > 1.0 + 1e-6).any():
        raise ValueError("Spot target_weights row sum cannot exceed 1.0 (zero leverage).")
    tw = np.where(row_sums > 1.0, tw / np.maximum(row_sums, 1e-12), tw)
    if ((~np.isfinite(cl)) & (tw > 1e-12)).any():
        raise ValueError("Cannot allocate non-zero target weight where close price is missing.")

    n_t, n_k = cl.shape
    rets = np.zeros((n_t, n_k), dtype=np.float64)
    valid_ret = np.isfinite(cl[1:]) & np.isfinite(cl[:-1]) & (cl[:-1] > 0.0) & (cl[1:] > 0.0)
    rets[1:][valid_ret] = cl[1:][valid_ret] / cl[:-1][valid_ret] - 1.0

    # Strict causal shift: target_weights[t-1] is the intended weight entering bar t.
    intended = np.zeros_like(tw)
    intended[1:] = tw[:-1]
    held = intended.copy()
    delist_w = np.zeros_like(tw)
    missing_next = (~valid_ret) & (tw[:-1] > 1e-12)
    delist_w[1:][missing_next] = tw[:-1][missing_next]
    held[1:][~valid_ret] = 0.0

    # Track post-return drifted weights entering bar t so constant-mix rebalancing
    # pays true drift turnover and enter-then-delist bars pay both entry and exit.
    gross = np.sum(held * rets, axis=1)
    w_drift = np.zeros_like(held)
    denom = 1.0 + gross[:-1]
    alive_denom = denom > 1e-12
    safe_denom = np.where(alive_denom, denom, 1.0)[:, None]
    w_drift[1:] = np.where(
        alive_denom[:, None],
        held[:-1] * (1.0 + rets[:-1]) / safe_denom,
        0.0,
    )

    turnover = np.sum(np.abs(intended - w_drift) + delist_w, axis=1)
    cost = turnover * ((fee_bps + slippage_bps) / 10_000.0) + np.sum(delist_w, axis=1) * (
        delist_haircut_bps / 10_000.0
    )
    net = gross - cost
    equity = np.cumprod(np.maximum(0.0, 1.0 + net))
    if (equity <= 0.0).any():
        first_zero = int(np.flatnonzero(equity <= 0.0)[0])
        net[first_zero] = -1.0
        equity[first_zero] = 0.0
        if first_zero + 1 < n_t:
            held[first_zero + 1 :] = 0.0
            gross[first_zero + 1 :] = 0.0
            cost[first_zero + 1 :] = 0.0
            turnover[first_zero + 1 :] = 0.0
            delist_w[first_zero + 1 :] = 0.0
            net[first_zero + 1 :] = 0.0
            equity[first_zero + 1 :] = 0.0
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    drawdown = np.where(peak > 0.0, equity / peak - 1.0, -1.0)
    gross_exposure = np.sum(held, axis=1)

    return pd.DataFrame(
        {
            "gross_return": gross,
            "cost": cost,
            "strategy_return": net,
            "turnover": turnover,
            "position": gross_exposure,
            "gross_exposure": gross_exposure,
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
    target_weights: Mapping[str, float],
    realized_pnl_vnd: float,
    unrealized_pnl_vnd: float,
    fees_vnd: float,
    slippage_bps: float,
    equity_vnd: float,
    reconciled: bool = True,
    healthy: bool = True,
    pause_reason: str | None = None,
    fills: int | None = None,
) -> PaperSoakDay:
    """Create a canonical-hashed PaperSoakDay observation for Sleeve #2 offline soak accounting."""
    if not isinstance(target_weights, Mapping):
        raise ValueError("target_weights must be a mapping.")
    if type(reconciled) is not bool or type(healthy) is not bool:
        raise ValueError("reconciled and healthy must be booleans.")
    equity_vnd = _validate_finite_float(equity_vnd, "equity_vnd", minimum=0.0)

    cleaned_weights: dict[str, float] = {}
    total_weight = 0.0
    for symbol, weight in sorted(target_weights.items()):
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("target_weights keys must be non-empty symbol strings.")
        parsed_w = _validate_finite_float(weight, f"target_weights[{symbol!r}]", minimum=0.0)
        if parsed_w > 1.0 + 1e-6:
            raise ValueError("Spot target_weights cannot exceed 1.0 per asset.")
        cleaned_weights[symbol] = round(parsed_w, 6)
        total_weight += parsed_w
    if total_weight > 1.0 + 1e-6:
        raise ValueError("Spot target_weights total cannot exceed 1.0 (zero leverage).")

    active_intents = sum(1 for v in cleaned_weights.values() if v > 1e-9)
    resolved_fills = (active_intents if reconciled else 0) if fills is None else fills
    equity_hash = _canonical_sha256({"date_utc": date_utc, "equity_vnd": round(equity_vnd, 2)})
    ledger_hash = _canonical_sha256({"date_utc": date_utc, "weights": cleaned_weights})
    return PaperSoakDay(
        date_utc=date_utc,
        input_cutoff_utc=input_cutoff_utc,
        candidate_count=len(cleaned_weights),
        paper_intents=active_intents,
        fills=resolved_fills,
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
    universe_name: str = "causal_top15_30d_qv",
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    max_gross: float = 1.0,
    max_coin: float = 0.25,
    dd_kill_pct: float = 12.0,
    owner_reviewed: bool = False,
    evidence_level: str = "offline_soak_evidence",
) -> PaperSoakReceipt:
    """Assemble a fail-closed PREP_ONLY_OFFLINE PaperSoakReceipt (execution_capability always False)."""
    if not days:
        if restart_checks or owner_reviewed:
            raise PaperSoakError("Cannot attach restart checks or owner review without observed soak days.")
        return PaperSoakReceipt()
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        raise PaperSoakError("strategy_name must be a non-empty string.")
    if not isinstance(universe_name, str) or not universe_name.strip():
        raise PaperSoakError("universe_name must be a non-empty string.")
    fee_bps = _validate_finite_float(fee_bps, "fee_bps", minimum=0.0)
    slippage_bps = _validate_finite_float(slippage_bps, "slippage_bps", minimum=0.0)
    max_gross = _validate_finite_float(max_gross, "max_gross", strictly_positive=True)
    max_coin = _validate_finite_float(max_coin, "max_coin", strictly_positive=True)
    dd_kill_pct = _validate_finite_float(dd_kill_pct, "dd_kill_pct", strictly_positive=True)
    if max_gross > 1.0 or max_coin > max_gross or dd_kill_pct > 100.0:
        raise PaperSoakError(
            "Risk budget must satisfy 0 < max_coin <= max_gross <= 1.0 and dd_kill_pct <= 100."
        )

    cfg_sha = _canonical_sha256(
        {"universe": universe_name.strip(), "fee_bps": fee_bps, "slippage_bps": slippage_bps}
    )
    strat_sha = _canonical_sha256({"strategy": strategy_name.strip(), "spot_only": True, "leverage": 1.0})
    risk_sha = _canonical_sha256(
        {"max_gross": max_gross, "max_coin": max_coin, "dd_kill_pct": dd_kill_pct}
    )

    synthetic_levels = {"template_only", "offline_fixture", "offline_smoke_fixture"}
    ordered_dates = [d.date_utc for d in days]
    is_contiguous = (
        (date.fromisoformat(ordered_dates[-1]) - date.fromisoformat(ordered_dates[0])).days + 1
        == len(ordered_dates)
    )
    has_hard_failure = (
        any(d.reconciliation_status != "reconciled" or d.fills > d.paper_intents for d in days)
        or any(c.status != "passed" or c.duplicate_intents > 0 for c in restart_checks)
    )
    all_ok = all(
        d.reconciliation_status == "reconciled"
        and d.health_status == "healthy"
        and d.fills <= d.paper_intents
        for d in days
    )
    restarts_ok = bool(restart_checks) and all(
        c.status == "passed" and c.duplicate_intents == 0 for c in restart_checks
    )
    can_accept = (
        len(days) >= MINIMUM_SOAK_DAYS
        and is_contiguous
        and all_ok
        and restarts_ok
        and owner_reviewed
        and evidence_level not in synthetic_levels
    )
    if has_hard_failure:
        status: Literal["OBSERVED", "ACCEPTED", "REJECTED"] = "REJECTED"
    elif can_accept:
        status = "ACCEPTED"
    elif len(days) >= MINIMUM_SOAK_DAYS and owner_reviewed:
        status = "REJECTED"
    else:
        status = "OBSERVED"
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
        evidence_level=evidence_level,
    )


def run_sleeve2_offline_smoke(
    *,
    strategy: Literal["f4", "f3"] = "f4",
    soak_days: int = 5,
    seed: int = 2026,
    initial_equity_vnd: float = 250_000_000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> dict[str, Any]:
    """Run a deterministic offline Sleeve #2 paper-soak smoke pipeline (zero network, OrderSend = 0).

    Emits an OBSERVED `offline_smoke_fixture` receipt that can never spoof an ACCEPTED 30-day soak.
    """
    if strategy not in {"f4", "f3"}:
        raise ValueError("strategy must be 'f4' or 'f3'.")
    _validate_positive_int(soak_days, "soak_days", minimum=1)
    if soak_days > MAXIMUM_SOAK_DAYS:
        raise ValueError(f"soak_days cannot exceed {MAXIMUM_SOAK_DAYS}.")
    initial_equity_vnd = _validate_finite_float(
        initial_equity_vnd, "initial_equity_vnd", strictly_positive=True
    )

    bars_per_day = 6
    warmup_bars = 360
    n_bars = warmup_bars + soak_days * bars_per_day
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="4h", tz="UTC")
    cols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "WAVESUSDT"]
    rets = rng.normal(0.0005, 0.014, size=(n_bars, len(cols)))
    rets[:, 0] += 0.0012
    close_arr = 100.0 * np.cumprod(1.0 + rets, axis=0)
    high_arr = close_arr * (1.0 + rng.uniform(0.002, 0.02, size=close_arr.shape))
    low_arr = close_arr * (1.0 - rng.uniform(0.002, 0.02, size=close_arr.shape))
    qv_arr = rng.uniform(5e6, 2e8, size=close_arr.shape)
    for arr in (close_arr, high_arr, low_arr, qv_arr):
        arr[:60, 2] = np.nan
        arr[warmup_bars - 30 :, 5] = np.nan

    close = pd.DataFrame(close_arr, index=idx, columns=cols)
    high = pd.DataFrame(high_arr, index=idx, columns=cols)
    low = pd.DataFrame(low_arr, index=idx, columns=cols)
    qv = pd.DataFrame(qv_arr, index=idx, columns=cols)

    mask = causal_rolling_top_n_mask(qv, close, lookback_bars=30, top_n=4)
    if strategy == "f4":
        weights = f4_ema_trend_btc_weights(close, mask, btc_fast=30, btc_slow=120, vol_window=60)
        strategy_name = "F4_4H_EMA_24_120_BTC"
    else:
        weights = f3_donchian_breakout_weights(
            close, high, low, mask, entry_window=60, exit_window=30, vol_window=60
        )
        strategy_name = "F3_DONCHIAN_4H_120_60"

    bt = run_multi_asset_spot_backtest(close, weights, fee_bps=fee_bps, slippage_bps=slippage_bps)
    metrics = performance_metrics(bt, periods_per_year=365 * bars_per_day)

    soak_bt = bt.iloc[warmup_bars:]
    soak_w = weights.iloc[warmup_bars:]
    day_groups = sorted(set(soak_bt.index.strftime("%Y-%m-%d")))[-soak_days:]

    days: list[PaperSoakDay] = []
    prev_equity_vnd = initial_equity_vnd * float(bt["equity"].iloc[warmup_bars - 1])
    for date_str in day_groups:
        day_mask = soak_bt.index.strftime("%Y-%m-%d") == date_str
        day_bt = soak_bt.loc[day_mask]
        day_w = soak_w.loc[day_mask]
        end_equity_vnd = initial_equity_vnd * float(day_bt["equity"].iloc[-1])
        day_pnl_vnd = end_equity_vnd - prev_equity_vnd
        day_fees_vnd = float(day_bt["cost"].sum()) * prev_equity_vnd
        cutoff_utc = int(day_bt.index[-1].timestamp())
        eod_weights = {col: float(day_w[col].iloc[-1]) for col in cols}
        days.append(
            build_sleeve2_paper_soak_day(
                date_utc=date_str,
                input_cutoff_utc=cutoff_utc,
                target_weights=eod_weights,
                realized_pnl_vnd=round(day_pnl_vnd, 2),
                unrealized_pnl_vnd=0.0,
                fees_vnd=round(day_fees_vnd, 2),
                slippage_bps=slippage_bps,
                equity_vnd=round(end_equity_vnd, 2),
            )
        )
        prev_equity_vnd = end_equity_vnd

    restart = PaperSoakRestartCheck(
        observed_at_utc=days[-1].input_cutoff_utc,
        attempt_no=1,
        checkpoint=f"smoke-day-{len(days)}-restart",
        duplicate_intents=0,
        status="passed",
        evidence_hash=_canonical_sha256({"strategy": strategy_name, "seed": seed, "days": len(days)}),
    )
    receipt = build_sleeve2_paper_soak_receipt(
        days,
        [restart],
        run_id=f"sleeve2-smoke-{strategy}-seed{seed}",
        attempt_no=1,
        fence_token="offline-smoke-only",
        strategy_name=strategy_name,
        universe_name="synthetic_top4_30b_qv",
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        owner_reviewed=False,
        evidence_level="offline_smoke_fixture",
    )
    health = audit_paper_soak_receipt(receipt)
    return {
        "strategy": strategy_name,
        "backtest_metrics": metrics,
        "receipt": receipt,
        "health_report": health,
    }
