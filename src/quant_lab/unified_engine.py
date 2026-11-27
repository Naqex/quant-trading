"""Unified strictly causal Spot Signal, Portfolio & Shadow Accounting Engine (pure numpy + pandas).

Single source of truth shared identically by offline Backtest mode and live read-only Shadow mode:
  - Pure `numpy` + `pandas` + Python stdlib ONLY (zero duckdb/pyarrow/scipy/root imports).
  - Strict `-O`-safe `enforce_pre_holdout_dates()` raising `ValueError` on any date >= `2025-01-01`.
  - Exact EMA warmup parity (`valid_streak >= span`, resetting on NaN or non-positive closes).
  - Exact rebalance deadband parity (`apply_deadband_step` matches `_apply_rebalance_deadband`).
  - Exact Shadow 4H bar idempotency lock (`last_4h_close_ms`) and full exited-coin MTM accounting.
  - Strictly offline / read-only (`ORDER_SEND_CAPABLE = False`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.causal_universe import (
    _validate_finite_float,
    _validate_panel,
    _validate_positive_int,
    _validate_positive_prices,
    run_multi_asset_spot_backtest,
)
from quant_lab.metrics import performance_metrics

ORDER_SEND_CAPABLE: bool = False
HOLDOUT_CUTOFF_DATE: str = "2025-01-01"
DEFAULT_SHADOW_4H_KLINE_LIMIT: int = 1000
MIN_RECOMMENDED_4H_KLINE_LIMIT: int = 600

_DOT_DATE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})(.*)$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|z|[+-]\d{2}:?\d{2})?$"
)


def _parse_utc_date(item: Any, *, context: str) -> date:
    """Strictly parse a calendar date / ISO datetime / pandas Timestamp into a UTC `date`."""
    if item is None or item is pd.NaT or isinstance(item, (bool, int, float, np.number, bytes, bytearray)):
        raise ValueError(
            f"HOLDOUT_LOCK_VIOLATION [{context}]: non-date/NaT value {item!r} "
            f"({type(item).__name__}) rejected."
        )
    if isinstance(item, pd.Timestamp):
        if pd.isna(item):
            raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: NaT timestamp rejected.")
        ts = item.tz_convert("UTC") if item.tzinfo is not None else item
        return ts.date()
    if isinstance(item, datetime):
        if pd.isna(item):
            raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: NaT datetime rejected.")
        return item.astimezone(UTC).date() if item.tzinfo is not None else item.date()
    if isinstance(item, date):
        return item
    if isinstance(item, np.datetime64):
        if bool(np.isnat(item)):
            raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: NaT datetime64 rejected.")
        py_dt = item.astype("datetime64[s]").item()
        if isinstance(py_dt, datetime):
            return py_dt.astimezone(UTC).date() if py_dt.tzinfo is not None else py_dt.date()
        if isinstance(py_dt, date):
            return py_dt
        raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: invalid numpy datetime {item!r}.")
    if not isinstance(item, str):
        raise ValueError(
            f"HOLDOUT_LOCK_VIOLATION [{context}]: unsupported date type {type(item).__name__}: {item!r}."
        )

    text = item.strip()
    if not text:
        raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: empty date string rejected.")
    dot_match = _DOT_DATE_RE.fullmatch(text)
    if dot_match is not None:
        text = f"{dot_match.group(1)}-{dot_match.group(2)}-{dot_match.group(3)}{dot_match.group(4)}"

    if _ISO_DATE_RE.fullmatch(text):
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"HOLDOUT_LOCK_VIOLATION [{context}]: invalid calendar date {item!r}: {exc}"
            ) from exc

    if _ISO_DATETIME_RE.fullmatch(text):
        norm_text = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
        try:
            dt = datetime.fromisoformat(norm_text)
        except ValueError as exc:
            raise ValueError(
                f"HOLDOUT_LOCK_VIOLATION [{context}]: invalid ISO datetime {item!r}: {exc}"
            ) from exc
        return dt.astimezone(UTC).date() if dt.tzinfo is not None else dt.date()

    raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: malformed/non-ISO date {item!r} rejected.")


def enforce_pre_holdout_dates(
    dates: str | date | datetime | pd.Timestamp | np.datetime64 | pd.Index | Iterable[Any],
    *,
    cutoff: str = HOLDOUT_CUTOFF_DATE,
    context: str = "unified_engine",
    allow_empty: bool = True,
) -> int:
    """Fail-closed `-O`-safe guard rejecting any date >= `cutoff` (`2025-01-01` UTC default).

    Never uses Python `assert`, so enforcement remains active under `python -O` and `python -OO`.
    Returns `0` (`holdout_rows_analysed == 0`) when all dates stay strictly before `cutoff`.
    """
    cutoff_dt = _parse_utc_date(cutoff, context=f"{context}.cutoff")
    if dates is None or dates is pd.NaT or isinstance(dates, (bool, int, float, np.number, bytes, bytearray)):
        raise ValueError(
            f"HOLDOUT_LOCK_VIOLATION [{context}]: invalid dates input {dates!r} ({type(dates).__name__})."
        )

    if isinstance(dates, pd.DatetimeIndex):
        if len(dates) == 0:
            if not allow_empty:
                raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: empty DatetimeIndex rejected.")
            return 0
        if dates.isna().any():
            raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: DatetimeIndex contains NaT.")
        utc_idx = dates.tz_convert("UTC") if dates.tz is not None else dates
        max_dt = utc_idx.max().date()
        if max_dt >= cutoff_dt:
            raise ValueError(
                f"HOLDOUT_LOCK_VIOLATION [{context}]: detected date {max_dt.isoformat()} >= "
                f"{cutoff_dt.isoformat()}."
            )
        return 0

    if isinstance(dates, (str, date, datetime, pd.Timestamp, np.datetime64)):
        raw_items: list[Any] = [dates]
    else:
        try:
            raw_items = list(dates)
        except TypeError as exc:
            raise ValueError(
                f"HOLDOUT_LOCK_VIOLATION [{context}]: non-iterable dates input {dates!r}."
            ) from exc

    if not raw_items:
        if not allow_empty:
            raise ValueError(f"HOLDOUT_LOCK_VIOLATION [{context}]: empty date series rejected.")
        return 0

    bad: list[str] = []
    for item in raw_items:
        parsed = _parse_utc_date(item, context=context)
        if parsed >= cutoff_dt:
            bad.append(parsed.isoformat())
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(
            f"HOLDOUT_LOCK_VIOLATION [{context}]: detected holdout date(s) >= {cutoff_dt.isoformat()}: {bad}"
        )
    return 0


def enforce_pre_holdout_frame(
    frame: pd.DataFrame,
    *,
    cutoff: str = HOLDOUT_CUTOFF_DATE,
    context: str = "unified_engine",
) -> int:
    """Validate that a DataFrame has a non-empty, sorted, strictly pre-holdout date index."""
    _validate_panel(frame, context)
    return enforce_pre_holdout_dates(frame.index, cutoff=cutoff, context=context, allow_empty=False)


def causal_ema_1d(values: np.ndarray, span: int) -> np.ndarray:
    """Strictly causal 1D EMA with warmup parity (`valid_streak >= span`) matching `causal_universe.py`."""
    _validate_positive_int(span, "span", minimum=1)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"causal_ema_1d expects a 1D array, got shape {arr.shape}.")
    n_t = arr.shape[0]
    out = np.full(n_t, np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    prev = np.nan
    valid_streak = 0
    for t in range(n_t):
        c = float(arr[t])
        if not math.isfinite(c) or c <= 0.0:
            prev = np.nan
            valid_streak = 0
            continue
        prev = c if not math.isfinite(prev) else (alpha * c + (1.0 - alpha) * prev)
        valid_streak += 1
        if valid_streak >= span:
            out[t] = prev
    return out


def causal_ema_matrix(closes: np.ndarray, span: int) -> np.ndarray:
    """Strictly causal 2D EMA (`T x K`) delegating column-wise to `causal_ema_1d`."""
    _validate_positive_int(span, "span", minimum=1)
    mat = np.asarray(closes, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"causal_ema_matrix expects a 2D array, got shape {mat.shape}.")
    out = np.full(mat.shape, np.nan, dtype=np.float64)
    for j in range(mat.shape[1]):
        out[:, j] = causal_ema_1d(mat[:, j], span)
    return out


def causal_rolling_vol_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Strictly causal 1D rolling sample std (`ddof=1`) of simple bar returns over `window` bars."""
    _validate_positive_int(window, "vol_window", minimum=2)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"causal_rolling_vol_1d expects a 1D array, got shape {arr.shape}.")
    n_t = arr.shape[0]
    out = np.full(n_t, np.nan, dtype=np.float64)
    if n_t <= window:
        return out

    rets = np.full(n_t, np.nan, dtype=np.float64)
    valid = np.isfinite(arr[1:]) & np.isfinite(arr[:-1]) & (arr[:-1] > 0.0) & (arr[1:] > 0.0)
    rets[1:][valid] = arr[1:][valid] / arr[:-1][valid] - 1.0

    for t in range(window, n_t):
        win = rets[t - window + 1 : t + 1]
        if np.all(np.isfinite(win)):
            out[t] = float(np.std(win, ddof=1))
    return out


def causal_rolling_vol_matrix(closes: np.ndarray, window: int) -> np.ndarray:
    """Strictly causal 2D rolling return std (`T x K`, `ddof=1`) matching `causal_universe.py`."""
    _validate_positive_int(window, "vol_window", minimum=2)
    mat = np.asarray(closes, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"causal_rolling_vol_matrix expects a 2D array, got shape {mat.shape}.")
    n_t, n_k = mat.shape
    rets = np.full((n_t, n_k), np.nan, dtype=np.float64)
    valid = np.isfinite(mat[1:]) & np.isfinite(mat[:-1]) & (mat[:-1] > 0.0) & (mat[1:] > 0.0)
    rets[1:][valid] = mat[1:][valid] / mat[:-1][valid] - 1.0

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
        mean = s1 / float(window)
        var = np.maximum(0.0, (s2 - float(window) * mean * mean) / float(max(1, window - 1)))
        out[window:] = np.where(full, np.sqrt(var), np.nan)
    return out


def apply_deadband_step(
    raw_weights: Mapping[str, float],
    prev_weights: Mapping[str, float] | None = None,
    *,
    deadband: float = 0.015,
) -> dict[str, float]:
    """Single-step rebalance deadband matching `causal_universe._apply_rebalance_deadband` bit-for-bit.

    Rules:
      1. Spot weights are finite and >= 0; raw vector is normalized to sum <= 1.0 first.
      2. Position entry (`0 -> >0`) or full exit (`>0 -> 0`) ALWAYS executes immediately (`flip`).
      3. Non-zero to non-zero adjustments smaller than `deadband` retain `prev_weights[sym]`.
      4. Resulting weights are renormalized to sum <= 1.0 (zero leverage invariant).
    """
    db = _validate_finite_float(deadband, "deadband", minimum=0.0)

    cleaned_raw: dict[str, float] = {}
    for sym, val in raw_weights.items():
        w = _validate_finite_float(val, f"raw_weights[{sym!r}]", minimum=0.0)
        if w > 0.0:
            cleaned_raw[str(sym)] = w
    raw_total = sum(cleaned_raw.values())
    if raw_total > 1.0:
        cleaned_raw = {k: v / raw_total for k, v in cleaned_raw.items()}

    if prev_weights is None:
        return dict(sorted(cleaned_raw.items()))

    cleaned_prev: dict[str, float] = {}
    for sym, val in prev_weights.items():
        w = _validate_finite_float(val, f"prev_weights[{sym!r}]", minimum=0.0)
        if w > 0.0:
            cleaned_prev[str(sym)] = w

    if db == 0.0:
        return dict(sorted(cleaned_raw.items()))

    updated: dict[str, float] = {}
    for sym in sorted(set(cleaned_raw) | set(cleaned_prev)):
        cur = cleaned_raw.get(sym, 0.0)
        prv = cleaned_prev.get(sym, 0.0)
        flip = (cur == 0.0) != (prv == 0.0)
        new_w = cur if (flip or abs(cur - prv) >= db) else prv
        if new_w > 0.0:
            updated[sym] = new_w

    total = sum(updated.values())
    if total > 1.0:
        updated = {k: v / total for k, v in updated.items()}
    return updated


@dataclass(frozen=True, slots=True)
class F4EmaConfig:
    """Immutable configuration for Sleeve #2 Primary (`F4_4H_EMA_24_120_BTC`) and 1H EMA variants."""

    btc_symbol: str = "BTCUSDT"
    fast_span: int = 24
    slow_span: int = 120
    btc_fast: int = 60
    btc_slow: int = 300
    vol_window: int = 120
    target_ann_vol: float = 0.15
    bars_per_year: float = 365.0 * 6.0
    deadband: float = 0.015
    min_recommended_history_bars: int = MIN_RECOMMENDED_4H_KLINE_LIMIT

    def __post_init__(self) -> None:
        if not isinstance(self.btc_symbol, str) or not self.btc_symbol.strip():
            raise ValueError("btc_symbol must be a non-empty string.")
        _validate_positive_int(self.fast_span, "fast_span", minimum=1)
        _validate_positive_int(self.slow_span, "slow_span", minimum=1)
        _validate_positive_int(self.btc_fast, "btc_fast", minimum=1)
        _validate_positive_int(self.btc_slow, "btc_slow", minimum=1)
        if self.slow_span <= self.fast_span or self.btc_slow <= self.btc_fast:
            raise ValueError("EMA spans must satisfy 0 < fast < slow.")
        _validate_positive_int(self.vol_window, "vol_window", minimum=2)
        _validate_finite_float(self.target_ann_vol, "target_ann_vol", strictly_positive=True)
        _validate_finite_float(self.bars_per_year, "bars_per_year", strictly_positive=True)
        _validate_finite_float(self.deadband, "deadband", minimum=0.0)
        _validate_positive_int(
            self.min_recommended_history_bars,
            "min_recommended_history_bars",
            minimum=1,
        )


def compute_f4_ema_step(
    closed_closes: Mapping[str, np.ndarray | Sequence[float]],
    eligible_symbols: Sequence[str],
    *,
    config: F4EmaConfig | None = None,
    prev_weights: Mapping[str, float] | None = None,
    round_digits: int | None = None,
) -> dict[str, Any]:
    """Compute exact causal single-bar `F4_4H_EMA_24_120_BTC` weights shared by Backtest & Shadow.

    Guarantees:
      - Exact EMA warmup parity (`valid_streak >= span`) on BTC macro filter and every coin.
      - Exact rebalance deadband parity (`apply_deadband_step`) when `prev_weights` is supplied.
      - Spot long/cash invariants (`0 <= w_j`, `sum(w) <= 1.0`, zero leverage).
    """
    cfg = config or F4EmaConfig()
    btc_raw = closed_closes.get(cfg.btc_symbol)
    if btc_raw is None:
        return {
            "btc_regime_bull": False,
            "btc_ema60": None,
            "btc_ema300": None,
            "btc_history_bars": 0,
            "ema300_drift_warning": True,
            "active_count": 0,
            "gross_exposure": 0.0,
            "weights": {},
            "raw_weights": {},
            "diagnostics": {},
            "reason": "INSUFFICIENT_BTC_BARS",
        }

    btc_arr = np.asarray(btc_raw, dtype=np.float64)
    btc_bars = int(btc_arr.shape[0]) if btc_arr.ndim == 1 else 0
    btc_ef_series = causal_ema_1d(btc_arr, cfg.btc_fast) if btc_bars > 0 else np.empty(0)
    btc_es_series = causal_ema_1d(btc_arr, cfg.btc_slow) if btc_bars > 0 else np.empty(0)
    btc_ef = float(btc_ef_series[-1]) if btc_bars > 0 else float("nan")
    btc_es = float(btc_es_series[-1]) if btc_bars > 0 else float("nan")

    if not math.isfinite(btc_ef) or not math.isfinite(btc_es):
        return {
            "btc_regime_bull": False,
            "btc_ema60": round(btc_ef, 2) if math.isfinite(btc_ef) else None,
            "btc_ema300": round(btc_es, 2) if math.isfinite(btc_es) else None,
            "btc_history_bars": btc_bars,
            "ema300_drift_warning": btc_bars < cfg.min_recommended_history_bars,
            "active_count": 0,
            "gross_exposure": 0.0,
            "weights": {},
            "raw_weights": {},
            "diagnostics": {},
            "reason": "INSUFFICIENT_BTC_BARS",
        }

    btc_bull = bool(btc_ef > btc_es)
    vt_bar = cfg.target_ann_vol / math.sqrt(cfg.bars_per_year)
    elig_unique = list(dict.fromkeys(str(s) for s in eligible_symbols))
    n_elig = max(1, len(elig_unique))

    raw: dict[str, float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    for sym in elig_unique:
        series_raw = closed_closes.get(sym)
        if series_raw is None:
            continue
        c = np.asarray(series_raw, dtype=np.float64)
        if c.ndim != 1 or c.size == 0:
            continue
        ef_val = float(causal_ema_1d(c, cfg.fast_span)[-1])
        es_val = float(causal_ema_1d(c, cfg.slow_span)[-1])
        vol_val = float(causal_rolling_vol_1d(c, cfg.vol_window)[-1])
        last_close = float(c[-1])

        sig_on = bool(
            btc_bull
            and math.isfinite(last_close)
            and last_close > 0.0
            and math.isfinite(ef_val)
            and math.isfinite(es_val)
            and (ef_val > es_val)
            and math.isfinite(vol_val)
            and (vol_val > 1e-6)
        )
        w_val = (vt_bar / vol_val) / float(n_elig) if sig_on else 0.0
        if w_val > 0.0:
            raw[sym] = w_val
        diagnostics[sym] = {
            "last_closed_4h": round(last_close, 6) if math.isfinite(last_close) else None,
            "ema24": round(ef_val, 6) if math.isfinite(ef_val) else None,
            "ema120": round(es_val, 6) if math.isfinite(es_val) else None,
            "vol120_ann_pct": (
                round(vol_val * math.sqrt(cfg.bars_per_year) * 100.0, 2) if math.isfinite(vol_val) else None
            ),
            "signal_long": sig_on,
        }

    tot_raw = sum(raw.values())
    capped_raw = {k: v / tot_raw for k, v in raw.items()} if tot_raw > 1.0 else dict(raw)
    weights = apply_deadband_step(capped_raw, prev_weights, deadband=cfg.deadband)

    if round_digits is not None:
        weights = {k: round(v, round_digits) for k, v in weights.items() if round(v, round_digits) > 0.0}
        tot_r = sum(weights.values())
        if tot_r > 1.0:
            weights = {k: round(v / tot_r, round_digits) for k, v in weights.items()}

    gross_exp = sum(weights.values())
    return {
        "btc_regime_bull": btc_bull,
        "btc_ema60": round(btc_ef, 2),
        "btc_ema300": round(btc_es, 2),
        "btc_history_bars": btc_bars,
        "ema300_drift_warning": btc_bars < cfg.min_recommended_history_bars,
        "active_count": len(weights),
        "gross_exposure": round(gross_exp, 6) if round_digits is not None else gross_exp,
        "weights": weights,
        "raw_weights": capped_raw,
        "diagnostics": diagnostics,
    }


def _parse_finite_positive_price(value: Any) -> float | None:
    """Return a strictly positive finite float price, or `None` for None/bool/NaN/Inf/<=0."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _parse_nonnegative_slip_bps(value: Any, fallback: float) -> float:
    """Return a finite non-negative slippage in bps, falling back safely on bool/NaN/Inf/<0."""
    if value is None or isinstance(value, bool):
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed < 0.0:
        return fallback
    return parsed


def _validate_positive_bar_ms(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer timestamp in ms, got {value!r}.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer timestamp in ms, got {parsed}.")
    return parsed


def update_shadow_spot_accounting(
    *,
    prev_weights: Mapping[str, float],
    prev_mids: Mapping[str, float],
    prev_equity_usd: float,
    prev_bar_close_ms: int | None,
    current_bar_close_ms: int,
    candidate_weights: Mapping[str, float],
    depth_metrics: Mapping[str, Mapping[str, Any]],
    taker_fee_bps: float = 10.0,
    fallback_slip_bps: float = 5.0,
) -> dict[str, Any]:
    """Deterministic Shadow spot portfolio accounting with 4H bar idempotency & full exit MTM.

    Fixes:
      - Idempotency lock keyed on `current_bar_close_ms`: repeated polls within the SAME 4H bar
        never re-rebalance or re-deduct turnover fees (`turnover = 0.0`, `cost_return = 0.0`).
      - Exit MTM completeness: marks every coin in `prev_weights` (including symbols dropped from
        Top-15) against live `depth_metrics[sym]["mid"]` and explicitly surfaces any missing/non-finite mids.
      - Fail-closed validation: rejects bool/non-positive `current_bar_close_ms` & `prev_bar_close_ms`,
        guards non-finite/bool prices, and never accepts negative/bool `slip_10k_bps`.
    """
    eq0 = _validate_finite_float(prev_equity_usd, "prev_equity_usd", minimum=0.0)
    fee_bps = _validate_finite_float(taker_fee_bps, "taker_fee_bps", minimum=0.0)
    fb_slip = _validate_finite_float(fallback_slip_bps, "fallback_slip_bps", minimum=0.0)
    cur_ms = _validate_positive_bar_ms(current_bar_close_ms, "current_bar_close_ms")
    prev_ms: int | None = (
        None
        if prev_bar_close_ms is None
        else _validate_positive_bar_ms(prev_bar_close_ms, "prev_bar_close_ms")
    )

    clean_prev: dict[str, float] = {}
    for sym, w_raw in prev_weights.items():
        w = _validate_finite_float(w_raw, f"prev_weights[{sym!r}]", minimum=0.0)
        if w > 0.0:
            clean_prev[str(sym)] = w

    clean_cand: dict[str, float] = {}
    for sym, w_raw in candidate_weights.items():
        w = _validate_finite_float(w_raw, f"candidate_weights[{sym!r}]", minimum=0.0)
        if w > 0.0:
            clean_cand[str(sym)] = w
    cand_sum = sum(clean_cand.values())
    if cand_sum > 1.0 + 1e-6:
        raise ValueError(f"Spot candidate_weights sum {cand_sum:.6f} exceeds 1.0 (zero leverage).")
    if cand_sum > 1.0:
        clean_cand = {k: v / cand_sum for k, v in clean_cand.items()}

    # 1. Mark-to-market EVERY previously held weight (including exited / dropped Top-15 coins!)
    mtm_ret = 0.0
    missing_exit_mids: list[str] = []
    for sym, w in sorted(clean_prev.items()):
        p0 = _parse_finite_positive_price(prev_mids.get(sym))
        p1 = _parse_finite_positive_price(depth_metrics.get(sym, {}).get("mid"))
        if p0 is not None and p1 is not None:
            mtm_ret += w * (p1 / p0 - 1.0)
        else:
            missing_exit_mids.append(sym)

    # 2. Strict 4H bar idempotency lock: only rebalance & charge fees on a NEW closed 4H bar!
    is_new_bar = (prev_ms is None) or (cur_ms > prev_ms)

    if is_new_bar:
        active_weights = clean_cand
        all_keys = sorted(set(clean_prev) | set(active_weights))
        turnover = sum(abs(active_weights.get(k, 0.0) - clean_prev.get(k, 0.0)) for k in all_keys)
        live_tca_bps = 0.0
        if turnover > 1e-9:
            w_slip = 0.0
            for k in all_keys:
                dw = abs(active_weights.get(k, 0.0) - clean_prev.get(k, 0.0))
                if dw > 1e-9:
                    raw_slip = depth_metrics.get(k, {}).get("slip_10k_bps")
                    slip_k = _parse_nonnegative_slip_bps(raw_slip, fb_slip)
                    w_slip += dw * (fee_bps + slip_k)
            live_tca_bps = w_slip / turnover
        cost_ret = turnover * (live_tca_bps / 10_000.0)
        locked_ms = cur_ms
    else:
        active_weights = clean_prev
        turnover = 0.0
        live_tca_bps = 0.0
        cost_ret = 0.0
        locked_ms = prev_ms  # type: ignore[assignment]

    new_equity_usd = max(0.0, eq0 * (1.0 + mtm_ret - cost_ret))
    return {
        "weights": active_weights,
        "equity_usd": new_equity_usd,
        "mtm_return": mtm_ret,
        "turnover": turnover,
        "live_tca_bps": live_tca_bps,
        "cost_return": cost_ret,
        "bar_rebalanced": is_new_bar,
        "last_4h_close_ms": locked_ms,
        "missing_exit_mids": missing_exit_mids,
    }


class UnifiedSpotEngine:
    """Single causal Spot signal + backtest + shadow engine shared identically across modes."""

    def __init__(self, config: F4EmaConfig | None = None) -> None:
        self.config = config or F4EmaConfig()

    def compute_f4_step(
        self,
        closed_closes: Mapping[str, np.ndarray | Sequence[float]],
        eligible_symbols: Sequence[str],
        *,
        prev_weights: Mapping[str, float] | None = None,
        round_digits: int | None = None,
    ) -> dict[str, Any]:
        """Shadow/step entry point delegating to canonical `compute_f4_ema_step`."""
        return compute_f4_ema_step(
            closed_closes,
            eligible_symbols,
            config=self.config,
            prev_weights=prev_weights,
            round_digits=round_digits,
        )

    def build_f4_weights(
        self,
        close: pd.DataFrame,
        eligible_mask: pd.DataFrame,
        *,
        enforce_pre_holdout: bool = True,
        holdout_cutoff: str = HOLDOUT_CUTOFF_DATE,
    ) -> pd.DataFrame:
        """Compute full-panel causal F4 EMA trend + BTC macro filter + deadband weights."""
        _validate_panel(close, "close")
        _validate_panel(eligible_mask, "eligible_mask")
        if not close.index.equals(eligible_mask.index) or list(close.columns) != list(eligible_mask.columns):
            raise ValueError("close and eligible_mask must share identical index and columns.")
        if enforce_pre_holdout:
            enforce_pre_holdout_dates(
                close.index,
                cutoff=holdout_cutoff,
                context="UnifiedSpotEngine.build_f4_weights",
                allow_empty=False,
            )

        cfg = self.config
        if cfg.btc_symbol not in close.columns:
            raise ValueError(f"btc_symbol '{cfg.btc_symbol}' must exist in close columns.")

        cl = close.to_numpy(dtype=np.float64, copy=False)
        _validate_positive_prices(cl, "close")
        elig = eligible_mask.to_numpy(dtype=bool, copy=False)
        cols = [str(c) for c in close.columns]
        btc_idx = cols.index(cfg.btc_symbol)

        ema_f = causal_ema_matrix(cl, cfg.fast_span)
        ema_s = causal_ema_matrix(cl, cfg.slow_span)
        btc_col = cl[:, btc_idx]
        btc_ef = causal_ema_1d(btc_col, cfg.btc_fast)
        btc_es = causal_ema_1d(btc_col, cfg.btc_slow)
        vol = causal_rolling_vol_matrix(cl, cfg.vol_window)

        vt_bar = cfg.target_ann_vol / math.sqrt(cfg.bars_per_year)
        valid_vol = np.isfinite(vol) & (vol > 1e-6)
        safe_vol = np.where(valid_vol, vol, 1.0)
        inv_vol = np.where(valid_vol, vt_bar / safe_vol, 0.0)
        n_elig = np.maximum(1, elig.sum(axis=1, keepdims=True))
        btc_bull = (np.isfinite(btc_ef) & np.isfinite(btc_es) & (btc_ef > btc_es))[:, None]

        sig = elig & btc_bull & np.isfinite(ema_f) & np.isfinite(ema_s) & (ema_f > ema_s)
        raw = np.where(sig, inv_vol, 0.0) / n_elig
        row_sum = raw.sum(axis=1, keepdims=True)
        capped = np.where(row_sum > 1.0, raw / np.maximum(row_sum, 1e-12), raw)

        # Apply exact same step-wise deadband transition used in Shadow mode!
        n_t, n_k = capped.shape
        out = np.zeros((n_t, n_k), dtype=np.float64)
        prev_map: dict[str, float] = {}
        col_to_j = {c: j for j, c in enumerate(cols)}
        for t in range(n_t):
            nz = np.flatnonzero(capped[t] > 0.0)
            raw_map = {cols[int(j)]: float(capped[t, int(j)]) for j in nz}
            upd = apply_deadband_step(raw_map, prev_map, deadband=cfg.deadband)
            for sym, w_val in upd.items():
                out[t, col_to_j[sym]] = w_val
            prev_map = upd

        return pd.DataFrame(out, index=close.index, columns=close.columns)

    def build_donchian_weights(
        self,
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
        btc_symbol: str | None = None,
        btc_fast: int = 60,
        btc_slow: int = 300,
        enforce_pre_holdout: bool = True,
        holdout_cutoff: str = HOLDOUT_CUTOFF_DATE,
    ) -> pd.DataFrame:
        """Causal Donchian breakout weights with optional BTC EMA macro filter (4H & 1H compatible)."""
        _validate_panel(close, "close")
        for name, df in (("high", high), ("low", low), ("eligible_mask", eligible_mask)):
            _validate_panel(df, name)
            if not close.index.equals(df.index) or list(close.columns) != list(df.columns):
                raise ValueError(f"{name} must align exactly with close.")
        if enforce_pre_holdout:
            enforce_pre_holdout_dates(
                close.index,
                cutoff=holdout_cutoff,
                context="UnifiedSpotEngine.build_donchian_weights",
                allow_empty=False,
            )

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
        cols = [str(c) for c in close.columns]

        btc_gate = np.ones(n_t, dtype=bool)
        if btc_symbol is not None:
            if btc_symbol not in close.columns:
                raise ValueError(f"btc_symbol '{btc_symbol}' must exist in close columns.")
            _validate_positive_int(btc_fast, "btc_fast", minimum=1)
            _validate_positive_int(btc_slow, "btc_slow", minimum=1)
            if btc_slow <= btc_fast:
                raise ValueError("BTC EMA spans must satisfy 0 < btc_fast < btc_slow.")
            b_idx = cols.index(btc_symbol)
            b_ef = causal_ema_1d(cl[:, b_idx], btc_fast)
            b_es = causal_ema_1d(cl[:, b_idx], btc_slow)
            btc_gate = np.isfinite(b_ef) & np.isfinite(b_es) & (b_ef > b_es)

        vol = causal_rolling_vol_matrix(cl, vol_window)
        vt_bar = target_ann_vol / math.sqrt(bars_per_year)
        valid_vol = np.isfinite(vol) & (vol > 1e-6)
        safe_vol = np.where(valid_vol, vol, 1.0)
        inv_vol = np.where(valid_vol, vt_bar / safe_vol, 0.0)

        prior_hi = np.full((n_t, n_k), np.nan, dtype=np.float64)
        prior_lo = np.full((n_t, n_k), np.nan, dtype=np.float64)
        if n_t > entry_window:
            sw_h = np.lib.stride_tricks.sliding_window_view(hi[:-1], entry_window, axis=0)
            h_ok = np.all(np.isfinite(sw_h), axis=2)
            h_max = np.max(np.where(h_ok[:, :, None], sw_h, 0.0), axis=2)
            prior_hi[entry_window:] = np.where(h_ok, h_max, np.nan)
        if n_t > exit_window:
            sw_l = np.lib.stride_tricks.sliding_window_view(lo[:-1], exit_window, axis=0)
            l_ok = np.all(np.isfinite(sw_l), axis=2)
            l_min = np.min(np.where(l_ok[:, :, None], sw_l, 0.0), axis=2)
            prior_lo[exit_window:] = np.where(l_ok, l_min, np.nan)

        state = np.zeros(n_k, dtype=bool)
        raw = np.zeros((n_t, n_k), dtype=np.float64)
        start_t = max(entry_window, exit_window)

        for t in range(start_t, n_t):
            c = cl[t]
            el = elig[t]
            g_ok = bool(btc_gate[t])
            below_lo = np.isfinite(prior_lo[t]) & (c < prior_lo[t])
            exit_cond = (~np.isfinite(c)) | (~el) | below_lo | (not g_ok)
            enter_cond = g_ok & el & np.isfinite(c) & np.isfinite(prior_hi[t]) & (c > prior_hi[t])
            state = np.where(state, ~exit_cond, enter_cond)
            if np.any(state):
                row = np.where(state, inv_vol[t], 0.0) / float(max(1, int(el.sum())))
                total = float(row.sum())
                raw[t] = row / total if total > 1.0 else row

        out = np.zeros((n_t, n_k), dtype=np.float64)
        prev_map: dict[str, float] = {}
        col_to_j = {c: j for j, c in enumerate(cols)}
        for t in range(n_t):
            nz = np.flatnonzero(raw[t] > 0.0)
            raw_map = {cols[int(j)]: float(raw[t, int(j)]) for j in nz}
            upd = apply_deadband_step(raw_map, prev_map, deadband=deadband)
            for sym, w_val in upd.items():
                out[t, col_to_j[sym]] = w_val
            prev_map = upd

        return pd.DataFrame(out, index=close.index, columns=close.columns)

    def run_backtest(
        self,
        close: pd.DataFrame,
        target_weights: pd.DataFrame,
        *,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        delist_haircut_bps: float = 100.0,
        periods_per_year: int | None = None,
        enforce_pre_holdout: bool = True,
        holdout_cutoff: str = HOLDOUT_CUTOFF_DATE,
    ) -> dict[str, Any]:
        """Run unified causal t -> t+1 spot backtest with strict pre-holdout date enforcement."""
        if enforce_pre_holdout:
            enforce_pre_holdout_dates(
                close.index,
                cutoff=holdout_cutoff,
                context="UnifiedSpotEngine.run_backtest",
                allow_empty=False,
            )
        bt = run_multi_asset_spot_backtest(
            close,
            target_weights,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            delist_haircut_bps=delist_haircut_bps,
        )
        ppy = int(round(self.config.bars_per_year)) if periods_per_year is None else int(periods_per_year)
        metrics = performance_metrics(bt, periods_per_year=ppy)
        return {
            "backtest": bt,
            "metrics": metrics,
            "holdout_rows_analysed": 0,
            "order_send_capable": ORDER_SEND_CAPABLE,
        }


__all__ = [
    "DEFAULT_SHADOW_4H_KLINE_LIMIT",
    "F4EmaConfig",
    "HOLDOUT_CUTOFF_DATE",
    "MIN_RECOMMENDED_4H_KLINE_LIMIT",
    "ORDER_SEND_CAPABLE",
    "UnifiedSpotEngine",
    "apply_deadband_step",
    "causal_ema_1d",
    "causal_ema_matrix",
    "causal_rolling_vol_1d",
    "causal_rolling_vol_matrix",
    "compute_f4_ema_step",
    "enforce_pre_holdout_dates",
    "enforce_pre_holdout_frame",
    "update_shadow_spot_accounting",
]
