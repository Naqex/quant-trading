"""Institutional offline purged walk-forward, fee sensitivity, and parameter plateau diagnostics.

Runs strictly causal (t -> t+1) spot long/cash evaluations on already-captured
CSV/Parquet OHLCV fixtures or a deterministic offline smoke series. Reuses
``quant_lab.backtest``, ``quant_lab.metrics``, ``quant_lab.strategies``,
``quant_trading.data_quality``, ``quant_trading.walk_forward``, and
``quant_trading.risk_metrics`` without duplicating core accounting or risk logic.

Every output is strictly ``PREP_ONLY_OFFLINE`` with ``execution_capability=False``
and ``promotion_allowed=False`` (OrderSend = 0, zero network/API keys).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant_lab.backtest import run_backtest
from quant_lab.data import _read_ohlcv_file
from quant_lab.metrics import performance_metrics
from quant_lab.strategies import (
    buy_and_hold_signal,
    donchian_breakout_signal,
    moving_average_trend_signal,
    time_series_momentum_signal,
)
from quant_trading.data_quality import DataQualityError, require_ohlcv_quality
from quant_trading.risk_metrics import expected_shortfall
from quant_trading.walk_forward import (
    SignalFactory,
    _metrics_with_boundary,
)

STRESS_REPORT_SCHEMA = "quant-stress-report-v1"
STRESS_REPORT_SCOPE = "PREP_ONLY_OFFLINE"
DEFAULT_FEE_LADDER_BPS: tuple[float, ...] = (10.0, 20.0, 35.0)
StrategyKind = Literal["ma", "donchian", "momentum"]


class StressReportError(ValueError):
    """Raised when a stress-report configuration or input fixture is invalid."""


def _strict_int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        hi = f" and <= {maximum}" if maximum is not None else ""
        raise StressReportError(f"{name} must be an integer >= {minimum}{hi}.")
    return value


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise StressReportError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StressReportError(f"{name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise StressReportError(f"{name} must be finite.")
    if minimum is not None and parsed < minimum:
        raise StressReportError(f"{name} must be >= {minimum}.")
    if maximum is not None and parsed > maximum:
        raise StressReportError(f"{name} must be <= {maximum}.")
    return parsed


def _clean_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    parsed = float(value)
    return None if not math.isfinite(parsed) else round(parsed, 8)


def _canonical_sha256(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _label(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    return str(value)


@dataclass(frozen=True)
class StressReportConfig:
    """Bounded, reproducible configuration for purged WFO + fee + plateau stress tests."""

    strategy: StrategyKind = "ma"
    fast: int = 20
    slow: int = 60
    entry_window: int = 40
    exit_window: int = 20
    lookback: int = 60
    train_bars: int = 180
    test_bars: int = 60
    purge_bars: int = 5
    step_bars: int | None = None
    minimum_folds: int = 2
    fee_ladder_bps: tuple[float, ...] = DEFAULT_FEE_LADDER_BPS
    slippage_bps: float = 5.0
    periods_per_year: int = 365
    bootstrap_resamples: int = 1_000
    block_length: int = 7
    seed: int = 20261125
    min_cost_breakeven_bps: float = 35.0
    min_plateau_pass_rate: float = 0.60
    min_plateau_median_sharpe: float = 0.0
    max_tail_drawdown_cvar: float | None = None

    def __post_init__(self) -> None:
        if self.strategy not in {"ma", "donchian", "momentum"}:
            raise StressReportError("strategy must be one of: ma, donchian, momentum.")
        fast = _strict_int(self.fast, "fast", minimum=2)
        slow = _strict_int(self.slow, "slow", minimum=3)
        if fast >= slow:
            raise StressReportError("fast must be strictly smaller than slow.")
        entry_window = _strict_int(self.entry_window, "entry_window", minimum=3)
        exit_window = _strict_int(self.exit_window, "exit_window", minimum=2)
        if exit_window >= entry_window:
            raise StressReportError("exit_window must be strictly smaller than entry_window.")
        lookback = _strict_int(self.lookback, "lookback", minimum=2)
        train_bars = _strict_int(self.train_bars, "train_bars", minimum=10)
        test_bars = _strict_int(self.test_bars, "test_bars", minimum=10)
        purge_bars = _strict_int(self.purge_bars, "purge_bars", minimum=0)
        step_bars = test_bars if self.step_bars is None else _strict_int(
            self.step_bars, "step_bars", minimum=1
        )
        minimum_folds = _strict_int(self.minimum_folds, "minimum_folds", minimum=1)
        if not isinstance(self.fee_ladder_bps, tuple) or not self.fee_ladder_bps:
            raise StressReportError("fee_ladder_bps must be a non-empty tuple.")
        fees = tuple(
            sorted(
                {
                    _finite_float(item, "fee_ladder_bps", minimum=0.0, maximum=9_999.0)
                    for item in self.fee_ladder_bps
                }
            )
        )
        slippage = _finite_float(self.slippage_bps, "slippage_bps", minimum=0.0, maximum=9_999.0)
        if max(fees) + slippage >= 10_000.0:
            raise StressReportError("Maximum fee_bps + slippage_bps must remain below 10,000 bps.")
        periods_per_year = _strict_int(self.periods_per_year, "periods_per_year", minimum=1)
        bootstrap_resamples = _strict_int(
            self.bootstrap_resamples, "bootstrap_resamples", minimum=100, maximum=20_000
        )
        block_length = _strict_int(self.block_length, "block_length", minimum=1)
        seed = _strict_int(self.seed, "seed", minimum=0)
        min_breakeven = _finite_float(
            self.min_cost_breakeven_bps, "min_cost_breakeven_bps", minimum=0.0
        )
        min_pass_rate = _finite_float(
            self.min_plateau_pass_rate, "min_plateau_pass_rate", minimum=0.0, maximum=1.0
        )
        min_med_sharpe = _finite_float(self.min_plateau_median_sharpe, "min_plateau_median_sharpe")
        max_tail_dd = (
            None
            if self.max_tail_drawdown_cvar is None
            else _finite_float(
                self.max_tail_drawdown_cvar, "max_tail_drawdown_cvar", minimum=-1.0, maximum=0.0
            )
        )
        object.__setattr__(self, "fast", fast)
        object.__setattr__(self, "slow", slow)
        object.__setattr__(self, "entry_window", entry_window)
        object.__setattr__(self, "exit_window", exit_window)
        object.__setattr__(self, "lookback", lookback)
        object.__setattr__(self, "train_bars", train_bars)
        object.__setattr__(self, "test_bars", test_bars)
        object.__setattr__(self, "purge_bars", purge_bars)
        object.__setattr__(self, "step_bars", step_bars)
        object.__setattr__(self, "minimum_folds", minimum_folds)
        object.__setattr__(self, "fee_ladder_bps", fees)
        object.__setattr__(self, "slippage_bps", slippage)
        object.__setattr__(self, "periods_per_year", periods_per_year)
        object.__setattr__(self, "bootstrap_resamples", bootstrap_resamples)
        object.__setattr__(self, "block_length", block_length)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "min_cost_breakeven_bps", min_breakeven)
        object.__setattr__(self, "min_plateau_pass_rate", min_pass_rate)
        object.__setattr__(self, "min_plateau_median_sharpe", min_med_sharpe)
        object.__setattr__(self, "max_tail_drawdown_cvar", max_tail_dd)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fast": self.fast,
            "slow": self.slow,
            "entry_window": self.entry_window,
            "exit_window": self.exit_window,
            "lookback": self.lookback,
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "purge_bars": self.purge_bars,
            "step_bars": self.step_bars,
            "minimum_folds": self.minimum_folds,
            "fee_ladder_bps": list(self.fee_ladder_bps),
            "slippage_bps": self.slippage_bps,
            "periods_per_year": self.periods_per_year,
            "bootstrap_resamples": self.bootstrap_resamples,
            "block_length": self.block_length,
            "seed": self.seed,
            "min_cost_breakeven_bps": self.min_cost_breakeven_bps,
            "min_plateau_pass_rate": self.min_plateau_pass_rate,
            "min_plateau_median_sharpe": self.min_plateau_median_sharpe,
            "max_tail_drawdown_cvar": self.max_tail_drawdown_cvar,
        }


def build_smoke_ohlcv(*, periods: int = 540, seed: int = 20261125) -> pd.DataFrame:
    """Build a deterministic offline OHLCV series with trending and pullback regimes."""
    _strict_int(periods, "periods", minimum=120)
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=periods, freq="1D", tz="UTC")
    # Alternating multi-month bull / bear regimes with realistic daily noise so trend
    # strategies experience real drawdowns and genuine OOS regime capture.
    regime = np.empty(periods, dtype=np.float64)
    for i in range(periods):
        cycle = i % 160
        if cycle < 95:
            regime[i] = 0.0055
        elif cycle < 135:
            regime[i] = -0.0045
        else:
            regime[i] = 0.0035
    noise = rng.normal(0.0, 0.011, size=periods)
    rets = np.clip(regime + noise, -0.15, 0.15)
    close = 20_000.0 * np.cumprod(1.0 + rets)
    open_p = np.r_[20_000.0, close[:-1]]
    span = np.abs(close - open_p) + close * rng.uniform(0.004, 0.014, size=periods)
    high = np.maximum(open_p, close) + span * 0.5
    low = np.maximum(1.0, np.minimum(open_p, close) - span * 0.5)
    volume = rng.uniform(500.0, 3_500.0, size=periods)
    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def load_close_from_ohlcv(path: Path | str) -> tuple[pd.Series, str]:
    """Load and fail-closed validate an offline CSV/Parquet OHLCV artifact."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"OHLCV file does not exist: {file_path}")
    frame = _read_ohlcv_file(file_path)
    try:
        receipt = require_ohlcv_quality(frame)
    except DataQualityError as exc:
        raise StressReportError(f"Input OHLCV quality preflight failed: {exc}") from exc
    timestamps = pd.to_datetime(frame["open_time"], utc=True)
    close = pd.Series(
        pd.to_numeric(frame["close"]).to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(timestamps),
        name="close",
    )
    assert receipt.data_sha256 is not None
    return close, receipt.data_sha256


def _build_purged_folds(
    index: pd.Index,
    *,
    train_bars: int,
    test_bars: int,
    purge_bars: int,
    step_bars: int,
    minimum_folds: int,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    fold_idx = 0
    train_end = train_bars - 1
    while True:
        test_start = train_end + 1 + purge_bars
        test_end = test_start + test_bars - 1
        if test_end >= len(index):
            break
        folds.append(
            {
                "fold_index": fold_idx,
                "train_start_idx": 0,
                "train_end_idx": train_end,
                "purge_start_idx": train_end + 1 if purge_bars > 0 else None,
                "purge_end_idx": train_end + purge_bars if purge_bars > 0 else None,
                "test_start_idx": test_start,
                "test_end_idx": test_end,
                "train_start": _label(index[0]),
                "train_end": _label(index[train_end]),
                "purge_bars": purge_bars,
                "test_start": _label(index[test_start]),
                "test_end": _label(index[test_end]),
                "train_count": train_end + 1,
                "test_count": test_bars,
            }
        )
        fold_idx += 1
        train_end += step_bars
    if len(folds) < minimum_folds:
        raise StressReportError(
            f"Insufficient rows for purged walk-forward: got {len(folds)} fold(s), need {minimum_folds}."
        )
    return folds


def _make_strategy_factory(
    strategy: StrategyKind,
    params: dict[str, int],
) -> SignalFactory:
    if strategy == "ma":
        fast, slow = params["fast"], params["slow"]
        return lambda close: moving_average_trend_signal(close, fast, slow)
    if strategy == "donchian":
        entry_w, exit_w = params["entry_window"], params["exit_window"]
        return lambda close: donchian_breakout_signal(close, entry_w, exit_w)
    lookback = params["lookback"]
    return lambda close: time_series_momentum_signal(close, lookback)


def _build_plateau_grid(config: StressReportConfig) -> tuple[str, list[tuple[str, dict[str, int]]]]:
    """Build deterministic bounded ±20% parameter neighbourhood around the registered center."""
    if config.strategy == "ma":
        center_id = f"ma_{config.fast}_{config.slow}"
        fasts = sorted({max(2, int(round(config.fast * m))) for m in (0.8, 1.0, 1.2)})
        slows = sorted({max(3, int(round(config.slow * m))) for m in (0.8, 1.0, 1.2)})
        variants = [(f"ma_{f}_{s}", {"fast": f, "slow": s}) for f in fasts for s in slows if f < s]
        return center_id, variants
    if config.strategy == "donchian":
        center_id = f"donchian_{config.entry_window}_{config.exit_window}"
        entries = sorted({max(3, int(round(config.entry_window * m))) for m in (0.8, 1.0, 1.2)})
        exits = sorted({max(2, int(round(config.exit_window * m))) for m in (0.8, 1.0, 1.2)})
        variants = [
            (f"donchian_{en}_{ex}", {"entry_window": en, "exit_window": ex})
            for en in entries
            for ex in exits
            if ex < en
        ]
        return center_id, variants
    center_id = f"momentum_{config.lookback}"
    lookbacks = sorted({max(2, int(round(config.lookback * m))) for m in (0.8, 0.9, 1.0, 1.1, 1.2)})
    variants = [(f"momentum_{lb}", {"lookback": lb}) for lb in lookbacks]
    return center_id, variants


def _precompute_pit_histories(
    close: pd.Series,
    folds: list[dict[str, Any]],
    signal_factory: SignalFactory,
    *,
    verify_prefix_causality: bool = False,
) -> list[tuple[dict[str, Any], pd.Series, pd.Series]]:
    histories: list[tuple[dict[str, Any], pd.Series, pd.Series]] = []
    for fold in folds:
        test_end = fold["test_end_idx"]
        history = close.iloc[: test_end + 1].copy()
        signal = signal_factory(history)
        if not isinstance(signal, pd.Series) or not signal.index.equals(history.index):
            raise StressReportError("signal_factory must return a Series matching history index.")
        if verify_prefix_causality and fold["fold_index"] == 0:
            cut = fold["test_start_idx"]
            prefix_sig = signal_factory(history.iloc[: cut + 1].copy())
            if not np.allclose(
                prefix_sig.fillna(0.0).to_numpy(dtype=np.float64),
                signal.iloc[: cut + 1].fillna(0.0).to_numpy(dtype=np.float64),
                atol=1e-12,
                rtol=0.0,
            ):
                raise StressReportError("Non-causal signal repainting detected across fold prefix.")
        histories.append((fold, history, signal))
    return histories


def _evaluate_histories_at_cost(
    histories: list[tuple[dict[str, Any], pd.Series, pd.Series]],
    *,
    fee_bps: float,
    slippage_bps: float,
    periods_per_year: int,
) -> tuple[list[dict[str, Any]], dict[str, float | int | None], np.ndarray]:
    fold_results: list[dict[str, Any]] = []
    stitched_frames: list[pd.DataFrame] = []
    total_oos_entries = 0

    for fold, history, signal in histories:
        bt = run_backtest(history, signal, fee_bps=fee_bps, slippage_bps=slippage_bps)
        metrics = _metrics_with_boundary(bt, fold["test_start_idx"], fold["test_end_idx"])
        total_oos_entries += int(metrics["entries"] or 0)
        fold_results.append(
            {
                "fold_index": fold["fold_index"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "purge_bars": fold["purge_bars"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "metrics": {k: _clean_number(v) for k, v in metrics.items()},
            }
        )
        stitched_frames.append(bt.iloc[fold["test_start_idx"] : fold["test_end_idx"] + 1].copy())

    stitched = pd.concat(stitched_frames, axis=0)
    stitched["equity"] = np.maximum(0.0, 1.0 + stitched["strategy_return"]).cumprod()
    peak = stitched["equity"].cummax()
    stitched["drawdown"] = pd.Series(
        np.where(peak > 0.0, stitched["equity"] / peak - 1.0, -1.0),
        index=stitched.index,
        dtype=float,
    )
    stitched_raw = performance_metrics(stitched, periods_per_year=periods_per_year)
    stitched_raw["entries"] = total_oos_entries
    stitched_metrics = {k: _clean_number(v) for k, v in stitched_raw.items()}
    oos_returns = stitched["strategy_return"].to_numpy(dtype=np.float64)
    return fold_results, stitched_metrics, oos_returns


def _cost_breakeven_bps(
    histories: list[tuple[dict[str, Any], pd.Series, pd.Series]],
    *,
    periods_per_year: int,
    upper_bps: float = 2_000.0,
) -> float:
    """Compute exact total one-way cost (bps) where stitched OOS total return crosses zero."""

    def _net_return(total_cost_bps: float) -> float:
        _, stitched, _ = _evaluate_histories_at_cost(
            histories,
            fee_bps=total_cost_bps,
            slippage_bps=0.0,
            periods_per_year=periods_per_year,
        )
        val = stitched["total_return"]
        return float(val) if val is not None else -1.0

    if _net_return(0.0) <= 0.0:
        return 0.0
    if _net_return(upper_bps) > 0.0:
        return round(upper_bps, 4)

    lo, hi = 0.0, upper_bps
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if _net_return(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return round(lo, 4)


def _block_bootstrap_summary(
    oos_returns: np.ndarray,
    *,
    periods_per_year: int,
    resamples: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    """Circular block-bootstrap 95% Sharpe/CAGR CIs and worst-5% tail drawdown metrics."""
    n = int(oos_returns.size)
    if n == 0:
        raise StressReportError("OOS return series cannot be empty for bootstrap.")
    eff_block = min(block_length, n)
    n_blocks = int(math.ceil(n / eff_block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, n_blocks))
    offsets = np.arange(eff_block, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    sampled = oos_returns[indices.reshape(resamples, n_blocks * eff_block)[:, :n]]

    years = n / float(periods_per_year)
    means = np.mean(sampled, axis=1)
    stds = np.std(sampled, axis=1, ddof=0)
    valid_std = stds > 1e-12
    safe_stds = np.where(valid_std, stds, 1.0)
    sharpes = np.where(valid_std, (means / safe_stds) * math.sqrt(periods_per_year), 0.0)

    equities = np.cumprod(np.maximum(0.0, 1.0 + sampled), axis=1)
    end_eq = equities[:, -1]
    cagrs = np.where(end_eq > 0.0, np.power(np.maximum(end_eq, 1e-18), 1.0 / years) - 1.0, -1.0)

    peaks = np.maximum.accumulate(
        np.concatenate([np.ones((resamples, 1), dtype=np.float64), equities], axis=1), axis=1
    )[:, 1:]
    max_dds = np.min(equities / peaks - 1.0, axis=1)

    obs_eq = np.cumprod(np.maximum(0.0, 1.0 + oos_returns))
    obs_peak = np.maximum.accumulate(np.r_[1.0, obs_eq])[1:]
    obs_max_dd = float(np.min(obs_eq / obs_peak - 1.0))

    s_lo, s_med, s_hi = np.quantile(sharpes, (0.025, 0.5, 0.975))
    c_lo, c_med, c_hi = np.quantile(cagrs, (0.025, 0.5, 0.975))
    dd_var_05, dd_med = np.quantile(max_dds, (0.05, 0.5))
    dd_cvar_05 = expected_shortfall(max_dds, alpha=0.05)
    ret_cvar_05 = expected_shortfall(oos_returns, alpha=0.05)

    return {
        "resamples": resamples,
        "block_length": eff_block,
        "seed": seed,
        "oos_bars": n,
        "sharpe_ci_95": {
            "lo": _clean_number(s_lo),
            "median": _clean_number(s_med),
            "hi": _clean_number(s_hi),
        },
        "cagr_ci_95": {
            "lo": _clean_number(c_lo),
            "median": _clean_number(c_med),
            "hi": _clean_number(c_hi),
        },
        "tail_drawdown": {
            "observed_max_drawdown": _clean_number(obs_max_dd),
            "bootstrap_median_max_drawdown": _clean_number(dd_med),
            "worst_5pct_drawdown_var": _clean_number(dd_var_05),
            "worst_5pct_drawdown_cvar": _clean_number(dd_cvar_05),
            "empirical_return_cvar_95": _clean_number(ret_cvar_05),
        },
    }


def run_stress_report(
    close: pd.Series,
    *,
    config: StressReportConfig | None = None,
    input_source: str = "in_memory",
    input_sha256: str | None = None,
) -> dict[str, Any]:
    """Run purged WFO + fee sensitivity ladder + ±20% plateau + block-bootstrap stress report."""
    cfg = config or StressReportConfig()
    if not isinstance(cfg, StressReportConfig):
        raise StressReportError("config must be StressReportConfig.")
    if not isinstance(close, pd.Series) or close.empty:
        raise StressReportError("close must be a non-empty pandas Series.")
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise StressReportError("close index must be unique and sorted ascending.")
    close_num = close.astype(float)
    arr = close_num.to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all() or (arr <= 0.0).any():
        raise StressReportError("close prices must be finite and strictly positive.")

    resolved_input_sha = input_sha256 or _canonical_sha256(
        {
            "index": [_label(item) for item in close_num.index],
            "close": [format(float(v), ".17g") for v in arr],
        }
    )
    folds = _build_purged_folds(
        close_num.index,
        train_bars=cfg.train_bars,
        test_bars=cfg.test_bars,
        purge_bars=cfg.purge_bars,
        step_bars=cfg.step_bars or cfg.test_bars,
        minimum_folds=cfg.minimum_folds,
    )

    center_id, plateau_specs = _build_plateau_grid(cfg)
    center_params = dict(plateau_specs)[center_id]
    center_factory = _make_strategy_factory(cfg.strategy, center_params)

    center_histories = _precompute_pit_histories(
        close_num, folds, center_factory, verify_prefix_causality=True
    )
    bh_histories = _precompute_pit_histories(close_num, folds, buy_and_hold_signal)

    base_fee = cfg.fee_ladder_bps[0]
    base_folds, base_stitched, base_oos_returns = _evaluate_histories_at_cost(
        center_histories,
        fee_bps=base_fee,
        slippage_bps=cfg.slippage_bps,
        periods_per_year=cfg.periods_per_year,
    )
    _, bh_stitched, _ = _evaluate_histories_at_cost(
        bh_histories,
        fee_bps=base_fee,
        slippage_bps=cfg.slippage_bps,
        periods_per_year=cfg.periods_per_year,
    )

    # 1) Fee sensitivity ladder (10 / 20 / 35 bps) + exact cost breakeven
    fee_ladder: list[dict[str, Any]] = []
    for fee_bps in cfg.fee_ladder_bps:
        tier_folds, tier_stitched, _ = _evaluate_histories_at_cost(
            center_histories,
            fee_bps=fee_bps,
            slippage_bps=cfg.slippage_bps,
            periods_per_year=cfg.periods_per_year,
        )
        fold_cagrs = [f["metrics"]["cagr"] for f in tier_folds if f["metrics"]["cagr"] is not None]
        fold_sharpes = [
            f["metrics"]["sharpe_zero_rate"]
            for f in tier_folds
            if f["metrics"]["sharpe_zero_rate"] is not None
        ]
        fold_dds = [
            f["metrics"]["max_drawdown"] for f in tier_folds if f["metrics"]["max_drawdown"] is not None
        ]
        fee_ladder.append(
            {
                "fee_bps": fee_bps,
                "slippage_bps": cfg.slippage_bps,
                "total_cost_bps": round(fee_bps + cfg.slippage_bps, 6),
                "stitched_total_return": tier_stitched["total_return"],
                "stitched_cagr": tier_stitched["cagr"],
                "stitched_sharpe": tier_stitched["sharpe_zero_rate"],
                "stitched_max_drawdown": tier_stitched["max_drawdown"],
                "median_fold_cagr": _clean_number(np.median(fold_cagrs)) if fold_cagrs else None,
                "median_fold_sharpe": _clean_number(np.median(fold_sharpes)) if fold_sharpes else None,
                "worst_fold_drawdown": _clean_number(min(fold_dds)) if fold_dds else None,
                "annual_turnover": tier_stitched["annual_turnover"],
                "entries": tier_stitched["entries"],
                "time_in_market": tier_stitched["time_in_market"],
            }
        )
    breakeven_bps = _cost_breakeven_bps(center_histories, periods_per_year=cfg.periods_per_year)

    # 2) Bounded ±20% parameter plateau grid
    plateau_variants: list[dict[str, Any]] = []
    for variant_id, params in plateau_specs:
        if variant_id == center_id:
            v_stitched = base_stitched
        else:
            v_factory = _make_strategy_factory(cfg.strategy, params)
            v_hist = _precompute_pit_histories(close_num, folds, v_factory)
            _, v_stitched, _ = _evaluate_histories_at_cost(
                v_hist,
                fee_bps=base_fee,
                slippage_bps=cfg.slippage_bps,
                periods_per_year=cfg.periods_per_year,
            )
        sharpe_v = v_stitched["sharpe_zero_rate"]
        cagr_v = v_stitched["cagr"]
        positive = (sharpe_v is not None and sharpe_v > 0.0) and (cagr_v is not None and cagr_v > 0.0)
        plateau_variants.append(
            {
                "variant_id": variant_id,
                "is_center": variant_id == center_id,
                "params": params,
                "stitched_cagr": cagr_v,
                "stitched_sharpe": sharpe_v,
                "stitched_max_drawdown": v_stitched["max_drawdown"],
                "annual_turnover": v_stitched["annual_turnover"],
                "positive_oos": positive,
            }
        )

    sharpe_vals = [
        float(v["stitched_sharpe"]) for v in plateau_variants if v["stitched_sharpe"] is not None
    ]
    cagr_vals = [float(v["stitched_cagr"]) for v in plateau_variants if v["stitched_cagr"] is not None]
    pos_count = sum(1 for v in plateau_variants if v["positive_oos"])
    pass_rate = pos_count / len(plateau_variants)
    sharpe_median = float(np.median(sharpe_vals)) if sharpe_vals else None
    center_sharpe = base_stitched["sharpe_zero_rate"]
    retention_ratio = (
        round(sharpe_median / float(center_sharpe), 6)
        if (sharpe_median is not None and center_sharpe is not None and float(center_sharpe) > 1e-9)
        else 0.0
    )
    plateau_summary = {
        "center_variant_id": center_id,
        "variant_count": len(plateau_variants),
        "positive_variant_count": pos_count,
        "positive_variant_fraction": round(pass_rate, 6),
        "sharpe_min": _clean_number(min(sharpe_vals)) if sharpe_vals else None,
        "sharpe_median": _clean_number(sharpe_median),
        "sharpe_max": _clean_number(max(sharpe_vals)) if sharpe_vals else None,
        "cagr_min": _clean_number(min(cagr_vals)) if cagr_vals else None,
        "cagr_median": _clean_number(np.median(cagr_vals)) if cagr_vals else None,
        "cagr_max": _clean_number(max(cagr_vals)) if cagr_vals else None,
        "center_sharpe": center_sharpe,
        "retention_ratio": retention_ratio,
        "variants": plateau_variants,
    }

    # 3) Block-bootstrap 95% Sharpe/CAGR CI + worst-5% tail drawdown
    bootstrap = _block_bootstrap_summary(
        base_oos_returns,
        periods_per_year=cfg.periods_per_year,
        resamples=cfg.bootstrap_resamples,
        block_length=cfg.block_length,
        seed=cfg.seed,
    )

    # 4) Fail-closed threshold gates
    blockers: list[str] = []
    if breakeven_bps < cfg.min_cost_breakeven_bps:
        blockers.append(
            f"cost_breakeven_below_threshold:{breakeven_bps:.2f}<{cfg.min_cost_breakeven_bps:.2f}"
        )
    if pass_rate < cfg.min_plateau_pass_rate:
        blockers.append(
            f"plateau_collapse_pass_rate:{pass_rate:.4f}<{cfg.min_plateau_pass_rate:.4f}"
        )
    if sharpe_median is None or sharpe_median < cfg.min_plateau_median_sharpe:
        med_str = "None" if sharpe_median is None else f"{sharpe_median:.4f}"
        blockers.append(
            f"plateau_median_sharpe_below_threshold:{med_str}<{cfg.min_plateau_median_sharpe:.4f}"
        )
    tail_dd_cvar = bootstrap["tail_drawdown"]["worst_5pct_drawdown_cvar"]
    if (
        cfg.max_tail_drawdown_cvar is not None
        and tail_dd_cvar is not None
        and float(tail_dd_cvar) < cfg.max_tail_drawdown_cvar
    ):
        blockers.append(
            f"tail_drawdown_cvar_exceeded:{float(tail_dd_cvar):.4f}<{cfg.max_tail_drawdown_cvar:.4f}"
        )

    status = "PASS" if not blockers else "FAIL"
    core: dict[str, Any] = {
        "schema": STRESS_REPORT_SCHEMA,
        "scope": STRESS_REPORT_SCOPE,
        "status": status,
        "input_source": input_source,
        "input_sha256": resolved_input_sha,
        "rows": int(len(close_num)),
        "start_utc": _label(close_num.index[0]),
        "end_utc": _label(close_num.index[-1]),
        "config": cfg.as_dict(),
        "purged_walk_forward": {
            "fold_count": len(base_folds),
            "purge_bars": cfg.purge_bars,
            "baseline_fee_bps": base_fee,
            "slippage_bps": cfg.slippage_bps,
            "stitched_strategy_metrics": base_stitched,
            "stitched_benchmark_buy_and_hold": bh_stitched,
            "folds": base_folds,
        },
        "fee_sensitivity": {
            "tiers": fee_ladder,
            "cost_breakeven_bps": breakeven_bps,
            "min_cost_breakeven_bps": cfg.min_cost_breakeven_bps,
        },
        "parameter_plateau": plateau_summary,
        "bootstrap": bootstrap,
        "blockers": blockers,
        "execution_capability": False,
        "promotion_allowed": False,
        "provider_access": False,
        "broker_access": False,
        "oauth_access": False,
    }
    report = dict(core)
    report["report_sha256"] = _canonical_sha256(core)
    return report


def render_stress_report_markdown(report: dict[str, Any]) -> str:
    """Render a concise, auditable Markdown summary (zero bare dollar signs)."""
    wfo = report["purged_walk_forward"]
    strat = wfo["stitched_strategy_metrics"]
    bh = wfo["stitched_benchmark_buy_and_hold"]
    fee = report["fee_sensitivity"]
    plat = report["parameter_plateau"]
    boot = report["bootstrap"]
    dd = boot["tail_drawdown"]

    def _fmt(val: Any, pct: bool = False) -> str:
        if val is None:
            return "n/a"
        return f"{float(val) * 100.0:.2f}%" if pct else f"{float(val):.4f}"

    lines = [
        f"# Quant Lab Offline Stress Report — `{report['status']}`",
        "",
        (
            f"- **Scope:** `{report['scope']}`"
            " (`OrderSend = 0`, `execution_capability = false`, `promotion_allowed = false`)"
        ),
        (
            f"- **Input:** `{report['input_source']}` ({report['rows']} bars,"
            f" `{report['start_utc']}` -> `{report['end_utc']}`)"
        ),
        (
            f"- **Strategy:** `{plat['center_variant_id']}` | Purged WFO folds:"
            f" `{wfo['fold_count']}` (purge `{wfo['purge_bars']}` bars)"
        ),
        f"- **Report SHA-256:** `{report['report_sha256']}`",
        "",
        "## 1. Purged Walk-Forward OOS vs Buy-and-Hold",
        "",
        "| Series | CAGR | Sharpe | Max Drawdown | Annual Turnover | Entries | Time in Market |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Strategy (`{plat['center_variant_id']}`) | {_fmt(strat['cagr'], True)} | "
            f"{_fmt(strat['sharpe_zero_rate'])} | {_fmt(strat['max_drawdown'], True)} | "
            f"{_fmt(strat['annual_turnover'])} | {strat['entries']} | "
            f"{_fmt(strat['time_in_market'], True)} |"
        ),
        (
            f"| Buy & Hold | {_fmt(bh['cagr'], True)} | {_fmt(bh['sharpe_zero_rate'])} | "
            f"{_fmt(bh['max_drawdown'], True)} | {_fmt(bh['annual_turnover'])} | {bh['entries']} | "
            f"{_fmt(bh['time_in_market'], True)} |"
        ),
        "",
        "## 2. Fee Sensitivity Ladder & Breakeven",
        "",
        (
            f"- **Exact Cost Breakeven (one-way):** `{fee['cost_breakeven_bps']:.2f} bps`"
            f" (gate `>= {fee['min_cost_breakeven_bps']:.2f} bps`)"
        ),
        "",
        "| Fee (bps) | Slippage (bps) | Total Cost (bps) | Stitched CAGR | Stitched Sharpe | Worst Fold DD |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in fee["tiers"]:
        lines.append(
            f"| {tier['fee_bps']:.1f} | {tier['slippage_bps']:.1f} | {tier['total_cost_bps']:.1f} | "
            f"{_fmt(tier['stitched_cagr'], True)} | {_fmt(tier['stitched_sharpe'])} | "
            f"{_fmt(tier['worst_fold_drawdown'], True)} |"
        )
    lines.extend(
        [
            "",
            "## 3. Bounded +-20% Parameter Plateau",
            "",
            (
                f"- **Positive Variants:** `{plat['positive_variant_count']}/{plat['variant_count']}` "
                f"(`{plat['positive_variant_fraction'] * 100.0:.1f}%`) | "
                f"**Median Sharpe:** `{_fmt(plat['sharpe_median'])}` | "
                f"**Retention Ratio:** `{plat['retention_ratio']:.3f}`"
            ),
            "",
            "| Variant | Center | OOS CAGR | OOS Sharpe | OOS Max DD | Positive |",
            "|---|:---:|---:|---:|---:|:---:|",
        ]
    )
    for var in plat["variants"]:
        lines.append(
            f"| `{var['variant_id']}` | {'yes' if var['is_center'] else ''} | "
            f"{_fmt(var['stitched_cagr'], True)} | {_fmt(var['stitched_sharpe'])} | "
            f"{_fmt(var['stitched_max_drawdown'], True)} | {'PASS' if var['positive_oos'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 4. Block-Bootstrap 95% CI & Worst-5% Tail Drawdown",
            "",
            (
                f"- **Bootstrap:** `{boot['resamples']}` circular block resamples"
                f" (block `{boot['block_length']}`, seed `{boot['seed']}`,"
                f" `{boot['oos_bars']}` OOS bars)"
            ),
            (
                f"- **Sharpe 95% CI:** `[{_fmt(boot['sharpe_ci_95']['lo'])},"
                f" {_fmt(boot['sharpe_ci_95']['hi'])}]`"
                f" (median `{_fmt(boot['sharpe_ci_95']['median'])}`)"
            ),
            (
                f"- **CAGR 95% CI:** `[{_fmt(boot['cagr_ci_95']['lo'], True)},"
                f" {_fmt(boot['cagr_ci_95']['hi'], True)}]`"
                f" (median `{_fmt(boot['cagr_ci_95']['median'], True)}`)"
            ),
            (
                f"- **Observed OOS Max DD:** `{_fmt(dd['observed_max_drawdown'], True)}` |"
                f" **Worst-5% DD VaR:** `{_fmt(dd['worst_5pct_drawdown_var'], True)}` |"
                f" **Worst-5% DD CVaR:** `{_fmt(dd['worst_5pct_drawdown_cvar'], True)}`"
            ),
            "",
            "## 5. Gate Verdict",
            "",
            f"- **Status:** `{report['status']}`",
            f"- **Blockers:** `{', '.join(report['blockers']) if report['blockers'] else 'none'}`",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_FEE_LADDER_BPS",
    "STRESS_REPORT_SCHEMA",
    "STRESS_REPORT_SCOPE",
    "StressReportConfig",
    "StressReportError",
    "build_smoke_ohlcv",
    "load_close_from_ohlcv",
    "render_stress_report_markdown",
    "run_stress_report",
]
