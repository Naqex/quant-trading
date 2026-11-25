from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from quant_lab.provenance import validate_candles
from quant_trading.data_quality import (
    REQUIRED_COLUMNS,
    OHLCVQualityPolicy,
    audit_ohlcv,
)

DATA_AUDIT_SCHEMA = "ohlcv-directory-audit-v1"
DATA_AUDIT_SCOPE = "PREP_ONLY_OFFLINE"
TIMEFRAME_SECONDS: dict[str, int] = {
    "1d": 86_400,
    "4h": 14_400,
    "1h": 3_600,
}
_SUPPORTED_EXTENSIONS = frozenset({".csv", ".parquet"})
_TIMEFRAME_SUFFIX_RE = re.compile(r"_(1d|4h|1h)$", re.IGNORECASE)
_PRICE_COLUMNS = ("open", "high", "low", "close")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "unused",
]


def _utc_milliseconds(value: str) -> int:
    parsed = datetime.fromisoformat(value).replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1_000)


def download_binance_klines(
    symbol: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download public Binance candles without credentials."""
    symbol = symbol.upper()
    cursor = _utc_milliseconds(start)
    end_ms = _utc_milliseconds(end) if end else int(datetime.now(UTC).timestamp() * 1_000)
    rows: list[list[object]] = []

    while cursor < end_ms:
        query = urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
        )
        request = Request(f"{BINANCE_KLINES_URL}?{query}", headers={"User-Agent": "quant-lab/0.1"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
            page = json.load(response)

        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][0]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("Binance returned a non-advancing page.")
        cursor = next_cursor
        time.sleep(0.1)

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        raise RuntimeError(f"No candles returned for {symbol}.")

    numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    frame["trade_count"] = frame["trade_count"].astype(int)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.sort_values("open_time").reset_index(drop=True)
    validate_candles(frame)
    return frame


def save_candles(frame: pd.DataFrame, path: Path) -> Path:
    validate_candles(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    validate_candles(frame)
    return frame.reset_index(drop=True)


def _infer_symbol_and_timeframe(path: Path, explicit_timeframe: str | None) -> tuple[str, str]:
    stem = path.stem
    match = _TIMEFRAME_SUFFIX_RE.search(stem)
    if match is not None:
        inferred_tf = match.group(1).lower()
        symbol = stem[: match.start()].upper() or stem.upper()
    else:
        inferred_tf = "1d"
        symbol = stem.upper()
    timeframe = explicit_timeframe.lower() if explicit_timeframe is not None else inferred_tf
    return symbol, timeframe


def _audit_symbol_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    rel_path: str,
    file_format: str,
    timeframe: str,
    max_gap_count: int = 0,
    allow_zero_volume: bool = False,
) -> dict[str, Any]:
    expected_seconds = TIMEFRAME_SECONDS[timeframe]
    policy = OHLCVQualityPolicy(
        expected_interval_seconds=expected_seconds,
        max_gap_count=max_gap_count,
    )
    receipt = audit_ohlcv(frame, policy=policy)
    errors = list(receipt.errors)
    warnings = list(receipt.warnings)

    duplicate_timestamps = 0
    monotonic_violations = 0
    missing_candle_gaps = 0
    missing_candles_estimated = 0
    non_positive_prices = 0
    non_finite_prices = 0
    ohlc_consistency_violations = 0
    negative_volume_bars = 0
    zero_volume_bars = 0
    first_utc = receipt.start_utc
    last_utc = receipt.end_utc
    observed_interval = receipt.observed_interval_seconds

    has_unique_cols = frame.columns.is_unique
    if has_unique_cols and "open_time" in frame.columns and not frame.empty:
        timestamps = pd.to_datetime(frame["open_time"], errors="coerce", utc=True)
        valid_ts = timestamps.dropna()
        if not valid_ts.empty:
            first_utc = valid_ts.iloc[0].isoformat()
            last_utc = valid_ts.iloc[-1].isoformat()
            duplicate_timestamps = int(valid_ts.duplicated().sum())
            if len(valid_ts) > 1:
                deltas = valid_ts.diff().iloc[1:].dt.total_seconds().to_numpy(dtype=np.float64)
                monotonic_violations = int(np.count_nonzero(deltas < 0.0))
                positive_deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
                if observed_interval is None and len(positive_deltas) > 0:
                    observed_interval = float(np.median(positive_deltas))
                gap_deltas = positive_deltas[positive_deltas > float(expected_seconds)]
                missing_candle_gaps = int(len(gap_deltas))
                if missing_candle_gaps > 0:
                    missing_candles_estimated = int(
                        sum(max(1, int(round(delta / expected_seconds)) - 1) for delta in gap_deltas)
                    )
                    if missing_candle_gaps > max_gap_count and "gap_budget_exceeded" not in errors:
                        errors.append("gap_budget_exceeded")

    if has_unique_cols and not frame.empty:
        present_prices = [col for col in _PRICE_COLUMNS if col in frame.columns]
        if present_prices:
            price_num = frame.loc[:, present_prices].apply(pd.to_numeric, errors="coerce")
            price_arr = price_num.to_numpy(dtype=np.float64)
            finite_mask = np.isfinite(price_arr)
            non_finite_prices = int(np.count_nonzero(~finite_mask))
            non_positive_prices = int(np.count_nonzero(finite_mask & (price_arr <= 0.0)))
        if all(col in frame.columns for col in _PRICE_COLUMNS):
            ohlc = frame.loc[:, list(_PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
            finite_rows = np.isfinite(ohlc.to_numpy(dtype=np.float64)).all(axis=1)
            if np.any(finite_rows):
                clean = ohlc.loc[finite_rows]
                high_floor = clean[["open", "close"]].max(axis=1)
                low_ceiling = clean[["open", "close"]].min(axis=1)
                bad_ohlc = (
                    (clean["high"] < clean["low"])
                    | (clean["high"] < high_floor)
                    | (clean["low"] > low_ceiling)
                )
                ohlc_consistency_violations = int(bad_ohlc.sum())
        if "volume" in frame.columns:
            vol = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=np.float64)
            vol_finite = np.isfinite(vol)
            negative_volume_bars = int(np.count_nonzero((~vol_finite) | (vol < 0.0)))
            zero_volume_bars = int(np.count_nonzero(vol_finite & (vol == 0.0)))
            if zero_volume_bars > 0:
                if allow_zero_volume:
                    warnings.append(f"zero_volume_halted_bars:{zero_volume_bars}")
                else:
                    errors.append(f"zero_volume_halted_bars:{zero_volume_bars}")

    if receipt.gap_count is not None and missing_candle_gaps == 0:
        missing_candle_gaps = receipt.gap_count

    symbol_status = "PASS" if not errors else "FAIL"
    return {
        "symbol": symbol,
        "path": rel_path,
        "format": file_format,
        "timeframe": timeframe,
        "status": symbol_status,
        "rows": int(len(frame)),
        "first_utc": first_utc,
        "last_utc": last_utc,
        "expected_interval_seconds": expected_seconds,
        "observed_interval_seconds": observed_interval,
        "duplicate_timestamps": duplicate_timestamps,
        "monotonic_violations": monotonic_violations,
        "missing_candle_gaps": missing_candle_gaps,
        "missing_candles_estimated": missing_candles_estimated,
        "non_positive_prices": non_positive_prices,
        "non_finite_prices": non_finite_prices,
        "ohlc_consistency_violations": ohlc_consistency_violations,
        "negative_volume_bars": negative_volume_bars,
        "zero_volume_bars": zero_volume_bars,
        "data_sha256": receipt.data_sha256,
        "quality_receipt_sha256": receipt.receipt_sha256,
        "errors": errors,
        "warnings": warnings,
    }


def _read_ohlcv_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported OHLCV extension: {suffix}")


def audit_ohlcv_directory(
    directory: Path | str,
    *,
    timeframe: str | None = None,
    max_gap_count: int = 0,
    allow_zero_volume: bool = False,
) -> dict[str, Any]:
    """Offline fail-closed OHLCV quality & gap audit across CSV/Parquet files in a directory."""
    if timeframe is not None and timeframe.lower() not in TIMEFRAME_SECONDS:
        supported = ", ".join(sorted(TIMEFRAME_SECONDS))
        raise ValueError(f"Unsupported timeframe {timeframe!r}; expected one of: {supported}.")
    if type(max_gap_count) is not int or max_gap_count < 0:
        raise ValueError("max_gap_count must be an integer >= 0.")
    if type(allow_zero_volume) is not bool:
        raise ValueError("allow_zero_volume must be a boolean.")

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"OHLCV directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"OHLCV path is not a directory: {root}")

    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS),
        key=lambda p: p.relative_to(root).as_posix().lower(),
    )

    top_errors: list[str] = []
    symbol_reports: list[dict[str, Any]] = []
    if not files:
        top_errors.append("no_ohlcv_files_found")

    for file_path in files:
        rel_path = file_path.relative_to(root).as_posix()
        file_format = file_path.suffix.lower().lstrip(".")
        symbol, resolved_tf = _infer_symbol_and_timeframe(file_path, timeframe)
        try:
            frame = _read_ohlcv_file(file_path)
        except Exception as exc:
            symbol_reports.append(
                {
                    "symbol": symbol,
                    "path": rel_path,
                    "format": file_format,
                    "timeframe": resolved_tf,
                    "status": "FAIL",
                    "rows": 0,
                    "first_utc": None,
                    "last_utc": None,
                    "expected_interval_seconds": TIMEFRAME_SECONDS[resolved_tf],
                    "observed_interval_seconds": None,
                    "duplicate_timestamps": 0,
                    "monotonic_violations": 0,
                    "missing_candle_gaps": 0,
                    "missing_candles_estimated": 0,
                    "non_positive_prices": 0,
                    "non_finite_prices": 0,
                    "ohlc_consistency_violations": 0,
                    "negative_volume_bars": 0,
                    "zero_volume_bars": 0,
                    "data_sha256": None,
                    "quality_receipt_sha256": None,
                    "errors": [f"read_error:{type(exc).__name__}:{exc}"],
                    "warnings": [],
                }
            )
            continue

        symbol_reports.append(
            _audit_symbol_frame(
                frame,
                symbol=symbol,
                rel_path=rel_path,
                file_format=file_format,
                timeframe=resolved_tf,
                max_gap_count=max_gap_count,
                allow_zero_volume=allow_zero_volume,
            )
        )

    passed = sum(1 for item in symbol_reports if item["status"] == "PASS")
    failed = len(symbol_reports) - passed
    overall_status = "PASS" if (symbol_reports and failed == 0 and not top_errors) else "FAIL"

    core: dict[str, Any] = {
        "schema": DATA_AUDIT_SCHEMA,
        "scope": DATA_AUDIT_SCOPE,
        "timeframe": timeframe.lower() if timeframe is not None else "auto",
        "required_columns": list(REQUIRED_COLUMNS),
        "status": overall_status,
        "summary": {
            "files_scanned": len(symbol_reports),
            "passed_symbols": passed,
            "failed_symbols": failed,
            "total_rows": sum(item["rows"] for item in symbol_reports),
            "total_gaps": sum(item["missing_candle_gaps"] for item in symbol_reports),
            "total_zero_volume_bars": sum(item["zero_volume_bars"] for item in symbol_reports),
        },
        "errors": top_errors,
        "symbols": symbol_reports,
        "execution_capability": False,
        "provider_access": False,
        "broker_access": False,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report = dict(core)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report
