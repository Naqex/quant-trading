from __future__ import annotations

import hashlib
from typing import Final

import numpy as np
import pandas as pd

REQUIRED_CANDLE_COLUMNS: Final[tuple[str, ...]] = ("open_time", "open", "high", "low", "close", "volume")
_NUMERIC_CANDLE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


def validate_candles(frame: pd.DataFrame) -> None:
    """Fail closed on malformed OHLCV data before it reaches a backtest.

    This deliberately does not require a gap-free calendar: crypto venues can
    have missing candles, but a duplicate, reordered, non-positive or internally
    inconsistent candle must never be silently repaired.
    """
    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Candle data is missing required columns: {', '.join(missing)}.")
    if frame.empty:
        raise ValueError("Candle data cannot be empty.")

    timestamps = pd.to_datetime(frame["open_time"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("Candle open_time values must be valid timestamps.")
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("Candle open_time values must be unique and sorted ascending.")

    numeric = frame.loc[:, _NUMERIC_CANDLE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Candle OHLCV values must be finite numbers.")
    if (numeric.loc[:, ("open", "high", "low", "close")] <= 0).any().any():
        raise ValueError("Candle OHLC prices must be strictly positive.")
    if (numeric["volume"] < 0).any():
        raise ValueError("Candle volume cannot be negative.")

    high_floor = numeric[["open", "close"]].max(axis=1)
    low_ceiling = numeric[["open", "close"]].min(axis=1)
    if (numeric["high"] < high_floor).any() or (numeric["low"] > low_ceiling).any():
        raise ValueError("Candle high/low values are inconsistent with open/close.")


def candle_fingerprint(frame: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 fingerprint for validated OHLCV rows."""
    validate_candles(frame)
    canonical = frame.loc[:, REQUIRED_CANDLE_COLUMNS].copy()
    canonical["open_time"] = pd.to_datetime(canonical["open_time"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    for column in _NUMERIC_CANDLE_COLUMNS:
        canonical[column] = canonical[column].map(lambda value: format(float(value), ".17g"))
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_provenance(
    frame: pd.DataFrame,
    *,
    source: str,
    symbol: str,
    interval: str,
) -> dict[str, str | int]:
    """Build report metadata that makes the exact input artifact auditable."""
    validate_candles(frame)
    timestamps = pd.to_datetime(frame["open_time"], utc=True)
    return {
        "schema_version": "candle-provenance-v1",
        "source": source,
        "symbol": symbol.upper(),
        "interval": interval,
        "rows": int(len(frame)),
        "start": timestamps.iloc[0].isoformat(),
        "end": timestamps.iloc[-1].isoformat(),
        "sha256": candle_fingerprint(frame),
    }
