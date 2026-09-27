from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from quant_lab.provenance import validate_candles

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
