from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_lab.cli import build_parser
from quant_lab.data import audit_ohlcv_directory


def _ohlcv_frame(
    *,
    periods: int = 8,
    freq: str = "1d",
    start: str = "2024-01-01",
    base_price: float = 100.0,
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = base_price + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "open_time": ts,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(periods, 500.0),
        }
    )


def _write_csv(directory: Path, filename: str, frame: pd.DataFrame) -> Path:
    path = directory / filename
    frame.to_csv(path, index=False)
    return path


def test_audit_clean_multi_asset_directory_passes_deterministically(tmp_path: Path) -> None:
    _write_csv(tmp_path, "BTCUSDT_4h.csv", _ohlcv_frame(periods=12, freq="4h", base_price=40_000.0))
    _write_csv(tmp_path, "ETHUSDT_4h.csv", _ohlcv_frame(periods=10, freq="4h", base_price=2_500.0))

    first = audit_ohlcv_directory(tmp_path, timeframe="4h")
    second = audit_ohlcv_directory(tmp_path, timeframe="4h")

    assert first["status"] == "PASS"
    assert first["report_sha256"] == second["report_sha256"]
    assert first["execution_capability"] is False
    assert first["provider_access"] is False
    assert first["broker_access"] is False
    assert first["summary"]["files_scanned"] == 2
    assert first["summary"]["passed_symbols"] == 2
    assert first["summary"]["failed_symbols"] == 0
    assert first["summary"]["total_rows"] == 22
    assert first["summary"]["total_gaps"] == 0
    assert first["summary"]["total_zero_volume_bars"] == 0

    by_symbol = {item["symbol"]: item for item in first["symbols"]}
    assert set(by_symbol) == {"BTCUSDT", "ETHUSDT"}
    assert by_symbol["BTCUSDT"]["rows"] == 12
    assert by_symbol["BTCUSDT"]["expected_interval_seconds"] == 14_400
    assert by_symbol["BTCUSDT"]["observed_interval_seconds"] == 14_400.0
    assert by_symbol["BTCUSDT"]["data_sha256"] is not None
    assert len(by_symbol["BTCUSDT"]["data_sha256"]) == 64


@pytest.mark.parametrize("timeframe,freq", [("1d", "1D"), ("4h", "4h"), ("1h", "1h")])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_audit_detects_missing_candle_gaps_across_resolutions_and_timeframes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeframe: str,
    freq: str,
    unit: str,
) -> None:
    frame = _ohlcv_frame(periods=6, freq=freq).drop(index=3).reset_index(drop=True)
    frame["open_time"] = frame["open_time"].dt.as_unit(unit)
    parquet_path = tmp_path / f"BTCUSDT_{timeframe}.parquet"
    parquet_path.write_bytes(b"placeholder")
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame.copy())

    report = audit_ohlcv_directory(tmp_path, timeframe=timeframe)
    assert report["status"] == "FAIL"
    item = report["symbols"][0]
    assert item["format"] == "parquet"
    assert item["status"] == "FAIL"
    assert item["missing_candle_gaps"] == 1
    assert item["missing_candles_estimated"] == 1
    assert "gap_budget_exceeded" in item["errors"]


def test_audit_fails_closed_on_every_ohlcv_anomaly(tmp_path: Path) -> None:
    # 1) Duplicate timestamps
    dup_dir = tmp_path / "dup"
    dup_dir.mkdir()
    dup = _ohlcv_frame()
    dup.loc[3, "open_time"] = dup.loc[2, "open_time"]
    _write_csv(dup_dir, "BTCUSDT_1d.csv", dup)
    r_dup = audit_ohlcv_directory(dup_dir, timeframe="1d")
    assert r_dup["status"] == "FAIL"
    assert r_dup["symbols"][0]["duplicate_timestamps"] == 1
    assert "duplicate_timestamp" in r_dup["symbols"][0]["errors"]

    # 2) Monotonic order violation
    order_dir = tmp_path / "order"
    order_dir.mkdir()
    reordered = _ohlcv_frame()
    reordered.loc[2, "open_time"], reordered.loc[4, "open_time"] = (
        reordered.loc[4, "open_time"],
        reordered.loc[2, "open_time"],
    )
    _write_csv(order_dir, "BTCUSDT_1d.csv", reordered)
    r_ord = audit_ohlcv_directory(order_dir, timeframe="1d")
    assert r_ord["status"] == "FAIL"
    assert r_ord["symbols"][0]["monotonic_violations"] >= 1
    assert "timestamp_order" in r_ord["symbols"][0]["errors"]

    # 3) Non-positive price (0 and negative)
    nonpos_dir = tmp_path / "nonpos"
    nonpos_dir.mkdir()
    nonpos = _ohlcv_frame()
    nonpos.loc[1, "close"] = 0.0
    nonpos.loc[2, "low"] = -5.0
    _write_csv(nonpos_dir, "BTCUSDT_1d.csv", nonpos)
    r_nonpos = audit_ohlcv_directory(nonpos_dir, timeframe="1d")
    assert r_nonpos["status"] == "FAIL"
    assert r_nonpos["symbols"][0]["non_positive_prices"] == 2
    assert "non_positive_price" in r_nonpos["symbols"][0]["errors"]

    # 4) NaN and Inf OHLC prices
    nonfin_dir = tmp_path / "nonfin"
    nonfin_dir.mkdir()
    nonfin = _ohlcv_frame()
    nonfin.loc[1, "open"] = np.nan
    nonfin.loc[2, "high"] = np.inf
    _write_csv(nonfin_dir, "BTCUSDT_1d.csv", nonfin)
    r_nonfin = audit_ohlcv_directory(nonfin_dir, timeframe="1d")
    assert r_nonfin["status"] == "FAIL"
    assert r_nonfin["symbols"][0]["non_finite_prices"] == 2
    assert "non_finite_numeric" in r_nonfin["symbols"][0]["errors"]

    # 5) High < Low / OHLC consistency violation
    ohlc_dir = tmp_path / "ohlc"
    ohlc_dir.mkdir()
    bad_ohlc = _ohlcv_frame()
    bad_ohlc.loc[2, "high"] = bad_ohlc.loc[2, "low"] - 2.0
    _write_csv(ohlc_dir, "BTCUSDT_1d.csv", bad_ohlc)
    r_ohlc = audit_ohlcv_directory(ohlc_dir, timeframe="1d")
    assert r_ohlc["status"] == "FAIL"
    assert r_ohlc["symbols"][0]["ohlc_consistency_violations"] == 1
    assert "inconsistent_high_low" in r_ohlc["symbols"][0]["errors"]

    # 6) Zero-volume halted bars
    zvol_dir = tmp_path / "zvol"
    zvol_dir.mkdir()
    zvol = _ohlcv_frame()
    zvol.loc[4, "volume"] = 0.0
    _write_csv(zvol_dir, "BTCUSDT_1d.csv", zvol)
    r_zvol = audit_ohlcv_directory(zvol_dir, timeframe="1d")
    assert r_zvol["status"] == "FAIL"
    assert r_zvol["symbols"][0]["zero_volume_bars"] == 1
    assert "zero_volume_halted_bars:1" in r_zvol["symbols"][0]["errors"]

    # 7) Negative volume
    nvol_dir = tmp_path / "nvol"
    nvol_dir.mkdir()
    nvol = _ohlcv_frame()
    nvol.loc[2, "volume"] = -10.0
    _write_csv(nvol_dir, "BTCUSDT_1d.csv", nvol)
    r_nvol = audit_ohlcv_directory(nvol_dir, timeframe="1d")
    assert r_nvol["status"] == "FAIL"
    assert r_nvol["symbols"][0]["negative_volume_bars"] == 1
    assert "negative_volume" in r_nvol["symbols"][0]["errors"]

    # 8) Empty directory fails closed
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    r_empty = audit_ohlcv_directory(empty_dir)
    assert r_empty["status"] == "FAIL"
    assert "no_ohlcv_files_found" in r_empty["errors"]

    # 9) Unreadable / missing parquet engine fails closed cleanly
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "BROKEN_1d.csv").write_text("", encoding="utf-8")
    r_corrupt = audit_ohlcv_directory(corrupt_dir, timeframe="1d")
    assert r_corrupt["status"] == "FAIL"
    assert any(err.startswith("read_error:") for err in r_corrupt["symbols"][0]["errors"])


def test_data_audit_cli_json_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    _write_csv(clean_dir, "BTCUSDT_1d.csv", _ohlcv_frame(periods=6, freq="1d"))
    _write_csv(clean_dir, "ETHUSDT_1d.csv", _ohlcv_frame(periods=6, freq="1d"))

    parser = build_parser()
    args = parser.parse_args(["data-audit", "--dir", str(clean_dir), "--timeframe", "1d", "--json"])
    args.handler(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["summary"]["files_scanned"] == 2
    assert payload["execution_capability"] is False

    # Text mode PASS
    text_args = parser.parse_args(["data-audit", "--dir", str(clean_dir)])
    text_args.handler(text_args)
    text_out = capsys.readouterr().out
    assert "-> PASS" in text_out
    assert "BTCUSDT" in text_out

    # Corrupt one asset -> CLI --json emits valid FAIL JSON AND exits non-zero (code 1)
    bad = _ohlcv_frame(periods=6, freq="1d").drop(index=2).reset_index(drop=True)
    _write_csv(clean_dir, "SOLUSDT_1d.csv", bad)
    fail_args = parser.parse_args(["data-audit", "--dir", str(clean_dir), "--timeframe", "1d", "--json"])
    with pytest.raises(SystemExit) as exc_info:
        fail_args.handler(fail_args)
    assert exc_info.value.code == 1
    fail_payload = json.loads(capsys.readouterr().out)
    assert fail_payload["status"] == "FAIL"
    assert fail_payload["summary"]["passed_symbols"] == 2
    assert fail_payload["summary"]["failed_symbols"] == 1

    # Missing directory exits non-zero
    missing_args = parser.parse_args(["data-audit", "--dir", str(tmp_path / "does_not_exist")])
    with pytest.raises(SystemExit, match="does not exist"):
        missing_args.handler(missing_args)
