import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from quant_lab import research


def _write_fixture(path) -> None:
    timestamps = pd.date_range("2023-01-01", periods=320, freq="D", tz="UTC")
    close = 100.0 + np.arange(len(timestamps), dtype=float) * 0.2 + np.sin(np.arange(len(timestamps)))
    frame = pd.DataFrame(
        {
            "open_time": timestamps,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000.0,
            "close_time": timestamps + pd.to_timedelta(23, unit="h"),
        }
    )
    frame.to_csv(path, index=False)


def test_quant_research_writes_provenance_sidecar_and_row_digest(tmp_path, monkeypatch) -> None:
    data_directory = tmp_path / "data"
    report_directory = tmp_path / "reports"
    data_directory.mkdir()
    _write_fixture(data_directory / "BTCUSDT_1d.csv")
    monkeypatch.setattr(research, "DATA_DIRECTORY", data_directory)
    monkeypatch.setattr(research, "REPORT_DIRECTORY", report_directory)

    research.run_research(
        Namespace(
            symbols=["btcusdt"],
            test_start="2023-09-01",
            fee_bps=10.0,
            slippage_bps=5.0,
        )
    )

    payload = json.loads((report_directory / "data_provenance.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research-report-provenance-v1"
    assert payload["run"]["symbols"] == ["BTCUSDT"]
    assert payload["run"]["test_start"] == "2023-09-01"
    dataset = payload["datasets"]["BTCUSDT"]
    assert dataset["schema_version"] == "candle-provenance-v1"
    assert dataset["rows"] == 320
    assert len(dataset["sha256"]) == 64

    summary = pd.read_csv(report_directory / "strategy_summary.csv")
    assert summary["data_sha256"].nunique() == 1
    assert summary["data_sha256"].iloc[0] == dataset["sha256"]
    assert set(payload["artifacts"]).issubset({path.name for path in report_directory.iterdir()})

    portfolio = pd.read_csv(report_directory / "portfolio_test.csv")
    assert json.loads(portfolio["data_provenance"].iloc[0])["BTCUSDT"]["sha256"] == dataset["sha256"]


def test_evaluate_portfolio_preserves_active_sleeve_returns_when_peer_has_gap_or_delists(
    tmp_path, monkeypatch
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    timestamps = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    btc_close = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 121.0, 133.1])
    btc = pd.DataFrame(
        {
            "open_time": timestamps,
            "open": btc_close,
            "high": btc_close * 1.01,
            "low": btc_close * 0.99,
            "close": btc_close,
            "volume": 1_000.0,
            "close_time": timestamps + pd.to_timedelta(23, unit="h"),
        }
    )
    btc.to_csv(data_directory / "BTCUSDT_1d.csv", index=False)

    # SOL misses day 7 and ends after day 8 (delists before days 9-10)
    sol_idx = [0, 1, 2, 3, 4, 5, 7]
    sol_close = np.full(len(sol_idx), 50.0)
    sol = pd.DataFrame(
        {
            "open_time": timestamps[sol_idx],
            "open": sol_close,
            "high": sol_close * 1.01,
            "low": sol_close * 0.99,
            "close": sol_close,
            "volume": 500.0,
            "close_time": timestamps[sol_idx] + pd.to_timedelta(23, unit="h"),
        }
    )
    sol.to_csv(data_directory / "SOLUSDT_1d.csv", index=False)
    monkeypatch.setattr(research, "DATA_DIRECTORY", data_directory)

    result = research.evaluate_portfolio(
        ["BTCUSDT", "SOLUSDT"],
        "buy_and_hold",
        research.STRATEGIES["buy_and_hold"],
        test_start="2023-01-06",
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    # Must span full BTC test window through 2023-01-10 and capture BTC's +33.1% gain (50% weight -> +16.55%)
    assert result["end"] == "2023-01-10"
    assert result["total_return"] == pytest.approx(0.1655, rel=1e-4)
