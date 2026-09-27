import json
from argparse import Namespace

import numpy as np
import pandas as pd

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
