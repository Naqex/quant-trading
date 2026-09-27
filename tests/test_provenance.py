import pandas as pd
import pytest

from quant_lab.data import load_candles, save_candles
from quant_lab.provenance import build_provenance, candle_fingerprint, validate_candles


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "open": [100.0, 105.0],
            "high": [110.0, 112.0],
            "low": [95.0, 101.0],
            "close": [105.0, 108.0],
            "volume": [10.0, 12.0],
        }
    )


def test_provenance_is_deterministic_and_auditable() -> None:
    frame = _candles()
    assert candle_fingerprint(frame) == candle_fingerprint(frame.copy())
    result = build_provenance(frame, source="fixture", symbol="btcusdt", interval="1d")
    assert result["schema_version"] == "candle-provenance-v1"
    assert result["symbol"] == "BTCUSDT"
    assert result["rows"] == 2
    assert len(result["sha256"]) == 64


def test_save_and_load_keep_validation_fail_closed(tmp_path) -> None:
    frame = _candles().assign(close_time=pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True))
    path = save_candles(frame, tmp_path / "candles.csv")
    loaded = load_candles(path)
    assert loaded["open_time"].tolist() == frame["open_time"].tolist()

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique and sorted"):
        load_candles(path)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.assign(high=[104.0, 112.0]), "high/low"),
        (lambda frame: frame.assign(volume=[-1.0, 12.0]), "volume"),
        (lambda frame: frame.assign(open_time=frame["open_time"].iloc[::-1].to_numpy()), "unique and sorted"),
        (lambda frame: frame.assign(close=[float("nan"), 108.0]), "finite"),
    ],
)
def test_validate_candles_rejects_malformed_input(mutator, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_candles(mutator(_candles()))
