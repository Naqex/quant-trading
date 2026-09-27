from __future__ import annotations

import math

import pytest

from quant_trading.conformal import (
    ConformalError,
    PredictionInterval,
    empirical_coverage,
    fit_split_conformal,
)
from quant_trading.discrete_optimizer import ResearchCutoffs

DATA_SHA = "a" * 64
CUTOFFS = ResearchCutoffs(train_end_utc=100, validation_end_utc=200, test_end_utc=300)


def test_split_conformal_is_deterministic_and_causal() -> None:
    calibration = fit_split_conformal(
        y_true=[10.0, 12.0, 9.0, 15.0],
        y_pred=[9.0, 10.0, 10.0, 12.0],
        timestamps_utc=[120, 130, 140, 150],
        alpha=0.25,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
    )
    # Residuals are [1, 2, 1, 3]; ceil(5 * .75) = 4, so q is 3.
    assert calibration.quantile == 3.0
    assert calibration.calibration_count == 4
    assert calibration.as_dict()["status"] == "PREP_ONLY"
    assert calibration.as_dict()["execution_capability"] is False
    intervals = calibration.predict([20.0, 21.0], [220, 230])
    assert intervals == (
        PredictionInterval(220, 20.0, 17.0, 23.0),
        PredictionInterval(230, 21.0, 18.0, 24.0),
    )


def test_coverage_is_evaluation_only_and_deterministic() -> None:
    calibration = fit_split_conformal(
        [0.0, 2.0, 5.0],
        [0.0, 1.0, 4.0],
        [110, 120, 130],
        alpha=0.2,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
    )
    intervals = calibration.predict([1.0, 10.0], [210, 220])
    assert empirical_coverage([2.0, 7.0], intervals) == 0.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestamps_utc": [100, 120, 130]}, "after train cutoff"),
        ({"timestamps_utc": [120, 220, 230]}, "validation cutoff"),
        ({"timestamps_utc": [120, 110, 130]}, "strictly increasing"),
        ({"alpha": 0.0}, "strictly between 0 and 1"),
        ({"data_sha256": "A" * 64}, "lowercase"),
    ],
)
def test_fit_rejects_leakage_and_malformed_contract(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "y_true": [1.0, 2.0, 3.0],
        "y_pred": [1.0, 2.0, 3.0],
        "timestamps_utc": [120, 130, 140],
        "alpha": 0.1,
        "data_sha256": DATA_SHA,
        "cutoffs": CUTOFFS,
    }
    values.update(kwargs)
    with pytest.raises(ConformalError, match=message):
        fit_split_conformal(**values)  # type: ignore[arg-type]


def test_predict_rejects_out_of_window_or_overflow() -> None:
    calibration = fit_split_conformal(
        [1.0e308], [0.0], [110], alpha=0.1, data_sha256=DATA_SHA, cutoffs=CUTOFFS
    )
    with pytest.raises(ConformalError, match="after validation"):
        calibration.predict([1.0], [200])
    with pytest.raises(ConformalError, match="test cutoff"):
        calibration.predict([1.0], [301])
    with pytest.raises(ConformalError, match="overflowed"):
        calibration.predict([math.nextafter(float("inf"), 0.0)], [220])


def test_small_calibration_is_conservative_and_rejects_bad_coverage_inputs() -> None:
    calibration = fit_split_conformal(
        [1.0], [0.0], [150], alpha=0.99, data_sha256=DATA_SHA, cutoffs=CUTOFFS
    )
    assert calibration.quantile == 1.0
    with pytest.raises(ConformalError, match="same non-empty length"):
        empirical_coverage([1.0], [])
