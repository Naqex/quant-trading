from __future__ import annotations

import json

import pytest

from quant_trading.discrete_optimizer import ResearchCutoffs
from quant_trading.selective_prediction import (
    SelectiveDecision,
    SelectivePredictionError,
    evaluate_selective,
    fit_selective_threshold,
)

DATA_SHA = "a" * 64
EVAL_SHA = "b" * 64
CUTOFFS = ResearchCutoffs(train_end_utc=100, validation_end_utc=200, test_end_utc=300)


def _calibration(**overrides):
    values: dict[str, object] = {
        "y_true": [10.0, 10.1, 10.2, 10.8, 10.9],
        "y_pred": [10.0, 10.0, 10.0, 10.0, 10.0],
        "uncertainty_scores": [0.1, 0.2, 0.3, 0.4, 0.5],
        "timestamps_utc": [110, 120, 130, 140, 150],
        "target_coverage": 0.6,
        "max_risk": 0.2,
        "data_sha256": DATA_SHA,
        "cutoffs": CUTOFFS,
    }
    values.update(overrides)
    return fit_selective_threshold(**values)  # type: ignore[arg-type]


def test_fit_is_deterministic_and_selects_largest_safe_threshold() -> None:
    first = _calibration()
    second = _calibration()

    assert first.as_dict() == second.as_dict()
    assert first.uncertainty_threshold == 0.3
    assert first.calibration_accepted_count == 3
    assert first.calibration_coverage == pytest.approx(0.6)
    assert first.calibration_risk == pytest.approx(0.1)
    assert first.receipt_sha256 and len(first.receipt_sha256) == 64
    assert first.as_dict()["status"] == "PREP_ONLY"
    assert first.as_dict()["execution_capability"] is False
    json.dumps(first.as_dict(), allow_nan=False)


def test_decide_and_evaluate_are_causal_and_auditable() -> None:
    calibration = _calibration()
    decisions = calibration.decide(
        point_predictions=[20.0, 21.0, 22.0, 23.0],
        uncertainty_scores=[0.1, 0.3, 0.31, 0.8],
        timestamps_utc=[210, 220, 230, 240],
    )
    assert [item.accepted for item in decisions] == [True, True, False, False]
    evaluation = evaluate_selective(calibration, [20.0, 21.1, 100.0, 100.0], decisions, data_sha256=EVAL_SHA)
    assert evaluation.accepted_count == 2
    assert evaluation.coverage == pytest.approx(0.5)
    assert evaluation.abstention_rate == pytest.approx(0.5)
    assert evaluation.selective_risk == pytest.approx(0.05)
    assert evaluation.policy_sha256 == calibration.receipt_sha256
    assert evaluation.receipt_sha256 and len(evaluation.receipt_sha256) == 64
    assert evaluation.as_dict()["provider_access"] is False


def test_ties_are_included_and_all_abstain_is_reported() -> None:
    calibration = fit_selective_threshold(
        y_true=[0.0, 0.0, 0.0, 1.0],
        y_pred=[0.0, 0.0, 0.0, 0.0],
        uncertainty_scores=[0.1, 0.2, 0.2, 0.3],
        timestamps_utc=[110, 120, 130, 140],
        target_coverage=0.5,
        max_risk=0.1,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
    )
    assert calibration.uncertainty_threshold == 0.2
    decisions = calibration.decide([1.0, 2.0], [0.21, 0.22], [210, 220])
    evaluation = evaluate_selective(calibration, [1.0, 2.0], decisions, data_sha256=EVAL_SHA)
    assert evaluation.accepted_count == 0
    assert evaluation.coverage == 0.0
    assert evaluation.selective_risk is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestamps_utc": [100, 120, 130, 140, 150]}, "after train cutoff"),
        ({"timestamps_utc": [110, 120, 130, 140, 210]}, "validation cutoff"),
        ({"timestamps_utc": [110, 120, 120, 140, 150]}, "strictly increasing"),
        ({"target_coverage": 0.0}, "target_coverage"),
        ({"max_risk": -1.0}, ">= 0"),
        ({"uncertainty_scores": [0.1, -0.2, 0.3, 0.4, 0.5]}, "non-negative"),
        ({"data_sha256": "A" * 64}, "lowercase"),
    ],
)

def test_fit_rejects_leakage_and_malformed_contract(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(SelectivePredictionError, match=message):
        _calibration(**kwargs)


def test_fit_rejects_no_safe_threshold_and_overflow() -> None:
    with pytest.raises(SelectivePredictionError, match="no uncertainty threshold"):
        fit_selective_threshold(
            y_true=[1.0, 2.0, 3.0],
            y_pred=[0.0, 0.0, 0.0],
            uncertainty_scores=[0.1, 0.2, 0.3],
            timestamps_utc=[110, 120, 130],
            target_coverage=0.5,
            max_risk=0.1,
            data_sha256=DATA_SHA,
            cutoffs=CUTOFFS,
        )
    with pytest.raises(SelectivePredictionError, match="overflowed"):
        fit_selective_threshold(
            y_true=[1.0e308],
            y_pred=[-1.0e308],
            uncertainty_scores=[0.1],
            timestamps_utc=[110],
            target_coverage=1.0,
            max_risk=1.0e308,
            data_sha256=DATA_SHA,
            cutoffs=CUTOFFS,
        )


def test_decide_and_evaluate_reject_future_window_and_malformed_rows() -> None:
    calibration = _calibration()
    with pytest.raises(SelectivePredictionError, match="after validation"):
        calibration.decide([1.0], [0.1], [200])
    with pytest.raises(SelectivePredictionError, match="test cutoff"):
        calibration.decide([1.0], [0.1], [301])
    with pytest.raises(SelectivePredictionError, match="same length"):
        calibration.decide([1.0], [0.1, 0.2], [210])
    with pytest.raises(SelectivePredictionError, match="strictly increasing"):
        evaluate_selective(
            calibration,
            [1.0, 2.0],
            [SelectiveDecision(210, 1.0, 0.1, True), SelectiveDecision(210, 2.0, 0.1, True)],
            data_sha256=EVAL_SHA,
        )



