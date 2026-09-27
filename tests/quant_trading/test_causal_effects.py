import math

import pytest

from quant_trading.causal_effects import (
    CausalEffectConfig,
    CausalEffectError,
    TemporalCausalSplit,
    TreatmentObservation,
    run_causal_effect_pilot,
    split_temporal_observations,
)


def _synthetic_rows(count: int = 72) -> list[TreatmentObservation]:
    rows: list[TreatmentObservation] = []
    for index in range(count):
        covariate = ((index % 9) - 4) / 4.0
        treatment = int(index % 2 == 0)
        outcome = 1.2 * treatment + 0.35 * covariate + (0.03 if index % 3 == 0 else -0.02)
        rows.append(
            TreatmentObservation(
                unit_id="BTCUSDT",
                event_time=index,
                known_at=index,
                outcome_known_at=index + 1,
                treatment=treatment,
                outcome=outcome,
                covariates={"state": covariate},
            )
        )
    return rows


def test_temporal_aipw_is_deterministic_and_keeps_execution_denied() -> None:
    rows = _synthetic_rows()
    split = TemporalCausalSplit(train_end=29, calibration_end=44, holdout_end=71)
    config = CausalEffectConfig(min_arm_count=3, min_effective_sample_size=3)
    first = run_causal_effect_pilot(rows, split=split, config=config)
    second = run_causal_effect_pilot(reversed(rows), split=split, config=config)
    default_result = run_causal_effect_pilot(rows, split=split)

    assert first.status == "accepted"
    assert default_result.status == "accepted"
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.ate_holdout is not None and first.ate_holdout > 0.5
    assert first.baseline_difference_in_means is not None
    assert first.placebo_difference_in_means is not None
    assert first.overlap_fraction == 1.0
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False
    assert first.as_dict()["receipt_sha256"] == first.receipt_sha256


def test_split_uses_outcome_availability_and_never_leaks_future_labels() -> None:
    rows = _synthetic_rows(12)
    train, calibration, holdout = split_temporal_observations(
        rows,
        TemporalCausalSplit(train_end=4, calibration_end=8),
    )
    assert [row.outcome_known_at for row in train] == [1, 2, 3, 4]
    assert [row.outcome_known_at for row in calibration] == [5, 6, 7, 8]
    assert [row.outcome_known_at for row in holdout] == [9, 10, 11, 12]
    assert max(row.known_at for row in train) <= 3
    assert min(row.outcome_known_at for row in holdout) > 8


def test_weak_training_or_holdout_arms_return_inconclusive() -> None:
    rows = _synthetic_rows()
    one_arm = [TreatmentObservation(**{**row.as_dict(), "treatment": 1}) for row in rows]
    result = run_causal_effect_pilot(
        one_arm,
        split=TemporalCausalSplit(train_end=29, calibration_end=44),
        config=CausalEffectConfig(min_arm_count=3, min_effective_sample_size=3),
    )
    assert result.status == "inconclusive"
    assert result.reason == "train split lacks both treatment arms"

    holdout_one_arm = [
        TreatmentObservation(**{**row.as_dict(), "treatment": (1 if row.event_time >= 45 else row.treatment)})
        for row in rows
    ]
    result = run_causal_effect_pilot(
        holdout_one_arm,
        split=TemporalCausalSplit(train_end=29, calibration_end=44),
        config=CausalEffectConfig(min_arm_count=3, min_effective_sample_size=3),
    )
    assert result.status == "inconclusive"
    assert result.reason == "holdout split lacks both treatment arms"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"known_at": 3, "event_time": 4}, "known_at cannot precede"),
        ({"outcome_known_at": 2, "known_at": 2}, "outcome_known_at must be after"),
        ({"treatment": 2}, "treatment must"),
        ({"outcome": math.inf}, "outcome must be finite"),
    ],
)
def test_observation_rejects_non_causal_or_non_finite_values(
    overrides: dict[str, object], message: str
) -> None:
    values = {
        "unit_id": "BTCUSDT",
        "event_time": 1,
        "known_at": 1,
        "outcome_known_at": 2,
        "treatment": 0,
        "outcome": 0.1,
        "covariates": {"state": 0.0},
    }
    values.update(overrides)
    with pytest.raises(CausalEffectError, match=message):
        TreatmentObservation(**values)


def test_observation_rejects_duplicate_point_in_time_keys_and_covariate_drift() -> None:
    first = _synthetic_rows(2)[0]
    duplicate = TreatmentObservation(**{**first.as_dict(), "outcome_known_at": 3})
    with pytest.raises(CausalEffectError, match="duplicate unit/event_time/sequence"):
        split_temporal_observations([first, duplicate], TemporalCausalSplit(train_end=2, calibration_end=4))
    known_first = _synthetic_rows(6)[5]
    known_duplicate = TreatmentObservation(
        unit_id="BTCUSDT",
        event_time=4,
        known_at=known_first.known_at,
        outcome_known_at=6,
        treatment=1,
        outcome=0.5,
        covariates={"state": 0.0},
    )
    with pytest.raises(CausalEffectError, match="duplicate unit/known_at/sequence"):
        split_temporal_observations(
            [known_first, known_duplicate],
            TemporalCausalSplit(train_end=2, calibration_end=4),
        )

    changed_covariates = TreatmentObservation(
        unit_id="BTCUSDT",
        event_time=2,
        known_at=2,
        outcome_known_at=3,
        treatment=1,
        outcome=0.5,
        covariates={"other": 0.0},
    )
    with pytest.raises(CausalEffectError, match="same covariate names"):
        split_temporal_observations(
            [first, changed_covariates],
            TemporalCausalSplit(train_end=2, calibration_end=4),
        )


def test_split_and_config_boundaries_are_validated() -> None:
    with pytest.raises(CausalEffectError, match="calibration_end"):
        TemporalCausalSplit(train_end=4, calibration_end=4)
    with pytest.raises(CausalEffectError, match="propensity_epsilon"):
        CausalEffectConfig(propensity_epsilon=0.5)

def test_propensity_overlap_uses_unclipped_scores_and_can_fail_closed() -> None:
    rows = []
    for index in range(100):
        covariate = -5.0 if index % 2 == 0 else 5.0
        treatment = int(covariate > 0)
        rows.append(
            TreatmentObservation(
                unit_id="BTCUSDT",
                event_time=index,
                known_at=index,
                outcome_known_at=index + 1,
                treatment=treatment,
                outcome=float(treatment) + 0.1 * covariate,
                covariates={"state": covariate},
            )
        )
    result = run_causal_effect_pilot(
        rows,
        split=TemporalCausalSplit(train_end=49, calibration_end=74),
        config=CausalEffectConfig(
            min_arm_count=3,
            min_effective_sample_size=3,
            propensity_epsilon=0.1,
            min_overlap=0.8,
        ),
    )
    assert result.status == "inconclusive"
    assert result.reason == "propensity overlap is below the pre-registered threshold"
    assert result.overlap_fraction == 0.0
