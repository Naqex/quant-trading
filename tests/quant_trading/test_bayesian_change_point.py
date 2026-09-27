import json

import pytest

from quant_trading.bayesian_change_point import (
    BayesianChangePointError,
    BOCPDConfig,
    run_bocpd,
)


def _obs(event_time: int, value: float, *, known_at: int | None = None, sequence: int = 0):
    return {
        "event_time": event_time,
        "known_at": event_time if known_at is None else known_at,
        "value": value,
        "sequence": sequence,
    }


def _config(**overrides) -> BOCPDConfig:
    values = {
        "prior_mean": 0.0,
        "prior_kappa": 1.0,
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "hazard": 0.25,
    }
    values.update(overrides)
    return BOCPDConfig(**values)


def test_bocpd_is_causal_normalized_and_detects_a_synthetic_shift() -> None:
    observations = [_obs(index, value) for index, value in enumerate((0.0, 0.0, 5.0, 5.0), 1)]
    result = run_bocpd(observations, config=_config(), cutoff_timestamp=4)

    assert result.execution_capability is False
    assert result.provider_access is False
    assert result.broker_access is False
    assert result.steps[0].change_probability == pytest.approx(0.25)
    assert result.steps[2].change_probability > result.steps[1].change_probability
    assert all(
        sum(item.probability for item in step.run_length_posterior) == pytest.approx(1.0)
        for step in result.steps
    )
    assert all(step.predictive_log_score <= 0.0 for step in result.steps)
    json.dumps(result.as_dict(), allow_nan=False)


def test_bocpd_is_deterministic_independent_of_input_order() -> None:
    observations = [_obs(index, value) for index, value in enumerate((0.0, 0.2, -0.1, 0.3), 1)]
    first = run_bocpd(observations, config=_config())
    second = run_bocpd(list(reversed(observations)), config=_config())

    assert first.as_dict() == second.as_dict()
    assert first.canonical_sha256() == second.canonical_sha256()


def test_bocpd_rejects_future_rows_noncausal_rows_and_duplicate_keys() -> None:
    with pytest.raises(BayesianChangePointError, match="future data"):
        run_bocpd([_obs(1, 0.0, known_at=5)], config=_config(), cutoff_timestamp=4)
    with pytest.raises(BayesianChangePointError, match="known_at cannot precede"):
        run_bocpd([_obs(4, 0.0, known_at=3)], config=_config())
    with pytest.raises(BayesianChangePointError, match="unique"):
        run_bocpd([_obs(1, 0.0, known_at=2), _obs(2, 1.0, known_at=2)], config=_config())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hazard": 0.0}, "hazard"),
        ({"hazard": 1.1}, "hazard"),
        ({"prior_kappa": 0.0}, "prior_kappa"),
        ({"prior_alpha": float("nan")}, "prior_alpha"),
        ({"prior_beta": -1.0}, "prior_beta"),
        ({"max_run_length": 0}, "max_run_length"),
    ],
)
def test_bocpd_config_rejects_unsafe_parameters(kwargs, message: str) -> None:
    with pytest.raises(BayesianChangePointError, match=message):
        _config(**kwargs)


def test_bocpd_max_run_length_fails_closed_instead_of_dropping_mass() -> None:
    observations = [_obs(index, 0.0) for index in range(1, 5)]
    with pytest.raises(BayesianChangePointError, match="max_run_length"):
        run_bocpd(observations, config=_config(max_run_length=2))


def test_bocpd_handles_extreme_finite_values_without_nonfinite_receipt() -> None:
    observations = [_obs(1, 1e300), _obs(2, -1e300), _obs(3, 0.0)]
    result = run_bocpd(observations, config=_config())

    assert all(0.0 <= value <= 1.0 for value in result.change_probabilities)
    assert all(step.predictive_log_score < 0.0 for step in result.steps)
    json.dumps(result.as_dict(), allow_nan=False)


def test_empty_bocpd_result_is_valid_and_deny_only() -> None:
    result = run_bocpd([], config=_config())

    assert result.steps == ()
    assert result.change_probabilities == ()
    assert result.execution_capability is False
    assert result.provider_access is False
    assert result.broker_access is False
