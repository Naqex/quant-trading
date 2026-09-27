import math

import pytest

from quant_trading.state_primitives import (
    BernoulliObservation,
    BetaPrior,
    CategoricalObservation,
    CausalObservation,
    Kalman1DConfig,
    StatePrimitiveError,
    compute_permutation_entropy,
    compute_shannon_entropy,
    run_kalman_1d,
    update_beta_bernoulli,
)


def test_kalman_is_forward_only_finite_and_deterministic() -> None:
    observations = [
        CausalObservation(event_time=10, known_at=12, value=100.0),
        CausalObservation(event_time=11, known_at=13, value=101.0),
        CausalObservation(event_time=12, known_at=14, value=102.0),
    ]
    config = Kalman1DConfig(
        transition=1.0,
        process_variance=1e-8,
        measurement_variance=1e-6,
        initial_mean=99.0,
        initial_variance=0.0,
    )
    first = run_kalman_1d(observations, config=config, cutoff_timestamp=14)
    second = run_kalman_1d(reversed(observations), config=config, cutoff_timestamp=14)

    assert first.canonical_sha256() == second.canonical_sha256()
    assert first.execution_capability is False
    # With near-zero process noise the filter moves conservatively from the
    # initial state; the exact value is deterministic for this configuration.
    assert first.final_mean == pytest.approx(99.13333603645825)
    assert all(item.filtered_variance >= 0 for item in first.observations)
    assert all(math.isfinite(item.filtered_variance) for item in first.observations)


def test_kalman_rejects_future_cutoff_and_noncausal_time() -> None:
    with pytest.raises(StatePrimitiveError, match="exceeds research cutoff"):
        run_kalman_1d([CausalObservation(event_time=1, known_at=5, value=1.0)], cutoff_timestamp=4)
    with pytest.raises(StatePrimitiveError, match="known_at cannot precede"):
        CausalObservation(event_time=4, known_at=3, value=1.0)


def test_entropy_is_categorical_and_records_correction() -> None:
    observations = [
        CategoricalObservation(event_time=1, known_at=3, category="up"),
        CategoricalObservation(event_time=2, known_at=4, category="down"),
        CategoricalObservation(event_time=3, known_at=5, category="up"),
    ]
    feature = compute_shannon_entropy(observations, cutoff_timestamp=5, miller_madow=True)

    assert feature.execution_capability is False
    assert feature.counts == {"down": 1, "up": 2}
    assert feature.sample_count == 3
    assert feature.miller_madow_correction == pytest.approx(1 / 6)
    expected_entropy = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3) + 1 / 6
    assert feature.entropy_nats == pytest.approx(expected_entropy)


def test_entropy_rejects_empty_and_future_observations() -> None:
    with pytest.raises(StatePrimitiveError, match="at least one"):
        compute_shannon_entropy([])
    with pytest.raises(StatePrimitiveError, match="exceeds research cutoff"):
        compute_shannon_entropy(
            [CategoricalObservation(event_time=1, known_at=3, category="up")],
            cutoff_timestamp=2,
        )


def test_permutation_entropy_is_causal_deterministic_and_bounded() -> None:
    observations = [
        CausalObservation(event_time=index, known_at=index + 1, value=value)
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0), start=1)
    ]
    feature = compute_permutation_entropy(reversed(observations), order=3, cutoff_timestamp=6)

    assert feature.execution_capability is False
    assert feature.pattern_counts == {"0,1,2": 3}
    assert feature.entropy_nats == pytest.approx(0.0)
    assert feature.normalized_entropy == pytest.approx(0.0)
    assert feature.tie_window_count == 0
    assert feature.canonical_sha256() == compute_permutation_entropy(
        observations, order=3, cutoff_timestamp=6
    ).canonical_sha256()


def test_permutation_entropy_records_ties_and_rejects_noncausal_windows() -> None:
    observations = [
        CausalObservation(event_time=index, known_at=index, value=value)
        for index, value in enumerate((1.0, 1.0, 2.0, 2.0), start=1)
    ]
    feature = compute_permutation_entropy(observations, order=3)

    assert feature.tie_window_count == 2
    assert feature.sample_count == 2
    assert 0.0 <= feature.normalized_entropy <= 1.0
    with pytest.raises(StatePrimitiveError, match="exceeds research cutoff"):
        compute_permutation_entropy(observations, cutoff_timestamp=3)
    with pytest.raises(StatePrimitiveError, match="requires at least"):
        compute_permutation_entropy(observations[:2], order=3)
    with pytest.raises(StatePrimitiveError, match="between 2 and 8"):
        compute_permutation_entropy(observations, order=1)


def test_beta_bernoulli_updates_are_causal_and_calibrated() -> None:
    observations = [
        BernoulliObservation(event_time=10, known_at=12, outcome=1),
        BernoulliObservation(event_time=11, known_at=13, outcome=0),
        BernoulliObservation(event_time=12, known_at=14, outcome=1),
    ]
    posterior = update_beta_bernoulli(observations, prior=BetaPrior(2.0, 3.0), cutoff_timestamp=14)

    assert posterior.execution_capability is False
    assert posterior.final_alpha == pytest.approx(4.0)
    assert posterior.final_beta == pytest.approx(4.0)
    assert posterior.final_mean == pytest.approx(0.5)
    assert posterior.observations[-1].variance == pytest.approx(16 / (8 * 8 * 9))


def test_state_inputs_require_unique_causal_keys_and_finite_values() -> None:
    duplicate = [
        CausalObservation(event_time=1, known_at=2, sequence=0, value=1.0),
        CausalObservation(event_time=2, known_at=2, sequence=0, value=2.0),
    ]
    with pytest.raises(StatePrimitiveError, match="unique"):
        run_kalman_1d(duplicate)
    with pytest.raises(StatePrimitiveError, match="finite"):
        CausalObservation(event_time=1, known_at=2, value=float("nan"))
