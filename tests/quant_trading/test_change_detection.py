import pytest

from quant_trading.change_detection import (
    ChangeDetectionError,
    CUSUMConfig,
    run_cusum,
)


def _obs(event_time: int, value: float, *, known_at: int | None = None, sequence: int = 0):
    return {
        "event_time": event_time,
        "known_at": event_time if known_at is None else known_at,
        "value": value,
        "sequence": sequence,
    }


def test_constant_baseline_has_no_detection_and_stable_hash() -> None:
    observations = [_obs(index, 10.0) for index in range(4)]
    config = CUSUMConfig(baseline=10.0, threshold=1.0)

    first = run_cusum(observations, config=config)
    second = run_cusum(list(reversed(observations)), config=config)

    assert first.detections == ()
    assert [step.positive_stat for step in first.steps] == [0.0] * 4
    assert [step.negative_stat for step in first.steps] == [0.0] * 4
    assert first.as_dict() == second.as_dict()
    assert first.canonical_sha256() == second.canonical_sha256()


def test_positive_shift_is_flagged_forward_only_and_reset() -> None:
    observations = [_obs(1, 0.0), _obs(2, 2.0), _obs(3, 2.0), _obs(4, 0.0)]
    result = run_cusum(
        observations,
        config=CUSUMConfig(baseline=0.0, threshold=3.0),
    )

    assert [step.detected for step in result.steps] == [False, False, True, False]
    assert result.steps[2].direction == "positive"
    assert result.steps[2].positive_stat == pytest.approx(4.0)
    assert result.steps[3].positive_stat == pytest.approx(0.0)


def test_negative_shift_and_drift_are_explicit() -> None:
    observations = [_obs(1, -1.0), _obs(2, -1.0), _obs(3, -1.0)]
    result = run_cusum(
        observations,
        config=CUSUMConfig(baseline=0.0, threshold=2.0, drift=0.25),
    )

    assert len(result.detections) == 1
    assert result.detections[0].direction == "negative"
    assert result.detections[0].negative_stat == pytest.approx(-2.25)


def test_cutoff_rejects_future_rows_and_accepts_boundary() -> None:
    observations = [_obs(1, 0.0), _obs(2, 1.0, known_at=5)]
    with pytest.raises(ChangeDetectionError, match="future data"):
        run_cusum(observations, config=CUSUMConfig(baseline=0.0, threshold=1.0), cutoff_timestamp=4)

    accepted = run_cusum(
        observations,
        config=CUSUMConfig(baseline=0.0, threshold=1.0),
        cutoff_timestamp=5,
    )
    assert len(accepted.steps) == 2
    assert accepted.cutoff_timestamp == 5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_input_is_rejected(value: float) -> None:
    with pytest.raises(ChangeDetectionError, match="finite"):
        run_cusum([_obs(1, value)], config=CUSUMConfig(baseline=0.0, threshold=1.0))


def test_known_time_order_and_duplicate_keys_are_deterministic() -> None:
    result = run_cusum(
        [_obs(3, 3.0, known_at=3), _obs(1, 1.0, known_at=1), _obs(2, 2.0, known_at=2)],
        config=CUSUMConfig(baseline=0.0, threshold=100.0),
    )
    assert [step.known_at for step in result.steps] == [1, 2, 3]
    assert [step.event_time for step in result.steps] == [1, 2, 3]

    with pytest.raises(ChangeDetectionError, match="unique"):
        run_cusum(
            [_obs(1, 1.0, known_at=1), _obs(1, 2.0, known_at=1)],
            config=CUSUMConfig(baseline=0.0, threshold=100.0),
        )


def test_empty_input_is_valid_and_deny_only() -> None:
    result = run_cusum([], config=CUSUMConfig(baseline=0.0, threshold=1.0))

    assert result.steps == ()
    assert result.detections == ()
    assert result.execution_capability is False
    assert result.provider_access is False
    assert result.broker_access is False


def test_non_iterable_input_is_rejected() -> None:
    with pytest.raises(ChangeDetectionError, match="iterable"):
        run_cusum(None, config=CUSUMConfig(baseline=0.0, threshold=1.0))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"drift": -1.0}, "drift"),
        ({"reset_on_detection": 1}, "reset_on_detection"),
    ],
)
def test_config_rejects_unsafe_parameters(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {"baseline": 0.0, "threshold": 1.0}
    values.update(kwargs)
    with pytest.raises(ChangeDetectionError, match=message):
        CUSUMConfig(**values)


def test_no_future_event_can_be_smuggled_through_known_at() -> None:
    with pytest.raises(ChangeDetectionError, match="known_at cannot precede"):
        run_cusum(
            [_obs(10, 1.0, known_at=9)],
            config=CUSUMConfig(baseline=0.0, threshold=1.0),
        )

    # A valid late-available row is still causal and is processed only at its
    # availability time; event time alone cannot make it appear earlier.
    result = run_cusum(
        [_obs(1, 0.0, known_at=1), _obs(2, 2.0, known_at=5)],
        config=CUSUMConfig(baseline=0.0, threshold=1.0),
        cutoff_timestamp=5,
    )
    assert [step.known_at for step in result.steps] == [1, 5]
    assert result.detections[0].known_at == 5
