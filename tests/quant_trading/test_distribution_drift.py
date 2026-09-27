import json

import pytest

from quant_trading.distribution_drift import (
    DistributionDriftError,
    PSIConfig,
    compute_psi,
)


def _obs(event_time: int, value: float, *, known_at: int | None = None, sequence: int = 0):
    return {
        "event_time": event_time,
        "known_at": event_time if known_at is None else known_at,
        "value": value,
        "sequence": sequence,
    }


def _config(**overrides) -> PSIConfig:
    values = {"bin_edges": (-2.0, -1.0, 0.0, 1.0, 2.0), "epsilon": 1e-6, "alert_threshold": 0.2}
    values.update(overrides)
    return PSIConfig(**values)


def test_psi_is_deterministic_causal_and_reports_a_shift() -> None:
    reference = [_obs(index, value) for index, value in enumerate((-0.5, -0.2, 0.1, 0.4), 1)]
    current = [_obs(index + 10, value) for index, value in enumerate((1.2, 1.4, 1.6, 1.8), 1)]
    first = compute_psi(reference, current, config=_config(), cutoff_timestamp=20)
    second = compute_psi(
        list(reversed(reference)),
        list(reversed(current)),
        config=_config(),
        cutoff_timestamp=20,
    )

    assert first.as_dict() == second.as_dict()
    assert first.canonical_sha256() == second.canonical_sha256()
    assert first.psi > 0.2
    assert first.drift_flag is True
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False
    json.dumps(first.as_dict(), allow_nan=False)


def test_identical_samples_have_zero_or_near_zero_psi() -> None:
    observations = [_obs(index, value) for index, value in enumerate((-1.5, -0.2, 0.2, 1.5), 1)]
    result = compute_psi(observations, observations, config=_config())
    assert result.psi == pytest.approx(0.0, abs=1e-15)
    assert result.drift_flag is False
    assert sum(item.reference_count for item in result.bins) == 4
    assert sum(item.current_count for item in result.bins) == 4


def test_cutoff_is_fail_closed_and_out_of_support_is_rejected() -> None:
    with pytest.raises(DistributionDriftError, match="exceeds cutoff"):
        compute_psi(
            [_obs(1, 0.0)],
            [_obs(2, 0.0, known_at=5)],
            config=_config(),
            cutoff_timestamp=4,
        )
    with pytest.raises(DistributionDriftError, match="outside declared bin support"):
        compute_psi([_obs(1, 0.0)], [_obs(2, 2.1)], config=_config())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"bin_edges": (0.0,)}, "at least two"),
        ({"bin_edges": (0.0, 1.0, 1.0)}, "strictly increasing"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"alert_threshold": float("nan")}, "alert_threshold"),
    ],
)
def test_config_rejects_unsafe_parameters(kwargs, message: str) -> None:
    with pytest.raises(DistributionDriftError, match=message):
        _config(**kwargs)


def test_empty_duplicate_and_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(DistributionDriftError, match="requires at least one"):
        compute_psi([], [_obs(1, 0.0)], config=_config())
    with pytest.raises(DistributionDriftError, match="unique"):
        compute_psi(
            [_obs(1, 0.0, known_at=2), _obs(2, 0.5, known_at=2)],
            [_obs(3, 0.0)],
            config=_config(),
        )
    with pytest.raises(DistributionDriftError, match="finite"):
        _config(bin_edges=(0.0, float("inf")))
