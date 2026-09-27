import math

import pytest

from quant_trading.ou_process import (
    OUConfig,
    OUObservation,
    OUProcessError,
    filter_ou_process,
    fit_ou_process,
)


def _synthetic_observations(kind: str = "spread") -> list[OUObservation]:
    values: list[float] = []
    value = 0.7
    event_time = 0
    event_times: list[int] = []
    for index in range(72):
        dt = 1 + (index % 4)
        event_time += dt
        event_times.append(event_time)
        mean_level = 0.4
        phi = math.exp(-0.3 * dt)
        innovation = (0.015, -0.008, 0.004)[index % 3]
        value = mean_level + phi * (value - mean_level) + innovation
        values.append(value)
    return [
        OUObservation(
            event_time=event_times[index],
            known_at=event_times[index] + 1,
            value=value,
            series_kind=kind,
        )
        for index, value in enumerate(values)
    ]


def test_ou_fit_handles_irregular_dt_and_is_deterministic() -> None:
    observations = _synthetic_observations()
    first = fit_ou_process(observations, cutoff_timestamp=observations[-1].known_at)
    second = fit_ou_process(reversed(observations), cutoff_timestamp=observations[-1].known_at)

    assert first.canonical_sha256() == second.canonical_sha256()
    assert first.series_kind == "spread"
    assert first.min_dt == 1.0
    assert first.max_dt == 4.0
    assert 0.05 < first.kappa < 2.0
    assert first.stationary_variance > 0
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False


def test_ou_filter_is_forward_only_and_records_provenance() -> None:
    observations = _synthetic_observations("residual")
    fit = fit_ou_process(observations, cutoff_timestamp=observations[-1].known_at)
    result = filter_ou_process(observations, fit=fit, cutoff_timestamp=observations[-1].known_at)

    assert len(result.estimates) == len(observations)
    assert result.estimates[0].dt == 0.0
    assert result.estimates[-1].dt == 4.0
    assert all(math.isfinite(item.standardized_innovation) for item in result.estimates)
    assert result.execution_capability is False
    assert result.provider_access is False
    assert result.broker_access is False
    assert result.canonical_sha256() == filter_ou_process(
        reversed(observations), fit=fit, cutoff_timestamp=observations[-1].known_at
    ).canonical_sha256()


def test_ou_rejects_raw_price_role_and_future_cutoff() -> None:
    with pytest.raises(OUProcessError, match="spread.*residual"):
        OUObservation(event_time=1, known_at=1, value=100.0, series_kind="price")
    with pytest.raises(OUProcessError, match="exceeds research cutoff"):
        fit_ou_process(_synthetic_observations()[:20], cutoff_timestamp=10)


def test_ou_stationarity_guard_rejects_random_walk_boundary() -> None:
    observations = [
        OUObservation(event_time=index + 1, known_at=index + 1, value=float(index), series_kind="residual")
        for index in range(32)
    ]
    with pytest.raises(OUProcessError, match="boundary|stationarity"):
        fit_ou_process(observations)


def test_ou_uses_stable_expm1_for_large_irregular_gaps() -> None:
    observations = [
        OUObservation(event_time=1, known_at=1, value=1.0, series_kind="spread"),
        OUObservation(event_time=10**6, known_at=10**6, value=-0.25, series_kind="spread"),
        OUObservation(event_time=2 * 10**6, known_at=2 * 10**6, value=0.15, series_kind="spread"),
        OUObservation(event_time=3 * 10**6, known_at=3 * 10**6, value=-0.05, series_kind="spread"),
        OUObservation(event_time=4 * 10**6, known_at=4 * 10**6, value=0.02, series_kind="spread"),
        OUObservation(event_time=5 * 10**6, known_at=5 * 10**6, value=-0.01, series_kind="spread"),
        OUObservation(event_time=6 * 10**6, known_at=6 * 10**6, value=0.005, series_kind="spread"),
        OUObservation(event_time=7 * 10**6, known_at=7 * 10**6, value=-0.002, series_kind="spread"),
        OUObservation(event_time=8 * 10**6, known_at=8 * 10**6, value=0.001, series_kind="spread"),
        OUObservation(event_time=9 * 10**6, known_at=9 * 10**6, value=-0.0005, series_kind="spread"),
        OUObservation(event_time=10**6 * 10, known_at=10**6 * 10, value=0.0002, series_kind="spread"),
        OUObservation(event_time=11 * 10**6, known_at=11 * 10**6, value=-0.0001, series_kind="spread"),
    ]
    config = OUConfig(min_kappa=1e-8, max_kappa=10.0)
    fit = fit_ou_process(observations, config=config)
    result = filter_ou_process(observations, fit=fit)

    assert math.isfinite(fit.kappa)
    assert all(math.isfinite(item.conditional_variance) for item in result.estimates)
