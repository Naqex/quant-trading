from __future__ import annotations

import math

import pytest

from quant_trading.discrete_optimizer import ResearchCutoffs
from quant_trading.risk_metrics import (
    RiskMetricsError,
    compute_risk_metrics,
    downside_deviation,
    expected_shortfall,
)

DATA_SHA = "a" * 64
CUTOFFS = ResearchCutoffs(train_end_utc=100, validation_end_utc=200, test_end_utc=300)


def test_expected_shortfall_uses_fractional_tail_mass_and_is_deterministic() -> None:
    result = compute_risk_metrics(
        returns=[-0.10, -0.04, 0.02, 0.08],
        timestamps_utc=[220, 230, 240, 250],
        alpha=0.50,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
        periods_per_year=1,
    )
    assert result.expected_shortfall_return == pytest.approx(-0.07)
    assert result.var_return == pytest.approx(-0.01)
    assert result.expected_shortfall_loss == pytest.approx(0.07)
    assert result.cvar_return == result.expected_shortfall_return
    assert result.cvar_loss == result.expected_shortfall_loss
    assert result.canonical_sha256() == compute_risk_metrics(
        returns=[-0.10, -0.04, 0.02, 0.08],
        timestamps_utc=[220, 230, 240, 250],
        alpha=0.50,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
        periods_per_year=1,
    ).canonical_sha256()


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, math.nan, math.inf, -math.inf])
def test_alpha_must_be_strictly_between_zero_and_one(alpha: float) -> None:
    with pytest.raises(RiskMetricsError, match="alpha must be (strictly between 0 and 1|finite)"):
        expected_shortfall([-0.1, 0.1], alpha=alpha)


def test_all_gains_are_finite_and_mark_sortino_undefined() -> None:
    result = compute_risk_metrics(
        returns=[0.01, 0.02, 0.03],
        timestamps_utc=[210, 220, 230],
        alpha=0.25,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
    )
    assert result.expected_shortfall_return == pytest.approx(0.01)
    assert result.expected_shortfall_loss == 0.0
    assert result.var_loss == 0.0
    assert result.downside_deviation == 0.0
    assert result.sortino_zero_target == 0.0
    assert result.sortino_defined is False
    assert all(
        math.isfinite(float(value))
        for value in result.as_dict().values()
        if isinstance(value, (int, float))
    )


def test_all_losses_have_finite_tail_and_sortino() -> None:
    result = compute_risk_metrics(
        returns=[-0.01, -0.02, -0.03],
        timestamps_utc=[210, 220, 230],
        alpha=0.50,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
    )
    assert result.expected_shortfall_return == pytest.approx(-0.02666666666666667)
    assert result.expected_shortfall_loss == pytest.approx(0.02666666666666667)
    assert result.var_loss <= result.expected_shortfall_loss
    assert result.downside_deviation > 0.0
    assert result.sortino_defined is True
    assert all(
        math.isfinite(float(value))
        for value in result.as_dict().values()
        if isinstance(value, (int, float))
    )


def test_short_sample_uses_one_observation_tail_without_nan() -> None:
    result = compute_risk_metrics(
        returns=[-0.125],
        timestamps_utc=[250],
        alpha=0.01,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
        periods_per_year=1,
    )
    assert result.sample_count == 1
    assert result.tail_mass == pytest.approx(0.01)
    assert result.tail_full_count == 0
    assert result.expected_shortfall_return == -0.125
    assert result.var_return == -0.125
    assert result.downside_deviation == 0.125
    assert result.sortino_zero_target == -1.0


def test_receipt_is_deny_only_and_test_window_is_causal() -> None:
    result = compute_risk_metrics(
        returns=[-0.02, 0.01],
        timestamps_utc=[220, 230],
        alpha=0.5,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
    )
    receipt = result.as_dict()
    assert receipt["status"] == "PREP_ONLY"
    assert receipt["scope"] == "crypto_spot_research"
    assert receipt["execution_capability"] is False
    assert receipt["provider_access"] is False
    assert receipt["broker_access"] is False
    assert receipt["cutoffs"] == CUTOFFS.as_dict()
    with pytest.raises(RiskMetricsError, match="timestamps must satisfy"):
        compute_risk_metrics(
            returns=[-0.02, 0.01],
            timestamps_utc=[190, 230],
            alpha=0.5,
            data_sha256=DATA_SHA,
            cutoffs=CUTOFFS,
        )


def test_malformed_input_and_non_finite_outputs_are_rejected() -> None:
    with pytest.raises(RiskMetricsError, match="same length"):
        compute_risk_metrics(
            returns=[0.01, 0.02],
            timestamps_utc=[220],
            alpha=0.5,
            data_sha256=DATA_SHA,
            cutoffs=CUTOFFS,
        )
    with pytest.raises(RiskMetricsError, match="finite values"):
        compute_risk_metrics(
            returns=[0.01, math.inf],
            timestamps_utc=[220, 230],
            alpha=0.5,
            data_sha256=DATA_SHA,
            cutoffs=CUTOFFS,
        )
    with pytest.raises(RiskMetricsError, match="finite"):
        downside_deviation([0.01, math.nan])
