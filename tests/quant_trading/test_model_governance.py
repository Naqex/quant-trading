from __future__ import annotations

import json

import pytest

from quant_trading.distribution_drift import PSIConfig, compute_psi
from quant_trading.model_governance import (
    GovernancePolicy,
    ModelEvaluation,
    ModelGovernanceError,
    evaluate_challenger,
)

FIXTURE_SHA = "a" * 64
CHAMPION_SHA = "b" * 64
CHALLENGER_SHA = "c" * 64
_MISSING = object()


def _psi(*, drift: bool = False):
    config = PSIConfig(bin_edges=(-2.0, 0.0, 2.0), alert_threshold=0.2)
    reference_values = (-1.0, -0.5, 0.5, 1.0)
    current_values = (1.0, 1.5, 1.8, 1.9) if drift else reference_values
    reference = [
        {"event_time": index, "known_at": index, "value": value}
        for index, value in enumerate(reference_values, 1)
    ]
    current = [
        {"event_time": index, "known_at": index, "value": value}
        for index, value in enumerate(current_values, 1)
    ]
    return compute_psi(reference, current, config=config, cutoff_timestamp=10)


def _evaluation(
    model_id: str,
    *,
    score: float,
    reference_score: float = 1.0,
    calibration_error: float = 0.02,
    status: str = "accepted",
    holdout_pass: bool = True,
    rollback_ref: str | None = "rollback/model-v1",
    psi=_MISSING,
    model_sha256: str = CHAMPION_SHA,
) -> ModelEvaluation:
    return ModelEvaluation(
        model_id=model_id,
        score=score,
        reference_score=reference_score,
        calibration_error=calibration_error,
        sample_count=4,
        cutoff_timestamp=10,
        fixture_sha256=FIXTURE_SHA,
        model_sha256=model_sha256,
        holdout_pass=holdout_pass,
        rollback_ref=rollback_ref,
        psi=_psi() if psi is _MISSING else psi,
        status=status,
    )


def _policy(**overrides) -> GovernancePolicy:
    values = {
        "minimum_sample_count": 4,
        "minimum_score_improvement": 0.05,
        "maximum_calibration_error": 0.1,
        "maximum_decay": 0.2,
    }
    values.update(overrides)
    return GovernancePolicy(**values)


def test_qualified_challenger_is_promoted_with_deterministic_receipt() -> None:
    champion = _evaluation("champion-v1", score=1.0, model_sha256=CHAMPION_SHA)
    candidate = _evaluation("challenger-v2", score=1.2, model_sha256=CHALLENGER_SHA)
    first = evaluate_challenger(candidate, champion, policy=_policy())
    second = evaluate_challenger(candidate, champion, policy=_policy())

    assert first.action == "promote_challenger"
    assert first.reason_codes == ("challenger_beats_champion",)
    assert first.candidate_healthy is True
    assert first.champion_healthy is True
    assert first.as_dict() == second.as_dict()
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False
    json.dumps(first.as_dict(), allow_nan=False)


def test_failed_challenger_retains_healthy_champion() -> None:
    champion = _evaluation("champion-v1", score=1.0, model_sha256=CHAMPION_SHA)
    candidate = _evaluation("challenger-v2", score=1.02, calibration_error=0.25, model_sha256=CHALLENGER_SHA)

    result = evaluate_challenger(candidate, champion, policy=_policy())

    assert result.action == "retain_champion"
    assert "candidate_calibration_error_exceeded" in result.reason_codes
    assert result.champion_healthy is True


def test_drifted_champion_returns_review_only_rollback_recommendation() -> None:
    champion = _evaluation("champion-v1", score=1.0, psi=_psi(drift=True), model_sha256=CHAMPION_SHA)
    candidate = _evaluation(
        "challenger-v2",
        score=0.9,
        calibration_error=0.2,
        model_sha256=CHALLENGER_SHA,
    )

    result = evaluate_challenger(candidate, champion, policy=_policy())

    assert result.action == "rollback_review"
    assert "champion_psi_drift_detected" in result.reason_codes
    assert "champion_rollback_reference_available" in result.reason_codes


def test_mismatched_fixture_or_cutoff_is_inconclusive() -> None:
    champion = _evaluation("champion-v1", score=1.0, model_sha256=CHAMPION_SHA)
    candidate = _evaluation("challenger-v2", score=1.2, model_sha256=CHALLENGER_SHA)
    object.__setattr__(candidate, "fixture_sha256", "d" * 64)
    result = evaluate_challenger(candidate, champion, policy=_policy())
    assert result.action == "inconclusive"
    assert result.reason_codes == ("fixture_mismatch",)

    object.__setattr__(candidate, "fixture_sha256", FIXTURE_SHA)
    object.__setattr__(candidate, "cutoff_timestamp", 11)
    result = evaluate_challenger(candidate, champion, policy=_policy())
    assert result.action == "inconclusive"
    assert result.reason_codes == ("cutoff_mismatch",)


def test_missing_psi_is_fail_closed_for_promotion() -> None:
    champion = _evaluation("champion-v1", score=1.0, model_sha256=CHAMPION_SHA)
    candidate = _evaluation("challenger-v2", score=1.2, model_sha256=CHALLENGER_SHA, psi=None)

    result = evaluate_challenger(candidate, champion, policy=_policy())

    assert result.action == "retain_champion"
    assert result.candidate_healthy is False
    assert "candidate_psi_missing" in result.reason_codes


def test_unaccepted_champion_cannot_be_silently_replaced() -> None:
    champion = _evaluation("champion-v1", score=1.0, model_sha256=CHAMPION_SHA, status="inconclusive")
    candidate = _evaluation("challenger-v2", score=1.2, model_sha256=CHALLENGER_SHA)

    result = evaluate_challenger(candidate, champion, policy=_policy())

    assert result.action == "inconclusive"
    assert "champion_evidence_inconclusive" in result.reason_codes
    assert "champion_evidence_not_accepted" in result.reason_codes


def test_contract_rejects_count_mismatch_and_unsafe_policy() -> None:
    with pytest.raises(ModelGovernanceError, match="minimum_sample_count"):
        GovernancePolicy(minimum_sample_count=0)
    with pytest.raises(ModelGovernanceError, match="calibration_error"):
        GovernancePolicy(maximum_calibration_error=-0.1)
    with pytest.raises(ModelGovernanceError, match="psi current sample count"):
        ModelEvaluation(
            model_id="x",
            score=1.0,
            reference_score=1.0,
            calibration_error=0.01,
            sample_count=3,
            cutoff_timestamp=10,
            fixture_sha256=FIXTURE_SHA,
            model_sha256=CHALLENGER_SHA,
            holdout_pass=True,
            psi=_psi(),
        )
