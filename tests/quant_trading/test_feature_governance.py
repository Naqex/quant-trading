from __future__ import annotations

import json

import pytest

from quant_trading.distribution_drift import PSIConfig
from quant_trading.feature_drift import compute_feature_snapshot_psi
from quant_trading.feature_governance import FeatureGovernanceError, review_feature_drift
from quant_trading.feature_registry import (
    FeatureObservation,
    FeatureRegistry,
    FeatureSpec,
    build_feature_snapshot,
)
from quant_trading.model_governance import GovernancePolicy, ModelEvaluation

DATA_SHA = "a" * 64
FIXTURE_SHA = "d" * 64
CHAMPION_SHA = "b" * 64
CHALLENGER_SHA = "c" * 64


def _registry() -> FeatureRegistry:
    return FeatureRegistry(
        (
            FeatureSpec(
                feature_id="return_1d",
                feature_version="v1",
                instrument_id="BTCUSDT",
                timeframe_seconds=86_400,
                source_data_sha256=DATA_SHA,
                source_schema="candle-provenance-v1",
                rule_version="return-close-v1",
                minimum_availability_lag_seconds=60,
            ),
        ),
        registry_version="2026-09-r1",
    )


def _snapshot(values: tuple[float, ...], *, cutoff: int):
    rows = tuple(
        FeatureObservation(
            feature_id="return_1d",
            event_time_utc=index * 100,
            known_at_utc=index * 100 + 10,
            available_at_utc=index * 100 + 70,
            value=value,
            source_row_id=f"row-{index}",
        )
        for index, value in enumerate(values, 1)
    )
    return build_feature_snapshot(_registry(), rows, cutoff_timestamp_utc=cutoff)


def _feature_psi(*, drift: bool = False):
    reference = _snapshot((-1.0, -0.5, 0.5, 1.0), cutoff=500)
    current_values = (0.7, 0.8, 1.1, 1.2) if drift else (-1.0, -0.5, 0.5, 1.0)
    current = _snapshot(current_values, cutoff=900)
    return compute_feature_snapshot_psi(
        reference,
        current,
        feature_id="return_1d",
        config=PSIConfig(bin_edges=(-2.0, 0.0, 2.0), alert_threshold=0.2),
    )


def _evaluation(model_id: str, *, score: float, psi, model_sha256: str) -> ModelEvaluation:
    return ModelEvaluation(
        model_id=model_id,
        score=score,
        reference_score=1.0,
        calibration_error=0.02,
        sample_count=4,
        cutoff_timestamp=900,
        fixture_sha256=FIXTURE_SHA,
        model_sha256=model_sha256,
        holdout_pass=True,
        rollback_ref="rollback/champion-v1",
        psi=psi.psi_result,
    )


def _policy() -> GovernancePolicy:
    return GovernancePolicy(
        minimum_sample_count=4,
        minimum_score_improvement=0.05,
        maximum_calibration_error=0.1,
        maximum_decay=0.2,
    )


def test_feature_governance_preserves_psi_lineage_and_allows_qualified_review() -> None:
    feature_psi = _feature_psi()
    champion = _evaluation("champion-v1", score=1.0, psi=feature_psi, model_sha256=CHAMPION_SHA)
    candidate = _evaluation("challenger-v2", score=1.2, psi=feature_psi, model_sha256=CHALLENGER_SHA)

    first = review_feature_drift(feature_psi, candidate, champion, policy=_policy())
    second = review_feature_drift(feature_psi, candidate, champion, policy=_policy())

    assert feature_psi.psi_result.drift_flag is False
    assert first.decision.action == "promote_challenger"
    assert first.as_dict() == second.as_dict()
    assert first.canonical_sha256() == second.canonical_sha256()
    assert first.as_dict()["feature_psi"]["registry_sha256"] == feature_psi.registry_sha256
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False
    json.dumps(first.as_dict(), allow_nan=False)


def test_feature_drift_is_carried_into_review_and_blocks_promotion() -> None:
    feature_psi = _feature_psi(drift=True)
    champion = _evaluation("champion-v1", score=1.0, psi=feature_psi, model_sha256=CHAMPION_SHA)
    candidate = _evaluation("challenger-v2", score=1.2, psi=feature_psi, model_sha256=CHALLENGER_SHA)

    review = review_feature_drift(feature_psi, candidate, champion, policy=_policy())

    assert review.decision.action == "rollback_review"
    assert "champion_psi_drift_detected" in review.decision.reason_codes
    assert "candidate_psi_drift_detected" in review.decision.reason_codes


def test_feature_governance_rejects_unrelated_or_missing_psi_evidence() -> None:
    feature_psi = _feature_psi()
    unrelated_psi = _feature_psi(drift=True)
    champion = _evaluation("champion-v1", score=1.0, psi=feature_psi, model_sha256=CHAMPION_SHA)
    candidate = _evaluation("challenger-v2", score=1.2, psi=unrelated_psi, model_sha256=CHALLENGER_SHA)
    with pytest.raises(FeatureGovernanceError, match="does not match"):
        review_feature_drift(feature_psi, candidate, champion, policy=_policy())

    missing = ModelEvaluation(
        model_id="challenger-v2",
        score=1.2,
        reference_score=1.0,
        calibration_error=0.02,
        sample_count=4,
        cutoff_timestamp=900,
        fixture_sha256=FIXTURE_SHA,
        model_sha256=CHALLENGER_SHA,
        holdout_pass=True,
        rollback_ref="rollback/challenger-v2",
        psi=None,
    )
    with pytest.raises(FeatureGovernanceError, match="must include"):
        review_feature_drift(feature_psi, missing, champion, policy=_policy())


def test_feature_governance_review_rejects_cutoff_and_tampered_receipt() -> None:
    feature_psi = _feature_psi()
    champion = _evaluation("champion-v1", score=1.0, psi=feature_psi, model_sha256=CHAMPION_SHA)
    candidate = _evaluation("challenger-v2", score=1.2, psi=feature_psi, model_sha256=CHALLENGER_SHA)
    object.__setattr__(candidate, "cutoff_timestamp", 901)
    with pytest.raises(FeatureGovernanceError, match="cutoff"):
        review_feature_drift(feature_psi, candidate, champion, policy=_policy())

    candidate = _evaluation("challenger-v2", score=1.2, psi=feature_psi, model_sha256=CHALLENGER_SHA)
    review = review_feature_drift(feature_psi, candidate, champion, policy=_policy())
    object.__setattr__(review, "receipt_sha256", "e" * 64)
    with pytest.raises(FeatureGovernanceError, match="receipt_sha256"):
        review.__post_init__()

