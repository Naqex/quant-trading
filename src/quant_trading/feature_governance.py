"""Provenance-preserving governance review for feature OOD diagnostics.

``model_governance`` already knows how a PSI result affects a frozen
champion/challenger review.  This module keeps the richer
``FeaturePSIResult`` (registry and snapshot hashes) attached to that decision
so a governance receipt cannot lose the point-in-time feature lineage.  The
review remains a recommendation only: it never mutates a model, changes a
selective threshold, calls a provider, or grants execution authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .feature_drift import FeaturePSIResult
from .model_governance import (
    GovernanceDecision,
    GovernancePolicy,
    ModelEvaluation,
    ModelGovernanceError,
    evaluate_challenger,
)

FEATURE_GOVERNANCE_SCHEMA = "feature-governance-review-v1"


class FeatureGovernanceError(ValueError):
    """Raised when feature provenance and governance evidence do not align."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FeatureGovernanceError("feature governance receipt must contain finite JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_evidence(
    feature_psi: FeaturePSIResult,
    candidate: ModelEvaluation,
    champion: ModelEvaluation,
    decision: GovernanceDecision,
) -> None:
    if not isinstance(feature_psi, FeaturePSIResult):
        raise FeatureGovernanceError("feature_psi must be a FeaturePSIResult")
    if not isinstance(candidate, ModelEvaluation) or not isinstance(champion, ModelEvaluation):
        raise FeatureGovernanceError("candidate and champion must be ModelEvaluation values")
    if not isinstance(decision, GovernanceDecision):
        raise FeatureGovernanceError("decision must be a GovernanceDecision")
    if decision.candidate_model_id != candidate.model_id:
        raise FeatureGovernanceError("decision candidate does not match candidate evaluation")
    if decision.champion_model_id != champion.model_id:
        raise FeatureGovernanceError("decision champion does not match champion evaluation")
    if candidate.fixture_sha256 != champion.fixture_sha256:
        raise FeatureGovernanceError("candidate and champion fixtures must match")
    if decision.fixture_sha256 != candidate.fixture_sha256:
        raise FeatureGovernanceError("decision fixture does not match model evaluations")
    expected_cutoff = feature_psi.cutoff_timestamp_utc
    if candidate.cutoff_timestamp != expected_cutoff or champion.cutoff_timestamp != expected_cutoff:
        raise FeatureGovernanceError("model evaluation cutoff must match feature PSI cutoff")
    if decision.cutoff_timestamp != expected_cutoff:
        raise FeatureGovernanceError("governance decision cutoff must match feature PSI cutoff")
    expected_psi_hash = feature_psi.psi_result.canonical_sha256()
    for role, evaluation in (("candidate", candidate), ("champion", champion)):
        if evaluation.psi is None:
            raise FeatureGovernanceError(f"{role} evaluation must include the feature PSI result")
        if evaluation.psi.canonical_sha256() != expected_psi_hash:
            raise FeatureGovernanceError(f"{role} PSI does not match feature provenance receipt")
        if evaluation.sample_count != feature_psi.psi_result.current_sample_count:
            raise FeatureGovernanceError(f"{role} sample count does not match feature PSI current sample")


@dataclass(frozen=True)
class FeatureGovernanceReview:
    """Immutable, lineage-preserving champion/challenger recommendation."""

    feature_psi: FeaturePSIResult
    candidate_evaluation: ModelEvaluation
    champion_evaluation: ModelEvaluation
    decision: GovernanceDecision
    schema: str = FEATURE_GOVERNANCE_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != FEATURE_GOVERNANCE_SCHEMA:
            raise FeatureGovernanceError(f"unsupported feature governance schema: {self.schema!r}")
        _validate_evidence(
            self.feature_psi,
            self.candidate_evaluation,
            self.champion_evaluation,
            self.decision,
        )
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise FeatureGovernanceError(f"{name} must remain false for offline governance")
        expected = _canonical_sha256(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise FeatureGovernanceError("receipt_sha256 does not match canonical feature governance review")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "feature_psi": self.feature_psi.as_dict(),
            "candidate_evaluation": self.candidate_evaluation.as_dict(),
            "champion_evaluation": self.champion_evaluation.as_dict(),
            "decision": self.decision.as_dict(),
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["status"] = "PREP_ONLY"
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def canonical_sha256(self) -> str:
        return self.receipt_sha256 or _canonical_sha256(self._core_dict())


def review_feature_drift(
    feature_psi: FeaturePSIResult,
    candidate: ModelEvaluation,
    champion: ModelEvaluation,
    *,
    policy: GovernancePolicy | None = None,
) -> FeatureGovernanceReview:
    """Run existing governance gates while preserving feature OOD lineage.

    Both model evaluations must explicitly carry the exact ``PSIResult`` from
    ``feature_psi``.  Requiring identity by canonical hash prevents a caller
    from attaching a convenient but unrelated drift receipt to the review.
    """

    if not isinstance(feature_psi, FeaturePSIResult):
        raise FeatureGovernanceError("feature_psi must be a FeaturePSIResult")
    if not isinstance(candidate, ModelEvaluation) or not isinstance(champion, ModelEvaluation):
        raise FeatureGovernanceError("candidate and champion must be ModelEvaluation values")
    try:
        decision = evaluate_challenger(candidate, champion, policy=policy)
    except ModelGovernanceError as exc:
        raise FeatureGovernanceError(str(exc)) from exc
    return FeatureGovernanceReview(
        feature_psi=feature_psi,
        candidate_evaluation=candidate,
        champion_evaluation=champion,
        decision=decision,
    )


__all__ = [
    "FEATURE_GOVERNANCE_SCHEMA",
    "FeatureGovernanceError",
    "FeatureGovernanceReview",
    "review_feature_drift",
]
