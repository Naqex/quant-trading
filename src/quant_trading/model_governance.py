"""Offline champion/challenger governance for research candidates.

This module makes an Adaptive Markets style review explicit without creating a
self-updating model service.  A challenger can only be promoted when it beats
the frozen champion on the same fixture and cutoff while passing pre-registered
sample, holdout, calibration, decay and drift checks.  A failed incumbent can
produce a ``rollback_review`` recommendation, but this module never changes a
model, writes an order, calls a provider or grants execution authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from .distribution_drift import PSIResult

GOVERNANCE_SCHEMA = "model-governance-receipt-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EvaluationStatus = Literal["accepted", "inconclusive", "rejected"]
GovernanceAction = Literal[
    "promote_challenger", "retain_champion", "rollback_review", "inconclusive"
]


class ModelGovernanceError(ValueError):
    """Raised when a governance contract is malformed or non-finite."""


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelGovernanceError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ModelGovernanceError(f"{name} must be at most {maximum} characters")
    return normalized


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ModelGovernanceError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ModelGovernanceError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelGovernanceError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ModelGovernanceError(f"{name} must be finite")
    return parsed


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ModelGovernanceError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _canonical_hash(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ModelGovernanceError("governance receipt must contain finite JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GovernancePolicy:
    """Pre-registered gates for one deterministic champion/challenger review.

    ``score`` is a higher-is-better, pre-registered utility in the caller's
    units.  ``reference_score - score`` measures recent decay in those same
    units; it is deliberately not converted to a ratio because ratios become
    unstable near zero and for negative utilities.
    """

    minimum_sample_count: int = 30
    minimum_score_improvement: float = 0.0
    maximum_calibration_error: float = 0.10
    maximum_decay: float = 0.0
    require_holdout_pass: bool = True
    require_psi_preflight: bool = True
    require_rollback_ref_for_promotion: bool = True

    def __post_init__(self) -> None:
        minimum_sample_count = _strict_int(self.minimum_sample_count, "minimum_sample_count", minimum=1)
        minimum_score_improvement = _finite(self.minimum_score_improvement, "minimum_score_improvement")
        maximum_calibration_error = _finite(self.maximum_calibration_error, "maximum_calibration_error")
        maximum_decay = _finite(self.maximum_decay, "maximum_decay")
        if minimum_score_improvement < 0.0:
            raise ModelGovernanceError("minimum_score_improvement must be >= 0")
        if maximum_calibration_error < 0.0:
            raise ModelGovernanceError("maximum_calibration_error must be >= 0")
        if maximum_decay < 0.0:
            raise ModelGovernanceError("maximum_decay must be >= 0")
        for name in ("require_holdout_pass", "require_psi_preflight", "require_rollback_ref_for_promotion"):
            if type(getattr(self, name)) is not bool:
                raise ModelGovernanceError(f"{name} must be a boolean")
        object.__setattr__(self, "minimum_sample_count", minimum_sample_count)
        object.__setattr__(self, "minimum_score_improvement", minimum_score_improvement)
        object.__setattr__(self, "maximum_calibration_error", maximum_calibration_error)
        object.__setattr__(self, "maximum_decay", maximum_decay)

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_sample_count": self.minimum_sample_count,
            "minimum_score_improvement": self.minimum_score_improvement,
            "maximum_calibration_error": self.maximum_calibration_error,
            "maximum_decay": self.maximum_decay,
            "require_holdout_pass": self.require_holdout_pass,
            "require_psi_preflight": self.require_psi_preflight,
            "require_rollback_ref_for_promotion": self.require_rollback_ref_for_promotion,
        }


@dataclass(frozen=True)
class ModelEvaluation:
    """One frozen evaluation window for a champion or challenger model."""

    model_id: str
    score: float
    reference_score: float
    calibration_error: float
    sample_count: int
    cutoff_timestamp: int
    fixture_sha256: str
    model_sha256: str
    holdout_pass: bool
    rollback_ref: str | None = None
    psi: PSIResult | None = None
    status: EvaluationStatus = "accepted"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(self, "reference_score", _finite(self.reference_score, "reference_score"))
        calibration_error = _finite(self.calibration_error, "calibration_error")
        if calibration_error < 0.0:
            raise ModelGovernanceError("calibration_error must be >= 0")
        object.__setattr__(self, "calibration_error", calibration_error)
        object.__setattr__(self, "sample_count", _strict_int(self.sample_count, "sample_count", minimum=1))
        object.__setattr__(
            self, "cutoff_timestamp", _strict_int(self.cutoff_timestamp, "cutoff_timestamp")
        )
        _sha256(self.fixture_sha256, "fixture_sha256")
        _sha256(self.model_sha256, "model_sha256")
        if type(self.holdout_pass) is not bool:
            raise ModelGovernanceError("holdout_pass must be a boolean")
        if self.rollback_ref is not None:
            object.__setattr__(self, "rollback_ref", _text(self.rollback_ref, "rollback_ref", maximum=1024))
        if self.status not in {"accepted", "inconclusive", "rejected"}:
            raise ModelGovernanceError("status is unsupported")
        if self.psi is not None:
            if not isinstance(self.psi, PSIResult):
                raise ModelGovernanceError("psi must be a PSIResult")
            if self.psi.current_sample_count != self.sample_count:
                raise ModelGovernanceError("psi current sample count must match sample_count")
            if self.psi.cutoff_timestamp is not None and self.psi.cutoff_timestamp != self.cutoff_timestamp:
                raise ModelGovernanceError("psi cutoff_timestamp must match model evaluation cutoff")

    @property
    def decay(self) -> float:
        """Positive values mean the recent score is below its reference score."""

        return self.reference_score - self.score

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "score": self.score,
            "reference_score": self.reference_score,
            "decay": self.decay,
            "calibration_error": self.calibration_error,
            "sample_count": self.sample_count,
            "cutoff_timestamp": self.cutoff_timestamp,
            "fixture_sha256": self.fixture_sha256,
            "model_sha256": self.model_sha256,
            "holdout_pass": self.holdout_pass,
            "rollback_ref": self.rollback_ref,
            "psi": None if self.psi is None else {
                "schema": self.psi.schema,
                "current_sample_count": self.psi.current_sample_count,
                "cutoff_timestamp": self.psi.cutoff_timestamp,
                "psi": self.psi.psi,
                "drift_flag": self.psi.drift_flag,
                "receipt_sha256": self.psi.canonical_sha256(),
            },
            "status": self.status,
        }


@dataclass(frozen=True)
class GovernanceDecision:
    """Immutable review result; actions are recommendations, never mutations."""

    action: GovernanceAction
    reason_codes: tuple[str, ...]
    candidate_model_id: str
    champion_model_id: str
    score_delta: float
    candidate_healthy: bool | None
    champion_healthy: bool | None
    fixture_sha256: str
    cutoff_timestamp: int
    policy: GovernancePolicy
    schema: str = GOVERNANCE_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != GOVERNANCE_SCHEMA:
            raise ModelGovernanceError(f"unsupported governance schema: {self.schema!r}")
        if self.action not in {"promote_challenger", "retain_champion", "rollback_review", "inconclusive"}:
            raise ModelGovernanceError("action is unsupported")
        if not isinstance(self.reason_codes, tuple) or not self.reason_codes:
            raise ModelGovernanceError("reason_codes must be a non-empty tuple")
        for code in self.reason_codes:
            _text(code, "reason_code", maximum=128)
        object.__setattr__(self, "candidate_model_id", _text(self.candidate_model_id, "candidate_model_id"))
        object.__setattr__(self, "champion_model_id", _text(self.champion_model_id, "champion_model_id"))
        object.__setattr__(self, "score_delta", _finite(self.score_delta, "score_delta"))
        for name in ("candidate_healthy", "champion_healthy"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ModelGovernanceError(f"{name} must be a boolean or None")
        _sha256(self.fixture_sha256, "fixture_sha256")
        object.__setattr__(self, "cutoff_timestamp", _strict_int(self.cutoff_timestamp, "cutoff_timestamp"))
        if not isinstance(self.policy, GovernancePolicy):
            raise ModelGovernanceError("policy must be GovernancePolicy")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise ModelGovernanceError(f"{name} must remain false for offline governance")
        core = self._core_dict()
        expected = _canonical_hash(core)
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise ModelGovernanceError("receipt_sha256 does not match canonical decision")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "candidate_model_id": self.candidate_model_id,
            "champion_model_id": self.champion_model_id,
            "score_delta": self.score_delta,
            "candidate_healthy": self.candidate_healthy,
            "champion_healthy": self.champion_healthy,
            "fixture_sha256": self.fixture_sha256,
            "cutoff_timestamp": self.cutoff_timestamp,
            "policy": self.policy.as_dict(),
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["receipt_sha256"] = self.receipt_sha256
        return result


def _health(evaluation: ModelEvaluation, policy: GovernancePolicy, role: str) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if evaluation.status != "accepted":
        reasons.append(f"{role}_evidence_{evaluation.status}")
    if evaluation.sample_count < policy.minimum_sample_count:
        reasons.append(f"{role}_sample_below_minimum")
    if policy.require_holdout_pass and not evaluation.holdout_pass:
        reasons.append(f"{role}_holdout_failed")
    if evaluation.calibration_error > policy.maximum_calibration_error:
        reasons.append(f"{role}_calibration_error_exceeded")
    if evaluation.decay > policy.maximum_decay:
        reasons.append(f"{role}_decay_exceeded")
    if policy.require_psi_preflight and evaluation.psi is None:
        reasons.append(f"{role}_psi_missing")
    elif evaluation.psi is not None and evaluation.psi.drift_flag:
        reasons.append(f"{role}_psi_drift_detected")
    return not reasons, tuple(reasons)


def evaluate_challenger(
    candidate: ModelEvaluation,
    champion: ModelEvaluation,
    *,
    policy: GovernancePolicy | None = None,
) -> GovernanceDecision:
    """Evaluate one challenger against one frozen champion.

    The function is deterministic and side-effect free.  ``inconclusive`` is
    used when the comparison cannot be trusted; ``rollback_review`` means the
    incumbent failed health gates and has a recorded rollback target, but a
    human or outer controller must still perform any restore.
    """

    if not isinstance(candidate, ModelEvaluation) or not isinstance(champion, ModelEvaluation):
        raise ModelGovernanceError("candidate and champion must be ModelEvaluation values")
    if policy is None:
        policy = GovernancePolicy()
    if not isinstance(policy, GovernancePolicy):
        raise ModelGovernanceError("policy must be GovernancePolicy")

    score_delta = candidate.score - champion.score
    if candidate.fixture_sha256 != champion.fixture_sha256:
        return GovernanceDecision(
            action="inconclusive",
            reason_codes=("fixture_mismatch",),
            candidate_model_id=candidate.model_id,
            champion_model_id=champion.model_id,
            score_delta=score_delta,
            candidate_healthy=None,
            champion_healthy=None,
            fixture_sha256=candidate.fixture_sha256,
            cutoff_timestamp=candidate.cutoff_timestamp,
            policy=policy,
        )
    if candidate.cutoff_timestamp != champion.cutoff_timestamp:
        return GovernanceDecision(
            action="inconclusive",
            reason_codes=("cutoff_mismatch",),
            candidate_model_id=candidate.model_id,
            champion_model_id=champion.model_id,
            score_delta=score_delta,
            candidate_healthy=None,
            champion_healthy=None,
            fixture_sha256=candidate.fixture_sha256,
            cutoff_timestamp=candidate.cutoff_timestamp,
            policy=policy,
        )

    candidate_healthy, candidate_reasons = _health(candidate, policy, "candidate")
    champion_healthy, champion_reasons = _health(champion, policy, "champion")
    if (
        candidate_healthy
        and champion.status == "accepted"
        and score_delta > policy.minimum_score_improvement
    ):
        reason_codes = ("challenger_beats_champion",) + candidate_reasons
        if policy.require_rollback_ref_for_promotion and candidate.rollback_ref is None:
            candidate_healthy = False
            candidate_reasons = ("candidate_rollback_reference_missing",)
        else:
            return GovernanceDecision(
                action="promote_challenger",
                reason_codes=reason_codes,
                candidate_model_id=candidate.model_id,
                champion_model_id=champion.model_id,
                score_delta=score_delta,
                candidate_healthy=True,
                champion_healthy=champion_healthy,
                fixture_sha256=candidate.fixture_sha256,
                cutoff_timestamp=candidate.cutoff_timestamp,
                policy=policy,
            )

    if candidate_healthy and score_delta > policy.minimum_score_improvement and champion.status != "accepted":
        candidate_reasons = candidate_reasons + ("champion_evidence_not_accepted",)

    if champion_healthy:
        reason_codes = candidate_reasons or ("challenger_improvement_below_threshold",)
        return GovernanceDecision(
            action="retain_champion",
            reason_codes=reason_codes,
            candidate_model_id=candidate.model_id,
            champion_model_id=champion.model_id,
            score_delta=score_delta,
            candidate_healthy=candidate_healthy,
            champion_healthy=True,
            fixture_sha256=candidate.fixture_sha256,
            cutoff_timestamp=candidate.cutoff_timestamp,
            policy=policy,
        )

    if champion.status == "inconclusive":
        action: GovernanceAction = "inconclusive"
        reason_codes = champion_reasons + candidate_reasons
    elif champion.rollback_ref is not None:
        action = "rollback_review"
        reason_codes = champion_reasons + candidate_reasons + ("champion_rollback_reference_available",)
    else:
        action = "inconclusive"
        reason_codes = champion_reasons + candidate_reasons + ("champion_rollback_reference_missing",)
    return GovernanceDecision(
        action=action,
        reason_codes=reason_codes or ("no_promotion_gate_passed",),
        candidate_model_id=candidate.model_id,
        champion_model_id=champion.model_id,
        score_delta=score_delta,
        candidate_healthy=candidate_healthy,
        champion_healthy=False,
        fixture_sha256=candidate.fixture_sha256,
        cutoff_timestamp=candidate.cutoff_timestamp,
        policy=policy,
    )


__all__ = [
    "GOVERNANCE_SCHEMA",
    "EvaluationStatus",
    "GovernanceAction",
    "GovernanceDecision",
    "GovernancePolicy",
    "ModelEvaluation",
    "ModelGovernanceError",
    "evaluate_challenger",
]
