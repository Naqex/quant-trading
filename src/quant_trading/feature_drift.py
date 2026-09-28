"""Point-in-time PSI diagnostics over feature provenance snapshots.

``distribution_drift`` already implements the numerical PSI calculation for
causal observations.  This module only supplies the missing provenance
boundary: it selects one feature from two validated ``FeatureSnapshot``
artifacts, preserves their registry and snapshot hashes, and refuses suspect
rows or incompatible registries.  It does not infer bins, label regimes,
make a forecast, call a provider, or grant execution authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .distribution_drift import DistributionDriftError, PSIConfig, PSIResult, compute_psi
from .feature_registry import FeatureObservation, FeatureRegistryError, FeatureSnapshot
from .state_primitives import CausalObservation

FEATURE_PSI_SCHEMA = "feature-population-stability-index-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FeatureDriftError(ValueError):
    """Raised when a feature-provenance PSI comparison is unsafe."""


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeatureDriftError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise FeatureDriftError(f"{name} must be at most {maximum} characters")
    return normalized


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FeatureDriftError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FeatureDriftError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FeatureDriftError("feature PSI receipt must contain finite JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_rows(snapshot: FeatureSnapshot, feature_id: str, name: str) -> tuple[FeatureObservation, ...]:
    rows = tuple(item for item in snapshot.observations if item.feature_id == feature_id)
    if not rows:
        raise FeatureDriftError(f"{name} snapshot has no observations for feature {feature_id!r}")
    suspect = [item.source_row_id for item in rows if item.quality_status != "accepted"]
    if suspect:
        raise FeatureDriftError(
            f"{name} snapshot contains non-accepted feature rows; review before PSI ({suspect[0]!r})"
        )
    return tuple(
        sorted(rows, key=lambda item: (item.available_at_utc, item.event_time_utc, item.source_row_id))
    )


def _as_causal(rows: Iterable[FeatureObservation]) -> tuple[CausalObservation, ...]:
    """Map feature availability to known time with a stable local sequence."""

    return tuple(
        CausalObservation(
            event_time=item.event_time_utc,
            known_at=item.available_at_utc,
            value=item.value,
            sequence=sequence,
        )
        for sequence, item in enumerate(rows)
    )


@dataclass(frozen=True)
class FeaturePSIResult:
    """Immutable OOD receipt anchored to two feature snapshots."""

    feature_id: str
    registry_sha256: str
    reference_snapshot_sha256: str
    current_snapshot_sha256: str
    cutoff_timestamp_utc: int
    psi_result: PSIResult
    schema: str = FEATURE_PSI_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False
    receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != FEATURE_PSI_SCHEMA:
            raise FeatureDriftError(f"unsupported feature PSI schema: {self.schema!r}")
        object.__setattr__(self, "feature_id", _text(self.feature_id, "feature_id"))
        for name in ("registry_sha256", "reference_snapshot_sha256", "current_snapshot_sha256"):
            _sha256(getattr(self, name), name)
        cutoff = _strict_int(self.cutoff_timestamp_utc, "cutoff_timestamp_utc")
        object.__setattr__(self, "cutoff_timestamp_utc", cutoff)
        if not isinstance(self.psi_result, PSIResult):
            raise FeatureDriftError("psi_result must be a PSIResult")
        if self.psi_result.cutoff_timestamp != cutoff:
            raise FeatureDriftError("PSI cutoff must match feature PSI cutoff")
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise FeatureDriftError(f"{name} must remain false for offline feature PSI")
        expected = _canonical_sha256(self._core_dict())
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise FeatureDriftError("receipt_sha256 does not match canonical feature PSI receipt")
        object.__setattr__(self, "receipt_sha256", expected)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "feature_id": self.feature_id,
            "registry_sha256": self.registry_sha256,
            "reference_snapshot_sha256": self.reference_snapshot_sha256,
            "current_snapshot_sha256": self.current_snapshot_sha256,
            "cutoff_timestamp_utc": self.cutoff_timestamp_utc,
            "psi_result": self.psi_result.as_dict(),
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


def compute_feature_snapshot_psi(
    reference_snapshot: FeatureSnapshot,
    current_snapshot: FeatureSnapshot,
    *,
    feature_id: str,
    config: PSIConfig,
    cutoff_timestamp_utc: int | None = None,
) -> FeaturePSIResult:
    """Compare one feature across two point-in-time snapshots.

    The snapshots must use the same registry definition.  The comparison
    cutoff defaults to the later snapshot cutoff so an older reference window
    can be compared with a newer current window; an explicit cutoff may only
    be earlier than that later boundary.  Rows after the effective cutoff are
    rejected by the underlying PSI contract rather than silently dropped.
    """

    if not isinstance(reference_snapshot, FeatureSnapshot):
        raise FeatureDriftError("reference_snapshot must be a FeatureSnapshot")
    if not isinstance(current_snapshot, FeatureSnapshot):
        raise FeatureDriftError("current_snapshot must be a FeatureSnapshot")
    if not isinstance(config, PSIConfig):
        raise FeatureDriftError("config must be a PSIConfig")
    feature_id = _text(feature_id, "feature_id")
    registry_sha256 = reference_snapshot.registry.canonical_sha256()
    if current_snapshot.registry.canonical_sha256() != registry_sha256:
        raise FeatureDriftError("reference and current snapshots must use the same feature registry")
    if feature_id not in reference_snapshot.registry.by_id:
        raise FeatureDriftError(f"feature registry does not define {feature_id!r}")
    reference_rows = _select_rows(reference_snapshot, feature_id, "reference")
    current_rows = _select_rows(current_snapshot, feature_id, "current")
    latest_boundary = max(
        reference_snapshot.cutoff_timestamp_utc, current_snapshot.cutoff_timestamp_utc
    )
    if cutoff_timestamp_utc is None:
        cutoff = latest_boundary
    else:
        cutoff = _strict_int(cutoff_timestamp_utc, "cutoff_timestamp_utc")
        if cutoff > latest_boundary:
            raise FeatureDriftError("cutoff_timestamp_utc exceeds both snapshot boundaries")
    try:
        psi_result = compute_psi(
            _as_causal(reference_rows),
            _as_causal(current_rows),
            config=config,
            cutoff_timestamp=cutoff,
        )
    except (DistributionDriftError, FeatureRegistryError, TypeError) as exc:
        raise FeatureDriftError(str(exc)) from exc
    return FeaturePSIResult(
        feature_id=feature_id,
        registry_sha256=registry_sha256,
        reference_snapshot_sha256=reference_snapshot.canonical_sha256(),
        current_snapshot_sha256=current_snapshot.canonical_sha256(),
        cutoff_timestamp_utc=cutoff,
        psi_result=psi_result,
    )


__all__ = [
    "FEATURE_PSI_SCHEMA",
    "FeatureDriftError",
    "FeaturePSIResult",
    "compute_feature_snapshot_psi",
]
