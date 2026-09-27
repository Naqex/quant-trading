"""Point-in-time feature provenance and temporal registry contracts.

The candle provenance helper identifies an input dataset and chart contracts
pin a feature-rule version, but neither records when each derived feature row
became available.  This module closes that boundary for offline research:
feature definitions form an acyclic registry, observations carry explicit
event and availability times, and a snapshot rejects rows that were not known
at its cutoff.  It is a receipt/validation primitive only; it never fetches
data, calls a model or grants execution capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

FEATURE_SPEC_SCHEMA = "feature-spec-v1"
FEATURE_REGISTRY_SCHEMA = "feature-registry-v1"
FEATURE_SNAPSHOT_SCHEMA = "feature-snapshot-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FeatureRegistryError(ValueError):
    """Raised when feature lineage or point-in-time availability is unsafe."""


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeatureRegistryError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise FeatureRegistryError(f"{name} must be at most {maximum} characters")
    return normalized


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FeatureRegistryError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise FeatureRegistryError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureRegistryError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise FeatureRegistryError(f"{name} must be finite")
    return parsed


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FeatureRegistryError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _json_value(value: Any, name: str) -> Any:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FeatureRegistryError(f"{name} must be finite JSON data") from exc
    return value


def _canonical_hash(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FeatureRegistryError("feature receipt must contain finite JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureSpec:
    """Definition and lineage for one numeric, point-in-time feature."""

    feature_id: str
    feature_version: str
    instrument_id: str
    timeframe_seconds: int
    source_data_sha256: str
    source_schema: str
    rule_version: str
    source_ref_ids: tuple[str, ...] = ()
    minimum_availability_lag_seconds: int = 0
    dependencies: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    execution_capability: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_id", _text(self.feature_id, "feature_id"))
        object.__setattr__(self, "feature_version", _text(self.feature_version, "feature_version"))
        object.__setattr__(self, "instrument_id", _text(self.instrument_id, "instrument_id"))
        object.__setattr__(
            self,
            "timeframe_seconds",
            _strict_int(self.timeframe_seconds, "timeframe_seconds", minimum=1),
        )
        object.__setattr__(self, "source_data_sha256", _sha256(self.source_data_sha256, "source_data_sha256"))
        object.__setattr__(self, "source_schema", _text(self.source_schema, "source_schema"))
        object.__setattr__(self, "rule_version", _text(self.rule_version, "rule_version"))
        if not isinstance(self.source_ref_ids, (tuple, list)):
            raise FeatureRegistryError("source_ref_ids must be a tuple or list")
        source_ref_ids = tuple(_text(value, "source_ref_id", maximum=512) for value in self.source_ref_ids)
        if len(set(source_ref_ids)) != len(source_ref_ids):
            raise FeatureRegistryError("source_ref_ids must be unique")
        object.__setattr__(self, "source_ref_ids", tuple(sorted(source_ref_ids)))
        object.__setattr__(
            self,
            "minimum_availability_lag_seconds",
            _strict_int(
                self.minimum_availability_lag_seconds,
                "minimum_availability_lag_seconds",
                minimum=0,
            ),
        )
        if not isinstance(self.dependencies, (tuple, list)):
            raise FeatureRegistryError("dependencies must be a tuple or list")
        dependencies = tuple(_text(value, "dependency", maximum=256) for value in self.dependencies)
        if len(set(dependencies)) != len(dependencies):
            raise FeatureRegistryError("dependencies must be unique")
        if self.feature_id in dependencies:
            raise FeatureRegistryError("feature cannot depend on itself")
        object.__setattr__(self, "dependencies", tuple(sorted(dependencies)))
        if not isinstance(self.parameters, Mapping):
            raise FeatureRegistryError("parameters must be a mapping")
        parameters = dict(self.parameters)
        _json_value(parameters, "parameters")
        object.__setattr__(self, "parameters", parameters)
        if self.execution_capability is not False:
            raise FeatureRegistryError("feature specs cannot grant execution capability")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        if not isinstance(payload, Mapping):
            raise FeatureRegistryError("feature spec must be a mapping")
        if payload.get("schema", FEATURE_SPEC_SCHEMA) != FEATURE_SPEC_SCHEMA:
            raise FeatureRegistryError("unsupported feature spec schema")
        try:
            return cls(
                feature_id=payload["feature_id"],
                feature_version=payload["feature_version"],
                instrument_id=payload["instrument_id"],
                timeframe_seconds=payload["timeframe_seconds"],
                source_data_sha256=payload["source_data_sha256"],
                source_schema=payload["source_schema"],
                rule_version=payload["rule_version"],
                source_ref_ids=tuple(payload.get("source_ref_ids", ())),
                minimum_availability_lag_seconds=payload.get("minimum_availability_lag_seconds", 0),
                dependencies=tuple(payload.get("dependencies", ())),
                parameters=payload.get("parameters", {}),
                execution_capability=payload.get("execution_capability", False),
            )
        except KeyError as exc:
            raise FeatureRegistryError(f"feature spec is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FEATURE_SPEC_SCHEMA,
            "feature_id": self.feature_id,
            "feature_version": self.feature_version,
            "instrument_id": self.instrument_id,
            "timeframe_seconds": self.timeframe_seconds,
            "source_data_sha256": self.source_data_sha256,
            "source_schema": self.source_schema,
            "rule_version": self.rule_version,
            "source_ref_ids": list(self.source_ref_ids),
            "minimum_availability_lag_seconds": self.minimum_availability_lag_seconds,
            "dependencies": list(self.dependencies),
            "parameters": dict(self.parameters),
            "execution_capability": self.execution_capability,
        }


@dataclass(frozen=True)
class FeatureRegistry:
    """Versioned feature DAG with deterministic lineage hash."""

    specs: tuple[FeatureSpec, ...]
    registry_version: str = "r1"
    execution_capability: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.specs, (tuple, list)) or not self.specs:
            raise FeatureRegistryError("feature registry requires at least one spec")
        if any(not isinstance(spec, FeatureSpec) for spec in self.specs):
            raise FeatureRegistryError("feature registry specs must be FeatureSpec values")
        specs = tuple(sorted(self.specs, key=lambda spec: spec.feature_id))
        ids = [spec.feature_id for spec in specs]
        if len(set(ids)) != len(ids):
            raise FeatureRegistryError("feature IDs must be unique")
        by_id = {spec.feature_id: spec for spec in specs}
        for spec in specs:
            missing = [dependency for dependency in spec.dependencies if dependency not in by_id]
            if missing:
                raise FeatureRegistryError(
                    f"feature {spec.feature_id!r} references unknown dependency {missing[0]!r}"
                )

        state: dict[str, int] = {}

        def visit(feature_id: str) -> None:
            status = state.get(feature_id, 0)
            if status == 1:
                raise FeatureRegistryError("feature dependencies must be acyclic")
            if status == 2:
                return
            state[feature_id] = 1
            for dependency in by_id[feature_id].dependencies:
                visit(dependency)
            state[feature_id] = 2

        for feature_id in ids:
            visit(feature_id)
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "registry_version", _text(self.registry_version, "registry_version"))
        if self.execution_capability is not False:
            raise FeatureRegistryError("feature registries cannot grant execution capability")

    @property
    def by_id(self) -> dict[str, FeatureSpec]:
        return {spec.feature_id: spec for spec in self.specs}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeatureRegistry:
        if not isinstance(payload, Mapping):
            raise FeatureRegistryError("feature registry must be a mapping")
        if payload.get("schema", FEATURE_REGISTRY_SCHEMA) != FEATURE_REGISTRY_SCHEMA:
            raise FeatureRegistryError("unsupported feature registry schema")
        raw_specs = payload.get("specs")
        if not isinstance(raw_specs, list):
            raise FeatureRegistryError("feature registry specs must be a list")
        return cls(
            specs=tuple(FeatureSpec.from_mapping(item) for item in raw_specs),
            registry_version=payload.get("registry_version", "r1"),
            execution_capability=payload.get("execution_capability", False),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FEATURE_REGISTRY_SCHEMA,
            "registry_version": self.registry_version,
            "execution_capability": self.execution_capability,
            "specs": [spec.as_dict() for spec in self.specs],
        }

    def canonical_sha256(self) -> str:
        return _canonical_hash(self.as_dict())


@dataclass(frozen=True)
class FeatureObservation:
    """One scalar feature value with explicit event and availability times."""

    feature_id: str
    event_time_utc: int
    known_at_utc: int
    available_at_utc: int
    value: float
    source_row_id: str
    revision: int = 0
    supersedes_source_row_id: str | None = None
    quality_status: str = "accepted"
    valid_until_utc: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_id", _text(self.feature_id, "feature_id"))
        event_time = _strict_int(self.event_time_utc, "event_time_utc")
        known_at = _strict_int(self.known_at_utc, "known_at_utc")
        available_at = _strict_int(self.available_at_utc, "available_at_utc")
        if known_at < event_time:
            raise FeatureRegistryError("known_at_utc cannot precede event_time_utc")
        if available_at < known_at:
            raise FeatureRegistryError("available_at_utc cannot precede known_at_utc")
        object.__setattr__(self, "event_time_utc", event_time)
        object.__setattr__(self, "known_at_utc", known_at)
        object.__setattr__(self, "available_at_utc", available_at)
        object.__setattr__(self, "value", _finite(self.value, "value"))
        object.__setattr__(self, "source_row_id", _text(self.source_row_id, "source_row_id", maximum=512))
        revision = _strict_int(self.revision, "revision")
        object.__setattr__(self, "revision", revision)
        if self.supersedes_source_row_id is not None:
            supersedes = _text(self.supersedes_source_row_id, "supersedes_source_row_id", maximum=512)
            if supersedes == self.source_row_id:
                raise FeatureRegistryError("observation cannot supersede itself")
            if revision == 0:
                raise FeatureRegistryError("revision zero cannot supersede another observation")
            object.__setattr__(self, "supersedes_source_row_id", supersedes)
        elif revision > 0:
            raise FeatureRegistryError("revised observations must identify the superseded source row")
        if self.quality_status not in {"accepted", "suspect", "invalidated"}:
            raise FeatureRegistryError("unsupported quality_status")
        if self.valid_until_utc is not None:
            valid_until = _strict_int(self.valid_until_utc, "valid_until_utc")
            if valid_until < available_at:
                raise FeatureRegistryError("valid_until_utc cannot precede available_at_utc")
            object.__setattr__(self, "valid_until_utc", valid_until)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeatureObservation:
        if not isinstance(payload, Mapping):
            raise FeatureRegistryError("feature observation must be a mapping")
        try:
            return cls(
                feature_id=payload["feature_id"],
                event_time_utc=payload["event_time_utc"],
                known_at_utc=payload["known_at_utc"],
                available_at_utc=payload["available_at_utc"],
                value=payload["value"],
                source_row_id=payload["source_row_id"],
                revision=payload.get("revision", 0),
                supersedes_source_row_id=payload.get("supersedes_source_row_id"),
                quality_status=payload.get("quality_status", "accepted"),
                valid_until_utc=payload.get("valid_until_utc"),
            )
        except KeyError as exc:
            raise FeatureRegistryError(f"feature observation is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "event_time_utc": self.event_time_utc,
            "known_at_utc": self.known_at_utc,
            "available_at_utc": self.available_at_utc,
            "value": self.value,
            "source_row_id": self.source_row_id,
            "revision": self.revision,
            "supersedes_source_row_id": self.supersedes_source_row_id,
            "quality_status": self.quality_status,
            "valid_until_utc": self.valid_until_utc,
        }


@dataclass(frozen=True)
class FeatureSnapshot:
    """A reproducible, cutoff-bounded view of registry observations."""

    registry: FeatureRegistry
    cutoff_timestamp_utc: int
    observations: tuple[FeatureObservation, ...]
    schema: str = FEATURE_SNAPSHOT_SCHEMA
    execution_capability: bool = False
    provider_access: bool = False
    broker_access: bool = False

    def __post_init__(self) -> None:
        if self.schema != FEATURE_SNAPSHOT_SCHEMA:
            raise FeatureRegistryError(f"unsupported feature snapshot schema: {self.schema!r}")
        if not isinstance(self.registry, FeatureRegistry):
            raise FeatureRegistryError("registry must be a FeatureRegistry")
        cutoff = _strict_int(self.cutoff_timestamp_utc, "cutoff_timestamp_utc")
        if not isinstance(self.observations, (tuple, list)):
            raise FeatureRegistryError("observations must be a tuple or list")
        by_id = self.registry.by_id
        normalized = tuple(self.observations)
        if any(not isinstance(item, FeatureObservation) for item in normalized):
            raise FeatureRegistryError("observations must be FeatureObservation values")
        keys: set[tuple[str, int]] = set()
        for observation in normalized:
            spec = by_id.get(observation.feature_id)
            if spec is None:
                raise FeatureRegistryError(
                    f"observation references unknown feature {observation.feature_id!r}"
                )
            if observation.available_at_utc > cutoff:
                raise FeatureRegistryError("observation exceeds snapshot cutoff")
            if observation.quality_status == "invalidated":
                raise FeatureRegistryError("invalidated observations cannot enter a feature snapshot")
            if observation.valid_until_utc is not None and cutoff > observation.valid_until_utc:
                raise FeatureRegistryError("observation is stale at snapshot cutoff")
            lag = observation.available_at_utc - observation.known_at_utc
            if lag < spec.minimum_availability_lag_seconds:
                raise FeatureRegistryError(
                    f"observation for {observation.feature_id!r} violates minimum availability lag"
                )
            key = (observation.feature_id, observation.event_time_utc)
            if key in keys:
                raise FeatureRegistryError("duplicate feature observation event")
            keys.add(key)
        normalized = tuple(
            sorted(
                normalized,
                key=lambda item: (
                    item.feature_id,
                    item.event_time_utc,
                    item.available_at_utc,
                    item.source_row_id,
                ),
            )
        )
        object.__setattr__(self, "cutoff_timestamp_utc", cutoff)
        object.__setattr__(self, "observations", normalized)
        for name in ("execution_capability", "provider_access", "broker_access"):
            if getattr(self, name) is not False:
                raise FeatureRegistryError(f"{name} must remain false for offline feature snapshots")

    @property
    def observation_sha256(self) -> str:
        return _canonical_hash([item.as_dict() for item in self.observations])

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "registry": self.registry.as_dict(),
            "registry_sha256": self.registry.canonical_sha256(),
            "cutoff_timestamp_utc": self.cutoff_timestamp_utc,
            "observations": [item.as_dict() for item in self.observations],
            "observation_sha256": self.observation_sha256,
            "execution_capability": self.execution_capability,
            "provider_access": self.provider_access,
            "broker_access": self.broker_access,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self._core_dict()
        result["snapshot_sha256"] = self.canonical_sha256()
        return result

    def canonical_sha256(self) -> str:
        return _canonical_hash(self._core_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeatureSnapshot:
        if not isinstance(payload, Mapping):
            raise FeatureRegistryError("feature snapshot must be a mapping")
        if payload.get("schema", FEATURE_SNAPSHOT_SCHEMA) != FEATURE_SNAPSHOT_SCHEMA:
            raise FeatureRegistryError("unsupported feature snapshot schema")
        raw_observations = payload.get("observations")
        if not isinstance(raw_observations, list):
            raise FeatureRegistryError("feature snapshot observations must be a list")
        snapshot = cls(
            registry=FeatureRegistry.from_mapping(payload["registry"]),
            cutoff_timestamp_utc=payload["cutoff_timestamp_utc"],
            observations=tuple(FeatureObservation.from_mapping(item) for item in raw_observations),
            schema=payload.get("schema", FEATURE_SNAPSHOT_SCHEMA),
            execution_capability=payload.get("execution_capability", False),
            provider_access=payload.get("provider_access", False),
            broker_access=payload.get("broker_access", False),
        )
        expected = payload.get("snapshot_sha256")
        if expected is not None and expected != snapshot.canonical_sha256():
            raise FeatureRegistryError("snapshot_sha256 does not match canonical snapshot")
        if payload.get("registry_sha256") not in (None, snapshot.registry.canonical_sha256()):
            raise FeatureRegistryError("registry_sha256 does not match registry")
        if payload.get("observation_sha256") not in (None, snapshot.observation_sha256):
            raise FeatureRegistryError("observation_sha256 does not match observations")
        return snapshot


def build_feature_snapshot(
    registry: FeatureRegistry,
    observations: Iterable[FeatureObservation | Mapping[str, Any]],
    *,
    cutoff_timestamp_utc: int,
) -> FeatureSnapshot:
    """Build a deterministic, provider-free feature snapshot at a cutoff."""

    if not isinstance(registry, FeatureRegistry):
        raise FeatureRegistryError("registry must be a FeatureRegistry")
    normalized = tuple(
        item if isinstance(item, FeatureObservation) else FeatureObservation.from_mapping(item)
        for item in observations
    )
    return FeatureSnapshot(
        registry=registry,
        cutoff_timestamp_utc=cutoff_timestamp_utc,
        observations=normalized,
    )


__all__ = [
    "FEATURE_REGISTRY_SCHEMA",
    "FEATURE_SNAPSHOT_SCHEMA",
    "FEATURE_SPEC_SCHEMA",
    "FeatureObservation",
    "FeatureRegistry",
    "FeatureRegistryError",
    "FeatureSnapshot",
    "FeatureSpec",
    "build_feature_snapshot",
]
