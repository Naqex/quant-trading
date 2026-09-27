"""Offline, deterministic event-stream contracts for Quant Lab.

This module is intentionally provider-free.  It accepts already captured event
facts, gives each event a reproducible identity, and exposes a fail-closed
normalizer for research replay.  It does not fetch a chain, subscribe to a
mempool, or grant execution capability.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

EVENT_ENVELOPE_SCHEMA = "event-envelope-v1"
FINALITY_STATES = ("pending", "confirmed", "finalized")
REORG_STATES = ("canonical", "unknown", "reorged")
FINALITY_RANK = {state: index for index, state in enumerate(FINALITY_STATES)}
DuplicatePolicy = Literal["drop", "error"]
LatePolicy = Literal["drop", "error", "keep"]


class EventEnvelopeError(ValueError):
    """Raised when an event violates the offline event-stream boundary."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EventEnvelopeError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventEnvelopeError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise EventEnvelopeError(f"{name} must be at most {maximum} characters")
    return normalized


def _canonical_json(value: Any) -> str:
    """Serialize finite JSON-like data with one stable representation."""
    try:
        return json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EventEnvelopeError("payload must be finite JSON data") from exc


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise EventEnvelopeError("payload must be finite JSON data")


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _event_core(event: EventEnvelope) -> dict[str, Any]:
    return {
        "schema": event.schema,
        "source": event.source,
        "instrument_id": event.instrument_id,
        "venue": event.venue,
        "event_type": event.event_type,
        "event_time": event.event_time,
        "known_at": event.known_at,
        "available_at": event.available_at,
        "sequence": event.sequence,
        "payload": _thaw(event.payload),
        "finality": event.finality,
        "reorg_status": event.reorg_status,
        "dedup_key": event.dedup_key,
    }


def _dedup_identity(event: EventEnvelope) -> dict[str, Any]:
    return {
        "source": event.source,
        "instrument_id": event.instrument_id,
        "venue": event.venue,
        "event_type": event.event_type,
        "event_time": event.event_time,
        "sequence": event.sequence,
        "payload": _thaw(event.payload),
    }


@dataclass(frozen=True)
class EventEnvelope:
    """A point-in-time event safe to use in offline replay.

    ``event_time`` is when the event occurred.  ``known_at`` is the first time
    the captured source could know it, and ``available_at`` is the first replay
    timestamp at which the consumer may use it.  Keeping both timestamps makes
    delayed data and look-ahead mistakes visible.
    """

    source: str
    instrument_id: str
    venue: str
    event_type: str
    event_time: int
    known_at: int
    available_at: int
    sequence: int
    payload: Mapping[str, Any]
    finality: str = "confirmed"
    reorg_status: str = "canonical"
    schema: str = EVENT_ENVELOPE_SCHEMA
    dedup_key: str | None = None
    event_id: str | None = None
    event_hash: str | None = None
    execution_capability: bool = False

    def __post_init__(self) -> None:
        if self.schema != EVENT_ENVELOPE_SCHEMA:
            raise EventEnvelopeError(f"unsupported event schema: {self.schema!r}")
        source = _text(self.source, "source")
        instrument = _text(self.instrument_id, "instrument_id")
        venue = _text(self.venue, "venue")
        event_type = _text(self.event_type, "event_type")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "instrument_id", instrument)
        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "event_type", event_type)
        event_time = _strict_int(self.event_time, "event_time")
        known_at = _strict_int(self.known_at, "known_at")
        available_at = _strict_int(self.available_at, "available_at")
        if known_at < event_time:
            raise EventEnvelopeError("known_at cannot precede event_time")
        if available_at < known_at:
            raise EventEnvelopeError("available_at cannot precede known_at")
        _strict_int(self.sequence, "sequence")
        if self.finality not in FINALITY_STATES:
            raise EventEnvelopeError(f"unsupported finality state: {self.finality!r}")
        if self.reorg_status not in REORG_STATES:
            raise EventEnvelopeError(f"unsupported reorg status: {self.reorg_status!r}")
        if self.execution_capability is not False:
            raise EventEnvelopeError("event envelopes cannot grant execution capability")
        if not isinstance(self.payload, Mapping):
            raise EventEnvelopeError("payload must be a mapping")
        frozen_payload = _freeze(self.payload)
        _canonical_json(frozen_payload)
        object.__setattr__(self, "payload", frozen_payload)
        expected_dedup = f"dedup-{_digest(_dedup_identity(self))}"
        if self.dedup_key is None:
            object.__setattr__(self, "dedup_key", expected_dedup)
        elif _text(self.dedup_key, "dedup_key", maximum=512) != self.dedup_key:
            raise EventEnvelopeError("dedup_key must not have surrounding whitespace")
        core_hash = _digest(_event_core(self))
        expected_id = f"evt-{core_hash[:32]}"
        if self.event_id is not None and self.event_id != expected_id:
            raise EventEnvelopeError("event_id does not match deterministic event hash")
        if self.event_hash is not None and self.event_hash != core_hash:
            raise EventEnvelopeError("event_hash does not match canonical event payload")
        object.__setattr__(self, "event_id", expected_id)
        object.__setattr__(self, "event_hash", core_hash)
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "available_at", available_at)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EventEnvelope:
        if not isinstance(payload, Mapping):
            raise EventEnvelopeError("event envelope must be a mapping")
        try:
            return cls(
                schema=payload.get("schema", EVENT_ENVELOPE_SCHEMA),
                source=payload["source"],
                instrument_id=payload["instrument_id"],
                venue=payload["venue"],
                event_type=payload["event_type"],
                event_time=payload["event_time"],
                known_at=payload["known_at"],
                available_at=payload["available_at"],
                sequence=payload["sequence"],
                payload=payload["payload"],
                finality=payload.get("finality", "confirmed"),
                reorg_status=payload.get("reorg_status", "canonical"),
                dedup_key=payload.get("dedup_key"),
                event_id=payload.get("event_id"),
                event_hash=payload.get("event_hash"),
                execution_capability=payload.get("execution_capability", False),
            )
        except KeyError as exc:
            raise EventEnvelopeError(f"event envelope is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "instrument_id": self.instrument_id,
            "venue": self.venue,
            "event_type": self.event_type,
            "event_time": self.event_time,
            "known_at": self.known_at,
            "available_at": self.available_at,
            "sequence": self.sequence,
            "payload": _thaw(self.payload),
            "finality": self.finality,
            "reorg_status": self.reorg_status,
            "dedup_key": self.dedup_key,
            "event_id": self.event_id,
            "event_hash": self.event_hash,
            "execution_capability": self.execution_capability,
        }


@dataclass(frozen=True)
class EventBatch:
    """Accepted events and explicit drop receipts from deterministic ingestion."""

    events: tuple[EventEnvelope, ...]
    duplicates: tuple[EventEnvelope, ...] = ()
    late: tuple[EventEnvelope, ...] = ()
    reorged: tuple[EventEnvelope, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "event-batch-v1",
            "events": [event.as_dict() for event in self.events],
            "duplicates": [event.as_dict() for event in self.duplicates],
            "late": [event.as_dict() for event in self.late],
            "reorged": [event.as_dict() for event in self.reorged],
            "execution_capability": False,
        }


def collect_events(
    events: Iterable[EventEnvelope | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None = None,
    watermark_timestamp: int | None = None,
    minimum_finality: str = "pending",
    allow_reorged: bool = False,
    duplicate_policy: DuplicatePolicy = "drop",
    late_policy: LatePolicy = "drop",
) -> EventBatch:
    """Normalize a replay batch and make duplicate/late handling explicit.

    ``cutoff_timestamp`` applies to ``available_at`` because that is the
    point-in-time boundary visible to a strategy.  ``watermark_timestamp``
    classifies events whose ``event_time`` is older than the stream watermark.
    No network or provider is contacted; all decisions derive from the input.
    """

    if cutoff_timestamp is not None:
        _strict_int(cutoff_timestamp, "cutoff_timestamp")
    if watermark_timestamp is not None:
        _strict_int(watermark_timestamp, "watermark_timestamp")
    if minimum_finality not in FINALITY_STATES:
        raise EventEnvelopeError(f"unsupported minimum_finality: {minimum_finality!r}")
    if duplicate_policy not in ("drop", "error"):
        raise EventEnvelopeError(f"unsupported duplicate_policy: {duplicate_policy!r}")
    if late_policy not in ("drop", "error", "keep"):
        raise EventEnvelopeError(f"unsupported late_policy: {late_policy!r}")

    normalized = tuple(
        raw if isinstance(raw, EventEnvelope) else EventEnvelope.from_mapping(raw) for raw in events
    )
    ordered = tuple(
        sorted(
            normalized,
            key=lambda item: (item.available_at, item.event_time, item.sequence, item.event_id),
        )
    )
    accepted: list[EventEnvelope] = []
    duplicates: list[EventEnvelope] = []
    late: list[EventEnvelope] = []
    reorged: list[EventEnvelope] = []
    seen: set[str] = set()
    for event in ordered:
        if cutoff_timestamp is not None and event.available_at > cutoff_timestamp:
            continue
        if FINALITY_RANK[event.finality] < FINALITY_RANK[minimum_finality]:
            continue
        if event.reorg_status == "reorged" and not allow_reorged:
            reorged.append(event)
            continue
        is_duplicate = event.dedup_key in seen
        if is_duplicate:
            duplicates.append(event)
            if duplicate_policy == "error":
                raise EventEnvelopeError(f"duplicate event dedup_key: {event.dedup_key}")
            continue
        is_late = watermark_timestamp is not None and event.event_time < watermark_timestamp
        if is_late:
            late.append(event)
            if late_policy == "error":
                raise EventEnvelopeError(f"late event precedes watermark: {event.event_id}")
            if late_policy == "drop":
                continue
        seen.add(event.dedup_key)
        accepted.append(event)
    return EventBatch(tuple(accepted), tuple(duplicates), tuple(late), tuple(reorged))


__all__ = [
    "EVENT_ENVELOPE_SCHEMA",
    "FINALITY_STATES",
    "REORG_STATES",
    "EventBatch",
    "EventEnvelope",
    "EventEnvelopeError",
    "collect_events",
]
