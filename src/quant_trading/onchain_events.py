"""Provider-free on-chain fixture adapters for the Quant Lab event kernel.

The adapters parse already captured JSON-like fixtures.  They never contact a
node, indexer, DEX, mempool, or wallet.  They convert chain facts into the
same deterministic :class:`EventEnvelope` used by other research events.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from quant_trading.event_envelope import (
    FINALITY_STATES,
    REORG_STATES,
    EventEnvelope,
    EventEnvelopeError,
)

ONCHAIN_EVENT_SCHEMA = "onchain-event-v1"
DEX_POOL_SNAPSHOT_SCHEMA = "dex-pool-snapshot-v1"


def _text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventEnvelopeError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise EventEnvelopeError(f"{name} must be at most {maximum} characters")
    return normalized


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EventEnvelopeError(f"{name} must be an integer >= {minimum}")
    return value


def _decimal_text(value: Any, name: str, *, minimum: Decimal = Decimal("0")) -> str:
    if isinstance(value, bool) or value is None:
        raise EventEnvelopeError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise EventEnvelopeError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or parsed < minimum:
        raise EventEnvelopeError(f"{name} must be a finite decimal >= {minimum}")
    # Fixed-point text avoids exponent/float ambiguity in fixture hashes.
    normalized = format(parsed, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _mapping(payload: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise EventEnvelopeError(f"{name} must be a mapping")
    return payload


@dataclass(frozen=True)
class OnChainEvent:
    """A captured chain log that can be projected into an event envelope."""

    chain_id: str
    block_number: int
    block_hash: str
    transaction_hash: str
    log_index: int
    event_signature: str
    event_time: int
    known_at: int
    available_at: int
    instrument_id: str
    venue: str
    args: Mapping[str, Any]
    source: str = "fixture:onchain"
    sequence: int | None = None
    finality: str = "confirmed"
    reorg_status: str = "canonical"
    schema: str = ONCHAIN_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ONCHAIN_EVENT_SCHEMA:
            raise EventEnvelopeError(f"unsupported on-chain event schema: {self.schema!r}")
        chain_id = _text(self.chain_id, "chain_id")
        block_hash = _text(self.block_hash, "block_hash", maximum=512)
        transaction_hash = _text(self.transaction_hash, "transaction_hash", maximum=512)
        event_signature = _text(self.event_signature, "event_signature")
        source = _text(self.source, "source")
        instrument_id = _text(self.instrument_id, "instrument_id")
        venue = _text(self.venue, "venue")
        _int(self.block_number, "block_number")
        _int(self.log_index, "log_index")
        _int(self.event_time, "event_time")
        _int(self.known_at, "known_at")
        _int(self.available_at, "available_at")
        if self.known_at < self.event_time:
            raise EventEnvelopeError("known_at cannot precede event_time")
        if self.available_at < self.known_at:
            raise EventEnvelopeError("available_at cannot precede known_at")
        if self.finality not in FINALITY_STATES:
            raise EventEnvelopeError(f"unsupported finality state: {self.finality!r}")
        if self.reorg_status not in REORG_STATES:
            raise EventEnvelopeError(f"unsupported reorg status: {self.reorg_status!r}")
        if self.sequence is not None:
            _int(self.sequence, "sequence")
        _mapping(self.args, "args")
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "block_hash", block_hash)
        object.__setattr__(self, "transaction_hash", transaction_hash)
        object.__setattr__(self, "event_signature", event_signature)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "instrument_id", instrument_id)
        object.__setattr__(self, "venue", venue)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> OnChainEvent:
        if not isinstance(payload, Mapping):
            raise EventEnvelopeError("on-chain fixture must be a mapping")
        if payload.get("schema", ONCHAIN_EVENT_SCHEMA) != ONCHAIN_EVENT_SCHEMA:
            raise EventEnvelopeError("unsupported on-chain fixture schema")
        try:
            return cls(
                schema=payload.get("schema", ONCHAIN_EVENT_SCHEMA),
                chain_id=payload["chain_id"],
                block_number=payload["block_number"],
                block_hash=payload["block_hash"],
                transaction_hash=payload["transaction_hash"],
                log_index=payload["log_index"],
                event_signature=payload["event_signature"],
                event_time=payload["event_time"],
                known_at=payload["known_at"],
                available_at=payload["available_at"],
                instrument_id=payload["instrument_id"],
                venue=payload["venue"],
                args=_mapping(payload.get("args", {}), "args"),
                source=payload.get("source", "fixture:onchain"),
                sequence=payload.get("sequence"),
                finality=payload.get("finality", "confirmed"),
                reorg_status=payload.get("reorg_status", "canonical"),
            )
        except KeyError as exc:
            raise EventEnvelopeError(f"on-chain fixture is missing {exc.args[0]}") from exc

    def to_envelope(self) -> EventEnvelope:
        sequence = self.sequence
        if sequence is None:
            # Six decimal digits for log order keeps block ordering deterministic
            # while remaining safely below Python's integer limits in practice.
            sequence = self.block_number * 1_000_000 + self.log_index
        payload = {
            "schema": ONCHAIN_EVENT_SCHEMA,
            "chain_id": self.chain_id,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "transaction_hash": self.transaction_hash,
            "log_index": self.log_index,
            "event_signature": self.event_signature,
            "args": dict(self.args),
        }
        return EventEnvelope(
            source=self.source,
            instrument_id=self.instrument_id,
            venue=self.venue,
            event_type=self.event_signature,
            event_time=self.event_time,
            known_at=self.known_at,
            available_at=self.available_at,
            sequence=sequence,
            payload=payload,
            finality=self.finality,
            reorg_status=self.reorg_status,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "chain_id": self.chain_id,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "transaction_hash": self.transaction_hash,
            "log_index": self.log_index,
            "event_signature": self.event_signature,
            "event_time": self.event_time,
            "known_at": self.known_at,
            "available_at": self.available_at,
            "instrument_id": self.instrument_id,
            "venue": self.venue,
            "args": dict(self.args),
            "source": self.source,
            "sequence": self.sequence,
            "finality": self.finality,
            "reorg_status": self.reorg_status,
        }


@dataclass(frozen=True)
class DexPoolSnapshot:
    """A deterministic point-in-time DEX pool reserve snapshot fixture."""

    chain_id: str
    block_number: int
    block_hash: str
    pool_address: str
    token0: str
    token1: str
    reserve0: str
    reserve1: str
    event_time: int
    known_at: int
    available_at: int
    venue: str
    fee_bps: str | None = None
    liquidity: str | None = None
    source: str = "fixture:dex"
    instrument_id: str | None = None
    sequence: int | None = None
    finality: str = "confirmed"
    reorg_status: str = "canonical"
    schema: str = DEX_POOL_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DEX_POOL_SNAPSHOT_SCHEMA:
            raise EventEnvelopeError(f"unsupported DEX snapshot schema: {self.schema!r}")
        chain_id = _text(self.chain_id, "chain_id")
        block_hash = _text(self.block_hash, "block_hash", maximum=512)
        pool = _text(self.pool_address, "pool_address", maximum=512)
        token0 = _text(self.token0, "token0", maximum=512)
        token1 = _text(self.token1, "token1", maximum=512)
        venue = _text(self.venue, "venue")
        source = _text(self.source, "source")
        _int(self.block_number, "block_number")
        _int(self.event_time, "event_time")
        _int(self.known_at, "known_at")
        _int(self.available_at, "available_at")
        if self.known_at < self.event_time:
            raise EventEnvelopeError("known_at cannot precede event_time")
        if self.available_at < self.known_at:
            raise EventEnvelopeError("available_at cannot precede known_at")
        if self.finality not in FINALITY_STATES:
            raise EventEnvelopeError(f"unsupported finality state: {self.finality!r}")
        if self.reorg_status not in REORG_STATES:
            raise EventEnvelopeError(f"unsupported reorg status: {self.reorg_status!r}")
        reserve0 = _decimal_text(self.reserve0, "reserve0")
        reserve1 = _decimal_text(self.reserve1, "reserve1")
        fee_bps = None if self.fee_bps is None else _decimal_text(self.fee_bps, "fee_bps")
        liquidity = None if self.liquidity is None else _decimal_text(self.liquidity, "liquidity")
        instrument = self.instrument_id or f"{token0}/{token1}"
        instrument = _text(instrument, "instrument_id", maximum=512)
        if self.sequence is not None:
            _int(self.sequence, "sequence")
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "block_hash", block_hash)
        object.__setattr__(self, "pool_address", pool)
        object.__setattr__(self, "token0", token0)
        object.__setattr__(self, "token1", token1)
        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "reserve0", reserve0)
        object.__setattr__(self, "reserve1", reserve1)
        object.__setattr__(self, "fee_bps", fee_bps)
        object.__setattr__(self, "liquidity", liquidity)
        object.__setattr__(self, "instrument_id", instrument)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DexPoolSnapshot:
        if not isinstance(payload, Mapping):
            raise EventEnvelopeError("DEX pool fixture must be a mapping")
        if payload.get("schema", DEX_POOL_SNAPSHOT_SCHEMA) != DEX_POOL_SNAPSHOT_SCHEMA:
            raise EventEnvelopeError("unsupported DEX pool fixture schema")
        try:
            return cls(
                schema=payload.get("schema", DEX_POOL_SNAPSHOT_SCHEMA),
                chain_id=payload["chain_id"],
                block_number=payload["block_number"],
                block_hash=payload["block_hash"],
                pool_address=payload["pool_address"],
                token0=payload["token0"],
                token1=payload["token1"],
                reserve0=payload["reserve0"],
                reserve1=payload["reserve1"],
                event_time=payload["event_time"],
                known_at=payload["known_at"],
                available_at=payload["available_at"],
                venue=payload["venue"],
                fee_bps=payload.get("fee_bps"),
                liquidity=payload.get("liquidity"),
                source=payload.get("source", "fixture:dex"),
                instrument_id=payload.get("instrument_id"),
                sequence=payload.get("sequence"),
                finality=payload.get("finality", "confirmed"),
                reorg_status=payload.get("reorg_status", "canonical"),
            )
        except KeyError as exc:
            raise EventEnvelopeError(f"DEX pool fixture is missing {exc.args[0]}") from exc

    def to_envelope(self) -> EventEnvelope:
        sequence = self.block_number if self.sequence is None else self.sequence
        payload = {
            "schema": DEX_POOL_SNAPSHOT_SCHEMA,
            "chain_id": self.chain_id,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "pool_address": self.pool_address,
            "token0": self.token0,
            "token1": self.token1,
            "reserve0": self.reserve0,
            "reserve1": self.reserve1,
            "fee_bps": self.fee_bps,
            "liquidity": self.liquidity,
        }
        return EventEnvelope(
            source=self.source,
            instrument_id=self.instrument_id,
            venue=self.venue,
            event_type="DEX_POOL_SNAPSHOT",
            event_time=self.event_time,
            known_at=self.known_at,
            available_at=self.available_at,
            sequence=sequence,
            payload=payload,
            finality=self.finality,
            reorg_status=self.reorg_status,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "chain_id": self.chain_id,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "pool_address": self.pool_address,
            "token0": self.token0,
            "token1": self.token1,
            "reserve0": self.reserve0,
            "reserve1": self.reserve1,
            "event_time": self.event_time,
            "known_at": self.known_at,
            "available_at": self.available_at,
            "venue": self.venue,
            "fee_bps": self.fee_bps,
            "liquidity": self.liquidity,
            "source": self.source,
            "instrument_id": self.instrument_id,
            "sequence": self.sequence,
            "finality": self.finality,
            "reorg_status": self.reorg_status,
        }


__all__ = [
    "DEX_POOL_SNAPSHOT_SCHEMA",
    "ONCHAIN_EVENT_SCHEMA",
    "DexPoolSnapshot",
    "OnChainEvent",
]
