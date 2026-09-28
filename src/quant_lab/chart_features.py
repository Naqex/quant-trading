"""Canonical, offline chart-event research contracts.

The chart engine lives in the MT5 workspace, while Quant Lab owns the small
spot backtest.  This module is the boundary between them: it accepts already
computed, point-in-time chart events and pins their provenance, evaluation
splits, and modeled costs.  It deliberately does not calculate SMC/ICT labels,
call a provider, or place an order.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

CHART_EVENT_SCHEMA = "chart-event-v1"
STRATEGY_RESEARCH_SCHEMA = "strategy-research-contract-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KINDS = {
    "BOS",
    "CHoCH",
    "FVG",
    "LIQUIDITY_SWEEP",
    "ORDER_BLOCK",
    "OTE",
    "SESSION",
    "SWING",
}
_DIRECTIONS = {"bullish", "bearish", "neutral"}
_STATES = {"confirmed", "tentative", "invalidated"}


class ChartResearchContractError(ValueError):
    """Raised when an event or research contract violates its boundary."""


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ChartResearchContractError(f"{name} must be an integer{suffix}")
    return value


def _non_empty_text(value: Any, name: str, *, maximum: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChartResearchContractError(f"{name} must be a non-empty string")
    if maximum is not None and len(value) > maximum:
        raise ChartResearchContractError(f"{name} must be at most {maximum} characters")
    return value


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ChartResearchContractError(f"{name} must be a finite positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ChartResearchContractError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ChartResearchContractError(f"{name} must be a finite positive number")
    return parsed


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ChartResearchContractError(f"{name} must be a finite nonnegative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ChartResearchContractError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ChartResearchContractError(f"{name} must be a finite nonnegative number")
    return parsed


@dataclass(frozen=True)
class ChartEvent:
    """A causal chart event that is safe to expose to a research consumer."""

    event_id: str
    kind: str
    direction: str
    instrument_id: str
    timeframe_seconds: int
    anchor_timestamp: int
    known_at: int
    source_bar_ids: tuple[str, ...]
    price_low: float
    price_high: float
    state: str
    confirmation_lag_bars: int
    rule_version: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty_text(self.event_id, "event_id", maximum=256)
        if self.kind not in _EVENT_KINDS:
            raise ChartResearchContractError(f"unsupported chart event kind: {self.kind!r}")
        if self.direction not in _DIRECTIONS:
            raise ChartResearchContractError(f"unsupported chart event direction: {self.direction!r}")
        _non_empty_text(self.instrument_id, "instrument_id", maximum=128)
        _strict_int(self.timeframe_seconds, "timeframe_seconds", minimum=1)
        anchor = _strict_int(self.anchor_timestamp, "anchor_timestamp", minimum=0)
        known_at = _strict_int(self.known_at, "known_at", minimum=0)
        if known_at < anchor:
            raise ChartResearchContractError("known_at cannot precede anchor_timestamp")
        if not isinstance(self.source_bar_ids, (tuple, list)):
            raise ChartResearchContractError("source_bar_ids must be a list or tuple")
        if not self.source_bar_ids or any(
            not isinstance(item, str) or not item.strip() for item in self.source_bar_ids
        ):
            raise ChartResearchContractError("source_bar_ids must contain non-empty strings")
        if len(set(self.source_bar_ids)) != len(self.source_bar_ids):
            raise ChartResearchContractError("source_bar_ids must be unique")
        low = _positive_finite(self.price_low, "price_low")
        high = _positive_finite(self.price_high, "price_high")
        if high < low:
            raise ChartResearchContractError("price_high must be >= price_low")
        if self.state not in _STATES:
            raise ChartResearchContractError(f"unsupported chart event state: {self.state!r}")
        _strict_int(self.confirmation_lag_bars, "confirmation_lag_bars", minimum=0)
        _non_empty_text(self.rule_version, "rule_version", maximum=128)
        if not isinstance(self.parameters, Mapping):
            raise ChartResearchContractError("parameters must be a mapping")
        try:
            json.dumps(dict(self.parameters), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ChartResearchContractError("parameters must be finite JSON data") from exc

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ChartEvent:
        if not isinstance(payload, Mapping):
            raise ChartResearchContractError("chart event must be a mapping")
        if payload.get("schema", CHART_EVENT_SCHEMA) != CHART_EVENT_SCHEMA:
            raise ChartResearchContractError("unsupported chart event schema")
        try:
            raw_source_ids = payload["source_bar_ids"]
            if not isinstance(raw_source_ids, (list, tuple)):
                raise ChartResearchContractError("source_bar_ids must be a list or tuple")
            source_ids = tuple(raw_source_ids)
            return cls(
                event_id=payload["event_id"],
                kind=payload["kind"],
                direction=payload["direction"],
                instrument_id=payload["instrument_id"],
                timeframe_seconds=payload["timeframe_seconds"],
                anchor_timestamp=payload["anchor_timestamp"],
                known_at=payload["known_at"],
                source_bar_ids=source_ids,
                price_low=payload["price_low"],
                price_high=payload["price_high"],
                state=payload["state"],
                confirmation_lag_bars=payload["confirmation_lag_bars"],
                rule_version=payload["rule_version"],
                parameters=payload.get("parameters", {}),
            )
        except KeyError as exc:
            raise ChartResearchContractError(f"chart event is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CHART_EVENT_SCHEMA,
            "event_id": self.event_id,
            "kind": self.kind,
            "direction": self.direction,
            "instrument_id": self.instrument_id,
            "timeframe_seconds": self.timeframe_seconds,
            "anchor_timestamp": self.anchor_timestamp,
            "known_at": self.known_at,
            "source_bar_ids": list(self.source_bar_ids),
            "price_low": self.price_low,
            "price_high": self.price_high,
            "state": self.state,
            "confirmation_lag_bars": self.confirmation_lag_bars,
            "rule_version": self.rule_version,
            "parameters": dict(self.parameters),
        }


def validate_chart_events(
    events: Iterable[ChartEvent | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None = None,
    holdout_from_utc: int | None = None,
) -> tuple[ChartEvent, ...]:
    """Normalize events and reject future or locked-holdout observations."""

    cutoff = (
        None if cutoff_timestamp is None else _strict_int(cutoff_timestamp, "cutoff_timestamp", minimum=0)
    )
    holdout = (
        None if holdout_from_utc is None else _strict_int(holdout_from_utc, "holdout_from_utc", minimum=0)
    )
    normalized: list[ChartEvent] = []
    seen: set[str] = set()
    for raw in events:
        event = raw if isinstance(raw, ChartEvent) else ChartEvent.from_mapping(raw)
        if event.event_id in seen:
            raise ChartResearchContractError(f"duplicate chart event ID: {event.event_id}")
        if cutoff is not None and event.known_at > cutoff:
            raise ChartResearchContractError("chart event exceeds research cutoff")
        if holdout is not None and event.known_at >= holdout:
            raise ChartResearchContractError("chart event reaches locked holdout data")
        seen.add(event.event_id)
        normalized.append(event)
    return tuple(sorted(normalized, key=lambda item: (item.known_at, item.event_id)))


@dataclass(frozen=True)
class CostModel:
    """Modeled, offline round-trip costs in basis points."""

    fee_bps: float
    slippage_bps: float

    def __post_init__(self) -> None:
        fee = _nonnegative_finite(self.fee_bps, "fee_bps")
        slippage = _nonnegative_finite(self.slippage_bps, "slippage_bps")
        if fee + slippage >= 10_000:
            raise ChartResearchContractError("fee_bps + slippage_bps must be less than 10000")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CostModel:
        if not isinstance(payload, Mapping):
            raise ChartResearchContractError("cost model must be a mapping")
        try:
            return cls(fee_bps=payload["fee_bps"], slippage_bps=payload["slippage_bps"])
        except KeyError as exc:
            raise ChartResearchContractError(f"cost model is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, float]:
        return {"fee_bps": float(self.fee_bps), "slippage_bps": float(self.slippage_bps)}


@dataclass(frozen=True)
class WalkForwardFold:
    """A half-open chronological train/purge/embargo/OOS fold."""

    train_start_utc: int
    train_end_utc: int
    purge_start_utc: int
    purge_end_utc: int
    embargo_start_utc: int
    embargo_end_utc: int
    oos_start_utc: int
    oos_end_utc: int

    def __post_init__(self) -> None:
        values = (
            ("train_start_utc", self.train_start_utc),
            ("train_end_utc", self.train_end_utc),
            ("purge_start_utc", self.purge_start_utc),
            ("purge_end_utc", self.purge_end_utc),
            ("embargo_start_utc", self.embargo_start_utc),
            ("embargo_end_utc", self.embargo_end_utc),
            ("oos_start_utc", self.oos_start_utc),
            ("oos_end_utc", self.oos_end_utc),
        )
        for name, value in values:
            _strict_int(value, name, minimum=0)
        ordered = [value for _, value in values]
        if not all(left <= right for left, right in zip(ordered, ordered[1:], strict=False)):
            raise ChartResearchContractError("walk-forward ranges must be chronological")
        if self.train_start_utc >= self.train_end_utc or self.oos_start_utc >= self.oos_end_utc:
            raise ChartResearchContractError("train and OOS ranges must contain at least one timestamp")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WalkForwardFold:
        if not isinstance(payload, Mapping):
            raise ChartResearchContractError("walk-forward fold must be a mapping")
        names = (
            "train_start_utc",
            "train_end_utc",
            "purge_start_utc",
            "purge_end_utc",
            "embargo_start_utc",
            "embargo_end_utc",
            "oos_start_utc",
            "oos_end_utc",
        )
        try:
            return cls(**{name: payload[name] for name in names})
        except KeyError as exc:
            raise ChartResearchContractError(f"walk-forward fold is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, int]:
        return {
            "train_start_utc": self.train_start_utc,
            "train_end_utc": self.train_end_utc,
            "purge_start_utc": self.purge_start_utc,
            "purge_end_utc": self.purge_end_utc,
            "embargo_start_utc": self.embargo_start_utc,
            "embargo_end_utc": self.embargo_end_utc,
            "oos_start_utc": self.oos_start_utc,
            "oos_end_utc": self.oos_end_utc,
        }


@dataclass(frozen=True)
class WalkForwardPlan:
    timeframe_seconds: int
    folds: tuple[WalkForwardFold, ...]
    holdout_from_utc: int | None = None

    def __post_init__(self) -> None:
        _strict_int(self.timeframe_seconds, "timeframe_seconds", minimum=1)
        if not self.folds:
            raise ChartResearchContractError("walk-forward plan requires at least one fold")
        previous_oos_end = -1
        for fold in self.folds:
            if fold.oos_start_utc < previous_oos_end:
                raise ChartResearchContractError("walk-forward OOS ranges must not overlap")
            previous_oos_end = fold.oos_end_utc
        if self.holdout_from_utc is not None:
            holdout = _strict_int(self.holdout_from_utc, "holdout_from_utc", minimum=0)
            if any(fold.oos_end_utc > holdout for fold in self.folds):
                raise ChartResearchContractError("walk-forward OOS reaches locked holdout data")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WalkForwardPlan:
        if not isinstance(payload, Mapping):
            raise ChartResearchContractError("walk-forward plan must be a mapping")
        if payload.get("schema", "walk-forward-plan-v1") != "walk-forward-plan-v1":
            raise ChartResearchContractError("unsupported walk-forward plan schema")
        if payload.get("holdout_access") is not False:
            raise ChartResearchContractError("walk-forward plan cannot authorize holdout access")
        raw_folds = payload.get("folds")
        if not isinstance(raw_folds, list):
            raise ChartResearchContractError("walk-forward plan folds must be a list")
        try:
            return cls(
                timeframe_seconds=payload["timeframe_seconds"],
                folds=tuple(WalkForwardFold.from_mapping(item) for item in raw_folds),
                holdout_from_utc=payload.get("holdout_from_utc"),
            )
        except KeyError as exc:
            raise ChartResearchContractError(f"walk-forward plan is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "walk-forward-plan-v1",
            "chronology": "train-purge-embargo-oos",
            "holdout_access": False,
            "holdout_from_utc": self.holdout_from_utc,
            "timeframe_seconds": self.timeframe_seconds,
            "folds": [fold.as_dict() for fold in self.folds],
        }


@dataclass(frozen=True)
class StrategyResearchContract:
    """A reusable point-in-time feature contract for offline strategy research."""

    strategy_id: str
    strategy_version: str
    instrument_id: str
    timeframe_seconds: int
    feature_rule_version: str
    data_sha256: str
    data_provenance: Mapping[str, Any]
    events: tuple[ChartEvent, ...]
    costs: CostModel
    walk_forward: WalkForwardPlan
    execution_timing: str = "closed_bar_next_open"
    execution_capability: bool = False
    # A replay cutoff is part of the contract when the upstream chart engine
    # captured a bounded historical view.  Keeping it here (instead of only
    # in a caller's metadata) makes the event boundary survive serialization
    # and canonical hashing.
    replay_cutoff_utc: int | None = None

    def __post_init__(self) -> None:
        _non_empty_text(self.strategy_id, "strategy_id", maximum=128)
        _non_empty_text(self.strategy_version, "strategy_version", maximum=128)
        _non_empty_text(self.instrument_id, "instrument_id", maximum=128)
        _strict_int(self.timeframe_seconds, "timeframe_seconds", minimum=1)
        _non_empty_text(self.feature_rule_version, "feature_rule_version", maximum=128)
        if not isinstance(self.data_sha256, str) or _HEX64.fullmatch(self.data_sha256) is None:
            raise ChartResearchContractError("data_sha256 must be a lowercase 64-character SHA-256")
        if not isinstance(self.data_provenance, Mapping):
            raise ChartResearchContractError("data_provenance must be a mapping")
        for name in ("schema_version", "source", "sha256"):
            _non_empty_text(self.data_provenance.get(name), f"data_provenance.{name}", maximum=512)
        if self.data_provenance["sha256"] != self.data_sha256:
            raise ChartResearchContractError("data_provenance.sha256 must match data_sha256")
        _strict_int(self.data_provenance.get("rows"), "data_provenance.rows", minimum=1)
        try:
            json.dumps(dict(self.data_provenance), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ChartResearchContractError("data_provenance must be finite JSON data") from exc
        if not isinstance(self.costs, CostModel):
            raise ChartResearchContractError("costs must be a CostModel")
        if not isinstance(self.walk_forward, WalkForwardPlan):
            raise ChartResearchContractError("walk_forward must be a WalkForwardPlan")
        if self.execution_timing != "closed_bar_next_open":
            raise ChartResearchContractError("execution_timing must be closed_bar_next_open")
        if self.execution_capability is not False:
            raise ChartResearchContractError("chart feature contracts cannot grant execution capability")
        if self.replay_cutoff_utc is not None:
            _strict_int(self.replay_cutoff_utc, "replay_cutoff_utc", minimum=0)
        events = validate_chart_events(
            self.events,
            cutoff_timestamp=self.replay_cutoff_utc,
            holdout_from_utc=self.walk_forward.holdout_from_utc,
        )
        if any(
            event.instrument_id != self.instrument_id or event.timeframe_seconds != self.timeframe_seconds
            for event in events
        ):
            raise ChartResearchContractError("chart events must match the contract instrument/timeframe")
        object.__setattr__(self, "events", events)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> StrategyResearchContract:
        if not isinstance(payload, Mapping):
            raise ChartResearchContractError("strategy research contract must be a mapping")
        if payload.get("schema", STRATEGY_RESEARCH_SCHEMA) != STRATEGY_RESEARCH_SCHEMA:
            raise ChartResearchContractError("unsupported strategy research contract schema")
        try:
            return cls(
                strategy_id=payload["strategy_id"],
                strategy_version=payload["strategy_version"],
                instrument_id=payload["instrument_id"],
                timeframe_seconds=payload["timeframe_seconds"],
                feature_rule_version=payload["feature_rule_version"],
                data_sha256=payload["data_sha256"],
                data_provenance=payload["data_provenance"],
                events=tuple(ChartEvent.from_mapping(item) for item in payload.get("events", [])),
                costs=CostModel.from_mapping(payload["costs"]),
                walk_forward=WalkForwardPlan.from_mapping(payload["walk_forward"]),
                execution_timing=payload.get("execution_timing", "closed_bar_next_open"),
                execution_capability=payload.get("execution_capability", False),
                replay_cutoff_utc=payload.get("replay_cutoff_utc"),
            )
        except KeyError as exc:
            raise ChartResearchContractError(f"strategy research contract is missing {exc.args[0]}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STRATEGY_RESEARCH_SCHEMA,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "instrument_id": self.instrument_id,
            "timeframe_seconds": self.timeframe_seconds,
            "feature_rule_version": self.feature_rule_version,
            "data_sha256": self.data_sha256,
            "data_provenance": dict(self.data_provenance),
            "execution_timing": self.execution_timing,
            "execution_capability": self.execution_capability,
            "replay_cutoff_utc": self.replay_cutoff_utc,
            "events": [event.as_dict() for event in self.events],
            "costs": self.costs.as_dict(),
            "walk_forward": self.walk_forward.as_dict(),
        }

    def canonical_sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_strategy_research_contract(
    *,
    strategy_id: str,
    strategy_version: str,
    instrument_id: str,
    timeframe_seconds: int,
    feature_rule_version: str,
    data_sha256: str,
    data_provenance: Mapping[str, Any],
    events: Iterable[ChartEvent | Mapping[str, Any]],
    costs: CostModel,
    walk_forward: WalkForwardPlan,
    replay_cutoff_utc: int | None = None,
) -> StrategyResearchContract:
    """Build one validated, provider-free chart feature research packet."""

    normalized_events = validate_chart_events(
        events,
        cutoff_timestamp=replay_cutoff_utc,
        holdout_from_utc=walk_forward.holdout_from_utc,
    )
    return StrategyResearchContract(
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        instrument_id=instrument_id,
        timeframe_seconds=timeframe_seconds,
        feature_rule_version=feature_rule_version,
        data_sha256=data_sha256,
        data_provenance=data_provenance,
        events=normalized_events,
        costs=costs,
        walk_forward=walk_forward,
        replay_cutoff_utc=replay_cutoff_utc,
    )


def build_causal_signal(
    timestamps: Iterable[int],
    events: Iterable[ChartEvent | Mapping[str, Any]],
    *,
    cutoff_timestamp: int | None = None,
    holdout_from_utc: int | None = None,
    instrument_id: str | None = None,
    timeframe_seconds: int | None = None,
) -> tuple[float, ...]:
    """Project confirmed chart events onto a long/cash next-bar signal.

    ``timestamps`` must use the same UTC-second unit as ``ChartEvent.known_at``.
    An event known at ``t`` can first affect the signal at the first timestamp
    strictly greater than ``t``.  This keeps the adapter causal; fills, fees,
    slippage and equity accounting remain owned by :func:`run_backtest`.
    Tentative and invalidated events are intentionally ignored.  A bearish
    confirmed event exits the spot position, while a bullish one enters it.
    """

    timeline = tuple(timestamps)
    previous: int | None = None
    for timestamp in timeline:
        _strict_int(timestamp, "timestamps", minimum=0)
        if previous is not None and timestamp <= previous:
            raise ChartResearchContractError("timestamps must be strictly increasing")
        previous = timestamp

    normalized_events = validate_chart_events(
        events,
        cutoff_timestamp=cutoff_timestamp,
        holdout_from_utc=holdout_from_utc,
    )
    if instrument_id is not None:
        _non_empty_text(instrument_id, "instrument_id", maximum=128)
    if timeframe_seconds is not None:
        _strict_int(timeframe_seconds, "timeframe_seconds", minimum=1)

    changes: dict[int, float] = {}
    for event in normalized_events:
        if instrument_id is not None and event.instrument_id != instrument_id:
            raise ChartResearchContractError("chart events must match instrument_id")
        if timeframe_seconds is not None and event.timeframe_seconds != timeframe_seconds:
            raise ChartResearchContractError("chart events must match timeframe_seconds")
        if event.state != "confirmed" or event.direction == "neutral":
            continue
        first_effective_index = bisect_right(timeline, event.known_at)
        if first_effective_index < len(timeline):
            changes[first_effective_index] = 1.0 if event.direction == "bullish" else 0.0

    signal: list[float] = []
    current = 0.0
    for index in range(len(timeline)):
        if index in changes:
            current = changes[index]
        signal.append(current)
    return tuple(signal)


__all__ = [
    "CHART_EVENT_SCHEMA",
    "STRATEGY_RESEARCH_SCHEMA",
    "ChartEvent",
    "ChartResearchContractError",
    "CostModel",
    "StrategyResearchContract",
    "WalkForwardFold",
    "WalkForwardPlan",
    "build_causal_signal",
    "build_strategy_research_contract",
    "validate_chart_events",
]
