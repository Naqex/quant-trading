from __future__ import annotations

import pytest

from quant_trading.event_envelope import EventEnvelope, EventEnvelopeError, collect_events
from quant_trading.onchain_events import DexPoolSnapshot, OnChainEvent


def make_event(**overrides) -> EventEnvelope:
    payload = {
        "price": "100.25",
        "side": "buy",
        "source_bar_ids": ["bar-1"],
    }
    payload.update(overrides.pop("payload", {}))
    values = {
        "source": "fixture:book",
        "instrument_id": "BTCUSDT",
        "venue": "synthetic",
        "event_type": "TRADE",
        "event_time": 100,
        "known_at": 101,
        "available_at": 102,
        "sequence": 7,
        "payload": payload,
    }
    values.update(overrides)
    return EventEnvelope(**values)


def test_event_identity_is_deterministic_and_round_trips() -> None:
    first = make_event()
    second = make_event(payload={"side": "buy", "source_bar_ids": ["bar-1"], "price": "100.25"})

    assert first.event_id == second.event_id
    assert first.event_hash == second.event_hash
    assert first.dedup_key == second.dedup_key
    assert first.as_dict() == second.as_dict()
    restored = EventEnvelope.from_mapping(first.as_dict())
    assert restored.as_dict() == first.as_dict()
    assert first.as_dict()["execution_capability"] is False


def test_event_is_causal_and_payload_is_snapshot_like() -> None:
    event = make_event(payload={"nested": {"values": [1, 2]}})
    with pytest.raises(TypeError):
        event.payload["new"] = "mutation"  # type: ignore[index]
    with pytest.raises(EventEnvelopeError, match="known_at"):
        make_event(known_at=99)
    with pytest.raises(EventEnvelopeError, match="available_at"):
        make_event(available_at=100)
    with pytest.raises(EventEnvelopeError, match="execution"):
        make_event(execution_capability=True)
    with pytest.raises(EventEnvelopeError, match="finite JSON"):
        make_event(payload={"bad": float("nan")})


def test_collect_events_applies_cutoff_finality_reorg_duplicate_and_late_policies() -> None:
    base = make_event()
    duplicate = EventEnvelope.from_mapping(base.as_dict())
    late = make_event(sequence=8, event_time=90, known_at=91, available_at=92)
    pending = make_event(sequence=9, finality="pending")
    reorged = make_event(sequence=10, reorg_status="reorged")
    future = make_event(sequence=11, available_at=999, known_at=998)

    result = collect_events(
        [future, reorged, pending, late, duplicate, base],
        cutoff_timestamp=200,
        watermark_timestamp=95,
        minimum_finality="confirmed",
    )
    assert [item.sequence for item in result.events] == [7]
    assert [item.sequence for item in result.duplicates] == [7]
    assert [item.sequence for item in result.late] == [8]
    assert [item.sequence for item in result.reorged] == [10]
    assert collect_events([late], watermark_timestamp=95, late_policy="keep").events == (late,)
    with pytest.raises(EventEnvelopeError, match="duplicate"):
        collect_events([base, duplicate], duplicate_policy="error")
    with pytest.raises(EventEnvelopeError, match="late"):
        collect_events([late], watermark_timestamp=95, late_policy="error")


def test_collect_events_is_order_independent() -> None:
    first = make_event(sequence=1, available_at=110)
    second = make_event(sequence=2, available_at=105)
    assert collect_events([first, second]).events == (second, first)


def onchain_fixture() -> dict[str, object]:
    return {
        "schema": "onchain-event-v1",
        "chain_id": "1",
        "block_number": 18_000_001,
        "block_hash": "0x" + "a" * 64,
        "transaction_hash": "0x" + "b" * 64,
        "log_index": 3,
        "event_signature": "Swap",
        "event_time": 1_700_000_000,
        "known_at": 1_700_000_012,
        "available_at": 1_700_000_013,
        "instrument_id": "ETH/USDC",
        "venue": "uniswap-v3",
        "args": {"amount0": "-100", "amount1": "250000"},
    }


def test_onchain_fixture_maps_to_canonical_envelope() -> None:
    parsed = OnChainEvent.from_mapping(onchain_fixture())
    envelope = parsed.to_envelope()
    restored = OnChainEvent.from_mapping(parsed.as_dict()).to_envelope()

    assert envelope.event_type == "Swap"
    assert envelope.sequence == 18_000_001_000_003
    assert envelope.payload["transaction_hash"] == onchain_fixture()["transaction_hash"]
    assert envelope.event_id == restored.event_id
    assert envelope.event_hash == restored.event_hash
    assert envelope.as_dict()["execution_capability"] is False

    with pytest.raises(EventEnvelopeError, match="missing transaction_hash"):
        OnChainEvent.from_mapping(
            {key: value for key, value in onchain_fixture().items() if key != "transaction_hash"}
        )
    with pytest.raises(EventEnvelopeError, match="schema"):
        OnChainEvent.from_mapping({**onchain_fixture(), "schema": "other-v1"})


def dex_fixture() -> dict[str, object]:
    return {
        "schema": "dex-pool-snapshot-v1",
        "chain_id": "1",
        "block_number": 18_000_002,
        "block_hash": "0x" + "c" * 64,
        "pool_address": "0x" + "d" * 40,
        "token0": "0x" + "e" * 40,
        "token1": "0x" + "f" * 40,
        "reserve0": "1000.0000",
        "reserve1": 2500000,
        "event_time": 1_700_000_100,
        "known_at": 1_700_000_101,
        "available_at": 1_700_000_102,
        "venue": "uniswap-v3",
        "fee_bps": "5.00",
        "liquidity": "1000000000",
    }


def test_dex_snapshot_normalizes_numeric_fixture_without_float_loss() -> None:
    snapshot = DexPoolSnapshot.from_mapping(dex_fixture())
    envelope = snapshot.to_envelope()

    assert snapshot.instrument_id == f"{snapshot.token0}/{snapshot.token1}"
    assert snapshot.reserve0 == "1000"
    assert snapshot.reserve1 == "2500000"
    assert snapshot.fee_bps == "5"
    assert envelope.event_type == "DEX_POOL_SNAPSHOT"
    assert envelope.payload["reserve0"] == "1000"
    assert envelope.payload["reserve1"] == "2500000"
    assert envelope.execution_capability is False

    with pytest.raises(EventEnvelopeError, match="reserve0"):
        DexPoolSnapshot.from_mapping({**dex_fixture(), "reserve0": "NaN"})
    with pytest.raises(EventEnvelopeError, match="reserve1"):
        DexPoolSnapshot.from_mapping({**dex_fixture(), "reserve1": -1})
    with pytest.raises(EventEnvelopeError, match="missing pool_address"):
        DexPoolSnapshot.from_mapping(
            {key: value for key, value in dex_fixture().items() if key != "pool_address"}
        )
