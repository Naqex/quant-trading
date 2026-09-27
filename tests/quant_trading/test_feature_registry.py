from __future__ import annotations

import json

import pytest

from quant_trading.feature_registry import (
    FeatureObservation,
    FeatureRegistry,
    FeatureRegistryError,
    FeatureSnapshot,
    FeatureSpec,
    build_feature_snapshot,
)

DATA_SHA = "a" * 64


def _registry() -> FeatureRegistry:
    base = FeatureSpec(
        feature_id="return_1d",
        feature_version="v1",
        instrument_id="BTCUSDT",
        timeframe_seconds=86_400,
        source_data_sha256=DATA_SHA,
        source_schema="candle-provenance-v1",
        rule_version="return-close-v1",
        source_ref_ids=("data/BTCUSDT_1d.csv",),
        minimum_availability_lag_seconds=60,
    )
    volatility = FeatureSpec(
        feature_id="vol_7d",
        feature_version="v1",
        instrument_id="BTCUSDT",
        timeframe_seconds=86_400,
        source_data_sha256=DATA_SHA,
        source_schema="candle-provenance-v1",
        rule_version="rolling-vol-v1",
        dependencies=("return_1d",),
        parameters={"window": 7},
    )
    return FeatureRegistry((volatility, base), registry_version="2026-09-r1")


def _observation(feature_id: str, event: int, value: float, *, source: str | None = None, **kwargs):
    return FeatureObservation(
        feature_id=feature_id,
        event_time_utc=event,
        known_at_utc=event + 10,
        available_at_utc=event + 70,
        value=value,
        source_row_id=source or f"row-{feature_id}-{event}",
        **kwargs,
    )


def test_registry_is_sorted_acyclic_and_hash_stable() -> None:
    registry = _registry()
    assert [spec.feature_id for spec in registry.specs] == ["return_1d", "vol_7d"]
    assert registry.by_id["vol_7d"].dependencies == ("return_1d",)
    assert registry.canonical_sha256() == FeatureRegistry.from_mapping(registry.as_dict()).canonical_sha256()
    json.dumps(registry.as_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "specs, message",
    [
        (
            (
                FeatureSpec(
                    feature_id="a",
                    feature_version="v1",
                    instrument_id="BTCUSDT",
                    timeframe_seconds=60,
                    source_data_sha256=DATA_SHA,
                    source_schema="x",
                    rule_version="r",
                    dependencies=("missing",),
                ),
            ),
            "unknown dependency",
        ),
    ],
)
def test_registry_rejects_unknown_dependency(specs, message: str) -> None:
    with pytest.raises(FeatureRegistryError, match=message):
        FeatureRegistry(specs)


def test_registry_rejects_cycles() -> None:
    common = dict(
        feature_version="v1",
        instrument_id="BTCUSDT",
        timeframe_seconds=60,
        source_data_sha256=DATA_SHA,
        source_schema="x",
        rule_version="r",
    )
    a = FeatureSpec(feature_id="a", dependencies=("b",), **common)
    b = FeatureSpec(feature_id="b", dependencies=("a",), **common)
    with pytest.raises(FeatureRegistryError, match="acyclic"):
        FeatureRegistry((a, b))


def test_snapshot_is_causal_sorted_and_round_trips() -> None:
    registry = _registry()
    rows = (
        _observation("vol_7d", 200, 0.3),
        _observation("return_1d", 100, 0.1),
        _observation("return_1d", 200, 0.2),
    )
    first = build_feature_snapshot(registry, rows, cutoff_timestamp_utc=300)
    second = build_feature_snapshot(registry, reversed(rows), cutoff_timestamp_utc=300)
    assert first.observations == tuple(
        sorted(
            rows,
            key=lambda item: (
                item.feature_id,
                item.event_time_utc,
                item.available_at_utc,
                item.source_row_id,
            ),
        )
    )
    assert first.canonical_sha256() == second.canonical_sha256()
    restored = first.from_mapping(first.as_dict())
    assert restored.as_dict() == first.as_dict()
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False


def test_snapshot_rejects_future_stale_invalidated_and_duplicate_rows() -> None:
    registry = _registry()
    with pytest.raises(FeatureRegistryError, match="snapshot cutoff"):
        build_feature_snapshot(registry, [_observation("return_1d", 100, 1.0)], cutoff_timestamp_utc=150)

    stale = _observation("return_1d", 100, 1.0, valid_until_utc=170)
    with pytest.raises(FeatureRegistryError, match="stale"):
        build_feature_snapshot(registry, [stale], cutoff_timestamp_utc=171)

    invalidated = _observation("return_1d", 100, 1.0, quality_status="invalidated")
    with pytest.raises(FeatureRegistryError, match="invalidated"):
        build_feature_snapshot(registry, [invalidated], cutoff_timestamp_utc=200)

    duplicate = _observation("return_1d", 100, 1.1, source="different-row")
    with pytest.raises(FeatureRegistryError, match="duplicate"):
        build_feature_snapshot(
            registry,
            [_observation("return_1d", 100, 1.0), duplicate],
            cutoff_timestamp_utc=200,
        )


def test_observation_enforces_temporal_and_revision_contract() -> None:
    with pytest.raises(FeatureRegistryError, match="known_at"):
        FeatureObservation(
            "x", event_time_utc=10, known_at_utc=9, available_at_utc=10, value=1.0, source_row_id="r"
        )
    with pytest.raises(FeatureRegistryError, match="available_at"):
        FeatureObservation(
            "x", event_time_utc=10, known_at_utc=10, available_at_utc=9, value=1.0, source_row_id="r"
        )
    with pytest.raises(FeatureRegistryError, match="revision zero"):
        _observation("return_1d", 100, 1.0, revision=0, supersedes_source_row_id="old")
    with pytest.raises(FeatureRegistryError, match="identify"):
        _observation("return_1d", 100, 1.0, revision=1)


def test_snapshot_enforces_minimum_consumer_delay_and_flags() -> None:
    registry = _registry()
    too_fast = FeatureObservation(
        feature_id="return_1d",
        event_time_utc=100,
        known_at_utc=150,
        available_at_utc=155,
        value=1.0,
        source_row_id="fast",
    )
    with pytest.raises(FeatureRegistryError, match="availability lag"):
        build_feature_snapshot(registry, [too_fast], cutoff_timestamp_utc=200)
    with pytest.raises(FeatureRegistryError, match="execution capability"):
        FeatureRegistry.from_mapping({**_registry().as_dict(), "execution_capability": True})


def test_snapshot_rejects_tampered_receipt_hashes() -> None:
    payload = build_feature_snapshot(
        _registry(), [_observation("return_1d", 100, 1.0)], cutoff_timestamp_utc=200
    ).as_dict()
    payload["observation_sha256"] = "b" * 64
    with pytest.raises(FeatureRegistryError, match="observation_sha256"):
        FeatureSnapshot.from_mapping(payload)
