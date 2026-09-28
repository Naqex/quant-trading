from __future__ import annotations

import json

import pytest

from quant_trading.distribution_drift import PSIConfig
from quant_trading.feature_drift import (
    FeatureDriftError,
    compute_feature_snapshot_psi,
)
from quant_trading.feature_registry import (
    FeatureObservation,
    FeatureRegistry,
    FeatureSpec,
    build_feature_snapshot,
)

DATA_SHA = "a" * 64


def _registry(*, source_sha: str = DATA_SHA) -> FeatureRegistry:
    return FeatureRegistry(
        (
            FeatureSpec(
                feature_id="return_1d",
                feature_version="v1",
                instrument_id="BTCUSDT",
                timeframe_seconds=86_400,
                source_data_sha256=source_sha,
                source_schema="candle-provenance-v1",
                rule_version="return-close-v1",
                minimum_availability_lag_seconds=60,
            ),
        ),
        registry_version="2026-09-r1",
    )


def _snapshot(
    values: tuple[float, ...],
    *,
    cutoff: int,
    registry: FeatureRegistry | None = None,
    quality_status: str = "accepted",
):
    rows = tuple(
        FeatureObservation(
            feature_id="return_1d",
            event_time_utc=index * 100,
            known_at_utc=index * 100 + 10,
            available_at_utc=index * 100 + 70,
            value=value,
            source_row_id=f"row-{index}",
            quality_status=quality_status,
        )
        for index, value in enumerate(values, 1)
    )
    return build_feature_snapshot(registry or _registry(), rows, cutoff_timestamp_utc=cutoff)


def test_feature_snapshot_psi_preserves_lineage_and_is_order_stable() -> None:
    reference = _snapshot((-1.0, -0.5, 0.5, 1.0), cutoff=500)
    current = _snapshot((0.7, 0.8, 1.1, 1.2), cutoff=900)
    config = PSIConfig(bin_edges=(-2.0, 0.0, 2.0), alert_threshold=0.2)

    first = compute_feature_snapshot_psi(reference, current, feature_id="return_1d", config=config)
    reversed_current = build_feature_snapshot(
        current.registry,
        reversed(current.observations),
        cutoff_timestamp_utc=current.cutoff_timestamp_utc,
    )
    second = compute_feature_snapshot_psi(
        reference,
        reversed_current,
        feature_id="return_1d",
        config=config,
    )

    assert first.cutoff_timestamp_utc == 900
    assert first.psi_result.current_sample_count == 4
    assert first.psi_result.drift_flag is True
    assert first.as_dict() == second.as_dict()
    assert first.canonical_sha256() == second.canonical_sha256()
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False
    json.dumps(first.as_dict(), allow_nan=False)


def test_feature_snapshot_psi_rejects_registry_mismatch() -> None:
    reference = _snapshot((-1.0, 0.0), cutoff=500)
    current = _snapshot((0.2, 0.5), cutoff=900, registry=_registry(source_sha="b" * 64))
    with pytest.raises(FeatureDriftError, match="same feature registry"):
        compute_feature_snapshot_psi(
            reference,
            current,
            feature_id="return_1d",
            config=PSIConfig(bin_edges=(-2.0, 0.0, 2.0)),
        )


def test_feature_snapshot_psi_rejects_suspect_rows_and_unknown_feature() -> None:
    suspect = _snapshot((-1.0, 0.0), cutoff=500, quality_status="suspect")
    clean = _snapshot((0.2, 0.5), cutoff=900)
    config = PSIConfig(bin_edges=(-2.0, 0.0, 2.0))
    with pytest.raises(FeatureDriftError, match="non-accepted"):
        compute_feature_snapshot_psi(suspect, clean, feature_id="return_1d", config=config)
    with pytest.raises(FeatureDriftError, match="does not define"):
        compute_feature_snapshot_psi(suspect, clean, feature_id="missing", config=config)


def test_feature_snapshot_psi_rejects_future_explicit_cutoff() -> None:
    reference = _snapshot((-1.0, 0.0), cutoff=500)
    current = _snapshot((0.2, 0.5), cutoff=900)
    with pytest.raises(FeatureDriftError, match="exceeds both snapshot boundaries"):
        compute_feature_snapshot_psi(
            reference,
            current,
            feature_id="return_1d",
            config=PSIConfig(bin_edges=(-2.0, 0.0, 2.0)),
            cutoff_timestamp_utc=901,
        )


def test_feature_snapshot_psi_rejects_rows_after_explicit_cutoff() -> None:
    reference = _snapshot((-1.0, 0.0), cutoff=500)
    current = _snapshot((0.2, 0.5), cutoff=900)
    with pytest.raises(FeatureDriftError, match="exceeds cutoff_timestamp"):
        compute_feature_snapshot_psi(
            reference,
            current,
            feature_id="return_1d",
            config=PSIConfig(bin_edges=(-2.0, 0.0, 2.0)),
            cutoff_timestamp_utc=250,
        )

