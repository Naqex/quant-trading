import pytest

from quant_lab.chart_features import (
    ChartEvent,
    ChartResearchContractError,
    CostModel,
    WalkForwardFold,
    WalkForwardPlan,
    build_strategy_research_contract,
    validate_chart_events,
)

DATA_SHA = "a" * 64
DATA_PROVENANCE = {
    "schema_version": "candle-provenance-v1",
    "source": "fixture:synthetic",
    "rows": 240,
    "sha256": DATA_SHA,
}


def event(event_id="evt-1", *, known_at=120, anchor_timestamp=120, instrument_id="BTCUSDT"):
    return ChartEvent(
        event_id=event_id,
        kind="BOS",
        direction="bullish",
        instrument_id=instrument_id,
        timeframe_seconds=60,
        anchor_timestamp=anchor_timestamp,
        known_at=known_at,
        source_bar_ids=("bar-1", "bar-2"),
        price_low=100.0,
        price_high=101.0,
        state="confirmed",
        confirmation_lag_bars=0,
        rule_version="smc-v1",
    )


def plan(holdout_from_utc=None):
    fold = WalkForwardFold(
        train_start_utc=0,
        train_end_utc=100,
        purge_start_utc=100,
        purge_end_utc=110,
        embargo_start_utc=110,
        embargo_end_utc=120,
        oos_start_utc=120,
        oos_end_utc=180,
    )
    return WalkForwardPlan(timeframe_seconds=60, folds=(fold,), holdout_from_utc=holdout_from_utc)


def test_contract_normalizes_causal_events_and_is_hashable():
    contract = build_strategy_research_contract(
        strategy_id="smc-bos-research",
        strategy_version="1",
        instrument_id="BTCUSDT",
        timeframe_seconds=60,
        feature_rule_version="smc-v1",
        data_sha256=DATA_SHA,
        data_provenance=DATA_PROVENANCE,
        events=[event()],
        costs=CostModel(fee_bps=10, slippage_bps=5),
        walk_forward=plan(),
    )

    assert contract.as_dict()["execution_capability"] is False
    assert contract.as_dict()["walk_forward"]["holdout_access"] is False
    assert len(contract.canonical_sha256()) == 64
    assert contract.events[0].event_id == "evt-1"
    restored = contract.from_mapping(contract.as_dict())
    assert restored.canonical_sha256() == contract.canonical_sha256()


def test_future_and_holdout_events_fail_closed():
    with pytest.raises(ChartResearchContractError, match="cutoff"):
        validate_chart_events([event(known_at=121)], cutoff_timestamp=120)
    with pytest.raises(ChartResearchContractError, match="holdout"):
        validate_chart_events([event(known_at=120)], holdout_from_utc=120)


def test_duplicate_ids_and_mismatched_event_identity_are_rejected():
    with pytest.raises(ChartResearchContractError, match="duplicate"):
        validate_chart_events([event(), event()])
    with pytest.raises(ChartResearchContractError, match="instrument/timeframe"):
        build_strategy_research_contract(
            strategy_id="smc",
            strategy_version="1",
            instrument_id="BTCUSDT",
            timeframe_seconds=60,
            feature_rule_version="smc-v1",
            data_sha256=DATA_SHA,
            data_provenance=DATA_PROVENANCE,
            events=[event(instrument_id="ETHUSDT")],
            costs=CostModel(10, 5),
            walk_forward=plan(),
        )


def test_provenance_digest_and_source_are_required():
    with pytest.raises(ChartResearchContractError, match="sha256"):
        build_strategy_research_contract(
            strategy_id="smc",
            strategy_version="1",
            instrument_id="BTCUSDT",
            timeframe_seconds=60,
            feature_rule_version="smc-v1",
            data_sha256=DATA_SHA,
            data_provenance={**DATA_PROVENANCE, "sha256": "b" * 64},
            events=[],
            costs=CostModel(10, 5),
            walk_forward=plan(),
        )

    with pytest.raises(ChartResearchContractError, match="lowercase"):
        build_strategy_research_contract(
            strategy_id="smc",
            strategy_version="1",
            instrument_id="BTCUSDT",
            timeframe_seconds=60,
            feature_rule_version="smc-v1",
            data_sha256=DATA_SHA.upper(),
            data_provenance={**DATA_PROVENANCE, "sha256": DATA_SHA.upper()},
            events=[],
            costs=CostModel(10, 5),
            walk_forward=plan(),
        )


@pytest.mark.parametrize("fee,slippage", [(float("nan"), 0), (0, float("inf")), (-1, 0), (9_999, 1)])
def test_cost_model_rejects_invalid_or_negative_assumptions(fee, slippage):
    with pytest.raises(ChartResearchContractError):
        CostModel(fee, slippage)


def test_walk_forward_rejects_oos_overlap_and_holdout_reach():
    first = plan().folds[0]
    second = WalkForwardFold(
        train_start_utc=0,
        train_end_utc=100,
        purge_start_utc=100,
        purge_end_utc=100,
        embargo_start_utc=100,
        embargo_end_utc=150,
        oos_start_utc=150,
        oos_end_utc=200,
    )
    with pytest.raises(ChartResearchContractError, match="overlap"):
        WalkForwardPlan(timeframe_seconds=60, folds=(first, second))
    with pytest.raises(ChartResearchContractError, match="holdout"):
        plan(holdout_from_utc=179)
