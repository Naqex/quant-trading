from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_trading.walk_forward import (
    WalkForwardConfig,
    WalkForwardError,
    build_walk_forward_folds,
    run_walk_forward_stress,
)


def _close(rows: int = 80) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    return pd.Series(100.0 + np.arange(rows, dtype=float), index=index, name="close")


def _alternating_signal(history: pd.Series) -> pd.Series:
    return pd.Series(np.arange(len(history)) % 2, index=history.index, dtype=float)


def _config(**overrides: object) -> WalkForwardConfig:
    values: dict[str, object] = {
        "train_bars": 20,
        "test_bars": 10,
        "step_bars": 10,
        "minimum_folds": 3,
        "stress_multipliers": (1.0, 2.0),
    }
    values.update(overrides)
    return WalkForwardConfig(**values)


def test_folds_are_ordered_disjoint_and_have_no_future_rows() -> None:
    close = _close()
    folds = build_walk_forward_folds(close.index, _config())

    assert len(folds) == 6
    assert folds[0].train_end < folds[0].test_start
    assert folds[0].test_end < folds[1].test_start
    assert folds[0].train_count == 20
    assert folds[-1].test_end == close.index[-1].isoformat()


def test_runner_is_causal_deterministic_and_preserves_offline_flags() -> None:
    close = _close()
    seen_end: list[pd.Timestamp] = []

    def factory(history: pd.Series) -> pd.Series:
        seen_end.append(history.index[-1])
        return _alternating_signal(history)

    config = _config()
    first = run_walk_forward_stress(
        close,
        factory,
        strategy_id="alternating-v1",
        fee_bps=10.0,
        slippage_bps=5.0,
        config=config,
    )
    second = run_walk_forward_stress(
        close,
        _alternating_signal,
        strategy_id="alternating-v1",
        fee_bps=10.0,
        slippage_bps=5.0,
        config=config,
    )

    assert len(seen_end) == 6 * 11
    for fold_index in range(6):
        fold_calls = seen_end[fold_index * 11 : (fold_index + 1) * 11]
        assert fold_calls[0] == close.index[19 + fold_index * 10]
        assert fold_calls[-1] == close.index[29 + fold_index * 10]
        assert fold_calls == sorted(fold_calls)
    assert first.as_dict() == second.as_dict()
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False
    assert first.status == "PREP_ONLY"


def test_adverse_cost_stress_changes_turnover_result() -> None:
    receipt = run_walk_forward_stress(
        _close(),
        _alternating_signal,
        strategy_id="alternating-v1",
        fee_bps=10.0,
        slippage_bps=5.0,
        config=_config(),
    )
    baseline, stressed = receipt.cases
    assert baseline.multiplier == 1.0
    assert stressed.multiplier == 2.0
    assert stressed.fee_bps == pytest.approx(20.0)
    assert stressed.slippage_bps == pytest.approx(10.0)
    assert stressed.aggregate_metrics["total_return"] < baseline.aggregate_metrics["total_return"]
    assert stressed.aggregate_metrics["fold_count"] == 6


def test_boundary_entry_is_counted_in_each_oos_window() -> None:
    receipt = run_walk_forward_stress(
        _close(70),
        lambda history: pd.Series(1.0, index=history.index),
        strategy_id="always-on-v1",
        config=_config(minimum_folds=2),
    )
    baseline = receipt.cases[0]
    assert all(fold.metrics["entries"] == 1 for fold in baseline.folds)


@pytest.mark.parametrize(
    "close, error",
    [
        (pd.Series([100.0, 101.0], index=[1, 0]), "sorted"),
        (_close(25), "insufficient"),
    ],
)
def test_contract_rejects_bad_fold_inputs(close: pd.Series, error: str) -> None:
    with pytest.raises(WalkForwardError, match=error):
        run_walk_forward_stress(
            close,
            _alternating_signal,
            strategy_id="bad",
            config=_config(minimum_folds=2),
        )


def test_config_requires_baseline_stress_case() -> None:
    with pytest.raises(WalkForwardError, match="include the 1.0"):
        WalkForwardConfig(stress_multipliers=(1.5,))


def test_contract_rejects_bad_signal_and_tampered_receipt() -> None:
    with pytest.raises(WalkForwardError, match="exact history index"):
        run_walk_forward_stress(
            _close(),
            lambda history: pd.Series(1.0, index=range(len(history))),
            strategy_id="wrong-index",
            config=_config(),
        )

    receipt = run_walk_forward_stress(
        _close(), _alternating_signal, strategy_id="tamper-test", config=_config()
    )
    with pytest.raises(WalkForwardError, match="does not match"):
        replace(receipt, receipt_sha256="0" * 64)


def test_cost_limit_and_nonfinite_prices_fail_closed() -> None:
    with pytest.raises(WalkForwardError, match="below 100%"):
        run_walk_forward_stress(
            _close(),
            _alternating_signal,
            strategy_id="unsafe-cost",
            fee_bps=6_000,
            slippage_bps=0,
            config=_config(stress_multipliers=(1.0, 2.0)),
        )
    broken = _close()
    broken.iloc[5] = np.nan
    with pytest.raises(WalkForwardError, match="finite"):
        run_walk_forward_stress(broken, _alternating_signal, strategy_id="nan", config=_config())
