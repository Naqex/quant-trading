from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_trading.null_baseline import (
    NullBaselineError,
    buy_and_hold_signal_factory,
    cash_signal_factory,
    run_null_baselines,
)
from quant_trading.walk_forward import WalkForwardConfig


def _close(rows: int = 80) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    return pd.Series(100.0 + np.arange(rows, dtype=float), index=index, name="close")


def _config() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_bars=20,
        test_bars=10,
        step_bars=10,
        minimum_folds=3,
        stress_multipliers=(1.0, 2.0),
    )


def test_factories_are_typed_spot_references() -> None:
    history = _close(3)
    assert cash_signal_factory(history).tolist() == [0.0, 0.0, 0.0]
    assert buy_and_hold_signal_factory(history).tolist() == [1.0, 1.0, 1.0]
    assert cash_signal_factory(history).index.equals(history.index)
    assert buy_and_hold_signal_factory(history).index.equals(history.index)


def test_null_baselines_are_deterministic_and_share_cost_stress() -> None:
    first = run_null_baselines(_close(), config=_config())
    second = run_null_baselines(_close(), baseline_ids=("buy_hold", "cash"), config=_config())

    assert first.as_dict() == second.as_dict()
    assert first.receipt_sha256 == second.receipt_sha256
    assert [evaluation.baseline_id for evaluation in first.evaluations] == ["buy_hold", "cash"]
    assert first.execution_capability is False
    assert first.provider_access is False
    assert first.broker_access is False
    assert first.status == "PREP_ONLY"
    buy_hold = first.evaluations[0]
    cash = first.evaluations[1]
    assert buy_hold.baseline_metrics["total_return"] > cash.baseline_metrics["total_return"]
    # The first entry can sit in the training prefix, so the OOS slice may
    # legitimately show no cost difference for a continuous hold.
    assert buy_hold.stress_metrics["total_return"] <= buy_hold.baseline_metrics["total_return"]
    assert cash.baseline_metrics["total_return"] == pytest.approx(0.0)
    assert [case.multiplier for case in buy_hold.receipt.cases] == [1.0, 2.0]


@pytest.mark.parametrize(
    "baseline_ids, error",
    [
        ((), "non-empty"),
        (("cash", "cash"), "unique"),
        (("random",), "unsupported"),
        (("cash", 1), "strings"),
    ],
)
def test_null_baseline_rejects_malformed_selection(
    baseline_ids: tuple[object, ...], error: str
) -> None:
    with pytest.raises(NullBaselineError, match=error):
        run_null_baselines(_close(), baseline_ids=baseline_ids)  # type: ignore[arg-type]


def test_null_baseline_receipt_rejects_tampered_hash() -> None:
    receipt = run_null_baselines(_close(), config=_config())
    with pytest.raises(NullBaselineError, match="does not match"):
        replace(receipt, receipt_sha256="0" * 64)
