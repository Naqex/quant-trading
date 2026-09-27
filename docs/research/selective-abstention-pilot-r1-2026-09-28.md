# Selective prediction / abstention pilot — 2026-09-28

Quant Lab now has a small, provider-free selective-prediction primitive at
`src/quant_trading/selective_prediction.py`. It calibrates an uncertainty
threshold on a frozen chronological calibration window, retains the largest
observed set whose mean absolute error is within a pre-registered risk limit,
and emits a deterministic receipt. A future row is either accepted or
abstained; the gate never changes the point predictor and never grants
execution authority.

## Contract

- Calibration rows must be strictly after `train_end_utc` and no later than
  `validation_end_utc`; future decisions must be strictly after validation and
  no later than `test_end_utc`.
- Uncertainty scores and predictions must be finite; scores must be
  non-negative; duplicate or out-of-order timestamps are rejected.
- `target_coverage` is a lower bound on the retained calibration fraction.
  `max_risk` is a pre-registered upper bound on retained calibration mean
  absolute error. The largest observed uncertainty threshold satisfying both
  is selected, with ties included deterministically.
- If no observed threshold satisfies both constraints, fitting fails closed.
  An evaluation can still report zero coverage when every future row is
  abstained; its selective risk is `null` because no row was accepted.
- `evaluate_selective` is evaluation-only. It reports accepted coverage,
  abstention rate, and mean absolute error on accepted future rows. It does
  not tune the threshold.
- Calibration and evaluation receipts contain canonical SHA-256 hashes,
  cutoff/data lineage and `execution_capability=false`, `provider_access=false`
  and `broker_access=false` invariants.

This is an offline uncertainty gate and coverage/risk diagnostic. It is not a
conformal guarantee, a market-edge claim, a regime label, or a trading order
policy. Promotion requires a fixed baseline, multiple chronological
walk-forward/OOS windows, costs and stress tests, and review of coverage/risk
stability under drift.

## Focused validation

```powershell
uv run pytest -q tests/quant_trading/test_selective_prediction.py
uv run ruff check src/quant_trading/selective_prediction.py tests/quant_trading/test_selective_prediction.py
uv run python -m compileall -q src/quant_trading/selective_prediction.py tests/quant_trading/test_selective_prediction.py
```

The synthetic pilot receipt is stored next to this note. It is mechanics-only
and must not be read as evidence of profitability or live safety.
