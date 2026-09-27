# Chart event edge evaluation r1

**Status:** `PREP_ONLY`

This slice makes the existing `chart-event-v1` contract usable for a bounded,
offline SMC/ICT hypothesis evaluation. It is an adapter and accounting layer;
it does not implement a second BOS/FVG/CHoCH detector and does not claim an
edge or profitability.

## What is now reusable

- `build_causal_signal(...)` projects confirmed chart events onto a spot
  long/cash signal. An event known at UTC timestamp `t` first affects the
  first supplied bar timestamp strictly greater than `t`; the event bar itself
  cannot trade on its own close.
- Tentative and invalidated events are ignored. Confirmed bullish events set a
  long target (`1.0`); confirmed bearish events exit (`0.0`). Equal-time events
  are applied in deterministic `(known_at, event_id)` order.
- The adapter reuses `validate_chart_events(...)`, including duplicate ID,
  replay cutoff and locked holdout checks. Optional instrument and timeframe
  bindings reject cross-series contamination.
- `run_backtest(...)` remains the only fill/accounting path and applies
  next-bar execution, fee, slippage and spot bounds. No leverage, shorting,
  broker, provider or network path is introduced.
- `performance_metrics(...)` now reports downside deviation, zero-rate
  Sortino, profit factor, 95% historical expected shortfall, worst-period
  return and maximum consecutive losing periods in addition to the existing
  CAGR, volatility, Sharpe, drawdown, Calmar and turnover fields.

## Validation

```text
uv run pytest -q tests/test_chart_features.py tests/test_backtest.py tests/test_research_provenance.py
24 passed

uv run pytest -q
36 passed

uv run ruff check src tests
All checks passed

uv run python -m compileall -q src
PASS
```

The fixtures are synthetic and only verify timing, accounting and contract
invariants. They do not demonstrate that SMC/ICT events predict returns.

## Deliberate limits and next gate

The next research gate is a pre-registered, multi-asset walk-forward run that
keeps the holdout locked, compares buy-and-hold, reports cost sensitivity and
records the exact event/data digests. It must be reviewed as research evidence
before any paper-trading promotion. No live execution permission is implied by
this receipt.
