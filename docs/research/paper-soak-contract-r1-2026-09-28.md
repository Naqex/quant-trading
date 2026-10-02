# Paper-soak contract r1 — 2026-09-28

`quant_trading.paper_soak` adds the typed offline boundary for the future
owner-absence paper run. It is deliberately separate from strategy selection,
broker connectivity and live execution.

## What it checks

- each UTC day has a canonical date, point-in-time input cutoff, accounting
  counts, finite VND amounts, non-negative fee/slippage and content hashes;
- a mismatch or unhealthy day must carry a pause/stop reason;
- dates are unique and sorted, and the evidence window cannot exceed 60 days;
- point-in-time input cutoffs advance strictly with the ordered UTC days, so a
  later observation cannot silently reuse an earlier information boundary;
- restart observations must report zero duplicate intents before they can be
  marked passed;
- a receipt marked `ACCEPTED` requires at least 30 days, all days healthy and
  reconciled, at least one passed restart check, and explicit owner review;
- canonical JSON hashing makes the receipt tamper-evident and round-trippable.

The mapping loader validates any supplied mode and capability fields through
the same constructor boundary. It rejects live mode and every capability
value other than literal `false`, rather than discarding them and applying
safe defaults. A supplied `window` must match the dates/count of the actual
observations and the fixed 30–60 day policy; an unchanged receipt hash cannot
endorse a modified summary. Legacy mappings that omit those fields remain
readable with paper-only, capability-free defaults.

Every receipt is `PREP_ONLY_OFFLINE`. `promotion_allowed`,
`execution_capability`, provider, broker and OAuth access are literal `false`,
including for an `ACCEPTED` observation. A paper result therefore cannot widen
permissions or silently become a live-trading decision.

## Example

```python
from quant_trading.paper_soak import PaperSoakReceipt

receipt = PaperSoakReceipt()  # honest NOT_RUN template
assert receipt.observed_days == 0
assert receipt.promotion_allowed is False
```

The contract does not run a 30–60 day forward observation, persist a host
lease, deliver alerts, or restore PostgreSQL/artifact roots. Those remain
external/operations gates in the departure audit. A future runner should emit
one `PaperSoakDay` per UTC date and one `PaperSoakRestartCheck` per recovery
exercise, then pass the receipt through the same hash and owner-review gate.

## Validation

```powershell
uv run pytest -q tests/quant_trading/test_paper_soak.py
uv run ruff check src/quant_trading/paper_soak.py tests/quant_trading/test_paper_soak.py
```

Both checks pass for this slice. No network, provider, broker, OAuth, wallet,
holdout or Job12 process was touched.

### Mapping validation repair — 2026-10-02

The loader previously ignored supplied mode/capabilities and the window
summary. A template carrying `mode="live"`, `execution_capability=true`, or
`observed_days=30` was silently reloaded as an unrun, safe template. The
regressions first reproduced 27 failures, then passed after the repair:
35 focused paper-soak cases and all 257 project tests passed. Ruff over
`src tests` passed. This verifies offline serialization only; the 30–60 day
forward observation and host-supervisor/restore/alert gates remain unrun.
