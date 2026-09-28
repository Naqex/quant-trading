# Chart feature research contract r1

**Scope:** offline interoperability between the causal MT5 SMC/ICT chart engine,
Quant Lab's spot backtest, and TradingAgents' decision-quality backtest.

## Authority and reuse

- The MT5 chart engine owns the meaning and timing of chart events. It emits
  `chart-event-v1` records with `known_at`, source bar IDs, rule version and
  replay cutoff. Quant Lab does not reimplement BOS/FVG/structure detection.
- Quant Lab owns the offline contract in
  `src/quant_lab/chart_features.py`. It validates event identity, closed-bar
  next-open timing, walk-forward boundaries, holdout exclusion, modeled fee and
  slippage, and dataset provenance. It does not fetch data, call an AI
  provider, connect a broker, or authorize execution.
- MT5 `StrategyResearchSpec` and `research_oos.py` remain the authority for
  methodology metadata and executable research-job planning. This contract is
  an adapter payload, not a second engine or playbook store.
- TradingAgents remains a decision-quality evaluator, not a fill simulator. A
  future adapter may pass only events whose `known_at` is at or before the
  analysis date; its LLM/provider path must remain outside this offline packet.

## Safety invariants

1. Event IDs are unique and events are sorted by `(known_at, event_id)`.
2. `known_at` never precedes the event anchor and cannot cross a replay cutoff
   or metadata-only holdout boundary. When the upstream replay is bounded,
   `StrategyResearchContract.replay_cutoff_utc` carries that cutoff into the
   serialized payload and canonical hash, so a consumer cannot silently widen
   the event window by changing out-of-band metadata.
3. Walk-forward folds are half-open and ordered as
   `train → purge → embargo → OOS`; OOS may not reach locked holdout data.
4. Fee and slippage are finite, non-negative basis-point assumptions whose
   combined rate cannot make one unit of turnover negative.
5. `execution_timing` is fixed to `closed_bar_next_open` and
   `execution_capability` is hard-coded false.
6. The contract carries a 64-character dataset SHA-256 plus source/schema/row
   provenance whose digest must match, and exposes a deterministic canonical
   contract hash for receipts.

## Validation

Focused command from `projects/quant-trading`:

```powershell
uv run pytest -q tests/test_chart_features.py
```

The test fixture is synthetic and offline. It proves contract validation only;
it does not prove SMC/ICT edge, profitability, provider quality, broker fills,
or live-trading readiness.
