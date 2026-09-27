# Offline event kernel pilot — 2026-09-27

This bounded P0 slice adds a provider-free event envelope and two fixture
adapters to Quant Lab. It is a research/replay seam for future on-chain DEX
analytics, not a node/indexer client, trading gateway, mempool listener, or
execution path.

## Contract

- `quant_trading.event_envelope.EventEnvelope` carries `source`,
  `instrument_id`, `venue`, `event_type`, `event_time`, `known_at`,
  `available_at`, `sequence`, payload, finality, and reorg status.
- Event identity is deterministic: canonical finite JSON produces an
  `event_hash`, a stable `evt-...` event ID, and a `dedup-...` key. The
  round-trip parser rejects a tampered ID/hash.
- Causality is fail-closed: `event_time <= known_at <= available_at`.
  A replay cutoff is applied to `available_at`, so delayed data cannot leak
  into an earlier research decision.
- `collect_events` sorts deterministically, enforces a minimum finality, drops
  reorged records by default, and reports duplicate/late/reorged records in an
  explicit `EventBatch`. Duplicate and late policies can be `drop` or `error`;
  late events can also be retained for an audit stream with `keep`.
- `execution_capability` is always false. No provider, network, wallet, broker,
  order, or API key is used.

## Fixture adapters

- `OnChainEvent.from_mapping` parses a captured `onchain-event-v1` log and
  maps it to the envelope with chain/block/transaction/log identity in the
  payload. If the fixture omits sequence, block number plus log index gives a
  stable ordering sequence.
- `DexPoolSnapshot.from_mapping` parses a captured `dex-pool-snapshot-v1`
  reserve snapshot. Decimal values are normalized as strings to avoid binary
  float loss; token pair becomes the default instrument ID.
- Both adapters retain finality/reorg state and reject malformed timestamps,
  negative chain indexes/reserves, unsupported schemas, or non-finite values.

## Validation

Commands run from `projects/quant-trading`:

```powershell
uv run pytest -q tests/quant_trading/test_event_kernel.py  # 6 passed
uv run pytest -q                                             # 46 passed
uv run ruff check src/quant_trading/event_envelope.py src/quant_trading/onchain_events.py tests/quant_trading/test_event_kernel.py  # All checks passed
uv run python -m compileall -q src/quant_trading/event_envelope.py src/quant_trading/onchain_events.py tests/quant_trading/test_event_kernel.py  # pass
git diff --check -- src/quant_trading/event_envelope.py src/quant_trading/onchain_events.py tests/quant_trading/test_event_kernel.py  # pass
```

The existing `quant_trading/discrete_optimizer.py` and its untracked test are
owned by another lane and were not changed or staged by this slice.

## Deliberate P0 limits

This packet does not claim a live chain parser, ABI decoding, block reorg
reconciliation, mempool ordering, DEX pricing, MEV detection, or strategy
edge. Those require captured fixtures and a separate research/evidence gate.
The next safe extension is a fixture corpus with chain-specific finality and
reorg replacement cases, then an offline feature projection that reuses this
same envelope rather than creating a second event schema.
