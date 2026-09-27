# CUSUM change-detection pilot (R1, 2026-09-27)

## Decision

Adopt a small **offline diagnostic primitive** for change-point research. The
implementation is a deterministic two-sided CUSUM with an explicit baseline,
drift allowance, threshold, and reset policy. It is a regime/quality-control
feature only: it does not produce a position, a trade signal, a forecast, or
execution authority.

The classical reference is Page's cumulative-sum procedure for sequential
analysis ([Page, 1954, *Biometrika*](https://doi.org/10.1093/biomet/41.1-2.100)).
This pilot deliberately keeps the implementation narrower than a production
sequential test: no estimated baseline, no p-value claim, no automatic
calibration, and no Bayesian online change-point model yet.

## Contract

- Module: `src/quant_trading/change_detection.py`
- Public entry point: `run_cusum(observations, config, cutoff_timestamp=...)`
- Observation contract: reuse `state_primitives.CausalObservation` or an
  equivalent mapping with finite `event_time`, `known_at`, `value`, and
  optional non-negative `sequence`.
- Processing order: `(known_at, sequence, event_time)`; duplicate
  `(known_at, sequence)` keys are rejected.
- Causality: `event_time <= known_at`; a row whose `known_at` is after the
  cutoff is rejected instead of silently dropped.
- Parameters: `baseline` and `threshold > 0` are mandatory; `drift >= 0` and
  `reset_on_detection` are explicit.
- Numerical guards: all inputs and intermediate statistics must be finite;
  negative/positive accumulators are bounded at zero in their respective
  directions.
- Security boundary: `execution_capability`, `provider_access`, and
  `broker_access` remain `false`; no network, wallet, broker, or model call.

For an observation deviation `d = value - baseline`, the detector updates:

```text
S+ = max(0, S+ + d - drift)
S- = min(0, S- + d + drift)
```

A row is marked `positive`, `negative`, or `both` when the relevant statistic
reaches `threshold` in magnitude. By default both accumulators reset after a
flag so one persistent shift does not produce a flag on every row. The result
retains the pre-reset statistics for auditability.

## Deterministic fixture

Fixture values `[0, 2, 2, 0]`, baseline `0`, threshold `3`, drift `0`, and a
cutoff at timestamp `4` produce one positive diagnostic crossing on the third
row. The result is **inconclusive** as a research hypothesis: one synthetic
fixture does not establish a trading edge or a robust regime detector.

Receipt: `cusum-change-detection-pilot-r1-2026-09-27.json`

- data hash: `98bfdd0e866ff1a589d7c04f97d6ec5fca3ca48ddf97e996cccc15dabb0a1705`
- model/config hash: `b6c26ecf1065ac3135c133055a4124ed2f40918b7b0213fd1609c15089350319`
- result hash: `f27609ada2949b50ae3907acc43f6ec934a07a27008e6a9c017f7a491f9959b3`
- receipt hash: `919a58b043d5c2cf32f47e4366cd769ad859e88e05a15f4460a3a9758b33b9eb`

## Validation and promotion gate

Focused tests cover constant baseline, positive and negative shifts, drift,
reset behavior, cutoff rejection, non-finite input, known-time ordering,
duplicate keys, empty input, invalid parameters, and prevention of a future
event being smuggled through `known_at`.

Before considering promotion beyond a diagnostic feature, use fixed train/
validation/test cutoffs, compare against a no-change and volatility/change
baseline, and evaluate detection delay, false-alarm rate, stability across
assets/regimes, and sensitivity to baseline/threshold choices. A future
Bayesian online change-point detector may be evaluated only under the same
causal receipt and out-of-sample gates.
