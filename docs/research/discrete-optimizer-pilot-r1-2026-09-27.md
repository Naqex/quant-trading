# Discrete optimizer pilot r1

**Scope:** bounded, provider-free QUBO research primitive for Quant Lab.

## What is implemented

`src/quant_trading/discrete_optimizer.py` contains two deterministic research
solvers over binary variables:

- `solve_exact_enum` is the small-problem global oracle. It hard-limits the
  caller to 24 variables and defaults to 20, validates every feasible
  assignment, and uses a deterministic lexicographic tie-break.
- `solve_simulated_annealing` is a NumPy-only baseline with explicit seed,
  temperature, step and restart controls. It never converts hard constraints
  into a hidden penalty: every returned assignment is checked before a result
  can be emitted.

`QUBOProblem` validates finite vectors, a square symmetric quadratic matrix,
and optional `BinaryConstraint` values (`<=`, `>=`, `==`) with explicit
numerical tolerance. `ResearchCutoffs` requires strict chronological
`train_end_utc < validation_end_utc < test_end_utc` boundaries.

Every result carries a receipt with solver/version, seed, selected bits,
objective value, objective/problem hash, data SHA-256, all three temporal
cutoffs, evaluation count, and hard-coded `execution_capability: false`.
Provider and broker access are also explicitly false. The module adds no
Qiskit, D-Wave, network, exchange, or live-trading dependency.

## Receipt and validation

The JSON receipt is a synthetic three-variable fixture only; it demonstrates
the exact-enumeration oracle and is not a strategy result or profitability
claim. The objective is the canonical binary QUBO objective
`linear @ x + x.T @ quadratic @ x`.

Focused validation:

```powershell
uv run pytest -q tests/test_discrete_optimizer.py
# 6 passed
uv run ruff check src/quant_trading tests/quant_trading tests/test_discrete_optimizer.py
# pass
uv run python -m compileall -q src/quant_trading
# pass
```

The normal Quant Lab full suite remains the project-level gate. This pilot is
offline-only and does not promote a result to paper/live execution. It is a
reusable seam for later portfolio/cardinality/scheduling experiments when a
specific economic hypothesis and leakage-safe data packet exist.
