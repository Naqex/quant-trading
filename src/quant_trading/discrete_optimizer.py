"""Small, auditable binary QUBO optimizers for offline research.

This module deliberately stops at a bounded research primitive.  It does not
fetch data, call a quantum service, or grant trading/execution capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

OPTIMIZER_SCHEMA = "discrete-optimizer-receipt-v1"
_EXACT_SOLVER_VERSION = "numpy-exact-enum-v1"
_SA_SOLVER_VERSION = "numpy-simulated-annealing-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HARD_EXACT_MAX_VARIABLES = 24


class DiscreteOptimizerError(ValueError):
    """Raised when a bounded optimizer contract is malformed or unsatisfiable."""


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DiscreteOptimizerError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise DiscreteOptimizerError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DiscreteOptimizerError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DiscreteOptimizerError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise DiscreteOptimizerError(f"{name} must be finite")
    return result


def _canonical_json(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DiscreteOptimizerError("contract payload must be finite JSON data") from exc


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DiscreteOptimizerError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _numeric_vector(value: Sequence[float] | np.ndarray, name: str) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DiscreteOptimizerError(f"{name} must be a one-dimensional numeric sequence") from exc
    if array.ndim != 1 or array.size == 0:
        raise DiscreteOptimizerError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise DiscreteOptimizerError(f"{name} must contain only finite values")
    return tuple(float(item) for item in array.tolist())


@dataclass(frozen=True)
class BinaryConstraint:
    """A finite-tolerance linear constraint over binary variables."""

    coefficients: Sequence[float] | np.ndarray
    relation: str
    rhs: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        coefficients = _numeric_vector(self.coefficients, "constraint.coefficients")
        if self.relation not in {"<=", ">=", "=="}:
            raise DiscreteOptimizerError("constraint.relation must be one of <=, >=, ==")
        rhs = _finite_float(self.rhs, "constraint.rhs")
        tolerance = _finite_float(self.tolerance, "constraint.tolerance")
        if tolerance < 0:
            raise DiscreteOptimizerError("constraint.tolerance must be >= 0")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "tolerance", tolerance)

    def __post_init_shape__(self, variables: int) -> None:
        if len(self.coefficients) != variables:
            raise DiscreteOptimizerError("constraint.coefficients must match the number of variables")

    def lhs(self, bits: Sequence[int] | np.ndarray) -> float:
        array = _validate_bits(bits, len(self.coefficients))
        return _safe_dot(self.coefficients, array, "constraint lhs")

    def is_satisfied(self, bits: Sequence[int] | np.ndarray) -> bool:
        value = self.lhs(bits)
        if self.relation == "<=":
            return value <= self.rhs + self.tolerance
        if self.relation == ">=":
            return value >= self.rhs - self.tolerance
        return abs(value - self.rhs) <= self.tolerance

    def as_dict(self) -> dict[str, Any]:
        return {
            "coefficients": list(self.coefficients),
            "relation": self.relation,
            "rhs": self.rhs,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class QUBOProblem:
    """A symmetric binary quadratic objective plus optional hard constraints.

    The objective is ``linear @ x + x.T @ quadratic @ x``.  Hard constraints
    are checked separately and are never silently converted into penalties.
    """

    linear: Sequence[float] | np.ndarray
    quadratic: Sequence[Sequence[float]] | np.ndarray
    constraints: tuple[BinaryConstraint, ...] = ()
    problem_id: str = "qubo"

    def __post_init__(self) -> None:
        linear = _numeric_vector(self.linear, "linear")
        try:
            quadratic_array = np.asarray(self.quadratic, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DiscreteOptimizerError("quadratic must be a square numeric matrix") from exc
        variables = len(linear)
        if quadratic_array.shape != (variables, variables):
            raise DiscreteOptimizerError("quadratic must be square and match linear")
        if not np.all(np.isfinite(quadratic_array)):
            raise DiscreteOptimizerError("quadratic must contain only finite values")
        if not np.allclose(quadratic_array, quadratic_array.T, rtol=0.0, atol=1e-12):
            raise DiscreteOptimizerError("quadratic must be symmetric")
        if not isinstance(self.constraints, tuple):
            try:
                constraints = tuple(self.constraints)
            except TypeError as exc:
                raise DiscreteOptimizerError("constraints must be iterable") from exc
        else:
            constraints = self.constraints
        for constraint in constraints:
            if not isinstance(constraint, BinaryConstraint):
                raise DiscreteOptimizerError("constraints must contain BinaryConstraint values")
            constraint.__post_init_shape__(variables)
        if not isinstance(self.problem_id, str):
            raise DiscreteOptimizerError("problem_id must be a string")
        problem_id = self.problem_id.strip()
        if not problem_id or len(problem_id) > 128:
            raise DiscreteOptimizerError("problem_id must be non-empty and at most 128 characters")
        object.__setattr__(self, "linear", linear)
        object.__setattr__(
            self, "quadratic", tuple(tuple(float(item) for item in row) for row in quadratic_array.tolist())
        )
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "problem_id", problem_id)

    @property
    def variables(self) -> int:
        return len(self.linear)

    def objective(self, bits: Sequence[int] | np.ndarray) -> float:
        array = _validate_bits(bits, self.variables)
        quadratic = np.asarray(self.quadratic, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            try:
                value = float(np.dot(np.asarray(self.linear, dtype=np.float64), array))
                value += float(np.dot(array, np.dot(quadratic, array)))
            except (FloatingPointError, ValueError) as exc:
                raise DiscreteOptimizerError("objective overflowed or is invalid") from exc
        if not math.isfinite(value):
            raise DiscreteOptimizerError("objective overflowed or is invalid")
        return value

    def is_feasible(self, bits: Sequence[int] | np.ndarray) -> bool:
        _validate_bits(bits, self.variables)
        return all(constraint.is_satisfied(bits) for constraint in self.constraints)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "binary-qubo-v1",
            "problem_id": self.problem_id,
            "linear": list(self.linear),
            "quadratic": [list(row) for row in self.quadratic],
            "constraints": [constraint.as_dict() for constraint in self.constraints],
        }

    def objective_sha256(self) -> str:
        return _sha256_json(self.as_dict())


def _validate_bits(bits: Sequence[int] | np.ndarray, variables: int) -> np.ndarray:
    try:
        array = np.asarray(bits)
    except (TypeError, ValueError) as exc:
        raise DiscreteOptimizerError("bits must be a binary vector") from exc
    if array.ndim != 1 or array.size != variables:
        raise DiscreteOptimizerError("bits must match the number of variables")
    if np.iscomplexobj(array):
        raise DiscreteOptimizerError("bits must contain finite binary values")
    try:
        numeric = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise DiscreteOptimizerError("bits must contain finite binary values") from exc
    if not np.all(np.isfinite(numeric)):
        raise DiscreteOptimizerError("bits must contain finite binary values")
    if not np.all((numeric == 0) | (numeric == 1)):
        raise DiscreteOptimizerError("bits must contain only 0 or 1")
    return numeric


def _safe_dot(left: Sequence[float], right: np.ndarray, label: str) -> float:
    with np.errstate(over="raise", invalid="raise"):
        try:
            value = float(np.dot(np.asarray(left, dtype=np.float64), right))
        except (FloatingPointError, ValueError) as exc:
            raise DiscreteOptimizerError(f"{label} overflowed or is invalid") from exc
    if not math.isfinite(value):
        raise DiscreteOptimizerError(f"{label} overflowed or is invalid")
    return value


@dataclass(frozen=True)
class ResearchCutoffs:
    """Strict chronological train/validation/test observation cutoffs."""

    train_end_utc: int
    validation_end_utc: int
    test_end_utc: int

    def __post_init__(self) -> None:
        train = _strict_int(self.train_end_utc, "train_end_utc", minimum=0)
        validation = _strict_int(self.validation_end_utc, "validation_end_utc", minimum=0)
        test = _strict_int(self.test_end_utc, "test_end_utc", minimum=0)
        if not train < validation < test:
            raise DiscreteOptimizerError("cutoffs must satisfy train < validation < test")
        object.__setattr__(self, "train_end_utc", train)
        object.__setattr__(self, "validation_end_utc", validation)
        object.__setattr__(self, "test_end_utc", test)

    def as_dict(self) -> dict[str, int]:
        return {
            "train_end_utc": self.train_end_utc,
            "validation_end_utc": self.validation_end_utc,
            "test_end_utc": self.test_end_utc,
        }


@dataclass(frozen=True)
class OptimizationReceipt:
    """Auditable, offline-only result metadata."""

    solver: str
    solver_version: str
    method: str
    seed: int
    selected_bits: tuple[int, ...]
    objective_value: float
    objective_sha256: str
    data_sha256: str
    cutoffs: ResearchCutoffs
    feasible: bool
    evaluations: int
    execution_capability: bool = False

    def __post_init__(self) -> None:
        if not self.solver or not self.solver_version or not self.method:
            raise DiscreteOptimizerError("solver metadata must be non-empty")
        _strict_int(self.seed, "seed", minimum=0)
        if not self.selected_bits:
            raise DiscreteOptimizerError("selected_bits must be non-empty")
        if any(bit not in (0, 1) for bit in self.selected_bits):
            raise DiscreteOptimizerError("selected_bits must be binary")
        _finite_float(self.objective_value, "objective_value")
        _validate_sha256(self.objective_sha256, "objective_sha256")
        _validate_sha256(self.data_sha256, "data_sha256")
        if not isinstance(self.cutoffs, ResearchCutoffs):
            raise DiscreteOptimizerError("cutoffs must be ResearchCutoffs")
        _strict_int(self.evaluations, "evaluations", minimum=1)
        if self.feasible is not True:
            raise DiscreteOptimizerError("optimizer receipts must contain a feasible result")
        if self.execution_capability is not False:
            raise DiscreteOptimizerError("discrete optimizer cannot grant execution capability")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OPTIMIZER_SCHEMA,
            "status": "PREP_ONLY",
            "solver": self.solver,
            "solver_version": self.solver_version,
            "method": self.method,
            "seed": self.seed,
            "selected_bits": list(self.selected_bits),
            "objective_value": self.objective_value,
            "objective_sha256": self.objective_sha256,
            "problem_sha256": self.objective_sha256,
            "data_sha256": self.data_sha256,
            "cutoffs": self.cutoffs.as_dict(),
            "feasible": self.feasible,
            "evaluations": self.evaluations,
            "execution_capability": self.execution_capability,
            "provider_access": False,
            "broker_access": False,
        }


@dataclass(frozen=True)
class OptimizationResult:
    bits: tuple[int, ...]
    objective_value: float
    receipt: OptimizationReceipt

    def as_dict(self) -> dict[str, Any]:
        return self.receipt.as_dict()


def _validate_context(data_sha256: str, cutoffs: ResearchCutoffs) -> None:
    _validate_sha256(data_sha256, "data_sha256")
    if not isinstance(cutoffs, ResearchCutoffs):
        raise DiscreteOptimizerError("cutoffs must be ResearchCutoffs")


def _result(
    *,
    problem: QUBOProblem,
    bits: tuple[int, ...],
    data_sha256: str,
    cutoffs: ResearchCutoffs,
    solver: str,
    solver_version: str,
    method: str,
    seed: int,
    evaluations: int,
) -> OptimizationResult:
    if not problem.is_feasible(bits):
        raise DiscreteOptimizerError("optimizer returned an infeasible result")
    objective_value = problem.objective(bits)
    receipt = OptimizationReceipt(
        solver=solver,
        solver_version=solver_version,
        method=method,
        seed=seed,
        selected_bits=bits,
        objective_value=objective_value,
        objective_sha256=problem.objective_sha256(),
        data_sha256=data_sha256,
        cutoffs=cutoffs,
        feasible=True,
        evaluations=evaluations,
    )
    return OptimizationResult(bits=bits, objective_value=objective_value, receipt=receipt)


def solve_exact_enum(
    problem: QUBOProblem,
    *,
    data_sha256: str,
    cutoffs: ResearchCutoffs,
    max_variables: int = 20,
) -> OptimizationResult:
    """Find the deterministic global optimum for a deliberately small QUBO."""

    if not isinstance(problem, QUBOProblem):
        raise DiscreteOptimizerError("problem must be QUBOProblem")
    _validate_context(data_sha256, cutoffs)
    max_variables = _strict_int(max_variables, "max_variables", minimum=1)
    if max_variables > _HARD_EXACT_MAX_VARIABLES:
        raise DiscreteOptimizerError("max_variables exceeds the hard exact-enumeration bound")
    if problem.variables > max_variables:
        raise DiscreteOptimizerError("exact enumeration is bounded by max_variables")
    best_bits: tuple[int, ...] | None = None
    best_value = math.inf
    evaluations = 0
    for mask in range(1 << problem.variables):
        bits = tuple((mask >> index) & 1 for index in range(problem.variables))
        evaluations += 1
        if not problem.is_feasible(bits):
            continue
        value = problem.objective(bits)
        if value < best_value or (value == best_value and (best_bits is None or bits < best_bits)):
            best_value = value
            best_bits = bits
    if best_bits is None:
        raise DiscreteOptimizerError("QUBO has no feasible binary assignment")
    return _result(
        problem=problem,
        bits=best_bits,
        data_sha256=data_sha256,
        cutoffs=cutoffs,
        solver="numpy",
        solver_version=_EXACT_SOLVER_VERSION,
        method="exact_enum",
        seed=0,
        evaluations=evaluations,
    )


def _random_feasible_start(problem: QUBOProblem, rng: np.random.Generator) -> tuple[int, ...]:
    zero = tuple(0 for _ in range(problem.variables))
    if problem.is_feasible(zero):
        return zero
    for _ in range(4096):
        candidate = tuple(int(value) for value in rng.integers(0, 2, size=problem.variables))
        if problem.is_feasible(candidate):
            return candidate
    if problem.variables <= 20:
        for mask in range(1 << problem.variables):
            candidate = tuple((mask >> index) & 1 for index in range(problem.variables))
            if problem.is_feasible(candidate):
                return candidate
    raise DiscreteOptimizerError(
        "could not find a feasible binary assignment within bounded initialization search"
    )


def _neighbor(
    problem: QUBOProblem, current: tuple[int, ...], rng: np.random.Generator
) -> tuple[int, ...] | None:
    candidates: list[tuple[int, ...]] = []
    for index in rng.permutation(problem.variables):
        candidate = list(current)
        candidate[int(index)] = 1 - candidate[int(index)]
        candidates.append(tuple(candidate))
    for _ in range(32):
        candidates.append(tuple(int(value) for value in rng.integers(0, 2, size=problem.variables)))
    for candidate in candidates:
        if problem.is_feasible(candidate):
            return candidate
    return None


def solve_simulated_annealing(
    problem: QUBOProblem,
    *,
    data_sha256: str,
    cutoffs: ResearchCutoffs,
    seed: int = 0,
    steps: int = 2_000,
    restarts: int = 4,
    initial_temperature: float = 1.0,
    final_temperature: float = 1e-3,
) -> OptimizationResult:
    """Run a deterministic NumPy-only simulated-annealing research baseline."""

    if not isinstance(problem, QUBOProblem):
        raise DiscreteOptimizerError("problem must be QUBOProblem")
    _validate_context(data_sha256, cutoffs)
    seed = _strict_int(seed, "seed", minimum=0)
    steps = _strict_int(steps, "steps", minimum=1)
    restarts = _strict_int(restarts, "restarts", minimum=1)
    initial_temperature = _finite_float(initial_temperature, "initial_temperature")
    final_temperature = _finite_float(final_temperature, "final_temperature")
    if initial_temperature <= 0 or final_temperature <= 0:
        raise DiscreteOptimizerError("temperatures must be > 0")
    if final_temperature > initial_temperature:
        raise DiscreteOptimizerError("final_temperature must be <= initial_temperature")

    rng = np.random.default_rng(seed)
    log_initial_temperature = math.log(initial_temperature)
    log_final_temperature = math.log(final_temperature)
    best_bits: tuple[int, ...] | None = None
    best_value = math.inf
    evaluations = 0
    for _ in range(restarts):
        current = _random_feasible_start(problem, rng)
        current_value = problem.objective(current)
        evaluations += 1
        if current_value < best_value or (
            current_value == best_value and (best_bits is None or current < best_bits)
        ):
            best_bits, best_value = current, current_value
        for step in range(steps):
            candidate = _neighbor(problem, current, rng)
            if candidate is None:
                continue
            candidate_value = problem.objective(candidate)
            evaluations += 1
            delta = candidate_value - current_value
            if steps == 1:
                temperature = initial_temperature
            else:
                fraction = step / (steps - 1)
                # Log-space interpolation keeps every temperature positive even
                # when the endpoint ratio underflows in float64.
                temperature = math.exp(
                    log_initial_temperature + (log_final_temperature - log_initial_temperature) * fraction
                )
            accept = delta <= 0
            if not accept:
                exponent = -delta / temperature
                accept = exponent >= 0 or (exponent > -745 and float(rng.random()) < math.exp(exponent))
            if accept:
                current, current_value = candidate, candidate_value
                if current_value < best_value or (
                    current_value == best_value and (best_bits is None or current < best_bits)
                ):
                    best_bits, best_value = current, current_value
    if best_bits is None:
        raise DiscreteOptimizerError(
            "could not find a feasible binary assignment within bounded initialization search"
        )
    return _result(
        problem=problem,
        bits=best_bits,
        data_sha256=data_sha256,
        cutoffs=cutoffs,
        solver="numpy",
        solver_version=_SA_SOLVER_VERSION,
        method="simulated_annealing",
        seed=seed,
        evaluations=evaluations,
    )


__all__ = [
    "BinaryConstraint",
    "DiscreteOptimizerError",
    "OptimizationReceipt",
    "OptimizationResult",
    "QUBOProblem",
    "ResearchCutoffs",
    "OPTIMIZER_SCHEMA",
    "solve_exact_enum",
    "solve_simulated_annealing",
]
