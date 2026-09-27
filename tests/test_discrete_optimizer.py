from __future__ import annotations

import numpy as np
import pytest

from quant_trading.discrete_optimizer import (
    BinaryConstraint,
    DiscreteOptimizerError,
    QUBOProblem,
    ResearchCutoffs,
    solve_exact_enum,
    solve_simulated_annealing,
)

DATA_SHA = "a" * 64
CUTOFFS = ResearchCutoffs(train_end_utc=100, validation_end_utc=200, test_end_utc=300)


def test_exact_enum_matches_global_oracle_and_receipt_is_offline_only() -> None:
    problem = QUBOProblem(
        linear=(-3.0, -2.0, 0.5),
        quadratic=((0.0, 0.5, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
        problem_id="toy-selection",
    )

    result = solve_exact_enum(problem, data_sha256=DATA_SHA, cutoffs=CUTOFFS)

    assert result.bits == (1, 1, 0)
    assert result.objective_value == pytest.approx(-4.0)
    receipt = result.receipt.as_dict()
    assert receipt["solver_version"] == "numpy-exact-enum-v1"
    assert receipt["objective_sha256"] == receipt["problem_sha256"]
    assert receipt["data_sha256"] == DATA_SHA
    assert receipt["cutoffs"] == {
        "train_end_utc": 100,
        "validation_end_utc": 200,
        "test_end_utc": 300,
    }
    assert receipt["execution_capability"] is False
    assert receipt["provider_access"] is False
    assert receipt["broker_access"] is False


def test_constraints_are_hard_and_sa_is_seed_deterministic() -> None:
    problem = QUBOProblem(
        linear=(-4.0, -3.0, -1.0, 0.0),
        quadratic=np.zeros((4, 4)),
        constraints=(BinaryConstraint((1.0, 1.0, 1.0, 1.0), "==", 1.0),),
    )

    exact = solve_exact_enum(problem, data_sha256=DATA_SHA, cutoffs=CUTOFFS)
    first = solve_simulated_annealing(
        problem,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
        seed=11,
        steps=300,
        restarts=3,
    )
    second = solve_simulated_annealing(
        problem,
        data_sha256=DATA_SHA,
        cutoffs=CUTOFFS,
        seed=11,
        steps=300,
        restarts=3,
    )

    assert exact.bits == (1, 0, 0, 0)
    assert first.bits == second.bits
    assert first.objective_value == second.objective_value
    assert sum(first.bits) == 1
    assert problem.is_feasible(first.bits)
    assert first.receipt.solver_version == "numpy-simulated-annealing-v1"


def test_validation_rejects_bad_shapes_numbers_cutoffs_and_provenance() -> None:
    with pytest.raises(DiscreteOptimizerError, match="symmetric"):
        QUBOProblem((1.0, 2.0), ((0.0, 1.0), (0.0, 0.0)))
    with pytest.raises(DiscreteOptimizerError, match="finite"):
        QUBOProblem((float("nan"),), ((0.0,),))
    with pytest.raises(DiscreteOptimizerError, match="coefficients"):
        QUBOProblem((1.0, 2.0), ((0.0, 0.0), (0.0, 0.0)), (BinaryConstraint((1.0,), "<=", 1.0),))
    with pytest.raises(DiscreteOptimizerError, match="train < validation < test"):
        ResearchCutoffs(10, 10, 20)
    with pytest.raises(DiscreteOptimizerError, match="lowercase"):
        solve_exact_enum(QUBOProblem((1.0,), ((0.0,),)), data_sha256="A" * 64, cutoffs=CUTOFFS)
    with pytest.raises(DiscreteOptimizerError, match="max_variables"):
        solve_exact_enum(
            QUBOProblem((1.0, 1.0), ((0.0, 0.0), (0.0, 0.0))),
            data_sha256=DATA_SHA,
            cutoffs=CUTOFFS,
            max_variables=1,
        )


def test_no_feasible_assignment_and_unsafe_sa_parameters_fail_closed() -> None:
    problem = QUBOProblem((0.0,), ((0.0,),), (BinaryConstraint((0.0,), "==", 1.0),))
    with pytest.raises(DiscreteOptimizerError, match="no feasible"):
        solve_exact_enum(problem, data_sha256=DATA_SHA, cutoffs=CUTOFFS)
    with pytest.raises(DiscreteOptimizerError, match="temperatures"):
        solve_simulated_annealing(
            QUBOProblem((1.0,), ((0.0,),)),
            data_sha256=DATA_SHA,
            cutoffs=CUTOFFS,
            initial_temperature=0,
        )
    with pytest.raises(DiscreteOptimizerError, match="final_temperature"):
        solve_simulated_annealing(
            QUBOProblem((1.0,), ((0.0,),)),
            data_sha256=DATA_SHA,
            cutoffs=CUTOFFS,
            initial_temperature=1,
            final_temperature=2,
        )
