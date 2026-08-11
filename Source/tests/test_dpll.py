from itertools import product
from pathlib import Path

import pytest

from core.game_engine import GameEngine
from core.models import Status
from core.puzzle_loader import load_puzzle
from logic.cnf_encoder import CNFEncoder
from logic.dpll import (
    DPLLSolver,
    InvalidAssumptionError,
    InvalidCNFError,
)


# ============================================================
# Helpers
# ============================================================

def formula_is_satisfied(
    clauses: list[list[int]],
    assignment: dict[int, bool],
) -> bool:
    """
    Evaluate a CNF formula under a complete Boolean assignment.
    """
    for clause in clauses:
        clause_satisfied = False

        for literal in clause:
            variable = abs(literal)
            value = assignment[variable]

            if literal > 0 and value:
                clause_satisfied = True
                break

            if literal < 0 and not value:
                clause_satisfied = True
                break

        if not clause_satisfied:
            return False

    return True


def brute_force_is_sat(
    clauses: list[list[int]],
    num_variables: int,
) -> bool:
    """
    Small reference SAT checker used only by tests.

    Exhaustively enumerates all Boolean assignments, so it is
    appropriate only for very small formulas.
    """
    for values in product(
        [False, True],
        repeat=num_variables,
    ):
        assignment = {
            variable: values[variable - 1]
            for variable in range(
                1,
                num_variables + 1,
            )
        }

        if formula_is_satisfied(
            clauses,
            assignment,
        ):
            return True

    return False


def puzzle_3x3_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "puzzles"
        / "puzzle_3x3_01.json"
    )


# ============================================================
# Basic SAT / UNSAT
# ============================================================

def test_simple_sat_formula():
    """
    A AND B

    Unique satisfying assignment:

        A = True
        B = True
    """
    solver = DPLLSolver()

    clauses = [
        [1],
        [2],
    ]

    result = solver.solve(
        clauses,
        num_variables=2,
    )

    assert result.satisfiable is True
    assert result.assignment is not None

    assert result.assignment == {
        1: True,
        2: True,
    }

    assert formula_is_satisfied(
        clauses,
        result.assignment,
    )


def test_simple_unsat_formula():
    """
    A AND NOT A
    """
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[
            [1],
            [-1],
        ],
        num_variables=1,
    )

    assert result.satisfiable is False
    assert result.assignment is None


# ============================================================
# Unit propagation
# ============================================================

def test_unit_propagation_chain():
    """
    CNF:

        A
        (NOT A OR B)
        (NOT B OR C)

    Unit propagation derives:

        A = True
        B = True
        C = True

    without branching.
    """
    solver = DPLLSolver()

    clauses = [
        [1],
        [-1, 2],
        [-2, 3],
    ]

    result = solver.solve(
        clauses,
        num_variables=3,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: True,
        2: True,
        3: True,
    }

    assert result.decisions == 0
    assert result.propagations == 3
    assert result.backtracks == 0


def test_unit_propagation_detects_conflict():
    """
    A
    (NOT A OR B)
    NOT B

    is inconsistent.
    """
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[
            [1],
            [-1, 2],
            [-2],
        ],
        num_variables=2,
    )

    assert result.satisfiable is False
    assert result.assignment is None

    assert result.decisions == 0


# ============================================================
# Pure-literal elimination
# ============================================================

def test_positive_pure_literal_eliminates_branching():
    """
    CNF:

        (A OR B)
        (NOT A OR B)

    B occurs only positively.

    Therefore B is a positive pure literal and DPLL can set:

        B = True

    without branching on A.
    """
    solver = DPLLSolver()

    clauses = [
        [1, 2],
        [-1, 2],
    ]

    result = solver.solve(
        clauses,
        num_variables=2,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: False,
        2: True,
    }

    assert result.decisions == 0
    assert result.propagations == 1
    assert result.backtracks == 0

    assert formula_is_satisfied(
        clauses,
        result.assignment,
    )


def test_negative_pure_literal_eliminates_branching():
    """
    CNF:

        (A OR NOT B)
        (NOT A OR NOT B)

    B occurs only negatively.

    Therefore:

        B = False

    satisfies both clauses without branching.
    """
    solver = DPLLSolver()

    clauses = [
        [1, -2],
        [-1, -2],
    ]

    result = solver.solve(
        clauses,
        num_variables=2,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: False,
        2: False,
    }

    assert result.decisions == 0
    assert result.propagations == 1
    assert result.backtracks == 0

    assert formula_is_satisfied(
        clauses,
        result.assignment,
    )


def test_pure_literal_elimination_reaches_fixed_point():
    """
    Initially:

        A is pure positive.

        B and C are not pure.

    CNF:

        (A OR B)
        (A OR NOT B)
        (B OR C)
        (B OR NOT C)

    First pure-literal step:

        A = True

    The first two clauses become satisfied.

    In the remaining unresolved clauses B now occurs only
    positively, so a second simplification round derives:

        B = True

    This verifies repeated pure-literal elimination until a
    fixed point.
    """
    solver = DPLLSolver()

    clauses = [
        [1, 2],
        [1, -2],
        [2, 3],
        [2, -3],
    ]

    result = solver.solve(
        clauses,
        num_variables=3,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: True,
        2: True,
        3: False,
    }

    assert result.decisions == 0

    # A and B are both assigned automatically.
    assert result.propagations == 2

    assert result.backtracks == 0

    assert formula_is_satisfied(
        clauses,
        result.assignment,
    )


def test_pure_literal_analysis_ignores_satisfied_clauses():
    """
    Under assumption:

        A = True

    clause:

        (A OR NOT B)

    is already satisfied and must no longer count the negative
    occurrence of B.

    Remaining unresolved clauses:

        (B OR C)
        (B OR NOT C)

    make B pure positive.

    Therefore B should be assigned automatically without a
    branching decision.
    """
    solver = DPLLSolver()

    clauses = [
        [1, -2],
        [2, 3],
        [2, -3],
    ]

    result = solver.solve(
        clauses,
        num_variables=3,
        assumptions=[
            1,
        ],
    )

    assert result.satisfiable is True
    assert result.assignment is not None

    assert result.assignment[1] is True
    assert result.assignment[2] is True

    assert result.decisions == 0

    # Assumption A=True is not counted.
    # Only pure-literal B=True is a propagation.
    assert result.propagations == 1

    assert formula_is_satisfied(
        clauses,
        result.assignment,
    )


def test_unit_propagation_can_enable_pure_literal_elimination():
    """
    CNF:

        A
        (A OR NOT B)
        (B OR C)
        (B OR NOT C)

    Unit propagation first derives:

        A = True

    This satisfies the second clause.

    B then becomes pure positive in the unresolved clauses and
    can be assigned automatically.
    """
    solver = DPLLSolver()

    clauses = [
        [1],
        [1, -2],
        [2, 3],
        [2, -3],
    ]

    result = solver.solve(
        clauses,
        num_variables=3,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: True,
        2: True,
        3: False,
    }

    assert result.decisions == 0

    # A by unit propagation + B by pure-literal elimination.
    assert result.propagations == 2

    assert result.backtracks == 0


# ============================================================
# Deterministic branching
# ============================================================

def test_branching_uses_smallest_unassigned_variable():
    """
    CNF:

        (A OR B)
        (NOT A OR B)
        (A OR NOT B)

    There are:

        - no unit clauses
        - no pure literals

    Both A and B occur positively and negatively.

    Therefore simplification cannot solve the formula and DPLL
    must branch.

    Deterministic selection must choose variable 1 (A) first,
    and deterministic branch order tries:

        A = True

    first.

    That forces:

        B = True

    and satisfies the formula.
    """
    solver = DPLLSolver()

    clauses = [
        [1, 2],
        [-1, 2],
        [1, -2],
    ]

    result = solver.solve(
        clauses,
        num_variables=2,
    )

    assert result.satisfiable is True
    assert result.assignment is not None

    assert result.assignment[1] is True
    assert result.assignment[2] is True

    assert result.decisions == 1
    assert result.backtracks == 0

    assert formula_is_satisfied(
        clauses,
        result.assignment,
    )


# ============================================================
# Backtracking
# ============================================================

def test_backtracking_after_failed_true_branch():
    """
    CNF:

        (A OR B)
        (NOT A OR B)
        (NOT A OR NOT B)

    Initially there are no unit clauses or pure literals.

    DPLL chooses A first.

    A=True leads to a conflict.

    DPLL must backtrack and try:

        A=False

    which gives a satisfying assignment with:

        B=True.
    """
    solver = DPLLSolver()

    clauses = [
        [1, 2],
        [-1, 2],
        [-1, -2],
    ]

    result = solver.solve(
        clauses,
        num_variables=2,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: False,
        2: True,
    }

    assert result.decisions == 1
    assert result.backtracks == 1

    assert formula_is_satisfied(
        clauses,
        result.assignment,
    )


def test_unsat_formula_requires_both_branches_to_fail():
    """
    These four clauses forbid all four assignments of A and B:

        (A OR B)
        (A OR NOT B)
        (NOT A OR B)
        (NOT A OR NOT B)

    No pure literals exist.

    Both branches of A must fail.
    """
    solver = DPLLSolver()

    clauses = [
        [1, 2],
        [1, -2],
        [-1, 2],
        [-1, -2],
    ]

    result = solver.solve(
        clauses,
        num_variables=2,
    )

    assert result.satisfiable is False
    assert result.assignment is None

    assert result.decisions == 1
    assert result.backtracks == 2


# ============================================================
# Complete assignment
# ============================================================

def test_sat_result_contains_complete_assignment():
    """
    Variables 2 and 3 do not occur in the formula.

    The returned SAT model must still assign all variables.
    """
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[
            [1],
        ],
        num_variables=3,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: True,
        2: False,
        3: False,
    }


def test_empty_formula_is_sat():
    """
    Empty conjunction is True.
    """
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[],
        num_variables=3,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: False,
        2: False,
        3: False,
    }


def test_empty_clause_is_unsat():
    """
    An empty clause represents False.
    """
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[
            [],
        ],
        num_variables=2,
    )

    assert result.satisfiable is False
    assert result.assignment is None


def test_zero_variable_empty_formula_is_sat():
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[],
        num_variables=0,
    )

    assert result.satisfiable is True
    assert result.assignment == {}


# ============================================================
# Assumptions
# ============================================================

def test_positive_assumption():
    """
    Formula:

        (A OR B)

    Under:

        NOT A

    B must become True.
    """
    solver = DPLLSolver()

    clauses = [
        [1, 2],
    ]

    result = solver.solve(
        clauses,
        num_variables=2,
        assumptions=[
            -1,
        ],
    )

    assert result.satisfiable is True
    assert result.assignment is not None

    assert result.assignment[1] is False
    assert result.assignment[2] is True


def test_assumptions_are_not_counted_as_propagations():
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[],
        num_variables=2,
        assumptions=[
            1,
        ],
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: True,
        2: False,
    }

    assert result.propagations == 0
    assert result.decisions == 0


def test_assumption_can_make_sat_formula_unsat():
    """
    A is SAT normally.

    Under assumption NOT A it becomes UNSAT.
    """
    solver = DPLLSolver()

    clauses = [
        [1],
    ]

    normal_result = solver.solve(
        clauses,
        num_variables=1,
    )

    assumed_result = solver.solve(
        clauses,
        num_variables=1,
        assumptions=[
            -1,
        ],
    )

    assert normal_result.satisfiable is True
    assert assumed_result.satisfiable is False


def test_conflicting_assumptions_are_unsat():
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[],
        num_variables=1,
        assumptions=[
            1,
            -1,
        ],
    )

    assert result.satisfiable is False
    assert result.assignment is None


def test_duplicate_same_assumption_is_allowed():
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[],
        num_variables=1,
        assumptions=[
            1,
            1,
        ],
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: True,
    }


# ============================================================
# CNF normalization
# ============================================================

def test_duplicate_literals_are_removed_semantically():
    """
    (A OR A) is equivalent to A.
    """
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[
            [1, 1],
        ],
        num_variables=1,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: True,
    }

    assert result.propagations == 1


def test_tautological_clause_is_ignored():
    """
    (A OR NOT A) is always True.

    With no remaining constraint, A is completed as False.
    """
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[
            [1, -1],
        ],
        num_variables=1,
    )

    assert result.satisfiable is True

    assert result.assignment == {
        1: False,
    }

    assert result.decisions == 0
    assert result.propagations == 0


def test_tautology_does_not_skip_validation_of_later_literal():
    """
    Even after detecting:

        A OR NOT A

    DPLL must still validate the rest of the original clause.

    Variable 3 is invalid when num_variables=2.
    """
    solver = DPLLSolver()

    with pytest.raises(
        InvalidCNFError
    ):
        solver.solve(
            clauses=[
                [1, -1, 3],
            ],
            num_variables=2,
        )


def test_tautology_does_not_skip_zero_literal_validation():
    solver = DPLLSolver()

    with pytest.raises(
        InvalidCNFError
    ):
        solver.solve(
            clauses=[
                [1, -1, 0],
            ],
            num_variables=2,
        )


# ============================================================
# Metrics
# ============================================================

def test_solver_records_metrics():
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[
            [1],
            [-1, 2],
        ],
        num_variables=2,
    )

    assert result.decisions >= 0
    assert result.propagations >= 0
    assert result.backtracks >= 0
    assert result.runtime >= 0.0


def test_pure_literal_assignments_count_as_propagations():
    solver = DPLLSolver()

    result = solver.solve(
        clauses=[
            [1, 2],
            [-1, 2],
        ],
        num_variables=2,
    )

    assert result.satisfiable is True

    # B=True is obtained by pure-literal elimination.
    assert result.propagations == 1

    # No branching is required.
    assert result.decisions == 0


# ============================================================
# Input validation
# ============================================================

def test_literal_zero_is_invalid():
    solver = DPLLSolver()

    with pytest.raises(
        InvalidCNFError
    ):
        solver.solve(
            clauses=[
                [0],
            ],
            num_variables=1,
        )


def test_literal_above_variable_count_is_invalid():
    solver = DPLLSolver()

    with pytest.raises(
        InvalidCNFError
    ):
        solver.solve(
            clauses=[
                [3],
            ],
            num_variables=2,
        )


def test_non_integer_literal_is_invalid():
    solver = DPLLSolver()

    with pytest.raises(
        InvalidCNFError
    ):
        solver.solve(
            clauses=[
                ["A"],
            ],
            num_variables=1,
        )


def test_boolean_literal_is_invalid():
    solver = DPLLSolver()

    with pytest.raises(
        InvalidCNFError
    ):
        solver.solve(
            clauses=[
                [True],
            ],
            num_variables=1,
        )


def test_invalid_assumption_zero():
    solver = DPLLSolver()

    with pytest.raises(
        InvalidAssumptionError
    ):
        solver.solve(
            clauses=[],
            num_variables=1,
            assumptions=[
                0,
            ],
        )


def test_invalid_assumption_variable():
    solver = DPLLSolver()

    with pytest.raises(
        InvalidAssumptionError
    ):
        solver.solve(
            clauses=[],
            num_variables=2,
            assumptions=[
                3,
            ],
        )


def test_boolean_assumption_is_invalid():
    solver = DPLLSolver()

    with pytest.raises(
        InvalidAssumptionError
    ):
        solver.solve(
            clauses=[],
            num_variables=1,
            assumptions=[
                True,
            ],
        )


def test_negative_num_variables_is_invalid():
    solver = DPLLSolver()

    with pytest.raises(
        ValueError
    ):
        solver.solve(
            clauses=[],
            num_variables=-1,
        )


def test_boolean_num_variables_is_invalid():
    solver = DPLLSolver()

    with pytest.raises(
        TypeError
    ):
        solver.solve(
            clauses=[],
            num_variables=True,
        )


# ============================================================
# Reference comparison against brute force
# ============================================================

@pytest.mark.parametrize(
    (
        "clauses",
        "num_variables",
    ),
    [
        (
            [],
            2,
        ),
        (
            [
                [1],
            ],
            1,
        ),
        (
            [
                [1],
                [-1],
            ],
            1,
        ),
        (
            [
                [1, 2],
                [-1, 2],
            ],
            2,
        ),
        (
            [
                [1, -2],
                [-1, -2],
            ],
            2,
        ),
        (
            [
                [1, 2],
                [-1, 2],
                [1, -2],
            ],
            2,
        ),
        (
            [
                [1, 2],
                [1, -2],
                [-1, 2],
                [-1, -2],
            ],
            2,
        ),
        (
            [
                [1, 2],
                [1, -2],
                [2, 3],
                [2, -3],
            ],
            3,
        ),
    ],
)
def test_dpll_matches_brute_force_on_small_formulas(
    clauses,
    num_variables,
):
    """
    Cross-check DPLL against exhaustive Boolean enumeration.

    This provides an implementation-independent reference for
    small formulas and helps ensure that optional simplifications
    such as pure-literal elimination preserve SAT semantics.
    """
    solver = DPLLSolver()

    expected_sat = brute_force_is_sat(
        clauses,
        num_variables,
    )

    result = solver.solve(
        clauses,
        num_variables=num_variables,
    )

    assert (
        result.satisfiable
        == expected_sat
    )

    if result.satisfiable:
        assert result.assignment is not None

        assert formula_is_satisfied(
            clauses,
            result.assignment,
        )

    else:
        assert result.assignment is None


# ============================================================
# Integration: CNF Encoder
# ============================================================

def test_dpll_solves_encoded_basic_3x3_full_clue_set():
    """
    Encode every clue in puzzle_3x3_01 and solve the resulting CNF.

    The complete clue set determines the designed hidden solution.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    encoder = CNFEncoder(
        characters=puzzle.characters,
        size=puzzle.size,
    )

    all_clues = [
        secret.clue
        for secret
        in puzzle.secrets.values()
    ]

    cnf = encoder.encode_clues(
        all_clues
    )

    solver = DPLLSolver()

    result = solver.solve(
        clauses=cnf,
        num_variables=encoder.total_variable_count,
    )

    assert result.satisfiable is True
    assert result.assignment is not None

    for character_id, secret in puzzle.secrets.items():
        variable = encoder.variable_for(
            character_id
        )

        expected_value = (
            secret.status
            == Status.CRIMINAL
        )

        assert (
            result.assignment[variable]
            == expected_value
        )


# ============================================================
# Integration: public KB + assumptions
# ============================================================

def test_initial_public_kb_is_sat():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    public_state = engine.get_public_state()

    encoder = CNFEncoder(
        characters=public_state.characters,
        size=puzzle.size,
    )

    kb = encoder.build_kb_from_public_state(
        public_state
    )

    solver = DPLLSolver()

    result = solver.solve(
        clauses=kb,
        num_variables=encoder.total_variable_count,
    )

    assert result.satisfiable is True


def test_initial_public_kb_forces_a1_criminal_using_assumption():
    """
    Initial public KB contains:

        B2 = CRIMINAL

    and B2's revealed clue:

        A1 = CRIMINAL

    Therefore:

        KB |= A1

    so:

        KB AND NOT A1

    must be UNSAT.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    public_state = engine.get_public_state()

    encoder = CNFEncoder(
        characters=public_state.characters,
        size=puzzle.size,
    )

    kb = encoder.build_kb_from_public_state(
        public_state
    )

    a1_variable = encoder.variable_for(
        "A1"
    )

    solver = DPLLSolver()

    result = solver.solve(
        clauses=kb,
        num_variables=encoder.total_variable_count,
        assumptions=[
            -a1_variable,
        ],
    )

    assert result.satisfiable is False


def test_initial_public_kb_accepts_a1_criminal_assumption():
    """
    Since A1 is forced Criminal:

        KB AND A1

    must remain SAT.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    public_state = engine.get_public_state()

    encoder = CNFEncoder(
        characters=public_state.characters,
        size=puzzle.size,
    )

    kb = encoder.build_kb_from_public_state(
        public_state
    )

    a1_variable = encoder.variable_for(
        "A1"
    )

    solver = DPLLSolver()

    result = solver.solve(
        clauses=kb,
        num_variables=encoder.total_variable_count,
        assumptions=[
            a1_variable,
        ],
    )

    assert result.satisfiable is True