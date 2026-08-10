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
    (A) AND (B)

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

    is immediately inconsistent.
    """
    solver = DPLLSolver()

    clauses = [
        [1],
        [-1],
    ]

    result = solver.solve(
        clauses,
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

    Unit propagation must derive:

        A = True
        B = True
        C = True

    without any branching.
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

    gives:

        A = True
        B = True
        B = False

    so the formula is UNSAT.
    """
    solver = DPLLSolver()

    clauses = [
        [1],
        [-1, 2],
        [-2],
    ]

    result = solver.solve(
        clauses,
        num_variables=2,
    )

    assert result.satisfiable is False
    assert result.assignment is None

    # No branching is required to discover the conflict.
    assert result.decisions == 0


# ============================================================
# Deterministic branching
# ============================================================

def test_branching_uses_smallest_unassigned_variable():
    """
    CNF:

        (A OR B)
        (NOT A OR B)

    There is no initial unit clause.

    Deterministic branching selects A (variable 1)
    before B (variable 2), trying True first.

    With A=True:
        B is forced True.
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
    assert result.assignment is not None

    assert result.assignment[1] is True
    assert result.assignment[2] is True

    assert result.decisions == 1
    assert result.backtracks == 0


# ============================================================
# Backtracking
# ============================================================

def test_backtracking_after_failed_true_branch():
    """
    CNF:

        (A OR B)
        (NOT A OR B)
        (NOT A OR NOT B)

    Branch order tries:

        A = True

    Then:
        second clause forces B=True
        third clause requires B=False

    -> conflict.

    DPLL must backtrack and try:

        A = False

    which forces:

        B = True

    and satisfies the formula.
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
    All four combinations of A and B are forbidden.

        (A OR B)
        (A OR NOT B)
        (NOT A OR B)
        (NOT A OR NOT B)

    Therefore UNSAT.

    DPLL must try both values of A.
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

    DPLL may solve the formula after setting only variable 1,
    but the returned SAT model must assign all variables.
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

    Temporary assumption:

        NOT A

    must force:

        B = True
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


def test_assumption_can_make_sat_formula_unsat():
    """
    Formula itself:

        A

    is SAT.

    Under assumption:

        NOT A

    it becomes UNSAT.
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

    With no remaining constraints, the complete assignment
    deterministically fills A=False.
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


def test_negative_num_variables_is_invalid():
    solver = DPLLSolver()

    with pytest.raises(
        ValueError
    ):
        solver.solve(
            clauses=[],
            num_variables=-1,
        )


# ============================================================
# Integration: CNF Encoder
# ============================================================

def test_dpll_solves_encoded_basic_3x3_full_clue_set():
    """
    Encode every clue in puzzle_3x3_01 and solve the resulting CNF.

    Because this sanity puzzle is designed so that its complete clue
    set determines all characters, DPLL should recover the known
    hidden assignment.
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

    # Compare DPLL's Boolean model with the puzzle's
    # known hidden assignment.
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
    Initial public knowledge in puzzle_3x3_01:

        B2 = CRIMINAL

        B2's revealed clue:
            A1 = CRIMINAL

    Therefore:

        KB |= A1

    which is equivalent to:

        KB AND NOT A1

    being UNSAT.
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
    The opposite assumption:

        A1 = CRIMINAL

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