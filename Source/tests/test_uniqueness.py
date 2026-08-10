from pathlib import Path

import pytest

from core.models import (
    Character,
    Puzzle,
    Status,
)
from core.puzzle_loader import load_puzzle
from logic.cnf_encoder import CNFEncoder
from logic.dpll import DPLLSolver, SATResult
from logic.uniqueness import (
    InvalidPuzzleSizeError,
    UniquenessChecker,
    UniquenessError,
    UniquenessResult,
)


# ============================================================
# Helpers
# ============================================================

def puzzle_3x3_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "puzzles"
        / "puzzle_3x3_01.json"
    )


def characters_1x1() -> dict[str, Character]:
    return {
        "A1": Character(
            id="A1",
            name="Alice",
            profession="Doctor",
        )
    }


def characters_2x2() -> dict[str, Character]:
    return {
        "A1": Character(
            id="A1",
            name="Alice",
            profession="Doctor",
        ),
        "B1": Character(
            id="B1",
            name="Ben",
            profession="Teacher",
        ),
        "A2": Character(
            id="A2",
            name="Cara",
            profession="Painter",
        ),
        "B2": Character(
            id="B2",
            name="Daniel",
            profession="Engineer",
        ),
    }


def make_sat_result(
    satisfiable: bool,
    assignment: dict[int, bool] | None = None,
    *,
    decisions: int = 0,
    propagations: int = 0,
    backtracks: int = 0,
    runtime: float = 0.0,
) -> SATResult:
    return SATResult(
        satisfiable=satisfiable,
        assignment=(
            assignment
            if satisfiable
            else None
        ),
        decisions=decisions,
        propagations=propagations,
        backtracks=backtracks,
        runtime=runtime,
    )


# ============================================================
# Recording solver
# ============================================================

class RecordingDPLLSolver(DPLLSolver):
    """
    Real DPLL solver that records every CNF passed to solve().

    This lets the tests verify:

        - an UNSAT KB requires only one solve,
        - a SAT KB requires a second solve,
        - the second solve contains the blocking clause.
    """

    def __init__(self):
        super().__init__()
        self.calls: list[
            tuple[tuple[int, ...], ...]
        ] = []

    def solve(
        self,
        clauses,
        num_variables,
        assumptions=None,
    ):
        self.calls.append(
            tuple(
                tuple(clause)
                for clause in clauses
            )
        )

        return super().solve(
            clauses=clauses,
            num_variables=num_variables,
            assumptions=assumptions,
        )


# ============================================================
# Multiple solutions
# ============================================================

def test_empty_1x1_kb_has_multiple_solutions():
    """
    With one character and no constraints:

        A1 = INNOCENT

    and

        A1 = CRIMINAL

    are both valid.

    Therefore the KB is satisfiable but not unique.
    """
    checker = UniquenessChecker(
        size=1
    )

    result = checker.check_clues(
        characters=characters_1x1(),
        clues=[],
    )

    assert result.satisfiable is True
    assert result.unique is False

    assert result.has_solution is True
    assert result.has_multiple_solutions is True

    assert (
        result.first_character_assignment
        is not None
    )

    assert (
        result.second_character_assignment
        is not None
    )

    assert (
        result.first_character_assignment
        != result.second_character_assignment
    )

    # Deterministic DPLL completes unconstrained variables
    # with False in the first model.
    assert (
        result.first_character_assignment["A1"]
        == Status.INNOCENT
    )

    # Blocking the first model forces the alternative.
    assert (
        result.second_character_assignment["A1"]
        == Status.CRIMINAL
    )

    assert result.blocking_clause == (1,)


# ============================================================
# Unique solution
# ============================================================

def test_known_status_makes_1x1_solution_unique():
    """
    Knowledge base:

        A1 = CRIMINAL

    has exactly one character-status assignment.
    """
    checker = UniquenessChecker(
        size=1
    )

    result = checker.check_clues(
        characters=characters_1x1(),
        clues=[],
        known_statuses={
            "A1": Status.CRIMINAL,
        },
    )

    assert result.satisfiable is True
    assert result.unique is True

    assert result.has_solution is True
    assert result.has_multiple_solutions is False

    assert result.first_character_assignment == {
        "A1": Status.CRIMINAL,
    }

    assert (
        result.second_character_assignment
        is None
    )

    # First model:
    #
    #     A1 = True
    #
    # so its blocking clause is:
    #
    #     NOT A1
    assert result.blocking_clause == (-1,)

    assert (
        result.second_solve_result
        is not None
    )

    assert (
        result.second_solve_result.satisfiable
        is False
    )


# ============================================================
# UNSAT knowledge base
# ============================================================

def test_contradictory_kb_has_no_solution():
    """
    Reuse the real B2 clue from puzzle_3x3_01:

        A1 = CRIMINAL

    then explicitly add:

        A1 = INNOCENT

    The resulting KB is inconsistent.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    b2_clue = puzzle.secrets[
        "B2"
    ].clue

    checker = UniquenessChecker(
        size=puzzle.size
    )

    result = checker.check_clues(
        characters=puzzle.characters,
        clues=[
            b2_clue,
        ],
        known_statuses={
            "A1": Status.INNOCENT,
        },
    )

    assert result.satisfiable is False
    assert result.unique is False

    assert result.has_solution is False
    assert result.has_multiple_solutions is False

    assert (
        result.first_character_assignment
        is None
    )

    assert (
        result.second_character_assignment
        is None
    )

    assert result.blocking_clause is None
    assert result.second_solve_result is None


# ============================================================
# Number of SAT calls
# ============================================================

def test_unsat_kb_requires_only_one_sat_call():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    b2_clue = puzzle.secrets[
        "B2"
    ].clue

    solver = RecordingDPLLSolver()

    checker = UniquenessChecker(
        size=puzzle.size,
        solver=solver,
    )

    result = checker.check_clues(
        characters=puzzle.characters,
        clues=[
            b2_clue,
        ],
        known_statuses={
            "A1": Status.INNOCENT,
        },
    )

    assert result.satisfiable is False

    # No first model exists, therefore there is nothing
    # to block and no second SAT call is required.
    assert len(solver.calls) == 1


def test_unique_kb_requires_two_sat_calls():
    solver = RecordingDPLLSolver()

    checker = UniquenessChecker(
        size=1,
        solver=solver,
    )

    result = checker.check_clues(
        characters=characters_1x1(),
        clues=[],
        known_statuses={
            "A1": Status.CRIMINAL,
        },
    )

    assert result.unique is True

    assert len(solver.calls) == 2


def test_multiple_solution_kb_requires_two_sat_calls():
    solver = RecordingDPLLSolver()

    checker = UniquenessChecker(
        size=1,
        solver=solver,
    )

    result = checker.check_clues(
        characters=characters_1x1(),
        clues=[],
    )

    assert result.satisfiable is True
    assert result.unique is False

    assert len(solver.calls) == 2


# ============================================================
# Blocking clause
# ============================================================

def test_second_solve_adds_blocking_clause():
    """
    First solve of an empty 1x1 KB returns:

        A1 = False

    so the blocking clause must be:

        A1

    The original KB is empty and must remain unchanged.
    """
    solver = RecordingDPLLSolver()

    checker = UniquenessChecker(
        size=1,
        solver=solver,
    )

    result = checker.check_clues(
        characters=characters_1x1(),
        clues=[],
    )

    assert result.blocking_clause == (1,)

    assert len(solver.calls) == 2

    # Original KB.
    assert solver.calls[0] == ()

    # Second KB = original KB + blocking clause.
    assert solver.calls[1] == (
        (1,),
    )


def test_blocking_clause_negates_complete_primary_assignment():
    """
    Assignment:

        A1 = True
        B1 = False
        A2 = True
        B2 = False

    must produce:

        NOT A1
        OR B1
        OR NOT A2
        OR B2
    """
    characters = characters_2x2()

    encoder = CNFEncoder(
        characters=characters,
        size=2,
    )

    assignment = {
        1: True,
        2: False,
        3: True,
        4: False,
    }

    clause = (
        UniquenessChecker
        ._build_blocking_clause(
            assignment=assignment,
            encoder=encoder,
            character_ids=tuple(
                characters.keys()
            ),
        )
    )

    assert clause == (
        -1,
        2,
        -3,
        4,
    )


def test_blocking_clause_ignores_non_character_variables():
    """
    Simulate a future SAT model containing an auxiliary variable.

    Variable 99 must not appear in the Griductive blocking clause.
    """
    characters = characters_2x2()

    encoder = CNFEncoder(
        characters=characters,
        size=2,
    )

    assignment = {
        1: True,
        2: False,
        3: True,
        4: False,

        # Simulated auxiliary variable.
        99: True,
    }

    clause = (
        UniquenessChecker
        ._build_blocking_clause(
            assignment=assignment,
            encoder=encoder,
            character_ids=tuple(
                characters.keys()
            ),
        )
    )

    assert clause == (
        -1,
        2,
        -3,
        4,
    )

    assert 99 not in clause
    assert -99 not in clause


def test_blocking_clause_rejects_missing_primary_variable():
    characters = characters_2x2()

    encoder = CNFEncoder(
        characters=characters,
        size=2,
    )

    incomplete_assignment = {
        1: True,
        2: False,
        3: True,

        # Variable 4 is missing.
    }

    with pytest.raises(
        UniquenessError
    ):
        (
            UniquenessChecker
            ._build_blocking_clause(
                assignment=incomplete_assignment,
                encoder=encoder,
                character_ids=tuple(
                    characters.keys()
                ),
            )
        )


# ============================================================
# SAT model -> Griductive status conversion
# ============================================================

def test_extract_character_assignment_converts_boolean_values():
    characters = characters_2x2()

    encoder = CNFEncoder(
        characters=characters,
        size=2,
    )

    assignment = {
        1: True,
        2: False,
        3: False,
        4: True,
    }

    result = (
        UniquenessChecker
        ._extract_character_assignment(
            assignment=assignment,
            encoder=encoder,
            character_ids=tuple(
                characters.keys()
            ),
        )
    )

    assert result == {
        "A1": Status.CRIMINAL,
        "B1": Status.INNOCENT,
        "A2": Status.INNOCENT,
        "B2": Status.CRIMINAL,
    }


def test_extract_character_assignment_rejects_missing_variable():
    characters = characters_1x1()

    encoder = CNFEncoder(
        characters=characters,
        size=1,
    )

    with pytest.raises(
        UniquenessError
    ):
        (
            UniquenessChecker
            ._extract_character_assignment(
                assignment={},
                encoder=encoder,
                character_ids=("A1",),
            )
        )


# ============================================================
# Metrics
# ============================================================

def test_uniqueness_result_aggregates_two_solver_runs():
    first = SATResult(
        satisfiable=True,
        assignment={
            1: True,
        },
        decisions=2,
        propagations=3,
        backtracks=1,
        runtime=0.01,
    )

    second = SATResult(
        satisfiable=False,
        assignment=None,
        decisions=4,
        propagations=5,
        backtracks=2,
        runtime=0.02,
    )

    result = UniquenessResult(
        satisfiable=True,
        unique=True,
        first_solve_result=first,
        second_solve_result=second,
        first_character_assignment={
            "A1": Status.CRIMINAL,
        },
        second_character_assignment=None,
        blocking_clause=(-1,),
        variable_count=1,
        kb_clause_count=1,
    )

    assert result.total_decisions == 6
    assert result.total_propagations == 8
    assert result.total_backtracks == 3

    assert result.total_runtime == pytest.approx(
        0.03
    )


def test_unsat_result_metrics_use_only_first_solve():
    first = SATResult(
        satisfiable=False,
        assignment=None,
        decisions=2,
        propagations=3,
        backtracks=1,
        runtime=0.01,
    )

    result = UniquenessResult(
        satisfiable=False,
        unique=False,
        first_solve_result=first,
        second_solve_result=None,
        first_character_assignment=None,
        second_character_assignment=None,
        blocking_clause=None,
        variable_count=1,
        kb_clause_count=2,
    )

    assert result.total_decisions == 2
    assert result.total_propagations == 3
    assert result.total_backtracks == 1

    assert result.total_runtime == pytest.approx(
        0.01
    )


# ============================================================
# Integration: puzzle_3x3_01
# ============================================================

def test_basic_3x3_puzzle_has_unique_solution():
    """
    Machine-check the complete clue set of puzzle_3x3_01.

    The puzzle was designed as a deterministic sanity puzzle and
    should have exactly one Criminal/Innocent assignment.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    checker = UniquenessChecker(
        size=puzzle.size
    )

    result = checker.check_puzzle(
        puzzle
    )

    assert result.satisfiable is True
    assert result.unique is True

    assert result.has_solution is True
    assert result.has_multiple_solutions is False

    assert (
        result.first_character_assignment
        is not None
    )

    assert (
        result.second_character_assignment
        is None
    )


def test_basic_3x3_unique_model_matches_hidden_solution():
    """
    The uniqueness checker does not use hidden statuses as the
    solution constraints.

    After solving from clues, however, the resulting unique model
    can be compared against the known puzzle answer as a test oracle.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    checker = UniquenessChecker(
        size=puzzle.size
    )

    result = checker.check_puzzle(
        puzzle
    )

    assert result.unique is True

    expected = {
        character_id:
            secret.status
        for character_id, secret
        in puzzle.secrets.items()
    }

    assert (
        result.first_character_assignment
        == expected
    )


def test_basic_3x3_unique_check_uses_second_unsat_solve():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    checker = UniquenessChecker(
        size=puzzle.size
    )

    result = checker.check_puzzle(
        puzzle
    )

    assert (
        result.first_solve_result.satisfiable
        is True
    )

    assert (
        result.second_solve_result
        is not None
    )

    assert (
        result.second_solve_result.satisfiable
        is False
    )


def test_check_puzzle_adds_only_initial_status_facts():
    """
    check_puzzle() includes all clues in both cases.

    The default call additionally includes exactly the statuses of
    puzzle.initial_revealed.

    Therefore the KB clause count should increase by exactly the
    number of initially revealed characters.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    checker = UniquenessChecker(
        size=puzzle.size
    )

    without_initial_statuses = (
        checker.check_puzzle(
            puzzle,
            include_initial_statuses=False,
        )
    )

    with_initial_statuses = (
        checker.check_puzzle(
            puzzle,
            include_initial_statuses=True,
        )
    )

    assert (
        with_initial_statuses.kb_clause_count
        ==
        without_initial_statuses.kb_clause_count
        + len(puzzle.initial_revealed)
    )


# ============================================================
# Constructor validation
# ============================================================

def test_checker_rejects_zero_size():
    with pytest.raises(
        ValueError
    ):
        UniquenessChecker(
            size=0
        )


def test_checker_rejects_negative_size():
    with pytest.raises(
        ValueError
    ):
        UniquenessChecker(
            size=-1
        )


def test_checker_rejects_non_integer_size():
    with pytest.raises(
        TypeError
    ):
        UniquenessChecker(
            size=3.0
        )


def test_checker_rejects_boolean_size():
    with pytest.raises(
        TypeError
    ):
        UniquenessChecker(
            size=True
        )


def test_checker_rejects_invalid_solver_type():
    with pytest.raises(
        TypeError
    ):
        UniquenessChecker(
            size=3,
            solver="not-a-solver",
        )


# ============================================================
# Puzzle validation
# ============================================================

def test_check_puzzle_rejects_wrong_checker_size():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    checker = UniquenessChecker(
        size=4
    )

    with pytest.raises(
        InvalidPuzzleSizeError
    ):
        checker.check_puzzle(
            puzzle
        )


def test_check_puzzle_rejects_non_puzzle_object():
    checker = UniquenessChecker(
        size=3
    )

    with pytest.raises(
        TypeError
    ):
        checker.check_puzzle(
            "not-a-puzzle"
        )