from pathlib import Path

import pytest

from core.models import Classification
from core.puzzle_loader import load_puzzle
from core.game_engine import GameEngine

from logic.cnf_encoder import UnknownCharacterError
from logic.dpll import DPLLSolver, SATResult
from logic.entailment import (
    EntailmentChecker,
    EntailmentResult,
    InvalidPublicStateError,
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


def load_initial_public_state():
    """
    Load puzzle_3x3_01 and return its initial public state.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    return (
        puzzle,
        engine.get_public_state(),
    )


def make_sat_result(
    satisfiable: bool,
) -> SATResult:
    """
    Create a minimal SATResult for testing classification logic.
    """
    return SATResult(
        satisfiable=satisfiable,
        assignment=(
            {}
            if satisfiable
            else None
        ),
        decisions=0,
        propagations=0,
        backtracks=0,
        runtime=0.0,
    )


# ============================================================
# Recording solver
# ============================================================

class RecordingDPLLSolver(DPLLSolver):
    """
    DPLL solver that records every assumption set while still
    delegating the actual SAT solving to DPLLSolver.

    Used to verify that entailment performs exactly two SAT calls:

        KB AND NOT C_i
        KB AND C_i
    """

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[int, ...]] = []

    def solve(
        self,
        clauses,
        num_variables,
        assumptions=None,
    ):
        normalized_assumptions = (
            tuple(assumptions)
            if assumptions is not None
            else ()
        )

        self.calls.append(
            normalized_assumptions
        )

        return super().solve(
            clauses=clauses,
            num_variables=num_variables,
            assumptions=assumptions,
        )


# ============================================================
# Classification truth table
# ============================================================

@pytest.mark.parametrize(
    (
        "innocent_query_sat",
        "criminal_query_sat",
        "expected",
    ),
    [
        # KB AND NOT C_i = UNSAT
        # KB AND C_i     = SAT
        #
        # Therefore C_i is forced.
        (
            False,
            True,
            Classification.CRIMINAL,
        ),

        # KB AND NOT C_i = SAT
        # KB AND C_i     = UNSAT
        #
        # Therefore NOT C_i is forced.
        (
            True,
            False,
            Classification.INNOCENT,
        ),

        # Both statuses remain possible.
        (
            True,
            True,
            Classification.UNKNOWN,
        ),

        # Neither status is possible.
        #
        # Therefore KB itself is inconsistent.
        (
            False,
            False,
            Classification.INCONSISTENT,
        ),
    ],
)
def test_classification_truth_table(
    innocent_query_sat,
    criminal_query_sat,
    expected,
):
    assume_innocent_result = make_sat_result(
        innocent_query_sat
    )

    assume_criminal_result = make_sat_result(
        criminal_query_sat
    )

    classification = (
        EntailmentChecker._classify_from_queries(
            assume_innocent_result,
            assume_criminal_result,
        )
    )

    assert classification == expected


# ============================================================
# EntailmentResult convenience properties
# ============================================================

def test_entailment_result_criminal_forced():
    result = EntailmentResult(
        character_id="A1",
        variable=1,
        classification=Classification.CRIMINAL,
        assume_innocent_result=make_sat_result(False),
        assume_criminal_result=make_sat_result(True),
        kb_clause_count=2,
        variable_count=9,
    )

    assert result.criminal_forced is True
    assert result.innocent_forced is False


def test_entailment_result_innocent_forced():
    result = EntailmentResult(
        character_id="A1",
        variable=1,
        classification=Classification.INNOCENT,
        assume_innocent_result=make_sat_result(True),
        assume_criminal_result=make_sat_result(False),
        kb_clause_count=2,
        variable_count=9,
    )

    assert result.criminal_forced is False
    assert result.innocent_forced is True


def test_entailment_result_unknown_has_no_forced_status():
    result = EntailmentResult(
        character_id="A1",
        variable=1,
        classification=Classification.UNKNOWN,
        assume_innocent_result=make_sat_result(True),
        assume_criminal_result=make_sat_result(True),
        kb_clause_count=2,
        variable_count=9,
    )

    assert result.criminal_forced is False
    assert result.innocent_forced is False


def test_entailment_result_inconsistent_forces_both_queries():
    result = EntailmentResult(
        character_id="A1",
        variable=1,
        classification=Classification.INCONSISTENT,
        assume_innocent_result=make_sat_result(False),
        assume_criminal_result=make_sat_result(False),
        kb_clause_count=2,
        variable_count=9,
    )

    assert result.criminal_forced is True
    assert result.innocent_forced is True


# ============================================================
# Metrics aggregation
# ============================================================

def test_entailment_result_aggregates_solver_metrics():
    first = SATResult(
        satisfiable=False,
        assignment=None,
        decisions=2,
        propagations=3,
        backtracks=1,
        runtime=0.01,
    )

    second = SATResult(
        satisfiable=True,
        assignment={1: True},
        decisions=4,
        propagations=5,
        backtracks=2,
        runtime=0.02,
    )

    result = EntailmentResult(
        character_id="A1",
        variable=1,
        classification=Classification.CRIMINAL,
        assume_innocent_result=first,
        assume_criminal_result=second,
        kb_clause_count=2,
        variable_count=9,
    )

    assert result.total_decisions == 6
    assert result.total_propagations == 8
    assert result.total_backtracks == 3
    assert result.total_runtime == pytest.approx(
        0.03
    )


# ============================================================
# Integration: initial public state
# ============================================================

def test_initial_public_state_forces_a1_criminal():
    """
    Initial puzzle knowledge:

        B2 = CRIMINAL

    and B2's revealed clue says:

        A1 = CRIMINAL

    Therefore:

        KB |= A1
    """
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    classification = (
        checker.classify_character(
            public_state,
            "A1",
        )
    )

    assert (
        classification
        == Classification.CRIMINAL
    )


def test_initial_public_state_leaves_b1_unknown():
    """
    A1's clue has not yet been revealed.

    Therefore the initial public KB does NOT yet know:

        SAME(A1, B1)

    so B1 must remain UNKNOWN.
    """
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    classification = (
        checker.classify_character(
            public_state,
            "B1",
        )
    )

    assert (
        classification
        == Classification.UNKNOWN
    )


def test_initial_public_state_leaves_c1_unknown():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    classification = (
        checker.classify_character(
            public_state,
            "C1",
        )
    )

    assert (
        classification
        == Classification.UNKNOWN
    )


# ============================================================
# Detailed analysis
# ============================================================

def test_analyze_a1_returns_expected_query_results():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    result = checker.analyze_character(
        public_state,
        "A1",
    )

    assert result.character_id == "A1"
    assert result.variable == 1

    assert (
        result.classification
        == Classification.CRIMINAL
    )

    # KB AND NOT A1
    assert (
        result.assume_innocent_result.satisfiable
        is False
    )

    # KB AND A1
    assert (
        result.assume_criminal_result.satisfiable
        is True
    )

    assert result.criminal_forced is True
    assert result.innocent_forced is False

    # Initial public KB is:
    #
    #     B2 = CRIMINAL
    #     A1 = CRIMINAL
    #
    assert result.kb_clause_count == 2
    assert result.variable_count == 9


# ============================================================
# Exactly two SAT calls
# ============================================================

def test_analyze_character_uses_exactly_two_sat_calls():
    puzzle, public_state = (
        load_initial_public_state()
    )

    solver = RecordingDPLLSolver()

    checker = EntailmentChecker(
        size=puzzle.size,
        solver=solver,
    )

    result = checker.analyze_character(
        public_state,
        "A1",
    )

    variable = result.variable

    assert solver.calls == [
        (-variable,),
        (variable,),
    ]


def test_classify_character_also_uses_two_sat_calls():
    puzzle, public_state = (
        load_initial_public_state()
    )

    solver = RecordingDPLLSolver()

    checker = EntailmentChecker(
        size=puzzle.size,
        solver=solver,
    )

    classification = (
        checker.classify_character(
            public_state,
            "A1",
        )
    )

    assert (
        classification
        == Classification.CRIMINAL
    )

    assert len(solver.calls) == 2


# ============================================================
# Analyze all unresolved characters
# ============================================================

def test_analyze_all_skips_already_proved_character_by_default():
    """
    B2 is initially known/proved and should therefore not be
    re-classified when only_unresolved=True.
    """
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    results = checker.analyze_all(
        public_state
    )

    assert "B2" not in results

    assert len(results) == 8


def test_analyze_all_initial_state_only_a1_is_forced():
    """
    Among unresolved characters in the initial public state:

        A1 -> CRIMINAL

    while the remaining unresolved characters are still UNKNOWN.
    """
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    results = checker.classify_all(
        public_state
    )

    assert (
        results["A1"]
        == Classification.CRIMINAL
    )

    for character_id, classification in results.items():
        if character_id == "A1":
            continue

        assert (
            classification
            == Classification.UNKNOWN
        )


def test_analyze_all_can_include_proved_characters():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    results = checker.classify_all(
        public_state,
        only_unresolved=False,
    )

    assert len(results) == 9

    assert (
        results["B2"]
        == Classification.CRIMINAL
    )

    assert (
        results["A1"]
        == Classification.CRIMINAL
    )


# ============================================================
# KB consistency
# ============================================================

def test_initial_public_kb_is_consistent():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    assert (
        checker.is_kb_consistent(
            public_state
        )
        is True
    )


# ============================================================
# Character normalization
# ============================================================

def test_character_id_is_normalized():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    result = checker.analyze_character(
        public_state,
        " a1 ",
    )

    assert result.character_id == "A1"

    assert (
        result.classification
        == Classification.CRIMINAL
    )


# ============================================================
# Error handling
# ============================================================

def test_unknown_character_raises_error():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    with pytest.raises(
        UnknownCharacterError
    ):
        checker.classify_character(
            public_state,
            "Z9",
        )


def test_wrong_checker_size_rejects_public_state():
    """
    A 3x3 public state contains 9 characters.

    A checker configured for 4x4 expects 16 and must reject it.
    """
    _, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=4
    )

    with pytest.raises(
        InvalidPublicStateError
    ):
        checker.classify_character(
            public_state,
            "A1",
        )


def test_non_public_state_is_rejected():
    checker = EntailmentChecker(
        size=3
    )

    with pytest.raises(
        TypeError
    ):
        checker.classify_character(
            None,
            "A1",
        )


def test_invalid_checker_size():
    with pytest.raises(
        ValueError
    ):
        EntailmentChecker(
            size=0
        )


def test_invalid_solver_type():
    with pytest.raises(
        TypeError
    ):
        EntailmentChecker(
            size=3,
            solver="not-a-solver",
        )


# ============================================================
# SAT-call metrics
# ============================================================

def test_initial_sat_call_count_is_zero():
    checker = EntailmentChecker(
        size=3
    )

    assert checker.sat_call_count == 0


def test_analyze_character_records_two_sat_calls():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    result = checker.analyze_character(
        public_state,
        "A1",
    )

    assert result.sat_calls == 2
    assert checker.sat_call_count == 2


def test_classify_character_records_two_sat_calls():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    checker.classify_character(
        public_state,
        "A1",
    )

    assert checker.sat_call_count == 2


def test_is_kb_consistent_records_one_sat_call():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    checker.is_kb_consistent(
        public_state
    )

    assert checker.sat_call_count == 1


def test_analyze_all_unresolved_records_sixteen_sat_calls():
    """
    puzzle_3x3_01 begins with B2 already proved.

    Therefore 8 characters remain unresolved.

        8 characters * 2 SAT calls = 16.
    """
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    results = checker.analyze_all(
        public_state
    )

    assert len(results) == 8
    assert checker.sat_call_count == 16


def test_analyze_all_characters_records_eighteen_sat_calls():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    results = checker.analyze_all(
        public_state,
        only_unresolved=False,
    )

    assert len(results) == 9

    # 9 characters * 2 SAT calls.
    assert checker.sat_call_count == 18


def test_sat_call_count_is_cumulative():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    checker.analyze_character(
        public_state,
        "A1",
    )

    assert checker.sat_call_count == 2

    checker.analyze_character(
        public_state,
        "B1",
    )

    assert checker.sat_call_count == 4

    checker.is_kb_consistent(
        public_state
    )

    assert checker.sat_call_count == 5


def test_reset_sat_call_count():
    puzzle, public_state = (
        load_initial_public_state()
    )

    checker = EntailmentChecker(
        size=puzzle.size
    )

    checker.analyze_character(
        public_state,
        "A1",
    )

    assert checker.sat_call_count == 2

    checker.reset_sat_call_count()

    assert checker.sat_call_count == 0


def test_sat_call_counter_matches_recording_solver():
    puzzle, public_state = (
        load_initial_public_state()
    )

    solver = RecordingDPLLSolver()

    checker = EntailmentChecker(
        size=puzzle.size,
        solver=solver,
    )

    checker.analyze_character(
        public_state,
        "A1",
    )

    checker.is_kb_consistent(
        public_state
    )

    assert checker.sat_call_count == 3
    assert len(solver.calls) == 3