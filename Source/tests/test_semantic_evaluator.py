from pathlib import Path

import pytest

from core.models import (
    Clue,
    ClueType,
    Region,
    RegionType,
    Status,
)
from core.puzzle_loader import load_puzzle
from logic.semantic_evaluator import (
    IncompleteAssignmentError,
    SemanticEvaluationError,
    UnsupportedClueTypeError,
    evaluate_at_least,
    evaluate_at_most,
    evaluate_clue,
    evaluate_clues,
    evaluate_different,
    evaluate_exactly,
    evaluate_fact,
    evaluate_implies,
    evaluate_parity,
    evaluate_same,
)


# ============================================================
# Shared fixtures
# ============================================================

@pytest.fixture
def assignment_3x3():
    """
    Complete semantic assignment used by unit tests.

    Grid:

            A           B           C
        +-----------+-----------+-----------+
    1   | Criminal  | Criminal  | Innocent  |
        +-----------+-----------+-----------+
    2   | Innocent  | Criminal  | Criminal  |
        +-----------+-----------+-----------+
    3   | Innocent  | Innocent  | Criminal  |
        +-----------+-----------+-----------+

    Criminals:
        A1, B1, B2, C2, C3
    """
    return {
        "A1": Status.CRIMINAL,
        "B1": Status.CRIMINAL,
        "C1": Status.INNOCENT,
        "A2": Status.INNOCENT,
        "B2": Status.CRIMINAL,
        "C2": Status.CRIMINAL,
        "A3": Status.INNOCENT,
        "B3": Status.INNOCENT,
        "C3": Status.CRIMINAL,
    }


# ============================================================
# FACT
# ============================================================

def test_fact_true(assignment_3x3):
    clue = Clue(
        id="TEST_FACT_TRUE",
        type=ClueType.FACT,
        params={
            "person": "A1",
            "status": Status.CRIMINAL,
        },
    )

    assert evaluate_fact(
        clue,
        assignment_3x3,
    ) is True


def test_fact_false(assignment_3x3):
    clue = Clue(
        id="TEST_FACT_FALSE",
        type=ClueType.FACT,
        params={
            "person": "A1",
            "status": Status.INNOCENT,
        },
    )

    assert evaluate_fact(
        clue,
        assignment_3x3,
    ) is False


# ============================================================
# SAME
# ============================================================

def test_same_true(assignment_3x3):
    clue = Clue(
        id="TEST_SAME_TRUE",
        type=ClueType.SAME,
        params={
            "people": ("A1", "B1"),
        },
    )

    assert evaluate_same(
        clue,
        assignment_3x3,
    ) is True


def test_same_false(assignment_3x3):
    clue = Clue(
        id="TEST_SAME_FALSE",
        type=ClueType.SAME,
        params={
            "people": ("A1", "C1"),
        },
    )

    assert evaluate_same(
        clue,
        assignment_3x3,
    ) is False


# ============================================================
# DIFFERENT
# ============================================================

def test_different_true(assignment_3x3):
    clue = Clue(
        id="TEST_DIFFERENT_TRUE",
        type=ClueType.DIFFERENT,
        params={
            "people": ("B1", "C1"),
        },
    )

    assert evaluate_different(
        clue,
        assignment_3x3,
    ) is True


def test_different_false(assignment_3x3):
    clue = Clue(
        id="TEST_DIFFERENT_FALSE",
        type=ClueType.DIFFERENT,
        params={
            "people": ("A1", "B1"),
        },
    )

    assert evaluate_different(
        clue,
        assignment_3x3,
    ) is False


# ============================================================
# EXACTLY
# ============================================================

def test_exactly_true_for_row(assignment_3x3):
    """
    Row 1:

        A1 = Criminal
        B1 = Criminal
        C1 = Innocent

    Exactly 2 Criminals.
    """
    clue = Clue(
        id="TEST_EXACTLY_ROW",
        type=ClueType.EXACTLY,
        params={
            "k": 2,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert evaluate_exactly(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_exactly_false_for_row(assignment_3x3):
    clue = Clue(
        id="TEST_EXACTLY_ROW_FALSE",
        type=ClueType.EXACTLY,
        params={
            "k": 1,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert evaluate_exactly(
        clue,
        assignment_3x3,
        size=3,
    ) is False


# ============================================================
# AT_LEAST
# ============================================================

def test_at_least_true_for_column(assignment_3x3):
    """
    Column B:

        B1 = Criminal
        B2 = Criminal
        B3 = Innocent

    There are 2 Criminals.
    """
    clue = Clue(
        id="TEST_AT_LEAST_COLUMN",
        type=ClueType.AT_LEAST,
        params={
            "k": 2,
            "region": Region(
                type=RegionType.COLUMN,
                value="B",
            ),
        },
    )

    assert evaluate_at_least(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_at_least_false_for_column(assignment_3x3):
    clue = Clue(
        id="TEST_AT_LEAST_COLUMN_FALSE",
        type=ClueType.AT_LEAST,
        params={
            "k": 3,
            "region": Region(
                type=RegionType.COLUMN,
                value="B",
            ),
        },
    )

    assert evaluate_at_least(
        clue,
        assignment_3x3,
        size=3,
    ) is False


# ============================================================
# AT_MOST
# ============================================================

def test_at_most_true_for_explicit_region(
    assignment_3x3,
):
    """
    Explicit region:

        A1 = Criminal
        A2 = Innocent
        A3 = Innocent

    Exactly 1 Criminal.
    """
    clue = Clue(
        id="TEST_AT_MOST_EXPLICIT",
        type=ClueType.AT_MOST,
        params={
            "k": 1,
            "region": Region(
                type=RegionType.EXPLICIT,
                value=("A1", "A2", "A3"),
            ),
        },
    )

    assert evaluate_at_most(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_at_most_false_for_explicit_region(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_AT_MOST_EXPLICIT_FALSE",
        type=ClueType.AT_MOST,
        params={
            "k": 0,
            "region": Region(
                type=RegionType.EXPLICIT,
                value=("A1", "A2", "A3"),
            ),
        },
    )

    assert evaluate_at_most(
        clue,
        assignment_3x3,
        size=3,
    ) is False


# ============================================================
# NEIGHBORS region
# ============================================================

def test_exactly_for_neighbors_region(
    assignment_3x3,
):
    """
    Neighbors of B2:

        A1 B1 C1
        A2    C2
        A3 B3 C3

    Criminals:
        A1, B1, C2, C3

    Therefore exactly 4 Criminals.
    """
    clue = Clue(
        id="TEST_NEIGHBORS",
        type=ClueType.EXACTLY,
        params={
            "k": 4,
            "region": Region(
                type=RegionType.NEIGHBORS,
                value="B2",
            ),
        },
    )

    assert evaluate_exactly(
        clue,
        assignment_3x3,
        size=3,
    ) is True


# ============================================================
# Extension 1: PARITY
# ============================================================

def test_parity_even_true(assignment_3x3):
    """
    Row 1 contains two Criminals:

        A1 = Criminal
        B1 = Criminal
        C1 = Innocent

    2 is EVEN.
    """
    clue = Clue(
        id="TEST_PARITY_EVEN_TRUE",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert evaluate_parity(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_parity_even_false(assignment_3x3):
    """
    Column A contains one Criminal:

        A1 = Criminal
        A2 = Innocent
        A3 = Innocent

    1 is not EVEN.
    """
    clue = Clue(
        id="TEST_PARITY_EVEN_FALSE",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.COLUMN,
                value="A",
            ),
        },
    )

    assert evaluate_parity(
        clue,
        assignment_3x3,
        size=3,
    ) is False


def test_parity_odd_true(assignment_3x3):
    """
    Column A contains exactly one Criminal.

    1 is ODD.
    """
    clue = Clue(
        id="TEST_PARITY_ODD_TRUE",
        type=ClueType.PARITY,
        params={
            "parity": "ODD",
            "region": Region(
                type=RegionType.COLUMN,
                value="A",
            ),
        },
    )

    assert evaluate_parity(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_parity_odd_false(assignment_3x3):
    """
    Row 1 contains two Criminals.

    2 is not ODD.
    """
    clue = Clue(
        id="TEST_PARITY_ODD_FALSE",
        type=ClueType.PARITY,
        params={
            "parity": "ODD",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert evaluate_parity(
        clue,
        assignment_3x3,
        size=3,
    ) is False


def test_parity_neighbors_region(assignment_3x3):
    """
    Neighbors of B2 contain four Criminals.

    Therefore the parity is EVEN.
    """
    clue = Clue(
        id="TEST_PARITY_NEIGHBORS",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.NEIGHBORS,
                value="B2",
            ),
        },
    )

    assert evaluate_parity(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_parity_explicit_region(assignment_3x3):
    """
    Explicit region:

        A1 = Criminal
        C1 = Innocent
        C3 = Criminal

    contains two Criminals -> EVEN.
    """
    clue = Clue(
        id="TEST_PARITY_EXPLICIT",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.EXPLICIT,
                value=("A1", "C1", "C3"),
            ),
        },
    )

    assert evaluate_parity(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_parity_normalizes_lowercase_value(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_PARITY_NORMALIZATION",
        type=ClueType.PARITY,
        params={
            "parity": " even ",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert evaluate_parity(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_invalid_parity_value_raises_error(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_INVALID_PARITY",
        type=ClueType.PARITY,
        params={
            "parity": "PRIME",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    with pytest.raises(
        SemanticEvaluationError
    ):
        evaluate_parity(
            clue,
            assignment_3x3,
            size=3,
        )


# ============================================================
# Extension 2: IMPLIES
# ============================================================

def test_implies_true_when_antecedent_and_consequent_true(
    assignment_3x3,
):
    """
    A1 is Criminal.
    C1 is Innocent.

    Criminal(A1) -> Innocent(C1)

    True -> True = True.
    """
    clue = Clue(
        id="TEST_IMPLIES_TRUE_TRUE",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "C1",
                "status": Status.INNOCENT,
            },
        },
    )

    assert evaluate_implies(
        clue,
        assignment_3x3,
    ) is True


def test_implies_false_when_antecedent_true_consequent_false(
    assignment_3x3,
):
    """
    A1 is Criminal.
    C1 is not Criminal.

    Criminal(A1) -> Criminal(C1)

    True -> False = False.
    """
    clue = Clue(
        id="TEST_IMPLIES_TRUE_FALSE",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "C1",
                "status": Status.CRIMINAL,
            },
        },
    )

    assert evaluate_implies(
        clue,
        assignment_3x3,
    ) is False


def test_implies_true_when_antecedent_false(
    assignment_3x3,
):
    """
    C1 is Innocent, so:

        Criminal(C1)

    is false.

    A false antecedent makes the implication true regardless
    of the consequent.
    """
    clue = Clue(
        id="TEST_IMPLIES_FALSE_ANTECEDENT",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "C1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "A2",
                "status": Status.CRIMINAL,
            },
        },
    )

    assert evaluate_implies(
        clue,
        assignment_3x3,
    ) is True


def test_implies_supports_innocent_antecedent(
    assignment_3x3,
):
    """
    C1 is Innocent and A1 is Criminal:

        Innocent(C1) -> Criminal(A1)

    True -> True = True.
    """
    clue = Clue(
        id="TEST_IMPLIES_INNOCENT_ANTECEDENT",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "C1",
                "status": Status.INNOCENT,
            },
            "consequent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
        },
    )

    assert evaluate_implies(
        clue,
        assignment_3x3,
    ) is True


def test_implies_supports_innocent_consequent(
    assignment_3x3,
):
    """
    B1 is Criminal and A2 is Innocent:

        Criminal(B1) -> Innocent(A2)

    True -> True = True.
    """
    clue = Clue(
        id="TEST_IMPLIES_INNOCENT_CONSEQUENT",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "B1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "A2",
                "status": Status.INNOCENT,
            },
        },
    )

    assert evaluate_implies(
        clue,
        assignment_3x3,
    ) is True


def test_implies_same_person_raises_error(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_IMPLIES_SAME_PERSON",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "A1",
                "status": Status.INNOCENT,
            },
        },
    )

    with pytest.raises(
        SemanticEvaluationError
    ):
        evaluate_implies(
            clue,
            assignment_3x3,
        )


def test_implies_missing_character_raises_error(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_IMPLIES_MISSING_CHARACTER",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "B1",
                "status": Status.CRIMINAL,
            },
        },
    )

    incomplete_assignment = dict(
        assignment_3x3
    )

    del incomplete_assignment[
        "B1"
    ]

    with pytest.raises(
        IncompleteAssignmentError
    ):
        evaluate_implies(
            clue,
            incomplete_assignment,
        )


# ============================================================
# Main dispatch function
# ============================================================

@pytest.mark.parametrize(
    ("clue", "expected"),
    [
        (
            Clue(
                id="DISPATCH_FACT",
                type=ClueType.FACT,
                params={
                    "person": "A1",
                    "status": Status.CRIMINAL,
                },
            ),
            True,
        ),
        (
            Clue(
                id="DISPATCH_SAME",
                type=ClueType.SAME,
                params={
                    "people": ("A1", "B1"),
                },
            ),
            True,
        ),
        (
            Clue(
                id="DISPATCH_DIFFERENT",
                type=ClueType.DIFFERENT,
                params={
                    "people": ("A1", "C1"),
                },
            ),
            True,
        ),
        (
            Clue(
                id="DISPATCH_EXACTLY",
                type=ClueType.EXACTLY,
                params={
                    "k": 2,
                    "region": Region(
                        type=RegionType.ROW,
                        value=1,
                    ),
                },
            ),
            True,
        ),
        (
            Clue(
                id="DISPATCH_AT_LEAST",
                type=ClueType.AT_LEAST,
                params={
                    "k": 2,
                    "region": Region(
                        type=RegionType.COLUMN,
                        value="B",
                    ),
                },
            ),
            True,
        ),
        (
            Clue(
                id="DISPATCH_AT_MOST",
                type=ClueType.AT_MOST,
                params={
                    "k": 1,
                    "region": Region(
                        type=RegionType.COLUMN,
                        value="A",
                    ),
                },
            ),
            True,
        ),
        (
            Clue(
                id="DISPATCH_PARITY",
                type=ClueType.PARITY,
                params={
                    "parity": "EVEN",
                    "region": Region(
                        type=RegionType.ROW,
                        value=1,
                    ),
                },
            ),
            True,
        ),
        (
            Clue(
                id="DISPATCH_IMPLIES",
                type=ClueType.IMPLIES,
                params={
                    "antecedent": {
                        "person": "A1",
                        "status": Status.CRIMINAL,
                    },
                    "consequent": {
                        "person": "C1",
                        "status": Status.INNOCENT,
                    },
                },
            ),
            True,
        ),
    ],
)
def test_evaluate_clue_dispatch(
    assignment_3x3,
    clue,
    expected,
):
    assert evaluate_clue(
        clue,
        assignment_3x3,
        size=3,
    ) is expected


# ============================================================
# String clue type normalization
# ============================================================

def test_parity_string_clue_type_is_supported(
    assignment_3x3,
):
    clue = Clue(
        id="STRING_PARITY",
        type="parity",
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert evaluate_clue(
        clue,
        assignment_3x3,
        size=3,
    ) is True


def test_implies_string_clue_type_is_supported(
    assignment_3x3,
):
    clue = Clue(
        id="STRING_IMPLIES",
        type="implies",
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "C1",
                "status": Status.INNOCENT,
            },
        },
    )

    assert evaluate_clue(
        clue,
        assignment_3x3,
        size=3,
    ) is True


# ============================================================
# Batch evaluation
# ============================================================

def test_evaluate_clues_batch(
    assignment_3x3,
):
    clues = [
        Clue(
            id="CL1",
            type=ClueType.FACT,
            params={
                "person": "A1",
                "status": Status.CRIMINAL,
            },
        ),
        Clue(
            id="CL2",
            type=ClueType.SAME,
            params={
                "people": ("A1", "B1"),
            },
        ),
        Clue(
            id="CL3",
            type=ClueType.DIFFERENT,
            params={
                "people": ("B1", "C1"),
            },
        ),
        Clue(
            id="CL4",
            type=ClueType.PARITY,
            params={
                "parity": "EVEN",
                "region": Region(
                    type=RegionType.ROW,
                    value=1,
                ),
            },
        ),
        Clue(
            id="CL5",
            type=ClueType.IMPLIES,
            params={
                "antecedent": {
                    "person": "A1",
                    "status": Status.CRIMINAL,
                },
                "consequent": {
                    "person": "C1",
                    "status": Status.INNOCENT,
                },
            },
        ),
    ]

    results = evaluate_clues(
        clues,
        assignment_3x3,
        size=3,
    )

    assert results == {
        "CL1": True,
        "CL2": True,
        "CL3": True,
        "CL4": True,
        "CL5": True,
    }


# ============================================================
# Error handling
# ============================================================

def test_missing_assignment_character_raises_error(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_MISSING_CHARACTER",
        type=ClueType.FACT,
        params={
            "person": "A1",
            "status": Status.CRIMINAL,
        },
    )

    incomplete_assignment = dict(
        assignment_3x3
    )

    del incomplete_assignment[
        "A1"
    ]

    with pytest.raises(
        IncompleteAssignmentError
    ):
        evaluate_clue(
            clue,
            incomplete_assignment,
            size=3,
        )


def test_unsupported_clue_type_raises_error(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_UNSUPPORTED_EXTENSION",
        type="UNSUPPORTED_EXTENSION",
        params={},
    )

    with pytest.raises(
        UnsupportedClueTypeError
    ):
        evaluate_clue(
            clue,
            assignment_3x3,
            size=3,
        )


def test_invalid_counting_k_raises_error(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_INVALID_K",
        type=ClueType.EXACTLY,
        params={
            "k": 4,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    with pytest.raises(
        SemanticEvaluationError
    ):
        evaluate_clue(
            clue,
            assignment_3x3,
            size=3,
        )


def test_invalid_parity_type_raises_error(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_INVALID_PARITY_TYPE",
        type=ClueType.PARITY,
        params={
            "parity": 2,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    with pytest.raises(
        SemanticEvaluationError
    ):
        evaluate_clue(
            clue,
            assignment_3x3,
            size=3,
        )


def test_implies_invalid_condition_status_raises_error(
    assignment_3x3,
):
    clue = Clue(
        id="TEST_INVALID_IMPLIES_STATUS",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": "CRIMINAL",
            },
            "consequent": {
                "person": "C1",
                "status": Status.INNOCENT,
            },
        },
    )

    with pytest.raises(
        SemanticEvaluationError
    ):
        evaluate_clue(
            clue,
            assignment_3x3,
            size=3,
        )


# ============================================================
# Integration test: puzzle_3x3_01
# ============================================================

def test_basic_3x3_puzzle_all_clues_are_true():
    """
    Load the first real project puzzle and verify that every clue is
    semantically consistent with its hidden solution.

    puzzle_3x3_01 intentionally contains only the basic core
    clue types and must remain valid after adding extensions.
    """
    puzzle_path = (
        Path(__file__).resolve().parents[1]
        / "puzzles"
        / "puzzle_3x3_01.json"
    )

    puzzle = load_puzzle(
        puzzle_path
    )

    hidden_assignment = {
        character_id:
            secret.status
        for character_id, secret
        in puzzle.secrets.items()
    }

    clues = [
        secret.clue
        for secret
        in puzzle.secrets.values()
    ]

    results = evaluate_clues(
        clues,
        hidden_assignment,
        puzzle.size,
    )

    assert len(results) == 9
    assert all(
        results.values()
    )