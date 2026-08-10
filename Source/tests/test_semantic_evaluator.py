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

    Therefore exactly 2 criminals.
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

    Therefore there are 2 criminals.
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

    Therefore there is only 1 criminal.
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
    Neighbors of B2 are:

        A1 B1 C1
        A2    C2
        A3 B3 C3

    Criminals:
        A1, B1, C2, C3

    Therefore exactly 4 criminals.
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
    del incomplete_assignment["A1"]

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
        id="TEST_EXTENSION",
        type="PARITY",
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


# ============================================================
# Integration test: puzzle_3x3_01
# ============================================================

def test_basic_3x3_puzzle_all_clues_are_true():
    """
    Load the first real project puzzle and verify that every clue is
    semantically consistent with its hidden solution.

    This is the first integration-level sanity check for the project.
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
        character_id: secret.status
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
    assert all(results.values())