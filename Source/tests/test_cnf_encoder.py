from itertools import product
from pathlib import Path

import pytest

from core.game_engine import GameEngine
from core.models import (
    Character,
    Clue,
    ClueType,
    Region,
    RegionType,
    Status,
)
from core.puzzle_loader import (
    expected_cell_ids,
    load_puzzle,
)
from logic.cnf_encoder import (
    CNFEncoder,
    CNFEncodingError,
    UnknownCharacterError,
    UnsupportedClueTypeError,
)
from logic.semantic_evaluator import (
    evaluate_clue,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def characters_3x3():
    """
    Public character mapping in deterministic row-major order:

        A1 B1 C1
        A2 B2 C2
        A3 B3 C3
    """
    return {
        cell_id: Character(
            id=cell_id,
            name=f"Person {cell_id}",
            profession="Tester",
        )
        for cell_id in expected_cell_ids(3)
    }


@pytest.fixture
def encoder_3x3(
    characters_3x3,
):
    return CNFEncoder(
        characters=characters_3x3,
        size=3,
    )


# ============================================================
# Helper functions
# ============================================================

def assignment_to_sat_values(
    encoder: CNFEncoder,
    assignment: dict[str, Status],
) -> dict[int, bool]:
    """
    Convert semantic character statuses to SAT Boolean values.

        CRIMINAL -> True
        INNOCENT -> False
    """
    return {
        encoder.variable_for(
            character_id
        ): (
            status
            == Status.CRIMINAL
        )
        for character_id, status
        in assignment.items()
    }


def clause_is_satisfied(
    clause: list[int],
    sat_values: dict[int, bool],
) -> bool:
    """
    Evaluate one CNF clause under a complete SAT assignment.
    """
    for literal in clause:
        variable = abs(
            literal
        )

        value = sat_values[
            variable
        ]

        if (
            literal > 0
            and value
        ):
            return True

        if (
            literal < 0
            and not value
        ):
            return True

    return False


def cnf_is_satisfied(
    cnf: list[list[int]],
    sat_values: dict[int, bool],
) -> bool:
    """
    Evaluate a complete CNF formula.

    An empty CNF represents True.
    An empty clause represents False.
    """
    return all(
        clause_is_satisfied(
            clause,
            sat_values,
        )
        for clause in cnf
    )


def all_assignments_3x3():
    """
    Enumerate all:

        2^9 = 512

    Criminal/Innocent assignments of a 3x3 board.
    """
    cells = expected_cell_ids(
        3
    )

    for values in product(
        (
            Status.INNOCENT,
            Status.CRIMINAL,
        ),
        repeat=len(cells),
    ):
        yield dict(
            zip(
                cells,
                values,
            )
        )


def assert_semantic_cnf_equivalence(
    encoder: CNFEncoder,
    clue: Clue,
):
    """
    Exhaustively verify:

        direct semantic evaluation
            ==
        generated CNF satisfaction

    for all 512 possible 3x3 assignments.

    This is the main correctness oracle for clue encoding.
    """
    cnf = encoder.encode_clue(
        clue
    )

    for assignment in all_assignments_3x3():

        semantic_result = (
            evaluate_clue(
                clue,
                assignment,
                size=3,
            )
        )

        sat_values = (
            assignment_to_sat_values(
                encoder,
                assignment,
            )
        )

        cnf_result = (
            cnf_is_satisfied(
                cnf,
                sat_values,
            )
        )

        assert (
            cnf_result
            == semantic_result
        ), (
            f"Semantic/CNF mismatch for clue "
            f"'{clue.id}' under assignment "
            f"{assignment}"
        )


def puzzle_3x3_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "puzzles"
        / "puzzle_3x3_01.json"
    )


# ============================================================
# Variable mapping
# ============================================================

def test_variable_mapping_is_row_major(
    encoder_3x3,
):
    assert encoder_3x3.variable_map == {
        "A1": 1,
        "B1": 2,
        "C1": 3,
        "A2": 4,
        "B2": 5,
        "C2": 6,
        "A3": 7,
        "B3": 8,
        "C3": 9,
    }


def test_reverse_variable_mapping(
    encoder_3x3,
):
    assert (
        encoder_3x3
        .character_for_variable(1)
        == "A1"
    )

    assert (
        encoder_3x3
        .character_for_variable(5)
        == "B2"
    )

    assert (
        encoder_3x3
        .character_for_variable(-9)
        == "C3"
    )


def test_literal_for_status(
    encoder_3x3,
):
    assert (
        encoder_3x3
        .literal_for_status(
            "A1",
            Status.CRIMINAL,
        )
        == 1
    )

    assert (
        encoder_3x3
        .literal_for_status(
            "A1",
            Status.INNOCENT,
        )
        == -1
    )


def test_variable_lookup_normalizes_character_id(
    encoder_3x3,
):
    assert (
        encoder_3x3.variable_for(
            " a1 "
        )
        == 1
    )


# ============================================================
# FACT encoding
# ============================================================

def test_fact_criminal_encoding(
    encoder_3x3,
):
    clue = Clue(
        id="FACT_CRIMINAL",
        type=ClueType.FACT,
        params={
            "person": "A1",
            "status": Status.CRIMINAL,
        },
    )

    assert (
        encoder_3x3.encode_fact(
            clue
        )
        == [
            [1],
        ]
    )


def test_fact_innocent_encoding(
    encoder_3x3,
):
    clue = Clue(
        id="FACT_INNOCENT",
        type=ClueType.FACT,
        params={
            "person": "B1",
            "status": Status.INNOCENT,
        },
    )

    assert (
        encoder_3x3.encode_fact(
            clue
        )
        == [
            [-2],
        ]
    )


# ============================================================
# SAME encoding
# ============================================================

def test_same_encoding(
    encoder_3x3,
):
    clue = Clue(
        id="SAME",
        type=ClueType.SAME,
        params={
            "people": (
                "A1",
                "B1",
            ),
        },
    )

    assert (
        encoder_3x3.encode_same(
            clue
        )
        == [
            [-1, 2],
            [1, -2],
        ]
    )


# ============================================================
# DIFFERENT encoding
# ============================================================

def test_different_encoding(
    encoder_3x3,
):
    clue = Clue(
        id="DIFFERENT",
        type=ClueType.DIFFERENT,
        params={
            "people": (
                "A1",
                "B1",
            ),
        },
    )

    assert (
        encoder_3x3.encode_different(
            clue
        )
        == [
            [1, 2],
            [-1, -2],
        ]
    )


# ============================================================
# AT_MOST encoding
# ============================================================

def test_at_most_one_encoding(
    encoder_3x3,
):
    """
    AT_MOST(1, row 1)

    For:

        A1, B1, C1

    every pair cannot simultaneously be Criminal:

        (-A1 OR -B1)
        (-A1 OR -C1)
        (-B1 OR -C1)
    """
    clue = Clue(
        id="AT_MOST_ONE",
        type=ClueType.AT_MOST,
        params={
            "k": 1,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_at_most(
            clue
        )
        == [
            [-1, -2],
            [-1, -3],
            [-2, -3],
        ]
    )


# ============================================================
# AT_LEAST encoding
# ============================================================

def test_at_least_two_encoding(
    encoder_3x3,
):
    """
    AT_LEAST(2, row 1)

    Every pair must contain at least one Criminal:

        (A1 OR B1)
        (A1 OR C1)
        (B1 OR C1)
    """
    clue = Clue(
        id="AT_LEAST_TWO",
        type=ClueType.AT_LEAST,
        params={
            "k": 2,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_at_least(
            clue
        )
        == [
            [1, 2],
            [1, 3],
            [2, 3],
        ]
    )


# ============================================================
# EXACTLY encoding
# ============================================================

def test_exactly_one_encoding(
    encoder_3x3,
):
    """
    EXACTLY(1, row 1)

    At least one:

        (A1 OR B1 OR C1)

    At most one:

        (-A1 OR -B1)
        (-A1 OR -C1)
        (-B1 OR -C1)
    """
    clue = Clue(
        id="EXACTLY_ONE",
        type=ClueType.EXACTLY,
        params={
            "k": 1,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_exactly(
            clue
        )
        == [
            [1, 2, 3],
            [-1, -2],
            [-1, -3],
            [-2, -3],
        ]
    )


# ============================================================
# Boundary cardinality cases
# ============================================================

def test_at_least_zero_is_tautology(
    encoder_3x3,
):
    clue = Clue(
        id="AT_LEAST_ZERO",
        type=ClueType.AT_LEAST,
        params={
            "k": 0,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_clue(
            clue
        )
        == []
    )


def test_at_most_region_size_is_tautology(
    encoder_3x3,
):
    clue = Clue(
        id="AT_MOST_ALL",
        type=ClueType.AT_MOST,
        params={
            "k": 3,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_clue(
            clue
        )
        == []
    )


def test_exactly_zero(
    encoder_3x3,
):
    clue = Clue(
        id="EXACTLY_ZERO",
        type=ClueType.EXACTLY,
        params={
            "k": 0,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_clue(
            clue
        )
        == [
            [-1],
            [-2],
            [-3],
        ]
    )


def test_exactly_region_size(
    encoder_3x3,
):
    clue = Clue(
        id="EXACTLY_ALL",
        type=ClueType.EXACTLY,
        params={
            "k": 3,
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_clue(
            clue
        )
        == [
            [1],
            [2],
            [3],
        ]
    )


# ============================================================
# Extension 1: PARITY exact encoding
# ============================================================

def test_parity_even_three_variables_encoding(
    encoder_3x3,
):
    """
    EVEN(A1, B1, C1)

    Valid assignments have 0 or 2 Criminals.

    Invalid assignments have 1 or 3 Criminals:

        F F T
        F T F
        T F F
        T T T

    Each invalid assignment is blocked by one clause.
    """
    clue = Clue(
        id="PARITY_EVEN_ROW_1",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_parity(
            clue
        )
        == [
            [1, 2, -3],
            [1, -2, 3],
            [-1, 2, 3],
            [-1, -2, -3],
        ]
    )


def test_parity_odd_three_variables_encoding(
    encoder_3x3,
):
    """
    ODD(A1, B1, C1)

    Invalid assignments have an even number of Criminals:

        F F F
        F T T
        T F T
        T T F
    """
    clue = Clue(
        id="PARITY_ODD_ROW_1",
        type=ClueType.PARITY,
        params={
            "parity": "ODD",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_parity(
            clue
        )
        == [
            [1, 2, 3],
            [1, -2, -3],
            [-1, 2, -3],
            [-1, -2, 3],
        ]
    )


def test_parity_dispatch_encoding(
    encoder_3x3,
):
    clue = Clue(
        id="PARITY_DISPATCH",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_clue(
            clue
        )
        == encoder_3x3.encode_parity(
            clue
        )
    )


def test_parity_lowercase_value_is_normalized(
    encoder_3x3,
):
    clue_upper = Clue(
        id="PARITY_UPPER",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    clue_lower = Clue(
        id="PARITY_LOWER",
        type=ClueType.PARITY,
        params={
            "parity": " even ",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    assert (
        encoder_3x3.encode_clue(
            clue_lower
        )
        == encoder_3x3.encode_clue(
            clue_upper
        )
    )


def test_parity_row_of_three_has_four_clauses(
    encoder_3x3,
):
    """
    For n variables, exactly half of all assignments have
    the wrong parity.

    Therefore PARITY on 3 variables blocks:

        2^(3-1) = 4

    assignments.
    """
    clue = Clue(
        id="PARITY_CLAUSE_COUNT",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    cnf = encoder_3x3.encode_clue(
        clue
    )

    assert len(cnf) == 4


def test_parity_neighbors_b2_has_128_clauses(
    encoder_3x3,
):
    """
    B2 has eight neighbors on a 3x3 board.

    PARITY therefore blocks:

        2^(8-1) = 128

    wrong-parity assignments.
    """
    clue = Clue(
        id="PARITY_NEIGHBORS",
        type=ClueType.PARITY,
        params={
            "parity": "ODD",
            "region": Region(
                type=RegionType.NEIGHBORS,
                value="B2",
            ),
        },
    )

    cnf = encoder_3x3.encode_clue(
        clue
    )

    assert len(cnf) == 128


# ============================================================
# Extension 2: IMPLIES exact encoding
# ============================================================

def test_implies_criminal_to_criminal_encoding(
    encoder_3x3,
):
    """
    Criminal(A1) -> Criminal(B1)

    A1 -> B1

    CNF:

        NOT A1 OR B1
    """
    clue = Clue(
        id="IMPLIES_CC",
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

    assert (
        encoder_3x3.encode_implies(
            clue
        )
        == [
            [-1, 2],
        ]
    )


def test_implies_criminal_to_innocent_encoding(
    encoder_3x3,
):
    """
    Criminal(A1) -> Innocent(B1)

    CNF:

        NOT A1 OR NOT B1
    """
    clue = Clue(
        id="IMPLIES_CI",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "B1",
                "status": Status.INNOCENT,
            },
        },
    )

    assert (
        encoder_3x3.encode_implies(
            clue
        )
        == [
            [-1, -2],
        ]
    )


def test_implies_innocent_to_criminal_encoding(
    encoder_3x3,
):
    """
    Innocent(A1) -> Criminal(B1)

    Antecedent literal:

        NOT A1

    Negating the antecedent gives:

        A1

    Therefore:

        A1 OR B1
    """
    clue = Clue(
        id="IMPLIES_IC",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.INNOCENT,
            },
            "consequent": {
                "person": "B1",
                "status": Status.CRIMINAL,
            },
        },
    )

    assert (
        encoder_3x3.encode_implies(
            clue
        )
        == [
            [1, 2],
        ]
    )


def test_implies_innocent_to_innocent_encoding(
    encoder_3x3,
):
    """
    Innocent(A1) -> Innocent(B1)

    CNF:

        A1 OR NOT B1
    """
    clue = Clue(
        id="IMPLIES_II",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.INNOCENT,
            },
            "consequent": {
                "person": "B1",
                "status": Status.INNOCENT,
            },
        },
    )

    assert (
        encoder_3x3.encode_implies(
            clue
        )
        == [
            [1, -2],
        ]
    )


def test_implies_dispatch_encoding(
    encoder_3x3,
):
    clue = Clue(
        id="IMPLIES_DISPATCH",
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

    assert (
        encoder_3x3.encode_clue(
            clue
        )
        == encoder_3x3.encode_implies(
            clue
        )
    )


# ============================================================
# String clue type normalization
# ============================================================

def test_parity_string_clue_type_is_supported(
    encoder_3x3,
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

    assert (
        encoder_3x3.encode_clue(
            clue
        )
        == [
            [1, 2, -3],
            [1, -2, 3],
            [-1, 2, 3],
            [-1, -2, -3],
        ]
    )


def test_implies_string_clue_type_is_supported(
    encoder_3x3,
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
                "person": "B1",
                "status": Status.CRIMINAL,
            },
        },
    )

    assert (
        encoder_3x3.encode_clue(
            clue
        )
        == [
            [-1, 2],
        ]
    )


# ============================================================
# Semantic equivalence: six core clue types
# ============================================================

@pytest.mark.parametrize(
    "clue",
    [
        Clue(
            id="EQ_FACT",
            type=ClueType.FACT,
            params={
                "person": "A1",
                "status": Status.CRIMINAL,
            },
        ),
        Clue(
            id="EQ_SAME",
            type=ClueType.SAME,
            params={
                "people": (
                    "A1",
                    "B2",
                ),
            },
        ),
        Clue(
            id="EQ_DIFFERENT",
            type=ClueType.DIFFERENT,
            params={
                "people": (
                    "A1",
                    "C3",
                ),
            },
        ),
        Clue(
            id="EQ_EXACTLY",
            type=ClueType.EXACTLY,
            params={
                "k": 2,
                "region": Region(
                    type=RegionType.ROW,
                    value=1,
                ),
            },
        ),
        Clue(
            id="EQ_AT_LEAST",
            type=ClueType.AT_LEAST,
            params={
                "k": 2,
                "region": Region(
                    type=RegionType.COLUMN,
                    value="B",
                ),
            },
        ),
        Clue(
            id="EQ_AT_MOST",
            type=ClueType.AT_MOST,
            params={
                "k": 2,
                "region": Region(
                    type=RegionType.NEIGHBORS,
                    value="B2",
                ),
            },
        ),
    ],
)
def test_core_clue_cnf_matches_semantics_for_all_assignments(
    encoder_3x3,
    clue,
):
    assert_semantic_cnf_equivalence(
        encoder_3x3,
        clue,
    )


# ============================================================
# Semantic equivalence: PARITY
# ============================================================

@pytest.mark.parametrize(
    "clue",
    [
        Clue(
            id="EQ_PARITY_EVEN_ROW",
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
            id="EQ_PARITY_ODD_ROW",
            type=ClueType.PARITY,
            params={
                "parity": "ODD",
                "region": Region(
                    type=RegionType.ROW,
                    value=2,
                ),
            },
        ),
        Clue(
            id="EQ_PARITY_COLUMN",
            type=ClueType.PARITY,
            params={
                "parity": "EVEN",
                "region": Region(
                    type=RegionType.COLUMN,
                    value="C",
                ),
            },
        ),
        Clue(
            id="EQ_PARITY_NEIGHBORS",
            type=ClueType.PARITY,
            params={
                "parity": "ODD",
                "region": Region(
                    type=RegionType.NEIGHBORS,
                    value="B2",
                ),
            },
        ),
        Clue(
            id="EQ_PARITY_EXPLICIT",
            type=ClueType.PARITY,
            params={
                "parity": "EVEN",
                "region": Region(
                    type=RegionType.EXPLICIT,
                    value=(
                        "A1",
                        "B2",
                        "C3",
                    ),
                ),
            },
        ),
    ],
)
def test_parity_cnf_matches_semantics_for_all_assignments(
    encoder_3x3,
    clue,
):
    assert_semantic_cnf_equivalence(
        encoder_3x3,
        clue,
    )


# ============================================================
# Semantic equivalence: IMPLIES
# ============================================================

@pytest.mark.parametrize(
    (
        "antecedent_status",
        "consequent_status",
    ),
    [
        (
            Status.CRIMINAL,
            Status.CRIMINAL,
        ),
        (
            Status.CRIMINAL,
            Status.INNOCENT,
        ),
        (
            Status.INNOCENT,
            Status.CRIMINAL,
        ),
        (
            Status.INNOCENT,
            Status.INNOCENT,
        ),
    ],
)
def test_implies_cnf_matches_semantics_for_all_assignments(
    encoder_3x3,
    antecedent_status,
    consequent_status,
):
    clue = Clue(
        id=(
            "EQ_IMPLIES_"
            f"{antecedent_status.value}_"
            f"{consequent_status.value}"
        ),
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": antecedent_status,
            },
            "consequent": {
                "person": "C3",
                "status": consequent_status,
            },
        },
    )

    assert_semantic_cnf_equivalence(
        encoder_3x3,
        clue,
    )


# ============================================================
# All eight clue types through dispatcher
# ============================================================

@pytest.mark.parametrize(
    "clue",
    [
        Clue(
            id="ALL_FACT",
            type=ClueType.FACT,
            params={
                "person": "A1",
                "status": Status.CRIMINAL,
            },
        ),
        Clue(
            id="ALL_SAME",
            type=ClueType.SAME,
            params={
                "people": (
                    "A1",
                    "B1",
                ),
            },
        ),
        Clue(
            id="ALL_DIFFERENT",
            type=ClueType.DIFFERENT,
            params={
                "people": (
                    "A1",
                    "B1",
                ),
            },
        ),
        Clue(
            id="ALL_EXACTLY",
            type=ClueType.EXACTLY,
            params={
                "k": 1,
                "region": Region(
                    type=RegionType.ROW,
                    value=1,
                ),
            },
        ),
        Clue(
            id="ALL_AT_LEAST",
            type=ClueType.AT_LEAST,
            params={
                "k": 1,
                "region": Region(
                    type=RegionType.ROW,
                    value=1,
                ),
            },
        ),
        Clue(
            id="ALL_AT_MOST",
            type=ClueType.AT_MOST,
            params={
                "k": 1,
                "region": Region(
                    type=RegionType.ROW,
                    value=1,
                ),
            },
        ),
        Clue(
            id="ALL_PARITY",
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
            id="ALL_IMPLIES",
            type=ClueType.IMPLIES,
            params={
                "antecedent": {
                    "person": "A1",
                    "status": Status.CRIMINAL,
                },
                "consequent": {
                    "person": "B1",
                    "status": Status.INNOCENT,
                },
            },
        ),
    ],
)
def test_all_supported_clue_types_encode(
    encoder_3x3,
    clue,
):
    cnf = encoder_3x3.encode_clue(
        clue
    )

    assert isinstance(
        cnf,
        list,
    )

    for clause in cnf:
        assert isinstance(
            clause,
            list,
        )

        assert all(
            isinstance(
                literal,
                int,
            )
            for literal in clause
        )


# ============================================================
# Region coverage
# ============================================================

@pytest.mark.parametrize(
    "region",
    [
        Region(
            type=RegionType.ROW,
            value=2,
        ),
        Region(
            type=RegionType.COLUMN,
            value="C",
        ),
        Region(
            type=RegionType.NEIGHBORS,
            value="B2",
        ),
        Region(
            type=RegionType.EXPLICIT,
            value=(
                "A1",
                "B2",
                "C3",
            ),
        ),
    ],
)
def test_counting_regions_match_semantics(
    encoder_3x3,
    region,
):
    clue = Clue(
        id="REGION_TEST",
        type=ClueType.EXACTLY,
        params={
            "k": 1,
            "region": region,
        },
    )

    assert_semantic_cnf_equivalence(
        encoder_3x3,
        clue,
    )


@pytest.mark.parametrize(
    "region",
    [
        Region(
            type=RegionType.ROW,
            value=2,
        ),
        Region(
            type=RegionType.COLUMN,
            value="C",
        ),
        Region(
            type=RegionType.NEIGHBORS,
            value="B2",
        ),
        Region(
            type=RegionType.EXPLICIT,
            value=(
                "A1",
                "B2",
                "C3",
            ),
        ),
    ],
)
def test_parity_supports_all_core_region_types(
    encoder_3x3,
    region,
):
    clue = Clue(
        id="PARITY_REGION_TEST",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": region,
        },
    )

    assert_semantic_cnf_equivalence(
        encoder_3x3,
        clue,
    )


# ============================================================
# Build KB
# ============================================================

def test_build_kb_uses_proved_statuses_and_revealed_clues(
    encoder_3x3,
):
    revealed_clue = Clue(
        id="PUBLIC_CLUE",
        type=ClueType.SAME,
        params={
            "people": (
                "A1",
                "B1",
            ),
        },
    )

    proved_statuses = {
        "B2": Status.CRIMINAL,
        "C1": Status.INNOCENT,
    }

    kb = encoder_3x3.build_kb(
        revealed_clues=[
            revealed_clue,
        ],
        proved_statuses=proved_statuses,
    )

    assert kb == [
        [5],
        [-3],
        [-1, 2],
        [1, -2],
    ]


def test_build_kb_supports_extension_clues(
    encoder_3x3,
):
    parity_clue = Clue(
        id="PUBLIC_PARITY",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    implies_clue = Clue(
        id="PUBLIC_IMPLIES",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "B1",
                "status": Status.INNOCENT,
            },
        },
    )

    kb = encoder_3x3.build_kb(
        revealed_clues=[
            parity_clue,
            implies_clue,
        ],
        proved_statuses={
            "B2": Status.CRIMINAL,
        },
    )

    assert kb == [
        # B2 = CRIMINAL
        [5],

        # PARITY(EVEN, row 1)
        [1, 2, -3],
        [1, -2, 3],
        [-1, 2, 3],
        [-1, -2, -3],

        # Criminal(A1) -> Innocent(B1)
        [-1, -2],
    ]


# ============================================================
# Statistics
# ============================================================

def test_statistics(
    encoder_3x3,
):
    clue = Clue(
        id="STATISTICS",
        type=ClueType.SAME,
        params={
            "people": (
                "A1",
                "B1",
            ),
        },
    )

    cnf = encoder_3x3.encode_clue(
        clue
    )

    stats = encoder_3x3.get_statistics(
        cnf
    )

    assert (
        stats.primary_variables
        == 9
    )

    assert (
        stats.auxiliary_variables
        == 0
    )

    assert (
        stats.clauses
        == 2
    )


def test_parity_statistics_use_no_auxiliary_variables(
    encoder_3x3,
):
    clue = Clue(
        id="PARITY_STATS",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.ROW,
                value=1,
            ),
        },
    )

    cnf = encoder_3x3.encode_clue(
        clue
    )

    stats = encoder_3x3.get_statistics(
        cnf
    )

    assert (
        stats.primary_variables
        == 9
    )

    assert (
        stats.auxiliary_variables
        == 0
    )

    assert (
        stats.clauses
        == 4
    )


def test_implies_statistics_use_one_clause(
    encoder_3x3,
):
    clue = Clue(
        id="IMPLIES_STATS",
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

    cnf = encoder_3x3.encode_clue(
        clue
    )

    stats = encoder_3x3.get_statistics(
        cnf
    )

    assert stats.clauses == 1
    assert stats.auxiliary_variables == 0


# ============================================================
# Extension error handling: PARITY
# ============================================================

def test_invalid_parity_value_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="INVALID_PARITY",
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
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_non_string_parity_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="INVALID_PARITY_TYPE",
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
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_parity_missing_region_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="PARITY_MISSING_REGION",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
        },
    )

    with pytest.raises(
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_parity_unknown_character_in_region_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="PARITY_UNKNOWN_CHARACTER",
        type=ClueType.PARITY,
        params={
            "parity": "EVEN",
            "region": Region(
                type=RegionType.EXPLICIT,
                value=(
                    "A1",
                    "Z9",
                ),
            ),
        },
    )

    with pytest.raises(
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


# ============================================================
# Extension error handling: IMPLIES
# ============================================================

def test_implies_same_person_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="IMPLIES_SAME_PERSON",
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
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_implies_unknown_antecedent_person_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="IMPLIES_UNKNOWN_ANTECEDENT",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "Z9",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "B1",
                "status": Status.CRIMINAL,
            },
        },
    )

    with pytest.raises(
        UnknownCharacterError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_implies_unknown_consequent_person_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="IMPLIES_UNKNOWN_CONSEQUENT",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
            "consequent": {
                "person": "Z9",
                "status": Status.CRIMINAL,
            },
        },
    )

    with pytest.raises(
        UnknownCharacterError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_implies_invalid_antecedent_status_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="IMPLIES_BAD_STATUS",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": "CRIMINAL",
            },
            "consequent": {
                "person": "B1",
                "status": Status.CRIMINAL,
            },
        },
    )

    with pytest.raises(
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_implies_condition_requires_exact_keys(
    encoder_3x3,
):
    clue = Clue(
        id="IMPLIES_EXTRA_KEY",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
                "extra": True,
            },
            "consequent": {
                "person": "B1",
                "status": Status.CRIMINAL,
            },
        },
    )

    with pytest.raises(
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_implies_missing_consequent_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="IMPLIES_MISSING_CONSEQUENT",
        type=ClueType.IMPLIES,
        params={
            "antecedent": {
                "person": "A1",
                "status": Status.CRIMINAL,
            },
        },
    )

    with pytest.raises(
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


# ============================================================
# General error handling
# ============================================================

def test_unknown_character_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="UNKNOWN",
        type=ClueType.FACT,
        params={
            "person": "Z9",
            "status": Status.CRIMINAL,
        },
    )

    with pytest.raises(
        UnknownCharacterError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_unsupported_clue_type_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="UNKNOWN_EXTENSION",
        type="UNSUPPORTED_EXTENSION",
        params={},
    )

    with pytest.raises(
        UnsupportedClueTypeError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_invalid_k_raises_error(
    encoder_3x3,
):
    clue = Clue(
        id="INVALID_K",
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
        CNFEncodingError
    ):
        encoder_3x3.encode_clue(
            clue
        )


def test_encode_clue_requires_clue_instance(
    encoder_3x3,
):
    with pytest.raises(
        TypeError
    ):
        encoder_3x3.encode_clue(
            "not-a-clue"
        )


# ============================================================
# Integration: puzzle_3x3_01
# ============================================================

def test_basic_3x3_full_clue_set_matches_hidden_solution():
    """
    Encode the complete clue set of puzzle_3x3_01 and verify
    that the designed hidden solution satisfies the CNF.

    puzzle_3x3_01 intentionally contains only core basic clues.
    Adding extensions must not change its behavior.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    encoder = CNFEncoder(
        characters=puzzle.characters,
        size=puzzle.size,
    )

    clues = [
        secret.clue
        for secret
        in puzzle.secrets.values()
    ]

    cnf = encoder.encode_clues(
        clues
    )

    hidden_assignment = {
        character_id:
            secret.status
        for character_id, secret
        in puzzle.secrets.items()
    }

    sat_values = (
        assignment_to_sat_values(
            encoder,
            hidden_assignment,
        )
    )

    assert cnf_is_satisfied(
        cnf,
        sat_values,
    )


def test_basic_3x3_initial_public_kb_contains_no_hidden_information():
    """
    Initial public KB must contain only:

        B2 = CRIMINAL
        CL_B2 = FACT(A1, CRIMINAL)

    Nothing from unrevealed cards may appear.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    public_state = (
        engine.get_public_state()
    )

    encoder = CNFEncoder(
        characters=public_state.characters,
        size=puzzle.size,
    )

    kb = (
        encoder
        .build_kb_from_public_state(
            public_state
        )
    )

    # B2 -> variable 5
    # B2 is initially CRIMINAL.
    #
    # Revealed clue of B2:
    #
    # FACT(A1, CRIMINAL)
    #
    # Therefore:
    #
    #     B2
    #     A1
    assert kb == [
        [5],
        [1],
    ]