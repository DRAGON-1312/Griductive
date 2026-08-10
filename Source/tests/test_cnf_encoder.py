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
from logic.semantic_evaluator import evaluate_clue


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
def encoder_3x3(characters_3x3):
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
    Convert semantic statuses to Boolean SAT values.

    CRIMINAL -> True
    INNOCENT -> False
    """
    return {
        encoder.variable_for(character_id):
            status == Status.CRIMINAL
        for character_id, status
        in assignment.items()
    }


def clause_is_satisfied(
    clause: list[int],
    sat_values: dict[int, bool],
) -> bool:
    """
    Check whether one CNF clause is satisfied.
    """
    for literal in clause:
        variable = abs(literal)
        value = sat_values[variable]

        if literal > 0 and value:
            return True

        if literal < 0 and not value:
            return True

    return False


def cnf_is_satisfied(
    cnf: list[list[int]],
    sat_values: dict[int, bool],
) -> bool:
    """
    A CNF formula is satisfied iff every clause is satisfied.

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
    Enumerate all 2^9 = 512 complete assignments
    for a 3x3 puzzle.
    """
    cells = expected_cell_ids(3)

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

        semantic_evaluator(clue, assignment)
            ==
        encoded_CNF(clue) satisfied by assignment

    for all 512 assignments of a 3x3 grid.
    """
    cnf = encoder.encode_clue(
        clue
    )

    for assignment in all_assignments_3x3():
        semantic_result = evaluate_clue(
            clue,
            assignment,
            size=3,
        )

        sat_values = assignment_to_sat_values(
            encoder,
            assignment,
        )

        cnf_result = cnf_is_satisfied(
            cnf,
            sat_values,
        )

        assert cnf_result == semantic_result, (
            f"Semantic/CNF mismatch for clue "
            f"'{clue.id}' under assignment "
            f"{assignment}"
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
        encoder_3x3.character_for_variable(1)
        == "A1"
    )

    assert (
        encoder_3x3.character_for_variable(5)
        == "B2"
    )

    assert (
        encoder_3x3.character_for_variable(-9)
        == "C3"
    )


def test_literal_for_status(
    encoder_3x3,
):
    assert (
        encoder_3x3.literal_for_status(
            "A1",
            Status.CRIMINAL,
        )
        == 1
    )

    assert (
        encoder_3x3.literal_for_status(
            "A1",
            Status.INNOCENT,
        )
        == -1
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

    assert encoder_3x3.encode_fact(
        clue
    ) == [
        [1]
    ]


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

    assert encoder_3x3.encode_fact(
        clue
    ) == [
        [-2]
    ]


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
            "people": ("A1", "B1"),
        },
    )

    assert encoder_3x3.encode_same(
        clue
    ) == [
        [-1, 2],
        [1, -2],
    ]


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
            "people": ("A1", "B1"),
        },
    )

    assert encoder_3x3.encode_different(
        clue
    ) == [
        [1, 2],
        [-1, -2],
    ]


# ============================================================
# AT_MOST encoding
# ============================================================

def test_at_most_one_encoding(
    encoder_3x3,
):
    """
    AT_MOST(1, row 1)

    For A1, B1, C1:

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

    assert encoder_3x3.encode_at_most(
        clue
    ) == [
        [-1, -2],
        [-1, -3],
        [-2, -3],
    ]


# ============================================================
# AT_LEAST encoding
# ============================================================

def test_at_least_two_encoding(
    encoder_3x3,
):
    """
    AT_LEAST(2, row 1)

    Every pair must contain a Criminal:

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

    assert encoder_3x3.encode_at_least(
        clue
    ) == [
        [1, 2],
        [1, 3],
        [2, 3],
    ]


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

    assert encoder_3x3.encode_exactly(
        clue
    ) == [
        [1, 2, 3],
        [-1, -2],
        [-1, -3],
        [-2, -3],
    ]


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

    assert encoder_3x3.encode_clue(
        clue
    ) == []


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

    assert encoder_3x3.encode_clue(
        clue
    ) == []


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

    assert encoder_3x3.encode_clue(
        clue
    ) == [
        [-1],
        [-2],
        [-3],
    ]


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

    assert encoder_3x3.encode_clue(
        clue
    ) == [
        [1],
        [2],
        [3],
    ]


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
                "people": ("A1", "B2"),
            },
        ),
        Clue(
            id="EQ_DIFFERENT",
            type=ClueType.DIFFERENT,
            params={
                "people": ("A1", "C3"),
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
            value=("A1", "B2", "C3"),
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
            "people": ("A1", "B1"),
        },
    )

    proved_statuses = {
        "B2": Status.CRIMINAL,
        "C1": Status.INNOCENT,
    }

    kb = encoder_3x3.build_kb(
        revealed_clues=[
            revealed_clue
        ],
        proved_statuses=proved_statuses,
    )

    assert kb == [
        [5],
        [-3],
        [-1, 2],
        [1, -2],
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
            "people": ("A1", "B1"),
        },
    )

    cnf = encoder_3x3.encode_clue(
        clue
    )

    stats = encoder_3x3.get_statistics(
        cnf
    )

    assert stats.primary_variables == 9
    assert stats.auxiliary_variables == 0
    assert stats.clauses == 2


# ============================================================
# Error handling
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
        id="EXTENSION",
        type="PARITY",
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


# ============================================================
# Integration: puzzle_3x3_01
# ============================================================

def test_basic_3x3_full_clue_set_matches_hidden_solution():
    """
    Encode the complete clue set of puzzle_3x3_01 and verify
    that the known hidden solution satisfies the resulting CNF.

    This does NOT yet prove uniqueness. That will be tested after
    the DPLL solver and uniqueness checker are implemented.
    """
    puzzle_path = (
        Path(__file__).resolve().parents[1]
        / "puzzles"
        / "puzzle_3x3_01.json"
    )

    puzzle = load_puzzle(
        puzzle_path
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
        character_id: secret.status
        for character_id, secret
        in puzzle.secrets.items()
    }

    sat_values = assignment_to_sat_values(
        encoder,
        hidden_assignment,
    )

    assert cnf_is_satisfied(
        cnf,
        sat_values,
    )


def test_basic_3x3_initial_public_kb_contains_no_hidden_information():
    """
    The initial public KB for puzzle_3x3_01 should contain only:

        B2 = CRIMINAL
        CL_B2 = FACT(A1, CRIMINAL)

    Nothing from unrevealed cards may appear in the KB.
    """
    puzzle_path = (
        Path(__file__).resolve().parents[1]
        / "puzzles"
        / "puzzle_3x3_01.json"
    )

    puzzle = load_puzzle(
        puzzle_path
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

    # B2 -> variable 5
    # B2 is initially known as CRIMINAL.
    #
    # B2's revealed clue:
    # FACT(A1, CRIMINAL)
    #
    # Therefore the initial KB must contain exactly:
    #
    #     B2
    #     A1
    assert kb == [
        [5],
        [1],
    ]