from copy import deepcopy
import json
from pathlib import Path

import pytest

from core.models import (
    ClueType,
    RegionType,
    Status,
)
from core.puzzle_loader import (
    PuzzleFormatError,
    load_puzzle,
    puzzle_from_dict,
)


# ============================================================
# Helpers
# ============================================================

def base_2x2_puzzle_data() -> dict:
    """
    Return a minimal valid 2x2 puzzle dictionary.

    Grid:

        A1 B1
        A2 B2

    All clues initially use FACT so individual tests can replace
    one clue with PARITY or IMPLIES without rebuilding the whole
    puzzle.
    """
    return {
        "name": "2x2 Loader Test",
        "size": 2,
        "characters": [
            {
                "id": "A1",
                "name": "Alice",
                "profession": "Doctor",
                "status": "CRIMINAL",
                "clue": {
                    "id": "CL_A1",
                    "type": "FACT",
                    "params": {
                        "person": "A1",
                        "status": "CRIMINAL",
                    },
                },
            },
            {
                "id": "B1",
                "name": "Ben",
                "profession": "Teacher",
                "status": "INNOCENT",
                "clue": {
                    "id": "CL_B1",
                    "type": "FACT",
                    "params": {
                        "person": "B1",
                        "status": "INNOCENT",
                    },
                },
            },
            {
                "id": "A2",
                "name": "Cara",
                "profession": "Engineer",
                "status": "CRIMINAL",
                "clue": {
                    "id": "CL_A2",
                    "type": "FACT",
                    "params": {
                        "person": "A2",
                        "status": "CRIMINAL",
                    },
                },
            },
            {
                "id": "B2",
                "name": "Daniel",
                "profession": "Writer",
                "status": "INNOCENT",
                "clue": {
                    "id": "CL_B2",
                    "type": "FACT",
                    "params": {
                        "person": "B2",
                        "status": "INNOCENT",
                    },
                },
            },
        ],
        "initial_revealed": [
            "A1",
        ],
    }


def with_a1_clue(
    clue: dict,
) -> dict:
    """
    Return a fresh valid puzzle with A1's clue replaced.
    """
    data = deepcopy(
        base_2x2_puzzle_data()
    )

    data["characters"][0]["clue"] = clue

    return data


def puzzle_3x3_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "puzzles"
        / "puzzle_3x3_01.json"
    )


# ============================================================
# Existing core behavior
# ============================================================

def test_base_2x2_puzzle_loads():
    puzzle = puzzle_from_dict(
        base_2x2_puzzle_data()
    )

    assert puzzle.name == "2x2 Loader Test"
    assert puzzle.size == 2

    assert tuple(
        puzzle.characters
    ) == (
        "A1",
        "B1",
        "A2",
        "B2",
    )

    assert puzzle.initial_revealed == (
        "A1",
    )


def test_existing_basic_3x3_puzzle_still_loads():
    """
    Adding extension parsing must not break the original
    core-clue puzzle.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    assert puzzle.size == 3
    assert len(
        puzzle.characters
    ) == 9

    assert len(
        puzzle.secrets
    ) == 9


# ============================================================
# PARITY - valid parsing
# ============================================================

def test_parity_clue_is_parsed():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": "EVEN",
                "region": {
                    "type": "ROW",
                    "value": 1,
                },
            },
        }
    )

    puzzle = puzzle_from_dict(
        data
    )

    clue = puzzle.secrets[
        "A1"
    ].clue

    assert (
        clue.type
        == ClueType.PARITY
    )

    assert (
        clue.params["parity"]
        == "EVEN"
    )

    region = clue.params[
        "region"
    ]

    assert (
        region.type
        == RegionType.ROW
    )

    assert region.value == 1


def test_parity_normalizes_case_and_whitespace():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": " parity ",
            "params": {
                "parity": " odd ",
                "region": {
                    "type": " column ",
                    "value": " b ",
                },
            },
        }
    )

    puzzle = puzzle_from_dict(
        data
    )

    clue = puzzle.secrets[
        "A1"
    ].clue

    assert (
        clue.type
        == ClueType.PARITY
    )

    assert (
        clue.params["parity"]
        == "ODD"
    )

    region = clue.params[
        "region"
    ]

    assert (
        region.type
        == RegionType.COLUMN
    )

    assert region.value == "B"


def test_parity_supports_neighbors_region():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": "EVEN",
                "region": {
                    "type": "NEIGHBORS",
                    "value": "B2",
                },
            },
        }
    )

    puzzle = puzzle_from_dict(
        data
    )

    region = (
        puzzle
        .secrets["A1"]
        .clue
        .params["region"]
    )

    assert (
        region.type
        == RegionType.NEIGHBORS
    )

    assert region.value == "B2"


def test_parity_supports_explicit_region():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": "ODD",
                "region": {
                    "type": "EXPLICIT",
                    "value": [
                        "A1",
                        "B2",
                    ],
                },
            },
        }
    )

    puzzle = puzzle_from_dict(
        data
    )

    region = (
        puzzle
        .secrets["A1"]
        .clue
        .params["region"]
    )

    assert (
        region.type
        == RegionType.EXPLICIT
    )

    assert region.value == (
        "A1",
        "B2",
    )


# ============================================================
# PARITY - invalid input
# ============================================================

def test_parity_rejects_invalid_value():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": "PRIME",
                "region": {
                    "type": "ROW",
                    "value": 1,
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_parity_rejects_non_string_value():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": 2,
                "region": {
                    "type": "ROW",
                    "value": 1,
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_parity_rejects_missing_region():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": "EVEN",
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_parity_rejects_unexpected_parameter():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": "EVEN",
                "region": {
                    "type": "ROW",
                    "value": 1,
                },
                "extra": True,
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_parity_rejects_invalid_region_reference():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": "ODD",
                "region": {
                    "type": "NEIGHBORS",
                    "value": "Z9",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_parity_rejects_duplicate_explicit_cells():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "PARITY",
            "params": {
                "parity": "EVEN",
                "region": {
                    "type": "EXPLICIT",
                    "value": [
                        "A1",
                        "A1",
                    ],
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


# ============================================================
# IMPLIES - valid parsing
# ============================================================

def test_implies_clue_is_parsed():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "A1",
                    "status": "CRIMINAL",
                },
                "consequent": {
                    "person": "B2",
                    "status": "INNOCENT",
                },
            },
        }
    )

    puzzle = puzzle_from_dict(
        data
    )

    clue = puzzle.secrets[
        "A1"
    ].clue

    assert (
        clue.type
        == ClueType.IMPLIES
    )

    assert clue.params[
        "antecedent"
    ] == {
        "person": "A1",
        "status": Status.CRIMINAL,
    }

    assert clue.params[
        "consequent"
    ] == {
        "person": "B2",
        "status": Status.INNOCENT,
    }


def test_implies_normalizes_case_and_whitespace():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": " implies ",
            "params": {
                "antecedent": {
                    "person": " a1 ",
                    "status": " criminal ",
                },
                "consequent": {
                    "person": " b2 ",
                    "status": " innocent ",
                },
            },
        }
    )

    puzzle = puzzle_from_dict(
        data
    )

    clue = puzzle.secrets[
        "A1"
    ].clue

    assert (
        clue.type
        == ClueType.IMPLIES
    )

    assert (
        clue.params[
            "antecedent"
        ]["person"]
        == "A1"
    )

    assert (
        clue.params[
            "antecedent"
        ]["status"]
        == Status.CRIMINAL
    )

    assert (
        clue.params[
            "consequent"
        ]["person"]
        == "B2"
    )

    assert (
        clue.params[
            "consequent"
        ]["status"]
        == Status.INNOCENT
    )


def test_implies_supports_innocent_antecedent():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "B1",
                    "status": "INNOCENT",
                },
                "consequent": {
                    "person": "A2",
                    "status": "CRIMINAL",
                },
            },
        }
    )

    puzzle = puzzle_from_dict(
        data
    )

    clue = puzzle.secrets[
        "A1"
    ].clue

    assert (
        clue.params[
            "antecedent"
        ]["status"]
        == Status.INNOCENT
    )

    assert (
        clue.params[
            "consequent"
        ]["status"]
        == Status.CRIMINAL
    )


# ============================================================
# IMPLIES - invalid input
# ============================================================

def test_implies_rejects_same_person():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "A1",
                    "status": "CRIMINAL",
                },
                "consequent": {
                    "person": "A1",
                    "status": "INNOCENT",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_implies_rejects_invalid_antecedent_person():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "Z9",
                    "status": "CRIMINAL",
                },
                "consequent": {
                    "person": "B1",
                    "status": "INNOCENT",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_implies_rejects_invalid_consequent_person():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "A1",
                    "status": "CRIMINAL",
                },
                "consequent": {
                    "person": "Z9",
                    "status": "INNOCENT",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_implies_rejects_invalid_status():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "A1",
                    "status": "UNKNOWN",
                },
                "consequent": {
                    "person": "B1",
                    "status": "INNOCENT",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_implies_rejects_missing_consequent():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "A1",
                    "status": "CRIMINAL",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_implies_rejects_condition_that_is_not_object():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": "A1 is Criminal",
                "consequent": {
                    "person": "B1",
                    "status": "INNOCENT",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_implies_rejects_missing_condition_status():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "A1",
                },
                "consequent": {
                    "person": "B1",
                    "status": "INNOCENT",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_implies_rejects_extra_condition_key():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "A1",
                    "status": "CRIMINAL",
                    "extra": True,
                },
                "consequent": {
                    "person": "B1",
                    "status": "INNOCENT",
                },
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


def test_implies_rejects_unexpected_top_level_parameter():
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "IMPLIES",
            "params": {
                "antecedent": {
                    "person": "A1",
                    "status": "CRIMINAL",
                },
                "consequent": {
                    "person": "B1",
                    "status": "INNOCENT",
                },
                "extra": True,
            },
        }
    )

    with pytest.raises(
        PuzzleFormatError
    ):
        puzzle_from_dict(
            data
        )


# ============================================================
# Unknown future extension behavior
# ============================================================

def test_unknown_future_extension_is_preserved_as_string():
    """
    The architecture intentionally permits unknown future clue
    types to pass through the loader as strings.

    SemanticEvaluator / CNFEncoder remain responsible for
    rejecting unsupported clue types during reasoning.
    """
    data = with_a1_clue(
        {
            "id": "CL_A1",
            "type": "FUTURE_EXTENSION",
            "params": {
                "anything": 123,
            },
        }
    )

    puzzle = puzzle_from_dict(
        data
    )

    clue = puzzle.secrets[
        "A1"
    ].clue

    assert (
        clue.type
        == "FUTURE_EXTENSION"
    )

    assert clue.params == {
        "anything": 123,
    }


# ============================================================
# JSON file integration
# ============================================================

def test_load_puzzle_from_json_supports_both_extensions(
    tmp_path,
):
    """
    Full JSON -> Puzzle integration check.

    A1 uses PARITY.
    B1 uses IMPLIES.
    """
    data = deepcopy(
        base_2x2_puzzle_data()
    )

    data[
        "characters"
    ][0]["clue"] = {
        "id": "CL_A1",
        "type": "PARITY",
        "params": {
            "parity": "EVEN",
            "region": {
                "type": "ROW",
                "value": 1,
            },
        },
    }

    data[
        "characters"
    ][1]["clue"] = {
        "id": "CL_B1",
        "type": "IMPLIES",
        "params": {
            "antecedent": {
                "person": "A1",
                "status": "CRIMINAL",
            },
            "consequent": {
                "person": "B2",
                "status": "INNOCENT",
            },
        },
    }

    file_path = (
        tmp_path
        / "extensions_2x2.json"
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
        )

    puzzle = load_puzzle(
        file_path
    )

    parity_clue = (
        puzzle
        .secrets["A1"]
        .clue
    )

    implies_clue = (
        puzzle
        .secrets["B1"]
        .clue
    )

    assert (
        parity_clue.type
        == ClueType.PARITY
    )

    assert (
        implies_clue.type
        == ClueType.IMPLIES
    )

    assert (
        parity_clue.params[
            "parity"
        ]
        == "EVEN"
    )

    assert (
        implies_clue.params[
            "antecedent"
        ]["status"]
        == Status.CRIMINAL
    )

    assert (
        implies_clue.params[
            "consequent"
        ]["status"]
        == Status.INNOCENT
    )