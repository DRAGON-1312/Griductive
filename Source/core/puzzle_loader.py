from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    Character,
    CharacterSecret,
    Clue,
    ClueType,
    Puzzle,
    Region,
    RegionType,
    Status,
)


# ============================================================
# Exceptions
# ============================================================

class PuzzleFormatError(ValueError):
    """
    Raised when a puzzle file has an invalid format or contains
    logically invalid references.
    """


# ============================================================
# Public API
# ============================================================

def load_puzzle(file_path: str | Path) -> Puzzle:
    """
    Load and validate a Griductive puzzle from a JSON file.

    Args:
        file_path:
            Path to the puzzle JSON file.

    Returns:
        A validated Puzzle object.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        PuzzleFormatError:
            If the JSON structure or puzzle content is invalid.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Puzzle file does not exist: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise PuzzleFormatError(
            f"Invalid JSON in '{path}': "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc

    return puzzle_from_dict(data)


def puzzle_from_dict(data: dict[str, Any]) -> Puzzle:
    """
    Build and validate a Puzzle object from a Python dictionary.

    This function is useful for unit tests because a test can construct
    puzzle data directly without creating a temporary JSON file.
    """
    if not isinstance(data, dict):
        raise PuzzleFormatError(
            "Puzzle root must be a JSON object."
        )

    name = _read_non_empty_string(
        data,
        "name",
        context="puzzle"
    )

    size = _read_integer(
        data,
        "size",
        context="puzzle"
    )

    if size <= 0:
        raise PuzzleFormatError(
            "'size' must be a positive integer."
        )

    # Griductive coordinates use letters for columns.
    # The project normally uses N = 3, 4, or 5, so this is more
    # than sufficient while still keeping the loader reusable.
    if size > 26:
        raise PuzzleFormatError(
            "'size' must not exceed 26 because columns use "
            "single letters A-Z."
        )

    valid_cells = expected_cell_ids(size)

    raw_characters = data.get("characters")

    if not isinstance(raw_characters, list):
        raise PuzzleFormatError(
            "'characters' must be a list."
        )

    expected_count = size * size

    if len(raw_characters) != expected_count:
        raise PuzzleFormatError(
            f"A {size}x{size} puzzle must contain exactly "
            f"{expected_count} characters, but "
            f"{len(raw_characters)} were provided."
        )

    characters: dict[str, Character] = {}
    secrets: dict[str, CharacterSecret] = {}

    clue_ids: set[str] = set()

    for index, raw_character in enumerate(raw_characters):
        character, secret = _parse_character(
            raw_character,
            index=index,
            size=size,
            valid_cells=valid_cells,
        )

        character_id = character.id

        if character_id in characters:
            raise PuzzleFormatError(
                f"Duplicate character id: '{character_id}'."
            )

        if secret.clue.id in clue_ids:
            raise PuzzleFormatError(
                f"Duplicate clue id: '{secret.clue.id}'."
            )

        characters[character_id] = character
        secrets[character_id] = secret
        clue_ids.add(secret.clue.id)

    # Ensure every board cell exists exactly once.
    missing_cells = [
        cell
        for cell in valid_cells
        if cell not in characters
    ]

    extra_cells = [
        cell
        for cell in characters
        if cell not in valid_cells
    ]

    if missing_cells:
        raise PuzzleFormatError(
            "Missing characters for cells: "
            + ", ".join(missing_cells)
        )

    if extra_cells:
        raise PuzzleFormatError(
            "Invalid character cells: "
            + ", ".join(extra_cells)
        )

    initial_revealed = _parse_initial_revealed(
        data.get("initial_revealed", []),
        valid_cells=set(valid_cells),
    )

    # Store characters in deterministic row-major order.
    ordered_characters = {
        cell: characters[cell]
        for cell in valid_cells
    }

    ordered_secrets = {
        cell: secrets[cell]
        for cell in valid_cells
    }

    return Puzzle(
        name=name,
        size=size,
        characters=ordered_characters,
        secrets=ordered_secrets,
        initial_revealed=initial_revealed,
    )


# ============================================================
# Character parsing
# ============================================================

def _parse_character(
    data: Any,
    *,
    index: int,
    size: int,
    valid_cells: tuple[str, ...],
) -> tuple[Character, CharacterSecret]:
    """
    Parse one character entry from puzzle JSON.
    """
    context = f"characters[{index}]"

    if not isinstance(data, dict):
        raise PuzzleFormatError(
            f"{context} must be an object."
        )

    character_id = _read_non_empty_string(
        data,
        "id",
        context=context,
    ).upper()

    if character_id not in valid_cells:
        raise PuzzleFormatError(
            f"{context}.id = '{character_id}' is not a valid "
            f"cell for a {size}x{size} board."
        )

    name = _read_non_empty_string(
        data,
        "name",
        context=context,
    )

    profession = _read_non_empty_string(
        data,
        "profession",
        context=context,
    )

    status = _parse_status(
        data.get("status"),
        context=f"{context}.status",
    )

    if "clue" not in data:
        raise PuzzleFormatError(
            f"{context} is missing required field 'clue'."
        )

    clue = _parse_clue(
        data["clue"],
        context=f"{context}.clue",
        size=size,
        valid_cells=set(valid_cells),
    )

    character = Character(
        id=character_id,
        name=name,
        profession=profession,
    )

    secret = CharacterSecret(
        status=status,
        clue=clue,
    )

    return character, secret


# ============================================================
# Clue parsing
# ============================================================

def _parse_clue(
    data: Any,
    *,
    context: str,
    size: int,
    valid_cells: set[str],
) -> Clue:
    """
    Parse and validate one structured clue.

    The six required core clue types are validated here.

    Unknown clue type strings are kept as strings so that extension
    clues can be added later without changing the data model.
    """
    if not isinstance(data, dict):
        raise PuzzleFormatError(
            f"{context} must be an object."
        )

    clue_id = _read_non_empty_string(
        data,
        "id",
        context=context,
    )

    raw_type = _read_non_empty_string(
        data,
        "type",
        context=context,
    ).upper()

    try:
        clue_type: ClueType | str = ClueType(raw_type)
    except ValueError:
        # Unknown future extension clue.
        clue_type = raw_type

    params = data.get("params")

    if not isinstance(params, dict):
        raise PuzzleFormatError(
            f"{context}.params must be an object."
        )

    if clue_type == ClueType.FACT:
        parsed_params = _parse_fact_params(
            params,
            context=f"{context}.params",
            valid_cells=valid_cells,
        )

    elif clue_type in {
        ClueType.SAME,
        ClueType.DIFFERENT,
    }:
        parsed_params = _parse_binary_params(
            params,
            context=f"{context}.params",
            valid_cells=valid_cells,
        )

    elif clue_type in {
        ClueType.EXACTLY,
        ClueType.AT_LEAST,
        ClueType.AT_MOST,
    }:
        parsed_params = _parse_counting_params(
            params,
            context=f"{context}.params",
            size=size,
            valid_cells=valid_cells,
        )

    elif clue_type == ClueType.PARITY:
        parsed_params = _parse_parity_params(
            params,
            context=f"{context}.params",
            size=size,
            valid_cells=valid_cells,
        )

    elif clue_type == ClueType.IMPLIES:
        parsed_params = _parse_implies_params(
            params,
            context=f"{context}.params",
            valid_cells=valid_cells,
        )

    else:
        # Extension clue.
        #
        # At this stage we only require it to contain a params
        # dictionary. Once the two project extensions are finalized,
        # their specific validation can be added here.
        parsed_params = dict(params)

    return Clue(
        id=clue_id,
        type=clue_type,
        params=parsed_params,
    )


# ============================================================
# Core clue parameters
# ============================================================

def _parse_fact_params(
    params: dict[str, Any],
    *,
    context: str,
    valid_cells: set[str],
) -> dict[str, Any]:
    """
    FACT:
        {
            "person": "A1",
            "status": "CRIMINAL"
        }
    """
    _require_exact_keys(
        params,
        required={"person", "status"},
        context=context,
    )

    person = _read_non_empty_string(
        params,
        "person",
        context=context,
    ).upper()

    _validate_cell_reference(
        person,
        valid_cells,
        context=f"{context}.person",
    )

    status = _parse_status(
        params["status"],
        context=f"{context}.status",
    )

    return {
        "person": person,
        "status": status,
    }


def _parse_binary_params(
    params: dict[str, Any],
    *,
    context: str,
    valid_cells: set[str],
) -> dict[str, Any]:
    """
    SAME / DIFFERENT:
        {
            "people": ["A1", "B2"]
        }
    """
    _require_exact_keys(
        params,
        required={"people"},
        context=context,
    )

    people = params["people"]

    if not isinstance(people, (list, tuple)):
        raise PuzzleFormatError(
            f"{context}.people must be a list of two cell ids."
        )

    if len(people) != 2:
        raise PuzzleFormatError(
            f"{context}.people must contain exactly two people."
        )

    first = _normalize_cell_id(
        people[0],
        context=f"{context}.people[0]",
    )

    second = _normalize_cell_id(
        people[1],
        context=f"{context}.people[1]",
    )

    _validate_cell_reference(
        first,
        valid_cells,
        context=f"{context}.people[0]",
    )

    _validate_cell_reference(
        second,
        valid_cells,
        context=f"{context}.people[1]",
    )

    if first == second:
        raise PuzzleFormatError(
            f"{context}.people must reference two distinct people."
        )

    return {
        "people": (first, second),
    }


def _parse_counting_params(
    params: dict[str, Any],
    *,
    context: str,
    size: int,
    valid_cells: set[str],
) -> dict[str, Any]:
    """
    EXACTLY / AT_LEAST / AT_MOST:
        {
            "k": 2,
            "region": {
                "type": "ROW",
                "value": 1
            }
        }
    """
    _require_exact_keys(
        params,
        required={"k", "region"},
        context=context,
    )

    k = params["k"]

    if not _is_integer(k):
        raise PuzzleFormatError(
            f"{context}.k must be an integer."
        )

    region = _parse_region(
        params["region"],
        context=f"{context}.region",
        size=size,
        valid_cells=valid_cells,
    )

    region_cells = resolve_region_cells(
        region,
        size,
    )

    if not 0 <= k <= len(region_cells):
        raise PuzzleFormatError(
            f"{context}.k must satisfy "
            f"0 <= k <= |R|. "
            f"Received k={k}, |R|={len(region_cells)}."
        )

    return {
        "k": k,
        "region": region,
    }

def _parse_parity_params(
    params: dict[str, Any],
    *,
    context: str,
    size: int,
    valid_cells: set[str],
) -> dict[str, Any]:
    """
    PARITY:
        {
            "parity": "EVEN" | "ODD",
            "region": {...}
        }
    """
    _require_exact_keys(
        params,
        required={"parity", "region"},
        context=context,
    )

    parity = _read_non_empty_string(
        params,
        "parity",
        context=context,
    ).upper()

    if parity not in {
        "EVEN",
        "ODD",
    }:
        raise PuzzleFormatError(
            f"{context}.parity must be "
            f"'EVEN' or 'ODD'."
        )

    region = _parse_region(
        params["region"],
        context=f"{context}.region",
        size=size,
        valid_cells=valid_cells,
    )

    return {
        "parity": parity,
        "region": region,
    }

# helper
def _parse_status_condition(
    data: Any,
    *,
    context: str,
    valid_cells: set[str],
) -> dict[str, Any]:
    """
    One status condition:

        {
            "person": "A1",
            "status": "CRIMINAL"
        }
    """
    if not isinstance(
        data,
        dict,
    ):
        raise PuzzleFormatError(
            f"{context} must be an object."
        )

    _require_exact_keys(
        data,
        required={
            "person",
            "status",
        },
        context=context,
    )

    person = _read_non_empty_string(
        data,
        "person",
        context=context,
    ).upper()

    _validate_cell_reference(
        person,
        valid_cells,
        context=f"{context}.person",
    )

    status = _parse_status(
        data["status"],
        context=f"{context}.status",
    )

    return {
        "person": person,
        "status": status,
    }

def _parse_implies_params(
    params: dict[str, Any],
    *,
    context: str,
    valid_cells: set[str],
) -> dict[str, Any]:
    """
    IMPLIES:

        antecedent -> consequent
    """
    _require_exact_keys(
        params,
        required={
            "antecedent",
            "consequent",
        },
        context=context,
    )

    antecedent = _parse_status_condition(
        params["antecedent"],
        context=f"{context}.antecedent",
        valid_cells=valid_cells,
    )

    consequent = _parse_status_condition(
        params["consequent"],
        context=f"{context}.consequent",
        valid_cells=valid_cells,
    )

    if (
        antecedent["person"]
        == consequent["person"]
    ):
        raise PuzzleFormatError(
            f"{context}: IMPLIES should reference "
            f"two distinct characters."
        )

    return {
        "antecedent": antecedent,
        "consequent": consequent,
    }

# ============================================================
# Region parsing
# ============================================================

def _parse_region(
    data: Any,
    *,
    context: str,
    size: int,
    valid_cells: set[str],
) -> Region:
    """
    Parse one of the four required core region types:

        ROW
        COLUMN
        NEIGHBORS
        EXPLICIT
    """
    if not isinstance(data, dict):
        raise PuzzleFormatError(
            f"{context} must be an object."
        )

    _require_exact_keys(
        data,
        required={"type", "value"},
        context=context,
    )

    raw_type = _read_non_empty_string(
        data,
        "type",
        context=context,
    ).upper()

    try:
        region_type = RegionType(raw_type)
    except ValueError as exc:
        allowed = ", ".join(
            region.value
            for region in RegionType
        )

        raise PuzzleFormatError(
            f"{context}.type = '{raw_type}' is invalid. "
            f"Expected one of: {allowed}."
        ) from exc

    value = data["value"]

    # --------------------------------------------------------
    # ROW
    # --------------------------------------------------------

    if region_type == RegionType.ROW:
        if not _is_integer(value):
            raise PuzzleFormatError(
                f"{context}.value must be an integer "
                f"for a ROW region."
            )

        if not 1 <= value <= size:
            raise PuzzleFormatError(
                f"{context}.value = {value} is outside "
                f"the valid row range 1..{size}."
            )

        return Region(
            type=region_type,
            value=value,
        )

    # --------------------------------------------------------
    # COLUMN
    # --------------------------------------------------------

    if region_type == RegionType.COLUMN:
        if not isinstance(value, str):
            raise PuzzleFormatError(
                f"{context}.value must be a column letter "
                f"for a COLUMN region."
            )

        column = value.strip().upper()

        valid_columns = {
            chr(ord("A") + i)
            for i in range(size)
        }

        if column not in valid_columns:
            raise PuzzleFormatError(
                f"{context}.value = '{column}' is not a valid "
                f"column for a {size}x{size} board."
            )

        return Region(
            type=region_type,
            value=column,
        )

    # --------------------------------------------------------
    # NEIGHBORS
    # --------------------------------------------------------

    if region_type == RegionType.NEIGHBORS:
        cell = _normalize_cell_id(
            value,
            context=f"{context}.value",
        )

        _validate_cell_reference(
            cell,
            valid_cells,
            context=f"{context}.value",
        )

        return Region(
            type=region_type,
            value=cell,
        )

    # --------------------------------------------------------
    # EXPLICIT
    # --------------------------------------------------------

    if not isinstance(value, (list, tuple)):
        raise PuzzleFormatError(
            f"{context}.value must be a list of cell ids "
            f"for an EXPLICIT region."
        )

    cells: list[str] = []

    for index, raw_cell in enumerate(value):
        cell = _normalize_cell_id(
            raw_cell,
            context=f"{context}.value[{index}]",
        )

        _validate_cell_reference(
            cell,
            valid_cells,
            context=f"{context}.value[{index}]",
        )

        cells.append(cell)

    if len(cells) != len(set(cells)):
        raise PuzzleFormatError(
            f"{context}.value contains duplicate cell ids."
        )

    return Region(
        type=region_type,
        value=tuple(cells),
    )


# ============================================================
# Region resolution
# ============================================================

def resolve_region_cells(
    region: Region,
    size: int,
) -> tuple[str, ...]:
    """
    Convert a structured Region into the concrete cell ids it contains.

    The returned order is deterministic.

    Examples:
        ROW 2 on 3x3
            -> ("A2", "B2", "C2")

        COLUMN B on 3x3
            -> ("B1", "B2", "B3")

        NEIGHBORS B2 on 3x3
            -> all eight surrounding cells

    The cell itself is excluded from a NEIGHBORS region.
    """
    if region.type == RegionType.ROW:
        row = int(region.value)

        return tuple(
            f"{chr(ord('A') + column)}{row}"
            for column in range(size)
        )

    if region.type == RegionType.COLUMN:
        column = str(region.value)

        return tuple(
            f"{column}{row}"
            for row in range(1, size + 1)
        )

    if region.type == RegionType.NEIGHBORS:
        center = str(region.value)

        column_index, row_index = _cell_to_indices(
            center,
            size,
        )

        neighbors: list[str] = []

        # Deterministic row-major order.
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):

                if row_delta == 0 and column_delta == 0:
                    continue

                new_row = row_index + row_delta
                new_column = column_index + column_delta

                if (
                    0 <= new_row < size
                    and 0 <= new_column < size
                ):
                    neighbors.append(
                        _indices_to_cell(
                            new_column,
                            new_row,
                        )
                    )

        return tuple(neighbors)

    if region.type == RegionType.EXPLICIT:
        return tuple(region.value)

    raise PuzzleFormatError(
        f"Unsupported region type: {region.type}"
    )


# ============================================================
# Initial revealed cards
# ============================================================

def _parse_initial_revealed(
    data: Any,
    *,
    valid_cells: set[str],
) -> tuple[str, ...]:
    """
    Parse the list of cards that are face-up when the puzzle starts.
    """
    if not isinstance(data, (list, tuple)):
        raise PuzzleFormatError(
            "'initial_revealed' must be a list."
        )

    revealed: list[str] = []

    for index, raw_cell in enumerate(data):
        cell = _normalize_cell_id(
            raw_cell,
            context=f"initial_revealed[{index}]",
        )

        _validate_cell_reference(
            cell,
            valid_cells,
            context=f"initial_revealed[{index}]",
        )

        revealed.append(cell)

    if len(revealed) != len(set(revealed)):
        raise PuzzleFormatError(
            "'initial_revealed' contains duplicate cell ids."
        )

    return tuple(revealed)


# ============================================================
# Coordinate helpers
# ============================================================

def expected_cell_ids(size: int) -> tuple[str, ...]:
    """
    Return all expected board cell ids in row-major order.

    Example for size = 3:
        A1, B1, C1,
        A2, B2, C2,
        A3, B3, C3
    """
    return tuple(
        f"{chr(ord('A') + column)}{row}"
        for row in range(1, size + 1)
        for column in range(size)
    )


def _cell_to_indices(
    cell_id: str,
    size: int,
) -> tuple[int, int]:
    """
    Convert a cell id such as B2 to zero-based:

        column_index = 1
        row_index = 1
    """
    valid_cells = set(expected_cell_ids(size))

    if cell_id not in valid_cells:
        raise PuzzleFormatError(
            f"Invalid cell id '{cell_id}' "
            f"for a {size}x{size} board."
        )

    column_index = ord(cell_id[0]) - ord("A")
    row_index = int(cell_id[1:]) - 1

    return column_index, row_index


def _indices_to_cell(
    column_index: int,
    row_index: int,
) -> str:
    """
    Convert zero-based board indices to a Griductive cell id.
    """
    column = chr(ord("A") + column_index)
    row = row_index + 1

    return f"{column}{row}"


# ============================================================
# Generic validation helpers
# ============================================================

def _parse_status(
    value: Any,
    *,
    context: str,
) -> Status:
    """
    Convert JSON status text to Status enum.
    """
    if not isinstance(value, str):
        raise PuzzleFormatError(
            f"{context} must be 'CRIMINAL' or 'INNOCENT'."
        )

    normalized = value.strip().upper()

    try:
        return Status(normalized)

    except ValueError as exc:
        raise PuzzleFormatError(
            f"{context} = '{value}' is invalid. "
            f"Expected 'CRIMINAL' or 'INNOCENT'."
        ) from exc


def _normalize_cell_id(
    value: Any,
    *,
    context: str,
) -> str:
    if not isinstance(value, str):
        raise PuzzleFormatError(
            f"{context} must be a cell id such as 'A1'."
        )

    cell = value.strip().upper()

    if not cell:
        raise PuzzleFormatError(
            f"{context} must not be empty."
        )

    return cell


def _validate_cell_reference(
    cell: str,
    valid_cells: set[str],
    *,
    context: str,
) -> None:
    if cell not in valid_cells:
        raise PuzzleFormatError(
            f"{context} references invalid cell '{cell}'."
        )


def _read_non_empty_string(
    data: dict[str, Any],
    key: str,
    *,
    context: str,
) -> str:
    if key not in data:
        raise PuzzleFormatError(
            f"{context} is missing required field '{key}'."
        )

    value = data[key]

    if not isinstance(value, str):
        raise PuzzleFormatError(
            f"{context}.{key} must be a string."
        )

    value = value.strip()

    if not value:
        raise PuzzleFormatError(
            f"{context}.{key} must not be empty."
        )

    return value


def _read_integer(
    data: dict[str, Any],
    key: str,
    *,
    context: str,
) -> int:
    if key not in data:
        raise PuzzleFormatError(
            f"{context} is missing required field '{key}'."
        )

    value = data[key]

    if not _is_integer(value):
        raise PuzzleFormatError(
            f"{context}.{key} must be an integer."
        )

    return value


def _is_integer(value: Any) -> bool:
    """
    bool is a subclass of int in Python, so explicitly exclude it.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_keys(
    data: dict[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    """
    Require exactly the expected parameter names.

    This helps detect common puzzle-authoring mistakes such as
    writing "persons" instead of "people".
    """
    actual = set(data)

    missing = required - actual
    unexpected = actual - required

    if missing:
        raise PuzzleFormatError(
            f"{context} is missing required parameter(s): "
            + ", ".join(sorted(missing))
        )

    if unexpected:
        raise PuzzleFormatError(
            f"{context} contains unexpected parameter(s): "
            + ", ".join(sorted(unexpected))
        )