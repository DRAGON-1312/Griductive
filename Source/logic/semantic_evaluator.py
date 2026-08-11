from __future__ import annotations

from collections.abc import Mapping

from core.models import (
    Clue,
    ClueType,
    Status,
)
from core.puzzle_loader import resolve_region_cells


# ============================================================
# Type aliases
# ============================================================

Assignment = Mapping[str, Status]
"""
Complete truth assignment for puzzle characters.

Example:

    {
        "A1": Status.CRIMINAL,
        "B1": Status.INNOCENT,
        "C1": Status.CRIMINAL,
        ...
    }

This evaluator works directly with semantic character statuses.
It does NOT use CNF variables or call the SAT solver.
"""


# ============================================================
# Exceptions
# ============================================================

class SemanticEvaluationError(ValueError):
    """
    Base exception for semantic clue evaluation errors.
    """


class IncompleteAssignmentError(SemanticEvaluationError):
    """
    Raised when a clue refers to a character that is missing
    from the supplied assignment.
    """


class UnsupportedClueTypeError(SemanticEvaluationError):
    """
    Raised when no semantic evaluator has been implemented for
    a clue type.
    """


# ============================================================
# Public API
# ============================================================

def evaluate_clue(
    clue: Clue,
    assignment: Assignment,
    size: int,
) -> bool:
    """
    Evaluate one clue directly under a complete assignment.

    This function defines the semantic meaning of each supported clue.
    It does NOT use the CNF encoder or DPLL solver.

    Args:
        clue:
            Structured clue to evaluate.

        assignment:
            Mapping from character/cell id to its actual Status.

        size:
            Board size N for an N x N puzzle.

    Returns:
        True if the clue is satisfied by the assignment.
        False otherwise.

    Raises:
        SemanticEvaluationError:
            If the clue is malformed.

        IncompleteAssignmentError:
            If a required character is missing from the assignment.

        UnsupportedClueTypeError:
            If the clue type has no implemented semantic evaluator.
    """
    if not isinstance(clue, Clue):
        raise TypeError(
            "clue must be an instance of Clue."
        )

    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError(
            "size must be an integer."
        )

    if size <= 0:
        raise ValueError(
            "size must be positive."
        )

    clue_type = _normalize_clue_type(
        clue.type
    )

    if clue_type == ClueType.FACT:
        return evaluate_fact(
            clue,
            assignment,
        )

    if clue_type == ClueType.SAME:
        return evaluate_same(
            clue,
            assignment,
        )

    if clue_type == ClueType.DIFFERENT:
        return evaluate_different(
            clue,
            assignment,
        )

    if clue_type == ClueType.EXACTLY:
        return evaluate_exactly(
            clue,
            assignment,
            size,
        )

    if clue_type == ClueType.AT_LEAST:
        return evaluate_at_least(
            clue,
            assignment,
            size,
        )

    if clue_type == ClueType.AT_MOST:
        return evaluate_at_most(
            clue,
            assignment,
            size,
        )

    if clue_type == ClueType.PARITY:
        return evaluate_parity(
            clue,
            assignment,
            size,
        )

    if clue_type == ClueType.IMPLIES:
        return evaluate_implies(
            clue,
            assignment,
        )

    raise UnsupportedClueTypeError(
        f"No semantic evaluator implemented for "
        f"clue type '{clue.type}'."
    )


def evaluate_clues(
    clues: list[Clue] | tuple[Clue, ...],
    assignment: Assignment,
    size: int,
) -> dict[str, bool]:
    """
    Evaluate multiple clues under the same assignment.

    Returns:
        Dictionary mapping clue id -> evaluation result.

    Example:

        {
            "CL_A1": True,
            "CL_B1": True,
            "CL_C1": False,
        }
    """
    return {
        clue.id: evaluate_clue(
            clue,
            assignment,
            size,
        )
        for clue in clues
    }


# ============================================================
# FACT
# ============================================================

def evaluate_fact(
    clue: Clue,
    assignment: Assignment,
) -> bool:
    """
    FACT(person, status)

    Semantic meaning:

        The specified person has the specified status.

    Example:

        FACT(A1, CRIMINAL)

    is true exactly when:

        assignment["A1"] == Status.CRIMINAL
    """
    person = _get_required_param(
        clue,
        "person",
    )

    expected_status = _get_required_param(
        clue,
        "status",
    )

    if not isinstance(person, str):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'person' must be a string."
        )

    if not isinstance(
        expected_status,
        Status,
    ):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'status' must be a Status value."
        )

    actual_status = _get_assignment_status(
        assignment,
        person,
    )

    return actual_status == expected_status


# ============================================================
# SAME
# ============================================================

def evaluate_same(
    clue: Clue,
    assignment: Assignment,
) -> bool:
    """
    SAME(A, B)

    Semantic meaning:

        A and B have the same status.

    Equivalent proposition:

        C_A <-> C_B
    """
    first, second = _get_binary_people(
        clue
    )

    first_status = _get_assignment_status(
        assignment,
        first,
    )

    second_status = _get_assignment_status(
        assignment,
        second,
    )

    return first_status == second_status


# ============================================================
# DIFFERENT
# ============================================================

def evaluate_different(
    clue: Clue,
    assignment: Assignment,
) -> bool:
    """
    DIFFERENT(A, B)

    Semantic meaning:

        A and B have opposite statuses.

    Equivalent proposition:

        C_A XOR C_B
    """
    first, second = _get_binary_people(
        clue
    )

    first_status = _get_assignment_status(
        assignment,
        first,
    )

    second_status = _get_assignment_status(
        assignment,
        second,
    )

    return first_status != second_status


# ============================================================
# EXACTLY
# ============================================================

def evaluate_exactly(
    clue: Clue,
    assignment: Assignment,
    size: int,
) -> bool:
    """
    EXACTLY(k, R)

    Semantic meaning:

        Exactly k characters in region R are criminals.

        sum(C_i for i in R) == k
    """
    k, region_cells = _get_counting_data(
        clue,
        size,
    )

    criminal_count = _count_criminals(
        region_cells,
        assignment,
    )

    return criminal_count == k


# ============================================================
# AT_LEAST
# ============================================================

def evaluate_at_least(
    clue: Clue,
    assignment: Assignment,
    size: int,
) -> bool:
    """
    AT_LEAST(k, R)

    Semantic meaning:

        At least k characters in region R are criminals.

        sum(C_i for i in R) >= k
    """
    k, region_cells = _get_counting_data(
        clue,
        size,
    )

    criminal_count = _count_criminals(
        region_cells,
        assignment,
    )

    return criminal_count >= k


# ============================================================
# AT_MOST
# ============================================================

def evaluate_at_most(
    clue: Clue,
    assignment: Assignment,
    size: int,
) -> bool:
    """
    AT_MOST(k, R)

    Semantic meaning:

        At most k characters in region R are criminals.

        sum(C_i for i in R) <= k
    """
    k, region_cells = _get_counting_data(
        clue,
        size,
    )

    criminal_count = _count_criminals(
        region_cells,
        assignment,
    )

    return criminal_count <= k


def evaluate_parity(
    clue: Clue,
    assignment: Assignment,
    size: int,
) -> bool:
    """
    PARITY(parity, region)

    EVEN:
        number of criminals is even.

    ODD:
        number of criminals is odd.
    """
    parity = _get_required_param(
        clue,
        "parity",
    )

    region = _get_required_param(
        clue,
        "region",
    )

    if not isinstance(
        parity,
        str,
    ):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'parity' must be a string."
        )

    parity = (
        parity
        .strip()
        .upper()
    )

    if parity not in {
        "EVEN",
        "ODD",
    }:
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'parity' must be EVEN or ODD."
        )

    try:
        cells = resolve_region_cells(
            region,
            size,
        )

    except Exception as exc:
        raise SemanticEvaluationError(
            f"Clue '{clue.id}' contains "
            f"an invalid region."
        ) from exc

    criminal_count = _count_criminals(
        cells,
        assignment,
    )

    if parity == "EVEN":
        return (
            criminal_count % 2
            == 0
        )

    return (
        criminal_count % 2
        == 1
    )


def evaluate_implies(
    clue: Clue,
    assignment: Assignment,
) -> bool:
    """
    IMPLIES(A=status1, B=status2)

    Logical meaning:

        antecedent -> consequent
    """
    (
        antecedent_person,
        antecedent_status,
    ) = _get_status_condition(
        clue,
        "antecedent",
    )

    (
        consequent_person,
        consequent_status,
    ) = _get_status_condition(
        clue,
        "consequent",
    )

    if (
        antecedent_person
        == consequent_person
    ):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"IMPLIES requires two distinct characters."
        )

    antecedent_holds = (
        _get_assignment_status(
            assignment,
            antecedent_person,
        )
        == antecedent_status
    )

    consequent_holds = (
        _get_assignment_status(
            assignment,
            consequent_person,
        )
        == consequent_status
    )

    return (
        not antecedent_holds
        or consequent_holds
    )


# ============================================================
# Counting helpers
# ============================================================

def _get_counting_data(
    clue: Clue,
    size: int,
) -> tuple[int, tuple[str, ...]]:
    """
    Extract k and concrete region cells from a counting clue.
    """
    k = _get_required_param(
        clue,
        "k",
    )

    region = _get_required_param(
        clue,
        "region",
    )

    if not isinstance(k, int) or isinstance(k, bool):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'k' must be an integer."
        )

    try:
        region_cells = resolve_region_cells(
            region,
            size,
        )

    except Exception as exc:
        raise SemanticEvaluationError(
            f"Clue '{clue.id}' contains "
            f"an invalid region."
        ) from exc

    if not 0 <= k <= len(region_cells):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"k={k} does not satisfy "
            f"0 <= k <= |R|={len(region_cells)}."
        )

    return k, region_cells


def _count_criminals(
    cells: tuple[str, ...],
    assignment: Assignment,
) -> int:
    """
    Count how many cells in a region are assigned CRIMINAL.
    """
    count = 0

    for cell in cells:
        status = _get_assignment_status(
            assignment,
            cell,
        )

        if status == Status.CRIMINAL:
            count += 1

    return count


# ============================================================
# Binary clue helpers
# ============================================================

def _get_binary_people(
    clue: Clue,
) -> tuple[str, str]:
    """
    Extract the two character ids used by SAME / DIFFERENT.
    """
    people = _get_required_param(
        clue,
        "people",
    )

    if not isinstance(
        people,
        (list, tuple),
    ):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'people' must contain two character ids."
        )

    if len(people) != 2:
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'people' must contain exactly two characters."
        )

    first, second = people

    if not isinstance(first, str) or not isinstance(second, str):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"character ids must be strings."
        )

    if first == second:
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"SAME/DIFFERENT requires two distinct characters."
        )

    return first, second


# ============================================================
# Assignment helpers
# ============================================================

def _get_assignment_status(
    assignment: Assignment,
    character_id: str,
) -> Status:
    """
    Safely obtain one character's status from a semantic assignment.
    """
    if character_id not in assignment:
        raise IncompleteAssignmentError(
            f"Assignment is missing character "
            f"'{character_id}'."
        )

    status = assignment[
        character_id
    ]

    if not isinstance(status, Status):
        raise SemanticEvaluationError(
            f"Assignment value for '{character_id}' "
            f"must be a Status value."
        )

    return status


# ============================================================
# Generic helpers
# ============================================================

def _get_required_param(
    clue: Clue,
    param_name: str,
):
    """
    Read one required clue parameter.
    """
    if param_name not in clue.params:
        raise SemanticEvaluationError(
            f"Clue '{clue.id}' is missing "
            f"required parameter '{param_name}'."
        )

    return clue.params[param_name]


def _normalize_clue_type(
    clue_type: ClueType | str,
) -> ClueType | str:
    """
    Normalize core clue type strings to ClueType.

    Extension clue strings remain unchanged and will be handled
    separately when their evaluators are implemented.
    """
    if isinstance(clue_type, ClueType):
        return clue_type

    if isinstance(clue_type, str):
        normalized = (
            clue_type
            .strip()
            .upper()
        )

        try:
            return ClueType(normalized)

        except ValueError:
            return normalized

    raise SemanticEvaluationError(
        "Clue type must be a ClueType or string."
    )


def _get_status_condition(
    clue: Clue,
    param_name: str,
) -> tuple[str, Status]:

    condition = _get_required_param(
        clue,
        param_name,
    )

    if not isinstance(
        condition,
        dict,
    ):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'{param_name}' must be an object."
        )

    if set(condition) != {
        "person",
        "status",
    }:
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"'{param_name}' must contain "
            f"'person' and 'status'."
        )

    person = condition[
        "person"
    ]

    status = condition[
        "status"
    ]

    if not isinstance(
        person,
        str,
    ):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"condition person must be a string."
        )

    if not isinstance(
        status,
        Status,
    ):
        raise SemanticEvaluationError(
            f"Clue '{clue.id}': "
            f"condition status must be a Status."
        )

    return person, status