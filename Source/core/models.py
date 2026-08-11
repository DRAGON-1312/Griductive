from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# Status
# ============================================================

class Status(str, Enum):
    """
    Actual status of a character in the puzzle.
    """
    CRIMINAL = "CRIMINAL"
    INNOCENT = "INNOCENT"


class Classification(str, Enum):
    """
    Logical classification returned by the Logic Agent.
    """
    CRIMINAL = "CRIMINAL"
    INNOCENT = "INNOCENT"
    UNKNOWN = "UNKNOWN"
    INCONSISTENT = "INCONSISTENT"


# ============================================================
# Clue types
# ============================================================

class ClueType(str, Enum):
    """
    Six core clue templates and two project extensions.
    """
    
    # Core clue types.
    FACT = "FACT"
    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    EXACTLY = "EXACTLY"
    AT_LEAST = "AT_LEAST"
    AT_MOST = "AT_MOST"

    # Project extensions.
    PARITY = "PARITY"
    IMPLIES = "IMPLIES"


class RegionType(str, Enum):
    """
    Core region types required by the project.
    """
    ROW = "ROW"
    COLUMN = "COLUMN"
    NEIGHBORS = "NEIGHBORS"
    EXPLICIT = "EXPLICIT"


# ============================================================
# Board / character models
# ============================================================

@dataclass(frozen=True)
class Character:
    """
    Public information of one character.

    The hidden Criminal/Innocent status is intentionally NOT stored here
    so this object can safely be shared with the Logic Agent and GUI.
    """
    id: str
    name: str
    profession: str


@dataclass(frozen=True)
class Region:
    """
    Structured region referenced by a counting clue.

    Examples:
        Region(type=RegionType.ROW, value=2)
        Region(type=RegionType.COLUMN, value="B")
        Region(type=RegionType.NEIGHBORS, value="C2")
        Region(type=RegionType.EXPLICIT, value=("A1", "B2", "C3"))
    """
    type: RegionType
    value: int | str | tuple[str, ...]


# ============================================================
# Clue model
# ============================================================

@dataclass(frozen=True)
class Clue:
    """
    Structured clue.

    `params` contains the parameters required by the clue type.

    Expected formats for the six core clue types:

    FACT:
        {
            "person": "A1",
            "status": Status.CRIMINAL
        }

    SAME:
        {
            "people": ("A1", "B2")
        }

    DIFFERENT:
        {
            "people": ("A1", "B2")
        }

    EXACTLY / AT_LEAST / AT_MOST:
        {
            "k": 2,
            "region": Region(...)
        }

    The flexible dictionary also makes it possible to add the required
    extension clue types later without redesigning the entire model.
    """
    id: str
    type: ClueType | str
    params: dict[str, Any]


# ============================================================
# Hidden puzzle data
# ============================================================

@dataclass(frozen=True)
class CharacterSecret:
    """
    Hidden information owned only by the Game Engine.

    Logic Agent must never receive this object directly.
    """
    status: Status
    clue: Clue


@dataclass
class Puzzle:
    """
    Complete puzzle definition loaded from a puzzle file.

    This object contains hidden information and therefore should remain
    inside the Game Engine.
    """
    name: str
    size: int

    characters: dict[str, Character]

    # Hidden solution + hidden clue for every character.
    secrets: dict[str, CharacterSecret]

    # Cards that are already face-up at the beginning.
    initial_revealed: tuple[str, ...] = field(default_factory=tuple)


# ============================================================
# Public knowledge state
# ============================================================

@dataclass
class PublicState:
    """
    Information that may safely be given to the Logic Agent.

    Important:
        - NO hidden solution
        - NO unrevealed clues
    """
    characters: dict[str, Character]

    # Statuses that have already been logically proved.
    proved_statuses: dict[str, Status] = field(default_factory=dict)

    # Only clues that have already been revealed.
    revealed_clues: list[Clue] = field(default_factory=list)


# ============================================================
# Game response
# ============================================================

class VerdictCode(str, Enum):
    """
    Result of submitting a verdict through the game interface.
    """
    ACCEPTED = "ACCEPTED"
    NOT_PROVABLE = "NOT_PROVABLE"
    CONTRADICTED = "CONTRADICTED"


@dataclass
class VerdictResult:
    """
    Result returned after submitting CRIMINAL / INNOCENT for a character.
    """
    character_id: str
    submitted_status: Status
    code: VerdictCode

    # Filled only when an accepted verdict reveals a new clue.
    revealed_clue: Clue | None = None