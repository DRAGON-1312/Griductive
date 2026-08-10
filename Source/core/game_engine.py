from __future__ import annotations

from copy import deepcopy
from typing import Callable

from .models import (
    Character,
    Classification,
    Clue,
    PublicState,
    Puzzle,
    Status,
    VerdictCode,
    VerdictResult,
)


# ============================================================
# Type aliases
# ============================================================

VerdictClassifier = Callable[
    [PublicState, str],
    Classification,
]
"""
A function that receives:

    1. The current PUBLIC knowledge state.
    2. A character id.

and returns the logical classification of that character.

Example later:

    engine.submit_verdict(
        "B2",
        Status.CRIMINAL,
        agent.classify_character,
    )

The classifier never receives the hidden Puzzle object.
"""


# ============================================================
# Exceptions
# ============================================================

class GameEngineError(RuntimeError):
    """
    Base exception for GameEngine-related errors.
    """


class InvalidCharacterError(GameEngineError):
    """
    Raised when an unknown character/cell id is requested.
    """


class InconsistentKnowledgeBaseError(GameEngineError):
    """
    Raised when the Logic Agent reports that the current public
    knowledge base is inconsistent.
    """


class EngineIntegrityError(GameEngineError):
    """
    Raised when the logical result conflicts with the hidden puzzle
    definition.

    This normally indicates:
        - a malformed puzzle,
        - an incorrect CNF encoding,
        - or a bug in the SAT / entailment logic.
    """


# ============================================================
# Game Engine
# ============================================================

class GameEngine:
    """
    Owns the complete Griductive puzzle.

    The GameEngine is the ONLY runtime component that owns:

        - hidden Criminal / Innocent statuses,
        - unrevealed clues.

    The Logic Agent must only receive PublicState objects produced by
    get_public_state().

    A verdict is accepted only when the supplied logical classifier
    proves that the submitted status is currently forced.
    """

    def __init__(self, puzzle: Puzzle):
        """
        Create a new game from a validated Puzzle.

        A deep copy is stored internally so outside code cannot mutate
        the hidden solution after the engine has been created.
        """
        self._puzzle = deepcopy(puzzle)

        self._validate_puzzle_integrity()

        self._proved_statuses: dict[str, Status] = {}
        self._revealed_ids: list[str] = []

        self.restart()

    # ========================================================
    # Safe public puzzle information
    # ========================================================

    @property
    def puzzle_name(self) -> str:
        """
        Public puzzle name.
        """
        return self._puzzle.name

    @property
    def size(self) -> int:
        """
        Board size N for an N x N puzzle.
        """
        return self._puzzle.size

    @property
    def total_characters(self) -> int:
        return len(self._puzzle.characters)

    def get_character(
        self,
        character_id: str,
    ) -> Character:
        """
        Return PUBLIC information about a character.

        Hidden status and hidden clue are never returned.
        """
        character_id = self._normalize_character_id(
            character_id
        )

        return deepcopy(
            self._puzzle.characters[character_id]
        )

    def get_characters(
        self,
    ) -> dict[str, Character]:
        """
        Return all public character information.

        The returned dictionary is a copy and is safe for GUI use.
        """
        return deepcopy(
            self._puzzle.characters
        )

    # ========================================================
    # Public knowledge state
    # ========================================================

    def get_public_state(self) -> PublicState:
        """
        Return exactly the information that the Logic Agent is allowed
        to know.

        Contains:
            - public character information,
            - already proved statuses,
            - already revealed clues.

        Does NOT contain:
            - hidden solution,
            - unrevealed clues.
        """
        revealed_clues = [
            deepcopy(
                self._puzzle.secrets[character_id].clue
            )
            for character_id in self._revealed_ids
        ]

        return PublicState(
            characters=deepcopy(
                self._puzzle.characters
            ),
            proved_statuses=deepcopy(
                self._proved_statuses
            ),
            revealed_clues=revealed_clues,
        )

    def get_proved_status(
        self,
        character_id: str,
    ) -> Status | None:
        """
        Return the currently public/proved status of a character.

        None means that the character has not yet been solved.
        """
        character_id = self._normalize_character_id(
            character_id
        )

        return self._proved_statuses.get(
            character_id
        )

    def get_revealed_clue(
        self,
        character_id: str,
    ) -> Clue | None:
        """
        Return a character's clue only if that card is already revealed.

        This method never exposes an unrevealed clue.
        """
        character_id = self._normalize_character_id(
            character_id
        )

        if character_id not in self._revealed_ids:
            return None

        return deepcopy(
            self._puzzle.secrets[character_id].clue
        )

    def get_revealed_character_ids(
        self,
    ) -> tuple[str, ...]:
        """
        Return revealed character ids in reveal order.
        """
        return tuple(self._revealed_ids)

    def get_unresolved_character_ids(
        self,
    ) -> tuple[str, ...]:
        """
        Return all characters whose statuses are not yet public/proved.

        Order follows the deterministic board order stored in Puzzle.
        """
        return tuple(
            character_id
            for character_id
            in self._puzzle.characters
            if character_id
            not in self._proved_statuses
        )

    # ========================================================
    # Game state
    # ========================================================

    def restart(self) -> None:
        """
        Restore the game to its initial public state.

        Initially face-up cards expose BOTH:
            - their status,
            - their clue.

        This follows the project specification: initially revealed
        cards form part of the public knowledge base.
        """
        self._proved_statuses.clear()
        self._revealed_ids.clear()

        for character_id in (
            self._puzzle.initial_revealed
        ):
            secret = self._puzzle.secrets[
                character_id
            ]

            self._proved_statuses[
                character_id
            ] = secret.status

            self._revealed_ids.append(
                character_id
            )

    def is_solved(self) -> bool:
        """
        A puzzle is solved when every character has a proved/public
        status.
        """
        return (
            len(self._proved_statuses)
            == len(self._puzzle.characters)
        )

    @property
    def solved_count(self) -> int:
        return len(self._proved_statuses)

    # ========================================================
    # Verdict submission
    # ========================================================

    def submit_verdict(
        self,
        character_id: str,
        submitted_status: Status | str,
        classifier: VerdictClassifier,
    ) -> VerdictResult:
        """
        Submit CRIMINAL or INNOCENT for one character.

        IMPORTANT:
        The GameEngine does NOT decide logical entailment from the
        hidden solution.

        Instead, it gives the current PublicState to `classifier`.

        The classifier must return one of:

            Classification.CRIMINAL
            Classification.INNOCENT
            Classification.UNKNOWN
            Classification.INCONSISTENT

        Result:

            forced == submitted
                -> ACCEPTED
                -> status becomes public
                -> card is revealed
                -> clue becomes public

            UNKNOWN
                -> NOT_PROVABLE
                -> game state unchanged

            opposite status forced
                -> CONTRADICTED
                -> game state unchanged

            INCONSISTENT
                -> raises InconsistentKnowledgeBaseError
        """
        character_id = self._normalize_character_id(
            character_id
        )

        submitted_status = self._normalize_status(
            submitted_status
        )

        if not callable(classifier):
            raise TypeError(
                "classifier must be callable."
            )

        # ----------------------------------------------------
        # Already solved character
        # ----------------------------------------------------

        if character_id in self._proved_statuses:
            known_status = self._proved_statuses[
                character_id
            ]

            if submitted_status == known_status:
                return VerdictResult(
                    character_id=character_id,
                    submitted_status=submitted_status,
                    code=VerdictCode.ACCEPTED,
                    revealed_clue=None,
                )

            return VerdictResult(
                character_id=character_id,
                submitted_status=submitted_status,
                code=VerdictCode.CONTRADICTED,
                revealed_clue=None,
            )

        # ----------------------------------------------------
        # Ask the logic layer using PUBLIC information only
        # ----------------------------------------------------

        public_state = self.get_public_state()

        classification = classifier(
            public_state,
            character_id,
        )

        if not isinstance(
            classification,
            Classification,
        ):
            raise TypeError(
                "classifier must return a Classification value."
            )

        # ----------------------------------------------------
        # Inconsistent KB
        # ----------------------------------------------------

        if (
            classification
            == Classification.INCONSISTENT
        ):
            raise InconsistentKnowledgeBaseError(
                "The current public knowledge base is "
                "inconsistent."
            )

        # ----------------------------------------------------
        # Nothing is logically forced
        # ----------------------------------------------------

        if classification == Classification.UNKNOWN:
            return VerdictResult(
                character_id=character_id,
                submitted_status=submitted_status,
                code=VerdictCode.NOT_PROVABLE,
                revealed_clue=None,
            )

        # ----------------------------------------------------
        # Determine which actual Status was logically forced
        # ----------------------------------------------------

        forced_status = self._classification_to_status(
            classification
        )

        # ----------------------------------------------------
        # Opposite verdict is logically forced
        # ----------------------------------------------------

        if submitted_status != forced_status:
            return VerdictResult(
                character_id=character_id,
                submitted_status=submitted_status,
                code=VerdictCode.CONTRADICTED,
                revealed_clue=None,
            )

        # ----------------------------------------------------
        # Integrity check
        #
        # The verdict was accepted by LOGIC first.
        #
        # Only now do we compare it with the hidden puzzle answer
        # to detect implementation / puzzle errors.
        # Hidden information is NOT used to perform deduction.
        # ----------------------------------------------------

        hidden_status = self._puzzle.secrets[
            character_id
        ].status

        if forced_status != hidden_status:
            raise EngineIntegrityError(
                f"Logic classified '{character_id}' as "
                f"{forced_status.value}, but the hidden puzzle "
                f"contains {hidden_status.value}. "
                f"Check the puzzle, CNF encoder, and SAT solver."
            )

        # ----------------------------------------------------
        # Accept verdict and reveal the clue
        # ----------------------------------------------------

        revealed_clue = self._accept_verdict(
            character_id,
            forced_status,
        )

        return VerdictResult(
            character_id=character_id,
            submitted_status=submitted_status,
            code=VerdictCode.ACCEPTED,
            revealed_clue=revealed_clue,
        )

    # ========================================================
    # Internal reveal logic
    # ========================================================

    def _accept_verdict(
        self,
        character_id: str,
        status: Status,
    ) -> Clue:
        """
        Make a proved status public and reveal its clue.

        This method must only be called after logical proof has already
        been established.
        """
        self._proved_statuses[
            character_id
        ] = status

        if character_id not in self._revealed_ids:
            self._revealed_ids.append(
                character_id
            )

        return deepcopy(
            self._puzzle.secrets[
                character_id
            ].clue
        )

    # ========================================================
    # Internal validation
    # ========================================================

    def _validate_puzzle_integrity(self) -> None:
        """
        Perform basic defensive checks.

        puzzle_loader.py already performs stronger JSON validation,
        but GameEngine should still protect itself in case somebody
        manually constructs a Puzzle object.
        """
        character_ids = set(
            self._puzzle.characters
        )

        secret_ids = set(
            self._puzzle.secrets
        )

        if character_ids != secret_ids:
            missing_secrets = (
                character_ids - secret_ids
            )

            extra_secrets = (
                secret_ids - character_ids
            )

            details: list[str] = []

            if missing_secrets:
                details.append(
                    "missing secrets for: "
                    + ", ".join(
                        sorted(missing_secrets)
                    )
                )

            if extra_secrets:
                details.append(
                    "secrets without characters: "
                    + ", ".join(
                        sorted(extra_secrets)
                    )
                )

            raise EngineIntegrityError(
                "Puzzle character/secret mapping mismatch: "
                + "; ".join(details)
            )

        for character_id in (
            self._puzzle.initial_revealed
        ):
            if character_id not in character_ids:
                raise EngineIntegrityError(
                    f"Initial revealed character "
                    f"'{character_id}' does not exist."
                )

        if (
            len(self._puzzle.initial_revealed)
            != len(
                set(
                    self._puzzle.initial_revealed
                )
            )
        ):
            raise EngineIntegrityError(
                "initial_revealed contains duplicates."
            )

    # ========================================================
    # Conversion / normalization helpers
    # ========================================================

    def _normalize_character_id(
        self,
        character_id: str,
    ) -> str:
        if not isinstance(character_id, str):
            raise InvalidCharacterError(
                "character_id must be a string."
            )

        normalized = (
            character_id
            .strip()
            .upper()
        )

        if normalized not in self._puzzle.characters:
            raise InvalidCharacterError(
                f"Unknown character id: "
                f"'{character_id}'."
            )

        return normalized

    @staticmethod
    def _normalize_status(
        status: Status | str,
    ) -> Status:
        if isinstance(status, Status):
            return status

        if isinstance(status, str):
            normalized = (
                status
                .strip()
                .upper()
            )

            try:
                return Status(normalized)

            except ValueError as exc:
                raise ValueError(
                    "submitted_status must be "
                    "'CRIMINAL' or 'INNOCENT'."
                ) from exc

        raise TypeError(
            "submitted_status must be "
            "a Status or string."
        )

    @staticmethod
    def _classification_to_status(
        classification: Classification,
    ) -> Status:
        if (
            classification
            == Classification.CRIMINAL
        ):
            return Status.CRIMINAL

        if (
            classification
            == Classification.INNOCENT
        ):
            return Status.INNOCENT

        raise ValueError(
            f"{classification.value} does not represent "
            f"a proved character status."
        )