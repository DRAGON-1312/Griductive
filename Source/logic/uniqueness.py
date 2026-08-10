from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from core.models import (
    Character,
    Clue,
    Puzzle,
    Status,
)
from logic.cnf_encoder import CNFEncoder
from logic.dpll import DPLLSolver, SATResult


# ============================================================
# Exceptions
# ============================================================

class UniquenessError(ValueError):
    """
    Base exception for puzzle uniqueness checking.
    """


class InvalidPuzzleSizeError(UniquenessError):
    """
    Raised when the puzzle size does not match the checker size.
    """


# ============================================================
# Result
# ============================================================

@dataclass(frozen=True)
class UniquenessResult:
    """
    Result of checking whether a Griductive knowledge base has
    exactly one assignment of character statuses.

    The check uses at most two SAT calls:

        1. Solve the original KB.
        2. If SAT, add a blocking clause for the first character
           assignment and solve again.

    Interpretation:

        first SAT = False
            -> KB has no solution.

        first SAT = True
        second SAT = False
            -> exactly one character-status assignment.

        first SAT = True
        second SAT = True
            -> at least two different character-status assignments.
    """

    satisfiable: bool
    unique: bool

    first_solve_result: SATResult
    second_solve_result: SATResult | None

    first_character_assignment: dict[str, Status] | None
    second_character_assignment: dict[str, Status] | None

    blocking_clause: tuple[int, ...] | None

    variable_count: int
    kb_clause_count: int

    # --------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------

    @property
    def has_solution(self) -> bool:
        return self.satisfiable

    @property
    def has_multiple_solutions(self) -> bool:
        return (
            self.satisfiable
            and not self.unique
        )

    @property
    def total_decisions(self) -> int:
        total = self.first_solve_result.decisions

        if self.second_solve_result is not None:
            total += self.second_solve_result.decisions

        return total

    @property
    def total_propagations(self) -> int:
        total = self.first_solve_result.propagations

        if self.second_solve_result is not None:
            total += self.second_solve_result.propagations

        return total

    @property
    def total_backtracks(self) -> int:
        total = self.first_solve_result.backtracks

        if self.second_solve_result is not None:
            total += self.second_solve_result.backtracks

        return total

    @property
    def total_runtime(self) -> float:
        total = self.first_solve_result.runtime

        if self.second_solve_result is not None:
            total += self.second_solve_result.runtime

        return total


# ============================================================
# Uniqueness checker
# ============================================================

class UniquenessChecker:
    """
    Check whether a Griductive knowledge base has exactly one
    assignment of Criminal/Innocent statuses.

    This module is intended for:

        - puzzle validation,
        - test cases,
        - experiment setup.

    It is separate from normal game entailment.

    Uniqueness algorithm
    --------------------

    Let the primary character variables be:

        C_1, C_2, ..., C_n

    Step 1:
        Solve KB.

    If UNSAT:
        the puzzle has no valid solution.

    Otherwise suppose the returned model is:

        C_1 = v_1
        ...
        C_n = v_n

    Step 2:
        Add one blocking clause that requires at least one
        character variable to differ from that model.

    Example first model:

        C_1 = True
        C_2 = False
        C_3 = True

    Blocking clause:

        NOT C_1 OR C_2 OR NOT C_3

    Then solve again.

    If the second solve is UNSAT:
        the character-status assignment is unique.

    If the second solve is SAT:
        another valid assignment exists, so the puzzle is not unique.

    Only primary character variables are included in the blocking
    clause. Auxiliary CNF variables, if introduced in the future,
    must not affect puzzle-solution uniqueness.
    """

    def __init__(
        self,
        size: int,
        solver: DPLLSolver | None = None,
    ):
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
        ):
            raise TypeError(
                "size must be an integer."
            )

        if size <= 0:
            raise ValueError(
                "size must be positive."
            )

        if (
            solver is not None
            and not isinstance(
                solver,
                DPLLSolver,
            )
        ):
            raise TypeError(
                "solver must be a DPLLSolver."
            )

        self.size = size

        self._solver = (
            solver
            if solver is not None
            else DPLLSolver()
        )

    # ========================================================
    # Generic clue-set uniqueness
    # ========================================================

    def check_clues(
        self,
        characters: Mapping[str, Character],
        clues: Sequence[Clue],
        known_statuses: Mapping[str, Status] | None = None,
    ) -> UniquenessResult:
        """
        Check uniqueness for an arbitrary set of clues and optional
        already-known character statuses.

        Args:
            characters:
                Public character definitions.

            clues:
                Clues that belong to the knowledge base.

            known_statuses:
                Optional status facts already known to be true.

        Returns:
            UniquenessResult.

        This method does not require or inspect a hidden solution.
        """
        encoder = CNFEncoder(
            characters=characters,
            size=self.size,
        )

        proved_statuses = (
            known_statuses
            if known_statuses is not None
            else {}
        )

        kb = encoder.build_kb(
            revealed_clues=clues,
            proved_statuses=proved_statuses,
        )

        character_ids = tuple(
            characters.keys()
        )

        return self._check_encoded_kb(
            kb=kb,
            encoder=encoder,
            character_ids=character_ids,
        )

    # ========================================================
    # Complete puzzle validation
    # ========================================================

    def check_puzzle(
        self,
        puzzle: Puzzle,
        include_initial_statuses: bool = True,
    ) -> UniquenessResult:
        """
        Check whether the complete puzzle information determines
        exactly one character-status assignment.

        All clues are included because this is an offline puzzle
        validation step rather than a live-agent reasoning step.

        By default, statuses of characters listed in
        puzzle.initial_revealed are also included because those
        statuses are public facts available at the beginning of the
        actual game.

        Hidden statuses of all other characters are NEVER added to
        the knowledge base.

        Args:
            puzzle:
                Puzzle to validate.

            include_initial_statuses:
                Whether initially revealed status facts should be
                included in the uniqueness check.

        Returns:
            UniquenessResult.
        """
        if not isinstance(
            puzzle,
            Puzzle,
        ):
            raise TypeError(
                "puzzle must be a Puzzle."
            )

        if puzzle.size != self.size:
            raise InvalidPuzzleSizeError(
                f"Checker size is {self.size}x{self.size}, "
                f"but puzzle size is "
                f"{puzzle.size}x{puzzle.size}."
            )

        all_clues = [
            secret.clue
            for secret
            in puzzle.secrets.values()
        ]

        initial_statuses: dict[
            str,
            Status,
        ] = {}

        if include_initial_statuses:
            for character_id in puzzle.initial_revealed:
                initial_statuses[
                    character_id
                ] = puzzle.secrets[
                    character_id
                ].status

        return self.check_clues(
            characters=puzzle.characters,
            clues=all_clues,
            known_statuses=initial_statuses,
        )

    # ========================================================
    # Core uniqueness algorithm
    # ========================================================

    def _check_encoded_kb(
        self,
        kb: list[list[int]],
        encoder: CNFEncoder,
        character_ids: Sequence[str],
    ) -> UniquenessResult:
        """
        Run the two-solve uniqueness algorithm on an encoded KB.
        """

        # ----------------------------------------------------
        # SAT call 1:
        #
        # Find one model of the original knowledge base.
        # ----------------------------------------------------

        first_result = self._solver.solve(
            clauses=kb,
            num_variables=encoder.total_variable_count,
        )

        # ----------------------------------------------------
        # No model -> puzzle / KB is inconsistent.
        # ----------------------------------------------------

        if not first_result.satisfiable:
            return UniquenessResult(
                satisfiable=False,
                unique=False,
                first_solve_result=first_result,
                second_solve_result=None,
                first_character_assignment=None,
                second_character_assignment=None,
                blocking_clause=None,
                variable_count=(
                    encoder.total_variable_count
                ),
                kb_clause_count=len(kb),
            )

        if first_result.assignment is None:
            raise UniquenessError(
                "DPLL returned SAT without an assignment."
            )

        first_character_assignment = (
            self._extract_character_assignment(
                assignment=first_result.assignment,
                encoder=encoder,
                character_ids=character_ids,
            )
        )

        # ----------------------------------------------------
        # Construct a blocking clause for the first assignment.
        #
        # If model contains:
        #
        #     C_i = True
        #
        # add:
        #
        #     -C_i
        #
        # If model contains:
        #
        #     C_i = False
        #
        # add:
        #
        #     C_i
        #
        # The resulting clause requires at least one primary
        # character variable to change.
        # ----------------------------------------------------

        blocking_clause = (
            self._build_blocking_clause(
                assignment=first_result.assignment,
                encoder=encoder,
                character_ids=character_ids,
            )
        )

        second_kb = [
            list(clause)
            for clause in kb
        ]

        second_kb.append(
            list(blocking_clause)
        )

        # ----------------------------------------------------
        # SAT call 2:
        #
        # Search for another character-status assignment.
        # ----------------------------------------------------

        second_result = self._solver.solve(
            clauses=second_kb,
            num_variables=encoder.total_variable_count,
        )

        # ----------------------------------------------------
        # Second solve UNSAT:
        #
        # no different primary assignment exists.
        #
        # Therefore the puzzle solution is unique.
        # ----------------------------------------------------

        if not second_result.satisfiable:
            return UniquenessResult(
                satisfiable=True,
                unique=True,
                first_solve_result=first_result,
                second_solve_result=second_result,
                first_character_assignment=(
                    first_character_assignment
                ),
                second_character_assignment=None,
                blocking_clause=blocking_clause,
                variable_count=(
                    encoder.total_variable_count
                ),
                kb_clause_count=len(kb),
            )

        # ----------------------------------------------------
        # Second solve SAT:
        #
        # at least one other primary assignment exists.
        # ----------------------------------------------------

        if second_result.assignment is None:
            raise UniquenessError(
                "DPLL returned SAT without an assignment."
            )

        second_character_assignment = (
            self._extract_character_assignment(
                assignment=second_result.assignment,
                encoder=encoder,
                character_ids=character_ids,
            )
        )

        return UniquenessResult(
            satisfiable=True,
            unique=False,
            first_solve_result=first_result,
            second_solve_result=second_result,
            first_character_assignment=(
                first_character_assignment
            ),
            second_character_assignment=(
                second_character_assignment
            ),
            blocking_clause=blocking_clause,
            variable_count=(
                encoder.total_variable_count
            ),
            kb_clause_count=len(kb),
        )

    # ========================================================
    # Blocking clause
    # ========================================================

    @staticmethod
    def _build_blocking_clause(
        assignment: Mapping[int, bool],
        encoder: CNFEncoder,
        character_ids: Sequence[str],
    ) -> tuple[int, ...]:
        """
        Build a clause that blocks exactly one assignment of the
        primary character variables.

        Example:

            A1 = True
            B1 = False
            C1 = True

        becomes:

            (-A1 OR B1 OR -C1)

        The clause contains only character variables.
        """
        literals: list[int] = []

        for character_id in character_ids:
            variable = encoder.variable_for(
                character_id
            )

            if variable not in assignment:
                raise UniquenessError(
                    f"SAT assignment is missing "
                    f"variable {variable} "
                    f"for character "
                    f"{character_id!r}."
                )

            value = assignment[
                variable
            ]

            if value:
                literals.append(
                    -variable
                )
            else:
                literals.append(
                    variable
                )

        if not literals:
            raise UniquenessError(
                "Cannot build a blocking clause "
                "without character variables."
            )

        return tuple(
            literals
        )

    # ========================================================
    # Model conversion
    # ========================================================

    @staticmethod
    def _extract_character_assignment(
        assignment: Mapping[int, bool],
        encoder: CNFEncoder,
        character_ids: Sequence[str],
    ) -> dict[str, Status]:
        """
        Convert a Boolean SAT model into Griductive statuses.

            True  -> CRIMINAL
            False -> INNOCENT
        """
        result: dict[
            str,
            Status,
        ] = {}

        for character_id in character_ids:
            variable = encoder.variable_for(
                character_id
            )

            if variable not in assignment:
                raise UniquenessError(
                    f"SAT assignment is missing "
                    f"variable {variable} "
                    f"for character "
                    f"{character_id!r}."
                )

            value = assignment[
                variable
            ]

            result[
                character_id
            ] = (
                Status.CRIMINAL
                if value
                else Status.INNOCENT
            )

        return result