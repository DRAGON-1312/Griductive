from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

from core.models import (
    Character,
    Clue,
    ClueType,
    PublicState,
    Status,
)
from core.puzzle_loader import resolve_region_cells


# ============================================================
# Type aliases
# ============================================================

Literal = int
Clause = list[Literal]
CNF = list[Clause]


# ============================================================
# Exceptions
# ============================================================

class CNFEncodingError(ValueError):
    """
    Base exception for CNF encoding errors.
    """


class UnknownCharacterError(CNFEncodingError):
    """
    Raised when a clue refers to a character that does not exist
    in the encoder's variable mapping.
    """


class UnsupportedClueTypeError(CNFEncodingError):
    """
    Raised when no CNF encoding has been implemented for a clue type.
    """


# ============================================================
# Statistics
# ============================================================

@dataclass(frozen=True)
class CNFStatistics:
    """
    Basic statistics required by the project report.
    """
    primary_variables: int
    auxiliary_variables: int
    clauses: int


# ============================================================
# CNF Encoder
# ============================================================

class CNFEncoder:
    """
    Convert structured Griductive knowledge into CNF clauses.

    Primary variable convention:

        C_i = True  -> character i is CRIMINAL
        C_i = False -> character i is INNOCENT

    CNF literal convention:

         n -> variable n is True
        -n -> variable n is False

    Example:

        A1 -> 1
        B1 -> 2

        SAME(A1, B1)

    becomes:

        (-A1 OR B1) AND (A1 OR -B1)

    represented as:

        [
            [-1, 2],
            [1, -2],
        ]

    The current implementation uses direct combinatorial encoding
    for cardinality constraints and therefore introduces no
    auxiliary variables.
    """

    def __init__(
        self,
        characters: Mapping[str, Character],
        size: int,
    ):
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(
                "size must be an integer."
            )

        if size <= 0:
            raise ValueError(
                "size must be positive."
            )

        if not isinstance(characters, Mapping):
            raise TypeError(
                "characters must be a mapping."
            )

        if not characters:
            raise ValueError(
                "characters must not be empty."
            )

        self.size = size

        # PuzzleLoader stores characters in deterministic row-major
        # order, so preserving mapping order gives deterministic
        # SAT variable identifiers.
        self._variable_map: dict[str, int] = {}
        self._reverse_map: dict[int, str] = {}

        for index, character_id in enumerate(
            characters,
            start=1,
        ):
            normalized_id = self._normalize_character_id(
                character_id
            )

            if normalized_id in self._variable_map:
                raise CNFEncodingError(
                    f"Duplicate character id "
                    f"'{normalized_id}'."
                )

            self._variable_map[
                normalized_id
            ] = index

            self._reverse_map[
                index
            ] = normalized_id

        self._primary_variable_count = len(
            self._variable_map
        )

        # Direct combinatorial encoding currently creates no
        # auxiliary variables.
        self._auxiliary_variable_count = 0

    # ========================================================
    # Variable mapping
    # ========================================================

    @property
    def variable_map(self) -> dict[str, int]:
        """
        Return a copy of character -> SAT variable mapping.
        """
        return dict(
            self._variable_map
        )

    @property
    def reverse_variable_map(self) -> dict[int, str]:
        """
        Return a copy of SAT variable -> character mapping.
        """
        return dict(
            self._reverse_map
        )

    @property
    def primary_variable_count(self) -> int:
        return self._primary_variable_count

    @property
    def auxiliary_variable_count(self) -> int:
        return self._auxiliary_variable_count

    @property
    def total_variable_count(self) -> int:
        return (
            self._primary_variable_count
            + self._auxiliary_variable_count
        )

    def variable_for(
        self,
        character_id: str,
    ) -> int:
        """
        Return the SAT variable representing one character.
        """
        character_id = self._normalize_character_id(
            character_id
        )

        if character_id not in self._variable_map:
            raise UnknownCharacterError(
                f"Unknown character id "
                f"'{character_id}'."
            )

        return self._variable_map[
            character_id
        ]

    def character_for_variable(
        self,
        variable: int,
    ) -> str:
        """
        Return the character id represented by a primary SAT variable.
        """
        if not isinstance(variable, int) or isinstance(
            variable,
            bool,
        ):
            raise TypeError(
                "variable must be an integer."
            )

        variable = abs(variable)

        if variable not in self._reverse_map:
            raise UnknownCharacterError(
                f"Unknown primary SAT variable "
                f"{variable}."
            )

        return self._reverse_map[
            variable
        ]

    def literal_for_status(
        self,
        character_id: str,
        status: Status,
    ) -> int:
        """
        Return the literal expressing a character status.

        CRIMINAL:
            +variable

        INNOCENT:
            -variable
        """
        if not isinstance(status, Status):
            raise TypeError(
                "status must be a Status value."
            )

        variable = self.variable_for(
            character_id
        )

        if status == Status.CRIMINAL:
            return variable

        if status == Status.INNOCENT:
            return -variable

        raise CNFEncodingError(
            f"Unsupported status '{status}'."
        )

    # ========================================================
    # Public API
    # ========================================================

    def encode_clue(
        self,
        clue: Clue,
    ) -> CNF:
        """
        Encode one structured clue into CNF.
        """
        if not isinstance(clue, Clue):
            raise TypeError(
                "clue must be an instance of Clue."
            )

        clue_type = self._normalize_clue_type(
            clue.type
        )

        if clue_type == ClueType.FACT:
            return self.encode_fact(
                clue
            )

        if clue_type == ClueType.SAME:
            return self.encode_same(
                clue
            )

        if clue_type == ClueType.DIFFERENT:
            return self.encode_different(
                clue
            )

        if clue_type == ClueType.EXACTLY:
            return self.encode_exactly(
                clue
            )

        if clue_type == ClueType.AT_LEAST:
            return self.encode_at_least(
                clue
            )

        if clue_type == ClueType.AT_MOST:
            return self.encode_at_most(
                clue
            )

        raise UnsupportedClueTypeError(
            f"No CNF encoding implemented for "
            f"clue type '{clue.type}'."
        )

    def encode_clues(
        self,
        clues: list[Clue] | tuple[Clue, ...],
    ) -> CNF:
        """
        Encode multiple clues into one conjunction of CNF clauses.
        """
        cnf: CNF = []

        for clue in clues:
            cnf.extend(
                self.encode_clue(
                    clue
                )
            )

        return cnf

    def build_kb(
        self,
        revealed_clues: list[Clue] | tuple[Clue, ...],
        proved_statuses: Mapping[str, Status],
    ) -> CNF:
        """
        Construct the current public knowledge base.

        KB contains ONLY:

            - revealed clues
            - previously proved statuses

        Hidden statuses and unrevealed clues are never used here.
        """
        if not isinstance(
            proved_statuses,
            Mapping,
        ):
            raise TypeError(
                "proved_statuses must be a mapping."
            )

        cnf: CNF = []

        # Previously proved verdicts become unit clauses.
        for character_id, status in proved_statuses.items():
            literal = self.literal_for_status(
                character_id,
                status,
            )

            cnf.append(
                [literal]
            )

        # Add all currently revealed clues.
        cnf.extend(
            self.encode_clues(
                revealed_clues
            )
        )

        return cnf

    def build_kb_from_public_state(
        self,
        public_state: PublicState,
    ) -> CNF:
        """
        Convenience wrapper for constructing KB directly from
        GameEngine.get_public_state().
        """
        if not isinstance(
            public_state,
            PublicState,
        ):
            raise TypeError(
                "public_state must be a PublicState."
            )

        return self.build_kb(
            revealed_clues=public_state.revealed_clues,
            proved_statuses=public_state.proved_statuses,
        )

    def get_statistics(
        self,
        cnf: CNF,
    ) -> CNFStatistics:
        """
        Return variable and clause statistics for one encoded KB.
        """
        return CNFStatistics(
            primary_variables=self.primary_variable_count,
            auxiliary_variables=self.auxiliary_variable_count,
            clauses=len(cnf),
        )

    # ========================================================
    # FACT
    # ========================================================

    def encode_fact(
        self,
        clue: Clue,
    ) -> CNF:
        """
        FACT(person, status)

        If A is CRIMINAL:

            C_A

        CNF:

            (A)

        If A is INNOCENT:

            NOT C_A

        CNF:

            (-A)
        """
        person = self._get_required_param(
            clue,
            "person",
        )

        status = self._get_required_param(
            clue,
            "status",
        )

        if not isinstance(person, str):
            raise CNFEncodingError(
                f"Clue '{clue.id}': "
                f"'person' must be a string."
            )

        if not isinstance(status, Status):
            raise CNFEncodingError(
                f"Clue '{clue.id}': "
                f"'status' must be a Status value."
            )

        literal = self.literal_for_status(
            person,
            status,
        )

        return [
            [literal]
        ]

    # ========================================================
    # SAME
    # ========================================================

    def encode_same(
        self,
        clue: Clue,
    ) -> CNF:
        """
        SAME(A, B)

        Semantics:

            A <-> B

        CNF:

            (NOT A OR B)
            AND
            (A OR NOT B)
        """
        first, second = self._get_binary_people(
            clue
        )

        a = self.variable_for(
            first
        )

        b = self.variable_for(
            second
        )

        return [
            [-a, b],
            [a, -b],
        ]

    # ========================================================
    # DIFFERENT
    # ========================================================

    def encode_different(
        self,
        clue: Clue,
    ) -> CNF:
        """
        DIFFERENT(A, B)

        Semantics:

            A XOR B

        CNF:

            (A OR B)
            AND
            (NOT A OR NOT B)
        """
        first, second = self._get_binary_people(
            clue
        )

        a = self.variable_for(
            first
        )

        b = self.variable_for(
            second
        )

        return [
            [a, b],
            [-a, -b],
        ]

    # ========================================================
    # EXACTLY
    # ========================================================

    def encode_exactly(
        self,
        clue: Clue,
    ) -> CNF:
        """
        EXACTLY(k, R)

        Exactly k variables in region R must be True.

        Encoded as:

            AT_LEAST(k, R)
            AND
            AT_MOST(k, R)
        """
        k, variables = self._get_counting_variables(
            clue
        )

        return (
            self._encode_at_least_k(
                variables,
                k,
            )
            +
            self._encode_at_most_k(
                variables,
                k,
            )
        )

    # ========================================================
    # AT_LEAST
    # ========================================================

    def encode_at_least(
        self,
        clue: Clue,
    ) -> CNF:
        """
        AT_LEAST(k, R)

        At least k variables must be True.

        For n region variables, at most n-k variables may be False.

        Therefore, for every subset of size:

            n - k + 1

        at least one variable in that subset must be True.
        """
        k, variables = self._get_counting_variables(
            clue
        )

        return self._encode_at_least_k(
            variables,
            k,
        )

    # ========================================================
    # AT_MOST
    # ========================================================

    def encode_at_most(
        self,
        clue: Clue,
    ) -> CNF:
        """
        AT_MOST(k, R)

        At most k variables may be True.

        Therefore every subset of k+1 variables must contain
        at least one False variable.
        """
        k, variables = self._get_counting_variables(
            clue
        )

        return self._encode_at_most_k(
            variables,
            k,
        )

    # ========================================================
    # Cardinality encoding
    # ========================================================

    @staticmethod
    def _encode_at_most_k(
        variables: tuple[int, ...],
        k: int,
    ) -> CNF:
        """
        Direct combinatorial encoding of:

            sum(variables) <= k

        For every subset of k+1 variables:

            NOT x1 OR NOT x2 OR ... OR NOT x_(k+1)

        This prevents any k+1 variables from all being True.
        """
        n = len(variables)

        if k < 0:
            # Impossible condition.
            return [
                []
            ]

        if k >= n:
            # Always true.
            return []

        clauses: CNF = []

        for subset in combinations(
            variables,
            k + 1,
        ):
            clauses.append(
                [
                    -variable
                    for variable in subset
                ]
            )

        return clauses

    @staticmethod
    def _encode_at_least_k(
        variables: tuple[int, ...],
        k: int,
    ) -> CNF:
        """
        Direct combinatorial encoding of:

            sum(variables) >= k

        At most n-k variables may be False.

        Therefore every subset of size n-k+1 must contain
        at least one True variable.
        """
        n = len(variables)

        if k <= 0:
            # Always true.
            return []

        if k > n:
            # Impossible condition.
            return [
                []
            ]

        subset_size = (
            n - k + 1
        )

        clauses: CNF = []

        for subset in combinations(
            variables,
            subset_size,
        ):
            clauses.append(
                list(subset)
            )

        return clauses

    # ========================================================
    # Counting clue helpers
    # ========================================================

    def _get_counting_variables(
        self,
        clue: Clue,
    ) -> tuple[int, tuple[int, ...]]:
        """
        Resolve a counting clue into:

            k
            tuple of SAT variables
        """
        k = self._get_required_param(
            clue,
            "k",
        )

        region = self._get_required_param(
            clue,
            "region",
        )

        if not isinstance(k, int) or isinstance(k, bool):
            raise CNFEncodingError(
                f"Clue '{clue.id}': "
                f"'k' must be an integer."
            )

        try:
            cells = resolve_region_cells(
                region,
                self.size,
            )

        except Exception as exc:
            raise CNFEncodingError(
                f"Clue '{clue.id}' contains "
                f"an invalid region."
            ) from exc

        if not 0 <= k <= len(cells):
            raise CNFEncodingError(
                f"Clue '{clue.id}': "
                f"k={k} does not satisfy "
                f"0 <= k <= |R|={len(cells)}."
            )

        variables = tuple(
            self.variable_for(
                cell
            )
            for cell in cells
        )

        return k, variables

    # ========================================================
    # Binary clue helpers
    # ========================================================

    def _get_binary_people(
        self,
        clue: Clue,
    ) -> tuple[str, str]:
        people = self._get_required_param(
            clue,
            "people",
        )

        if not isinstance(
            people,
            (list, tuple),
        ):
            raise CNFEncodingError(
                f"Clue '{clue.id}': "
                f"'people' must contain two character ids."
            )

        if len(people) != 2:
            raise CNFEncodingError(
                f"Clue '{clue.id}': "
                f"'people' must contain exactly two characters."
            )

        first, second = people

        if not isinstance(first, str) or not isinstance(
            second,
            str,
        ):
            raise CNFEncodingError(
                f"Clue '{clue.id}': "
                f"character ids must be strings."
            )

        if first == second:
            raise CNFEncodingError(
                f"Clue '{clue.id}': "
                f"SAME/DIFFERENT requires two distinct characters."
            )

        # Validate both references now.
        self.variable_for(
            first
        )
        self.variable_for(
            second
        )

        return first, second

    # ========================================================
    # Generic helpers
    # ========================================================

    @staticmethod
    def _get_required_param(
        clue: Clue,
        param_name: str,
    ):
        if param_name not in clue.params:
            raise CNFEncodingError(
                f"Clue '{clue.id}' is missing "
                f"required parameter '{param_name}'."
            )

        return clue.params[
            param_name
        ]

    @staticmethod
    def _normalize_character_id(
        character_id: str,
    ) -> str:
        if not isinstance(character_id, str):
            raise TypeError(
                "character id must be a string."
            )

        normalized = (
            character_id
            .strip()
            .upper()
        )

        if not normalized:
            raise CNFEncodingError(
                "character id must not be empty."
            )

        return normalized

    @staticmethod
    def _normalize_clue_type(
        clue_type: ClueType | str,
    ) -> ClueType | str:
        if isinstance(
            clue_type,
            ClueType,
        ):
            return clue_type

        if isinstance(
            clue_type,
            str,
        ):
            normalized = (
                clue_type
                .strip()
                .upper()
            )

            try:
                return ClueType(
                    normalized
                )

            except ValueError:
                return normalized

        raise CNFEncodingError(
            "Clue type must be a ClueType or string."
        )