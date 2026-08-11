from __future__ import annotations

from dataclasses import dataclass

from core.models import (
    Classification,
    PublicState,
)
from logic.cnf_encoder import (
    CNFEncoder,
)
from logic.dpll import (
    DPLLSolver,
    SATResult,
)


# ============================================================
# Exceptions
# ============================================================

class EntailmentError(ValueError):
    """
    Base exception for entailment-related errors.
    """


class InvalidPublicStateError(EntailmentError):
    """
    Raised when a PublicState is incompatible with the configured
    puzzle size.
    """


# ============================================================
# SAT workload metrics
# ============================================================

@dataclass(frozen=True)
class SATMetrics:
    """
    Aggregate workload of one or more SAT solver calls.

    The metrics measure actual DPLL computation:

        - sat_calls
        - decisions
        - propagations
        - backtracks
        - runtime

    Instances are immutable so they can safely be used as
    snapshots before and after a reasoning operation.
    """

    sat_calls: int = 0
    decisions: int = 0
    propagations: int = 0
    backtracks: int = 0
    runtime: float = 0.0

    def __sub__(
        self,
        other: SATMetrics,
    ) -> SATMetrics:
        """
        Return the workload performed between two cumulative
        metric snapshots.

        Typical use:

            before = checker.metrics

            ... perform reasoning ...

            used = checker.metrics - before
        """
        if not isinstance(
            other,
            SATMetrics,
        ):
            return NotImplemented

        return SATMetrics(
            sat_calls=(
                self.sat_calls
                - other.sat_calls
            ),
            decisions=(
                self.decisions
                - other.decisions
            ),
            propagations=(
                self.propagations
                - other.propagations
            ),
            backtracks=(
                self.backtracks
                - other.backtracks
            ),
            runtime=(
                self.runtime
                - other.runtime
            ),
        )


# ============================================================
# Entailment result
# ============================================================

@dataclass(frozen=True)
class EntailmentResult:
    """
    Detailed logical classification result for one character.

    Two SAT queries are recorded:

        assume_innocent_result:
            solves:

                KB AND NOT C_i

            If UNSAT, then:

                KB |= C_i

            so the character is forced CRIMINAL.

        assume_criminal_result:
            solves:

                KB AND C_i

            If UNSAT, then:

                KB |= NOT C_i

            so the character is forced INNOCENT.

    If both are SAT:
        UNKNOWN

    If both are UNSAT:
        INCONSISTENT
    """

    character_id: str
    variable: int
    classification: Classification

    assume_innocent_result: SATResult
    assume_criminal_result: SATResult

    kb_clause_count: int
    variable_count: int

    # --------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------

    @property
    def criminal_forced(self) -> bool:
        """
        True iff:

            KB AND NOT C_i

        is UNSAT.
        """
        return (
            not self.assume_innocent_result.satisfiable
        )

    @property
    def innocent_forced(self) -> bool:
        """
        True iff:

            KB AND C_i

        is UNSAT.
        """
        return (
            not self.assume_criminal_result.satisfiable
        )

    @property
    def sat_calls(self) -> int:
        """
        Every character entailment analysis consists of exactly
        two SAT queries:

            KB AND NOT C_i
            KB AND C_i
        """
        return 2

    @property
    def total_decisions(self) -> int:
        return (
            self.assume_innocent_result.decisions
            + self.assume_criminal_result.decisions
        )

    @property
    def total_propagations(self) -> int:
        return (
            self.assume_innocent_result.propagations
            + self.assume_criminal_result.propagations
        )

    @property
    def total_backtracks(self) -> int:
        return (
            self.assume_innocent_result.backtracks
            + self.assume_criminal_result.backtracks
        )

    @property
    def total_runtime(self) -> float:
        return (
            self.assume_innocent_result.runtime
            + self.assume_criminal_result.runtime
        )


# ============================================================
# Entailment checker
# ============================================================

class EntailmentChecker:
    """
    Perform logical entailment queries over the current public
    Griductive knowledge base.

    For every character i:

        C_i = True
            means CRIMINAL

        C_i = False
            means INNOCENT

    Classification uses exactly two SAT calls:

        Query 1:
            KB AND NOT C_i

        Query 2:
            KB AND C_i

    Classification table:

        Query 1      Query 2      Result
        ---------------------------------------
        UNSAT        SAT          CRIMINAL
        SAT          UNSAT        INNOCENT
        SAT          SAT          UNKNOWN
        UNSAT        UNSAT        INCONSISTENT

    The checker receives PublicState only. It never has access to:

        - hidden statuses,
        - unrevealed clues,
        - the complete Puzzle solution.
    """

    def __init__(
        self,
        size: int,
        solver: DPLLSolver | None = None,
    ):
        """
        Create an entailment checker for an N x N puzzle.

        Args:
            size:
                Puzzle size N.

            solver:
                Optional DPLL solver instance.

                Dependency injection is supported mainly for testing.
                If omitted, a normal DPLLSolver is created.
        """
        if not isinstance(
            size,
            int,
        ) or isinstance(
            size,
            bool,
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

        # ----------------------------------------------------
        # Cumulative SAT workload.
        #
        # Every DPLL solve performed through this checker is
        # aggregated here.
        # ----------------------------------------------------

        self._sat_call_count = 0
        self._decision_count = 0
        self._propagation_count = 0
        self._backtrack_count = 0
        self._solver_runtime = 0.0

    # ========================================================
    # SAT workload metrics
    # ========================================================

    @property
    def metrics(self) -> SATMetrics:
        """
        Immutable snapshot of the cumulative SAT workload
        performed by this checker.

        The snapshot includes every DPLL.solve() call made through
        the checker since construction or the latest reset_metrics().
        """
        return SATMetrics(
            sat_calls=self._sat_call_count,
            decisions=self._decision_count,
            propagations=self._propagation_count,
            backtracks=self._backtrack_count,
            runtime=self._solver_runtime,
        )

    @property
    def sat_call_count(self) -> int:
        """
        Cumulative number of actual SAT solver calls performed
        by this EntailmentChecker instance.

        This includes calls made by:

            - analyze_character()
            - classify_character()
            - analyze_all()
            - classify_all()
            - is_kb_consistent()

        Rebuilding or encoding the KB is not a SAT call.
        """
        return self._sat_call_count

    def reset_sat_call_count(self) -> None:
        """
        Reset only the cumulative SAT-call counter to zero.

        Other workload counters are intentionally preserved.
        Use reset_metrics() when a fresh measurement window is
        required for every metric.

        This does not modify the solver, puzzle state, or KB.
        """
        self._sat_call_count = 0

    def reset_metrics(self) -> None:
        """
        Reset all cumulative SAT workload metrics.

        This does not modify:

            - the SAT solver,
            - the public knowledge base,
            - the puzzle,
            - the game state.
        """
        self._sat_call_count = 0
        self._decision_count = 0
        self._propagation_count = 0
        self._backtrack_count = 0
        self._solver_runtime = 0.0

    def _solve(
        self,
        *,
        clauses: list[list[int]],
        num_variables: int,
        assumptions: list[int] | None = None,
    ) -> SATResult:
        """
        Execute exactly one SAT call and aggregate its workload.

        All EntailmentChecker SAT queries must pass through this
        helper so every experiment metric uses the same
        measurement scope.
        """
        self._sat_call_count += 1

        result = self._solver.solve(
            clauses=clauses,
            num_variables=num_variables,
            assumptions=assumptions,
        )

        self._decision_count += (
            result.decisions
        )

        self._propagation_count += (
            result.propagations
        )

        self._backtrack_count += (
            result.backtracks
        )

        self._solver_runtime += (
            result.runtime
        )

        return result

    # ========================================================
    # Main classifier API
    # ========================================================

    def classify_character(
        self,
        public_state: PublicState,
        character_id: str,
    ) -> Classification:
        """
        Return only the final logical classification.

        This method intentionally matches the callable interface
        expected by GameEngine.submit_verdict():

            classifier(
                public_state,
                character_id,
            ) -> Classification

        Therefore it can later be passed directly as:

            engine.submit_verdict(
                "B2",
                status,
                checker.classify_character,
            )
        """
        result = self.analyze_character(
            public_state,
            character_id,
        )

        return result.classification

    # ========================================================
    # Detailed entailment analysis
    # ========================================================

    def analyze_character(
        self,
        public_state: PublicState,
        character_id: str,
    ) -> EntailmentResult:
        """
        Classify one character and return the two underlying
        SAT-query results.

        This detailed form is useful for:

            - deduction traces,
            - GUI explanations,
            - experiment metrics,
            - debugging.

        Exactly two SAT calls are used.
        """
        encoder, kb = self._prepare_kb(
            public_state
        )

        variable = encoder.variable_for(
            character_id
        )

        # ----------------------------------------------------
        # Query 1
        #
        # Test:
        #
        #     KB AND NOT C_i
        #
        # If UNSAT:
        #
        #     KB |= C_i
        #
        # therefore the character is forced CRIMINAL.
        # ----------------------------------------------------

        assume_innocent_result = self._solve(
            clauses=kb,
            num_variables=encoder.total_variable_count,
            assumptions=[
                -variable,
            ],
        )

        # ----------------------------------------------------
        # Query 2
        #
        # Test:
        #
        #     KB AND C_i
        #
        # If UNSAT:
        #
        #     KB |= NOT C_i
        #
        # therefore the character is forced INNOCENT.
        # ----------------------------------------------------

        assume_criminal_result = self._solve(
            clauses=kb,
            num_variables=encoder.total_variable_count,
            assumptions=[
                variable,
            ],
        )

        classification = self._classify_from_queries(
            assume_innocent_result,
            assume_criminal_result,
        )

        return EntailmentResult(
            character_id=(
                character_id
                .strip()
                .upper()
            ),
            variable=variable,
            classification=classification,
            assume_innocent_result=assume_innocent_result,
            assume_criminal_result=assume_criminal_result,
            kb_clause_count=len(kb),
            variable_count=encoder.total_variable_count,
        )

    # ========================================================
    # Classify multiple characters
    # ========================================================

    def analyze_all(
        self,
        public_state: PublicState,
        only_unresolved: bool = True,
    ) -> dict[str, EntailmentResult]:
        """
        Analyze multiple characters under the same public state.

        Args:
            public_state:
                Current public knowledge.

            only_unresolved:
                True:
                    skip characters whose statuses have already
                    been proved.

                False:
                    classify every character.

        Returns:
            Mapping:

                character_id -> EntailmentResult

        The same encoded KB is reused for all characters.
        """
        encoder, kb = self._prepare_kb(
            public_state
        )

        results: dict[
            str,
            EntailmentResult,
        ] = {}

        for character_id in public_state.characters:

            if (
                only_unresolved
                and character_id
                in public_state.proved_statuses
            ):
                continue

            variable = encoder.variable_for(
                character_id
            )

            assume_innocent_result = self._solve(
                clauses=kb,
                num_variables=encoder.total_variable_count,
                assumptions=[
                    -variable,
                ],
            )

            assume_criminal_result = self._solve(
                clauses=kb,
                num_variables=encoder.total_variable_count,
                assumptions=[
                    variable,
                ],
            )

            classification = self._classify_from_queries(
                assume_innocent_result,
                assume_criminal_result,
            )

            results[
                character_id
            ] = EntailmentResult(
                character_id=character_id,
                variable=variable,
                classification=classification,
                assume_innocent_result=assume_innocent_result,
                assume_criminal_result=assume_criminal_result,
                kb_clause_count=len(kb),
                variable_count=encoder.total_variable_count,
            )

        return results

    def classify_all(
        self,
        public_state: PublicState,
        only_unresolved: bool = True,
    ) -> dict[str, Classification]:
        """
        Convenience wrapper returning only classifications.
        """
        analyses = self.analyze_all(
            public_state,
            only_unresolved=only_unresolved,
        )

        return {
            character_id:
                result.classification
            for character_id, result
            in analyses.items()
        }

    # ========================================================
    # KB consistency
    # ========================================================

    def is_kb_consistent(
        self,
        public_state: PublicState,
    ) -> bool:
        """
        Explicitly check whether the current public KB is SAT.

        This method requires one SAT call.

        Normal character classification does NOT call this method,
        because analyze_character() can detect inconsistency directly
        from its required two queries:

            KB AND NOT C_i
            KB AND C_i

        If both are UNSAT, the KB itself is inconsistent.
        """
        encoder, kb = self._prepare_kb(
            public_state
        )

        result = self._solve(
            clauses=kb,
            num_variables=encoder.total_variable_count,
        )

        return result.satisfiable

    # ========================================================
    # KB preparation
    # ========================================================

    def _prepare_kb(
        self,
        public_state: PublicState,
    ) -> tuple[
        CNFEncoder,
        list[list[int]],
    ]:
        """
        Validate public state and construct its CNF knowledge base.

        Only public information is encoded.
        """
        self._validate_public_state(
            public_state
        )

        encoder = CNFEncoder(
            characters=public_state.characters,
            size=self.size,
        )

        kb = encoder.build_kb_from_public_state(
            public_state
        )

        return encoder, kb

    # ========================================================
    # Classification logic
    # ========================================================

    @staticmethod
    def _classify_from_queries(
        assume_innocent_result: SATResult,
        assume_criminal_result: SATResult,
    ) -> Classification:
        """
        Convert the two SAT outcomes into the four required
        Griductive classifications.

        Let:

            Q_I = SAT(KB AND NOT C_i)
            Q_C = SAT(KB AND C_i)

        Cases:

            Q_I = UNSAT
            Q_C = SAT

                -> CRIMINAL

            Q_I = SAT
            Q_C = UNSAT

                -> INNOCENT

            Q_I = SAT
            Q_C = SAT

                -> UNKNOWN

            Q_I = UNSAT
            Q_C = UNSAT

                -> INCONSISTENT
        """
        innocent_possible = (
            assume_innocent_result.satisfiable
        )

        criminal_possible = (
            assume_criminal_result.satisfiable
        )

        # ----------------------------------------------------
        # KB itself is inconsistent.
        #
        # If KB had any model, C_i would necessarily be either
        # True or False in that model, so at least one of the two
        # assumption queries would be SAT.
        # ----------------------------------------------------

        if (
            not innocent_possible
            and not criminal_possible
        ):
            return Classification.INCONSISTENT

        # ----------------------------------------------------
        # Assuming Innocent creates a contradiction.
        #
        # Therefore Criminal is logically forced.
        # ----------------------------------------------------

        if not innocent_possible:
            return Classification.CRIMINAL

        # ----------------------------------------------------
        # Assuming Criminal creates a contradiction.
        #
        # Therefore Innocent is logically forced.
        # ----------------------------------------------------

        if not criminal_possible:
            return Classification.INNOCENT

        # ----------------------------------------------------
        # Both assignments remain possible.
        # ----------------------------------------------------

        return Classification.UNKNOWN

    # ========================================================
    # Validation
    # ========================================================

    def _validate_public_state(
        self,
        public_state: PublicState,
    ) -> None:
        if not isinstance(
            public_state,
            PublicState,
        ):
            raise TypeError(
                "public_state must be a PublicState."
            )

        expected_character_count = (
            self.size * self.size
        )

        actual_character_count = len(
            public_state.characters
        )

        if (
            actual_character_count
            != expected_character_count
        ):
            raise InvalidPublicStateError(
                f"Expected {expected_character_count} "
                f"characters for a {self.size}x{self.size} "
                f"puzzle, but received "
                f"{actual_character_count}."
            )