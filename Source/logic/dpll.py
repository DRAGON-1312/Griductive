from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter


# ============================================================
# Type aliases
# ============================================================

Literal = int
Clause = list[Literal]
CNF = list[Clause]
Assignment = dict[int, bool]


# ============================================================
# Exceptions
# ============================================================

class DPLLError(ValueError):
    """
    Base exception for invalid DPLL input.
    """


class InvalidCNFError(DPLLError):
    """
    Raised when the supplied CNF has an invalid representation.
    """


class InvalidAssumptionError(DPLLError):
    """
    Raised when an assumption literal is invalid.
    """


# ============================================================
# Public result
# ============================================================

@dataclass(frozen=True)
class SATResult:
    """
    Result returned by the DPLL solver.

    Attributes:
        satisfiable:
            True if the CNF is SAT, otherwise False.

        assignment:
            Complete variable assignment when SAT.
            None when UNSAT.

        decisions:
            Number of branching decisions made by DPLL.

        propagations:
            Number of variable assignments produced by
            unit propagation.

        backtracks:
            Number of failed search branches that caused
            the solver to backtrack.

        runtime:
            Solver runtime in seconds.
    """

    satisfiable: bool
    assignment: Assignment | None

    decisions: int
    propagations: int
    backtracks: int

    runtime: float


# ============================================================
# Internal metrics
# ============================================================

@dataclass
class _SearchMetrics:
    """
    Mutable metrics used internally during one solve() call.
    """

    decisions: int = 0
    propagations: int = 0
    backtracks: int = 0


# ============================================================
# DPLL Solver
# ============================================================

class DPLLSolver:
    """
    Deterministic DPLL SAT solver.

    CNF representation:

        [
            [1, -2],
            [-1, 3],
            [2, 3, -4],
        ]

    Literal convention:

         n -> variable n is True
        -n -> variable n is False

    The solver implements:

        1. Unit propagation
        2. Conflict detection
        3. Deterministic variable selection
        4. Recursive branching
        5. Backtracking

    Variable selection is deterministic:

        choose the smallest unassigned variable that occurs
        in an unresolved clause.

    Branch order is also deterministic:

        True first, then False.

    This implementation intentionally does not perform
    Griductive-specific reasoning. It is a generic SAT solver.
    """

    # ========================================================
    # Public API
    # ========================================================

    def solve(
        self,
        clauses: Sequence[Sequence[int]],
        num_variables: int,
        assumptions: Iterable[int] | None = None,
    ) -> SATResult:
        """
        Solve a CNF formula using DPLL.

        Args:
            clauses:
                CNF formula represented as a sequence of clauses.

            num_variables:
                Number of SAT variables.

                Valid variable ids are:

                    1, 2, ..., num_variables

            assumptions:
                Optional literals that are temporarily forced for
                this solve call.

                Examples:

                    [3]
                        -> assume variable 3 is True

                    [-3]
                        -> assume variable 3 is False

                Assumptions are useful later for entailment:

                    KB AND NOT C_i

                can be solved using:

                    assumptions=[-variable_i]

        Returns:
            SATResult.

            If SAT:
                result.assignment is a complete assignment.

            If UNSAT:
                result.assignment is None.
        """
        start_time = perf_counter()

        normalized_clauses = self._validate_and_normalize_cnf(
            clauses,
            num_variables,
        )

        normalized_assumptions = self._validate_assumptions(
            assumptions,
            num_variables,
        )

        metrics = _SearchMetrics()

        initial_assignment: Assignment = {}

        # ----------------------------------------------------
        # Apply assumptions
        # ----------------------------------------------------

        for literal in normalized_assumptions:
            variable = abs(literal)
            value = literal > 0

            if variable in initial_assignment:
                if initial_assignment[variable] != value:
                    runtime = perf_counter() - start_time

                    return SATResult(
                        satisfiable=False,
                        assignment=None,
                        decisions=metrics.decisions,
                        propagations=metrics.propagations,
                        backtracks=metrics.backtracks,
                        runtime=runtime,
                    )

                continue

            initial_assignment[variable] = value

        # ----------------------------------------------------
        # DPLL search
        # ----------------------------------------------------

        solution = self._dpll(
            normalized_clauses,
            initial_assignment,
            num_variables,
            metrics,
        )

        runtime = perf_counter() - start_time

        if solution is None:
            return SATResult(
                satisfiable=False,
                assignment=None,
                decisions=metrics.decisions,
                propagations=metrics.propagations,
                backtracks=metrics.backtracks,
                runtime=runtime,
            )

        complete_assignment = self._complete_assignment(
            solution,
            num_variables,
        )

        return SATResult(
            satisfiable=True,
            assignment=complete_assignment,
            decisions=metrics.decisions,
            propagations=metrics.propagations,
            backtracks=metrics.backtracks,
            runtime=runtime,
        )

    # ========================================================
    # Recursive DPLL
    # ========================================================

    def _dpll(
        self,
        clauses: tuple[tuple[int, ...], ...],
        assignment: Assignment,
        num_variables: int,
        metrics: _SearchMetrics,
    ) -> Assignment | None:
        """
        Recursive DPLL procedure.

        Returns:
            A satisfying assignment if SAT.
            None if UNSAT.
        """

        # ----------------------------------------------------
        # Step 1: Unit propagation
        # ----------------------------------------------------

        propagated_assignment = dict(
            assignment
        )

        propagation_success = self._unit_propagate(
            clauses,
            propagated_assignment,
            metrics,
        )

        if not propagation_success:
            return None

        # ----------------------------------------------------
        # Step 2: Check whether all clauses are satisfied
        # ----------------------------------------------------

        if self._all_clauses_satisfied(
            clauses,
            propagated_assignment,
        ):
            return propagated_assignment

        # ----------------------------------------------------
        # Step 3: Choose deterministic branching variable
        # ----------------------------------------------------

        variable = self._choose_variable(
            clauses,
            propagated_assignment,
            num_variables,
        )

        if variable is None:
            # If no variable can be selected while the formula
            # is not satisfied, the branch cannot produce a model.
            return None

        metrics.decisions += 1

        # ----------------------------------------------------
        # Step 4: Branch True
        # ----------------------------------------------------

        true_assignment = dict(
            propagated_assignment
        )

        true_assignment[
            variable
        ] = True

        result = self._dpll(
            clauses,
            true_assignment,
            num_variables,
            metrics,
        )

        if result is not None:
            return result

        metrics.backtracks += 1

        # ----------------------------------------------------
        # Step 5: Branch False
        # ----------------------------------------------------

        false_assignment = dict(
            propagated_assignment
        )

        false_assignment[
            variable
        ] = False

        result = self._dpll(
            clauses,
            false_assignment,
            num_variables,
            metrics,
        )

        if result is not None:
            return result

        metrics.backtracks += 1

        return None

    # ========================================================
    # Unit propagation
    # ========================================================

    def _unit_propagate(
        self,
        clauses: tuple[tuple[int, ...], ...],
        assignment: Assignment,
        metrics: _SearchMetrics,
    ) -> bool:
        """
        Repeatedly apply unit-clause propagation until:

            - no more unit clauses exist, or
            - a conflict is detected.

        Returns:
            True:
                propagation finished without conflict.

            False:
                conflict detected.

        Example:

            clauses:
                (A)
                (NOT A OR B)

            Unit propagation:

                A = True
                B = True
        """

        while True:
            changed = False

            for clause in clauses:

                # --------------------------------------------
                # Determine clause state
                # --------------------------------------------

                clause_satisfied = False
                unassigned_literals: list[int] = []

                for literal in clause:
                    variable = abs(
                        literal
                    )

                    if variable not in assignment:
                        unassigned_literals.append(
                            literal
                        )
                        continue

                    value = assignment[
                        variable
                    ]

                    if self._literal_is_true(
                        literal,
                        value,
                    ):
                        clause_satisfied = True
                        break

                # --------------------------------------------
                # Already satisfied
                # --------------------------------------------

                if clause_satisfied:
                    continue

                # --------------------------------------------
                # No satisfying literal and no unassigned
                # literal -> conflict
                # --------------------------------------------

                if not unassigned_literals:
                    return False

                # --------------------------------------------
                # More than one unassigned literal:
                # not a unit clause yet
                # --------------------------------------------

                if len(
                    unassigned_literals
                ) != 1:
                    continue

                # --------------------------------------------
                # Unit clause
                # --------------------------------------------

                unit_literal = unassigned_literals[
                    0
                ]

                variable = abs(
                    unit_literal
                )

                required_value = (
                    unit_literal > 0
                )

                if variable in assignment:
                    if (
                        assignment[variable]
                        != required_value
                    ):
                        return False

                    continue

                assignment[
                    variable
                ] = required_value

                metrics.propagations += 1
                changed = True

            if not changed:
                return True

    # ========================================================
    # Deterministic variable selection
    # ========================================================

    def _choose_variable(
        self,
        clauses: tuple[tuple[int, ...], ...],
        assignment: Assignment,
        num_variables: int,
    ) -> int | None:
        """
        Choose the smallest unassigned variable occurring in an
        unresolved clause.

        This makes DPLL deterministic.

        Example:

            unresolved variables:
                {2, 5, 7}

            selected:
                2
        """
        candidates: set[int] = set()

        for clause in clauses:

            if self._clause_is_satisfied(
                clause,
                assignment,
            ):
                continue

            for literal in clause:
                variable = abs(
                    literal
                )

                if variable not in assignment:
                    candidates.add(
                        variable
                    )

        if candidates:
            return min(
                candidates
            )

        # Defensive fallback.
        #
        # Normally every unresolved formula has at least one
        # unassigned variable in an unresolved clause.
        for variable in range(
            1,
            num_variables + 1,
        ):
            if variable not in assignment:
                return variable

        return None

    # ========================================================
    # Clause evaluation
    # ========================================================

    def _all_clauses_satisfied(
        self,
        clauses: tuple[tuple[int, ...], ...],
        assignment: Assignment,
    ) -> bool:
        """
        Return True iff every clause already has a true literal.
        """
        return all(
            self._clause_is_satisfied(
                clause,
                assignment,
            )
            for clause in clauses
        )

    @staticmethod
    def _clause_is_satisfied(
        clause: tuple[int, ...],
        assignment: Assignment,
    ) -> bool:
        """
        Return True iff the clause currently contains a literal
        that evaluates to True.
        """
        for literal in clause:
            variable = abs(
                literal
            )

            if variable not in assignment:
                continue

            value = assignment[
                variable
            ]

            if DPLLSolver._literal_is_true(
                literal,
                value,
            ):
                return True

        return False

    @staticmethod
    def _literal_is_true(
        literal: int,
        variable_value: bool,
    ) -> bool:
        """
        Evaluate one literal under a Boolean variable value.

        Examples:

            literal = 3
            variable_value = True
                -> True

            literal = -3
            variable_value = False
                -> True
        """
        if literal > 0:
            return variable_value

        return not variable_value

    # ========================================================
    # Complete SAT assignment
    # ========================================================

    @staticmethod
    def _complete_assignment(
        assignment: Assignment,
        num_variables: int,
    ) -> Assignment:
        """
        Complete a partial satisfying assignment.

        DPLL may satisfy all clauses before every variable has
        received a value.

        Remaining variables are deterministically assigned False.

        Since all clauses are already satisfied, these additional
        values cannot invalidate the discovered model.
        """
        complete = dict(
            assignment
        )

        for variable in range(
            1,
            num_variables + 1,
        ):
            if variable not in complete:
                complete[
                    variable
                ] = False

        return complete

    # ========================================================
    # CNF validation
    # ========================================================

    @staticmethod
    def _validate_and_normalize_cnf(
        clauses: Sequence[Sequence[int]],
        num_variables: int,
    ) -> tuple[tuple[int, ...], ...]:
        """
        Validate CNF input and convert it to immutable tuples.

        Also removes duplicate literals from individual clauses.

        Tautological clauses such as:

            (A OR NOT A OR B)

        are removed because they are always satisfied.
        """
        if not isinstance(
            num_variables,
            int,
        ) or isinstance(
            num_variables,
            bool,
        ):
            raise TypeError(
                "num_variables must be an integer."
            )

        if num_variables < 0:
            raise ValueError(
                "num_variables must be non-negative."
            )

        if not isinstance(
            clauses,
            Sequence,
        ):
            raise TypeError(
                "clauses must be a sequence."
            )

        normalized_cnf: list[
            tuple[int, ...]
        ] = []

        for clause_index, clause in enumerate(
            clauses
        ):
            if not isinstance(
                clause,
                Sequence,
            ):
                raise InvalidCNFError(
                    f"Clause {clause_index} "
                    f"must be a sequence."
                )

            seen_literals: set[int] = set()
            tautology = False

            for literal in clause:
                if not isinstance(
                    literal,
                    int,
                ) or isinstance(
                    literal,
                    bool,
                ):
                    raise InvalidCNFError(
                        f"Clause {clause_index} contains "
                        f"a non-integer literal."
                    )

                if literal == 0:
                    raise InvalidCNFError(
                        f"Clause {clause_index} contains "
                        f"literal 0, which is invalid."
                    )

                variable = abs(
                    literal
                )

                if variable > num_variables:
                    raise InvalidCNFError(
                        f"Clause {clause_index} references "
                        f"variable {variable}, but "
                        f"num_variables={num_variables}."
                    )

                if -literal in seen_literals:
                    tautology = True
                    break

                seen_literals.add(
                    literal
                )

            if tautology:
                # Always-satisfied clause can safely be removed.
                continue

            # Deterministic ordering while preserving the first
            # occurrence of each literal.
            unique_literals: list[int] = []
            already_added: set[int] = set()

            for literal in clause:
                if literal not in already_added:
                    already_added.add(
                        literal
                    )

                    unique_literals.append(
                        literal
                    )

            normalized_cnf.append(
                tuple(
                    unique_literals
                )
            )

        return tuple(
            normalized_cnf
        )

    # ========================================================
    # Assumption validation
    # ========================================================

    @staticmethod
    def _validate_assumptions(
        assumptions: Iterable[int] | None,
        num_variables: int,
    ) -> tuple[int, ...]:
        """
        Validate temporary assumption literals.
        """
        if assumptions is None:
            return ()

        try:
            assumptions_tuple = tuple(
                assumptions
            )

        except TypeError as exc:
            raise TypeError(
                "assumptions must be an iterable of integers."
            ) from exc

        normalized: list[int] = []

        for literal in assumptions_tuple:
            if not isinstance(
                literal,
                int,
            ) or isinstance(
                literal,
                bool,
            ):
                raise InvalidAssumptionError(
                    "Assumption literals must be integers."
                )

            if literal == 0:
                raise InvalidAssumptionError(
                    "Assumption literal 0 is invalid."
                )

            variable = abs(
                literal
            )

            if variable > num_variables:
                raise InvalidAssumptionError(
                    f"Assumption references variable "
                    f"{variable}, but "
                    f"num_variables={num_variables}."
                )

            normalized.append(
                literal
            )

        return tuple(
            normalized
        )