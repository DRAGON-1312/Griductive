from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isqrt

from core.game_engine import GameEngine
from core.models import (
    Classification,
    Clue,
    PublicState,
    Status,
    VerdictCode,
)
from logic.entailment import (
    EntailmentChecker,
    EntailmentResult,
)


# ============================================================
# Exceptions
# ============================================================

class LogicAgentError(RuntimeError):
    """
    Base exception for Logic Agent errors.
    """


class InvalidAgentConfigurationError(LogicAgentError):
    """
    Raised when the GameEngine and EntailmentChecker are
    configured inconsistently.
    """


class AgentKnowledgeBaseError(LogicAgentError):
    """
    Raised when the public knowledge base is inconsistent.
    """


class AgentIntegrityError(LogicAgentError):
    """
    Raised when a logically proved verdict is unexpectedly rejected
    by GameEngine.
    """


# ============================================================
# Stop reason
# ============================================================

class AgentStopReason(str, Enum):
    """
    Reason why an auto-solve run stopped.
    """

    SOLVED = "SOLVED"
    NO_PROVABLE_MOVE = "NO_PROVABLE_MOVE"
    STEP_LIMIT = "STEP_LIMIT"


# ============================================================
# Hint
# ============================================================

@dataclass(frozen=True)
class AgentHint:
    """
    One logically provable move found by the agent.

    A hint does NOT mutate the game state.

    The corresponding EntailmentResult contains the two SAT queries
    proving why the character can be classified.
    """

    character_id: str
    status: Status
    classification: Classification
    analysis: EntailmentResult


# ============================================================
# Deduction step
# ============================================================

@dataclass(frozen=True)
class AgentStep:
    """
    One accepted deduction performed by the Logic Agent.

    This object is suitable for:

        - deduction traces,
        - GUI explanations,
        - experiments,
        - debugging.
    """

    step_number: int

    character_id: str
    status: Status
    classification: Classification

    verdict_code: VerdictCode

    revealed_clue: Clue | None

    analysis: EntailmentResult

    # --------------------------------------------------------
    # Convenience metrics
    # --------------------------------------------------------

    @property
    def decisions(self) -> int:
        """
        Number of DPLL decisions used by the entailment analysis
        that justified this deduction.
        """
        return self.analysis.total_decisions

    @property
    def propagations(self) -> int:
        return self.analysis.total_propagations

    @property
    def backtracks(self) -> int:
        return self.analysis.total_backtracks

    @property
    def runtime(self) -> float:
        return self.analysis.total_runtime


# ============================================================
# Auto-solve result
# ============================================================

@dataclass(frozen=True)
class AgentRunResult:
    """
    Result of one LogicAgent.auto_solve() execution.
    """

    solved: bool
    stop_reason: AgentStopReason

    steps: tuple[AgentStep, ...]

    unresolved_character_ids: tuple[str, ...]

    # --------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------

    @property
    def deduction_count(self) -> int:
        return len(self.steps)

    @property
    def total_decisions(self) -> int:
        return sum(
            step.decisions
            for step in self.steps
        )

    @property
    def total_propagations(self) -> int:
        return sum(
            step.propagations
            for step in self.steps
        )

    @property
    def total_backtracks(self) -> int:
        return sum(
            step.backtracks
            for step in self.steps
        )

    @property
    def total_runtime(self) -> float:
        return sum(
            step.runtime
            for step in self.steps
        )


# ============================================================
# Logic Agent
# ============================================================

class LogicAgent:
    """
    Deductive Griductive agent.

    The agent interacts with GameEngine using public information only.

    It never reads:

        - hidden statuses,
        - unrevealed clues,
        - Puzzle.secrets,
        - the hidden solution.

    At each deduction step:

        1. Read current PublicState.
        2. Examine unresolved characters deterministically.
        3. Run SAT-based entailment.
        4. Ignore UNKNOWN characters.
        5. If CRIMINAL or INNOCENT is logically forced,
           submit that verdict.
        6. GameEngine accepts the verdict and reveals the
           corresponding clue.
        7. Repeat with the enlarged public KB.

    Therefore the agent never guesses.
    """

    def __init__(
        self,
        engine: GameEngine,
        checker: EntailmentChecker | None = None,
    ):
        if not isinstance(
            engine,
            GameEngine,
        ):
            raise TypeError(
                "engine must be a GameEngine."
            )

        self._engine = engine

        # ----------------------------------------------------
        # Infer N using public information only.
        #
        # A valid Griductive board contains N^2 characters.
        # ----------------------------------------------------

        public_state = (
            self._engine.get_public_state()
        )

        character_count = len(
            public_state.characters
        )

        inferred_size = isqrt(
            character_count
        )

        if (
            inferred_size <= 0
            or inferred_size * inferred_size
            != character_count
        ):
            raise InvalidAgentConfigurationError(
                "The public character count does not "
                "represent a square N x N puzzle."
            )

        self.size = inferred_size

        if checker is None:
            checker = EntailmentChecker(
                size=self.size
            )

        elif not isinstance(
            checker,
            EntailmentChecker,
        ):
            raise TypeError(
                "checker must be an EntailmentChecker."
            )

        elif checker.size != self.size:
            raise InvalidAgentConfigurationError(
                f"Agent inferred puzzle size "
                f"{self.size}x{self.size}, "
                f"but EntailmentChecker is configured "
                f"for {checker.size}x{checker.size}."
            )

        self._checker = checker

        self._trace: list[
            AgentStep
        ] = []

    # ========================================================
    # Public properties
    # ========================================================

    @property
    def engine(self) -> GameEngine:
        return self._engine

    @property
    def checker(self) -> EntailmentChecker:
        return self._checker

    @property
    def deduction_trace(
        self,
    ) -> tuple[AgentStep, ...]:
        """
        Immutable snapshot of all deductions performed by this agent.
        """
        return tuple(
            self._trace
        )

    # ========================================================
    # Public-state helpers
    # ========================================================

    @staticmethod
    def _is_solved(
        public_state: PublicState,
    ) -> bool:
        """
        A game is solved when every public character has a proved
        status.

        This check uses PublicState only.
        """
        return (
            len(public_state.proved_statuses)
            ==
            len(public_state.characters)
        )

    @staticmethod
    def _get_unresolved_character_ids(
        public_state: PublicState,
    ) -> tuple[str, ...]:
        """
        Return unresolved characters in deterministic character order.
        """
        return tuple(
            character_id
            for character_id
            in public_state.characters
            if character_id
            not in public_state.proved_statuses
        )

    # ========================================================
    # Classification -> Status
    # ========================================================

    @staticmethod
    def _classification_to_status(
        classification: Classification,
    ) -> Status | None:
        """
        Convert a logically forced classification to a verdict status.

        UNKNOWN produces no verdict.

        INCONSISTENT is handled separately and therefore raises.
        """
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

        if (
            classification
            == Classification.UNKNOWN
        ):
            return None

        if (
            classification
            == Classification.INCONSISTENT
        ):
            raise AgentKnowledgeBaseError(
                "The public knowledge base is inconsistent."
            )

        raise LogicAgentError(
            f"Unsupported classification: "
            f"{classification!r}."
        )

    # ========================================================
    # Hint
    # ========================================================

    def find_hint(
        self,
    ) -> AgentHint | None:
        """
        Find the first logically provable unresolved character.

        This method does NOT modify the game.

        Deterministic strategy:
            scan unresolved characters in PublicState character order
            and return the first CRIMINAL / INNOCENT result.

        UNKNOWN characters are skipped.

        Returns:
            AgentHint:
                when a provable deduction exists.

            None:
                when no unresolved character is currently provable.

        Raises:
            AgentKnowledgeBaseError:
                if entailment detects an inconsistent KB.
        """
        public_state = (
            self._engine.get_public_state()
        )

        return self._find_hint_from_state(
            public_state
        )

    def _find_hint_from_state(
        self,
        public_state: PublicState,
    ) -> AgentHint | None:
        """
        Internal hint search using an already obtained PublicState.
        """
        unresolved_ids = (
            self._get_unresolved_character_ids(
                public_state
            )
        )

        for character_id in unresolved_ids:

            analysis = (
                self._checker.analyze_character(
                    public_state,
                    character_id,
                )
            )

            if (
                analysis.classification
                == Classification.INCONSISTENT
            ):
                raise AgentKnowledgeBaseError(
                    "The public knowledge base is inconsistent."
                )

            status = (
                self._classification_to_status(
                    analysis.classification
                )
            )

            if status is None:
                # UNKNOWN:
                #
                # Both Criminal and Innocent remain logically
                # possible, so submitting a verdict would be
                # guessing.
                continue

            return AgentHint(
                character_id=character_id,
                status=status,
                classification=(
                    analysis.classification
                ),
                analysis=analysis,
            )

        return None

    # ========================================================
    # One deduction step
    # ========================================================

    def step(
        self,
    ) -> AgentStep | None:
        """
        Perform exactly one logically justified deduction.

        Returns:
            AgentStep:
                if a provable verdict was found and accepted.

            None:
                if the puzzle is already solved or no currently
                provable unresolved character exists.

        The method never submits UNKNOWN.
        """
        public_state = (
            self._engine.get_public_state()
        )

        if self._is_solved(
            public_state
        ):
            return None

        hint = self._find_hint_from_state(
            public_state
        )

        if hint is None:
            return None

        # ----------------------------------------------------
        # Defensive validation by GameEngine.
        #
        # GameEngine receives the public-state entailment classifier
        # rather than trusting the hidden answer.
        # ----------------------------------------------------

        verdict_result = (
            self._engine.submit_verdict(
                character_id=hint.character_id,
                submitted_status=hint.status,
                classifier=(
                    self._checker.classify_character
                ),
            )
        )

        # ----------------------------------------------------
        # Since the agent only submits an entailed verdict,
        # GameEngine must accept it.
        #
        # Any other result indicates an integration or state
        # consistency error.
        # ----------------------------------------------------

        if (
            verdict_result.code
            != VerdictCode.ACCEPTED
        ):
            raise AgentIntegrityError(
                f"Entailment proved "
                f"{hint.character_id} as "
                f"{hint.status.value}, "
                f"but GameEngine returned "
                f"{verdict_result.code.value}."
            )

        step_result = AgentStep(
            step_number=(
                len(self._trace) + 1
            ),
            character_id=(
                hint.character_id
            ),
            status=hint.status,
            classification=(
                hint.classification
            ),
            verdict_code=(
                verdict_result.code
            ),
            revealed_clue=(
                verdict_result.revealed_clue
            ),
            analysis=hint.analysis,
        )

        self._trace.append(
            step_result
        )

        return step_result

    # ========================================================
    # Auto solve
    # ========================================================

    def auto_solve(
        self,
        max_steps: int | None = None,
    ) -> AgentRunResult:
        """
        Repeatedly perform logically justified deductions.

        The agent stops when:

            SOLVED
                every character has a proved status.

            NO_PROVABLE_MOVE
                unresolved characters remain, but all are UNKNOWN.

            STEP_LIMIT
                max_steps deductions have been performed.

        No guessing is ever used.

        Args:
            max_steps:
                Optional maximum number of deductions to perform.

                None:
                    allow enough deductions to resolve every currently
                    unresolved character.

                0:
                    perform no deductions.

        Returns:
            AgentRunResult.
        """
        if max_steps is not None:

            if (
                not isinstance(
                    max_steps,
                    int,
                )
                or isinstance(
                    max_steps,
                    bool,
                )
            ):
                raise TypeError(
                    "max_steps must be an integer or None."
                )

            if max_steps < 0:
                raise ValueError(
                    "max_steps must be non-negative."
                )

        initial_public_state = (
            self._engine.get_public_state()
        )

        if self._is_solved(
            initial_public_state
        ):
            return AgentRunResult(
                solved=True,
                stop_reason=(
                    AgentStopReason.SOLVED
                ),
                steps=(),
                unresolved_character_ids=(),
            )

        initial_unresolved = (
            self._get_unresolved_character_ids(
                initial_public_state
            )
        )

        # Every successful deduction resolves exactly one character.
        #
        # Therefore the number of initially unresolved characters is
        # a natural finite upper bound.
        natural_limit = len(
            initial_unresolved
        )

        step_limit = (
            natural_limit
            if max_steps is None
            else min(
                max_steps,
                natural_limit,
            )
        )

        run_steps: list[
            AgentStep
        ] = []

        for _ in range(
            step_limit
        ):
            current_state = (
                self._engine.get_public_state()
            )

            if self._is_solved(
                current_state
            ):
                return AgentRunResult(
                    solved=True,
                    stop_reason=(
                        AgentStopReason.SOLVED
                    ),
                    steps=tuple(
                        run_steps
                    ),
                    unresolved_character_ids=(),
                )

            step_result = self.step()

            if step_result is None:
                final_state = (
                    self._engine.get_public_state()
                )

                unresolved = (
                    self._get_unresolved_character_ids(
                        final_state
                    )
                )

                return AgentRunResult(
                    solved=False,
                    stop_reason=(
                        AgentStopReason.NO_PROVABLE_MOVE
                    ),
                    steps=tuple(
                        run_steps
                    ),
                    unresolved_character_ids=(
                        unresolved
                    ),
                )

            run_steps.append(
                step_result
            )

        # ----------------------------------------------------
        # Re-check state after reaching the step bound.
        # ----------------------------------------------------

        final_state = (
            self._engine.get_public_state()
        )

        solved = self._is_solved(
            final_state
        )

        unresolved = (
            self._get_unresolved_character_ids(
                final_state
            )
        )

        if solved:
            stop_reason = (
                AgentStopReason.SOLVED
            )
        else:
            stop_reason = (
                AgentStopReason.STEP_LIMIT
            )

        return AgentRunResult(
            solved=solved,
            stop_reason=stop_reason,
            steps=tuple(
                run_steps
            ),
            unresolved_character_ids=(
                unresolved
            ),
        )

    # ========================================================
    # Trace management
    # ========================================================

    def clear_trace(
        self,
    ) -> None:
        """
        Clear the agent's stored deduction trace.

        This does not modify GameEngine or restart the puzzle.
        """
        self._trace.clear()