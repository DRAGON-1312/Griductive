from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median, pstdev
from typing import Any, Sequence

from core.game_engine import GameEngine
from core.models import Puzzle
from logic.agent import AgentRunResult
from logic.cnf_encoder import CNFEncoder
from logic.entailment import SATMetrics


class ExperimentMetricsError(ValueError):
    """Base exception for experiment-metric errors."""


class MetricsConsistencyError(ExperimentMetricsError):
    """Raised when repeated deterministic runs disagree."""


@dataclass(frozen=True)
class PuzzleMetrics:
    """Static CNF metrics of one benchmark puzzle.

    ``complete_clue_clauses`` is the main clause-count metric used for
    cross-puzzle comparison. Hidden statuses are not included in it.
    ``initial_kb_clauses`` is the real public KB at step 0.
    """

    puzzle_file: str
    puzzle_name: str
    size: int
    total_characters: int
    initial_revealed: int
    primary_variables: int
    auxiliary_variables: int
    total_variables: int
    complete_clue_clauses: int
    initial_kb_clauses: int
    final_public_kb_clauses: int

    @property
    def cnf_clauses(self) -> int:
        return self.complete_clue_clauses

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cnf_clauses"] = self.cnf_clauses
        return data


@dataclass(frozen=True)
class ExperimentRunMetrics:
    """Metrics from one fresh ``LogicAgent.auto_solve()`` run.

    ``solver_runtime`` is the sum of DPLL runtimes recorded by the
    standardized SAT metrics. ``wall_runtime`` is measured externally
    around the whole auto-solve call by ``run_experiments.py``.

    The candidate/UNKNOWN analysis counts are diagnostic values derived
    from the current architecture: each entailment classification uses
    two SAT calls, and each accepted verdict is re-verified by the
    GameEngine using one additional two-query classification.
    """

    puzzle_file: str
    run_index: int
    solved: bool
    stop_reason: str
    deduction_steps: int
    unresolved_characters: int
    sat_calls: int
    decisions: int
    propagations: int
    backtracks: int
    solver_runtime: float
    wall_runtime: float
    candidate_analyses: int | None = None
    unknown_analyses: int | None = None
    timed_out: bool = False
    error_type: str | None = None
    error_message: str | None = None

    @property
    def successful(self) -> bool:
        return self.solved and not self.timed_out and self.error_type is None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["successful"] = self.successful
        return data


@dataclass(frozen=True)
class RuntimeStatistics:
    """Descriptive statistics for repeated runtime measurements."""

    repetitions: int
    mean: float
    median: float
    minimum: float
    maximum: float
    stddev: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "RuntimeStatistics":
        normalized = tuple(float(value) for value in values)

        if not normalized:
            raise ExperimentMetricsError(
                "Runtime statistics require at least one value."
            )

        if any(value < 0.0 for value in normalized):
            raise ExperimentMetricsError(
                "Runtime values must be non-negative."
            )

        return cls(
            repetitions=len(normalized),
            mean=mean(normalized),
            median=median(normalized),
            minimum=min(normalized),
            maximum=max(normalized),
            stddev=pstdev(normalized) if len(normalized) > 1 else 0.0,
        )


@dataclass(frozen=True)
class ExperimentSummary:
    """Report-ready summary of repeated runs for one puzzle.

    Integer workload counters must remain deterministic across
    successful repetitions. Runtime is summarized statistically because
    it is affected by normal OS/interpreter timing noise.
    """

    puzzle_file: str
    puzzle_name: str
    size: int
    repetitions: int
    successful_runs: int
    failed_runs: int
    primary_variables: int
    auxiliary_variables: int
    total_variables: int
    cnf_clauses: int
    initial_kb_clauses: int
    final_public_kb_clauses: int
    deduction_steps: int
    sat_calls: int
    decisions: int
    propagations: int
    backtracks: int
    candidate_analyses: int | None
    unknown_analyses: int | None
    solver_runtime: RuntimeStatistics
    wall_runtime: RuntimeStatistics

    def to_dict(self) -> dict[str, Any]:
        return {
            "puzzle_file": self.puzzle_file,
            "puzzle_name": self.puzzle_name,
            "size": self.size,
            "repetitions": self.repetitions,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "primary_variables": self.primary_variables,
            "auxiliary_variables": self.auxiliary_variables,
            "total_variables": self.total_variables,
            "cnf_clauses": self.cnf_clauses,
            "initial_kb_clauses": self.initial_kb_clauses,
            "final_public_kb_clauses": self.final_public_kb_clauses,
            "deduction_steps": self.deduction_steps,
            "sat_calls": self.sat_calls,
            "decisions": self.decisions,
            "propagations": self.propagations,
            "backtracks": self.backtracks,
            "candidate_analyses": self.candidate_analyses,
            "unknown_analyses": self.unknown_analyses,
            "solver_runtime_mean": self.solver_runtime.mean,
            "solver_runtime_median": self.solver_runtime.median,
            "solver_runtime_min": self.solver_runtime.minimum,
            "solver_runtime_max": self.solver_runtime.maximum,
            "solver_runtime_stddev": self.solver_runtime.stddev,
            "wall_runtime_mean": self.wall_runtime.mean,
            "wall_runtime_median": self.wall_runtime.median,
            "wall_runtime_min": self.wall_runtime.minimum,
            "wall_runtime_max": self.wall_runtime.maximum,
            "wall_runtime_stddev": self.wall_runtime.stddev,
        }


def collect_puzzle_metrics(
    puzzle: Puzzle,
    *,
    puzzle_file: str,
) -> PuzzleMetrics:
    """Collect static CNF metrics for one validated puzzle.

    The experiment harness may inspect the complete puzzle definition
    to measure benchmark complexity. The LogicAgent itself still receives
    only ``PublicState`` and therefore does not gain hidden information.
    """
    if not isinstance(puzzle, Puzzle):
        raise TypeError("puzzle must be a Puzzle.")

    if not isinstance(puzzle_file, str) or not puzzle_file.strip():
        raise ValueError("puzzle_file must be a non-empty string.")

    encoder = CNFEncoder(
        characters=puzzle.characters,
        size=puzzle.size,
    )

    complete_clues = tuple(
        secret.clue for secret in puzzle.secrets.values()
    )
    complete_cnf = encoder.encode_clues(complete_clues)
    complete_stats = encoder.get_statistics(complete_cnf)

    # Measure the actual public KB_0 through GameEngine.
    engine = GameEngine(puzzle)
    initial_state = engine.get_public_state()
    initial_kb = encoder.build_kb_from_public_state(initial_state)

    total_characters = len(puzzle.characters)

    return PuzzleMetrics(
        puzzle_file=puzzle_file,
        puzzle_name=puzzle.name,
        size=puzzle.size,
        total_characters=total_characters,
        initial_revealed=len(puzzle.initial_revealed),
        primary_variables=complete_stats.primary_variables,
        auxiliary_variables=complete_stats.auxiliary_variables,
        total_variables=(
            complete_stats.primary_variables
            + complete_stats.auxiliary_variables
        ),
        complete_clue_clauses=complete_stats.clauses,
        initial_kb_clauses=len(initial_kb),
        final_public_kb_clauses=(
            complete_stats.clauses + total_characters
        ),
    )


def _derive_analysis_counts(
    *,
    sat_calls: int,
    deduction_steps: int,
) -> tuple[int | None, int | None]:
    """Derive candidate and UNKNOWN analyses for the current protocol."""
    if sat_calls < 0 or deduction_steps < 0 or sat_calls % 2 != 0:
        return None, None

    total_classifications = sat_calls // 2

    # One GameEngine classification verifies every accepted deduction.
    candidate_analyses = total_classifications - deduction_steps

    # One forced candidate analysis produces every accepted deduction;
    # all remaining candidate analyses were UNKNOWN scans.
    unknown_analyses = candidate_analyses - deduction_steps

    if candidate_analyses < 0 or unknown_analyses < 0:
        return None, None

    return candidate_analyses, unknown_analyses


def make_run_metrics(
    *,
    puzzle_file: str,
    run_index: int,
    result: AgentRunResult,
    wall_runtime: float,
) -> ExperimentRunMetrics:
    """Convert one successful/normal AgentRunResult into a run row."""
    if not isinstance(puzzle_file, str) or not puzzle_file.strip():
        raise ValueError("puzzle_file must be a non-empty string.")

    if (
        not isinstance(run_index, int)
        or isinstance(run_index, bool)
        or run_index <= 0
    ):
        raise ValueError("run_index must be a positive integer.")

    if not isinstance(result, AgentRunResult):
        raise TypeError("result must be an AgentRunResult.")

    if wall_runtime < 0.0:
        raise ValueError("wall_runtime must be non-negative.")

    candidate_analyses, unknown_analyses = _derive_analysis_counts(
        sat_calls=result.total_sat_calls,
        deduction_steps=result.deduction_count,
    )

    return ExperimentRunMetrics(
        puzzle_file=puzzle_file,
        run_index=run_index,
        solved=result.solved,
        stop_reason=result.stop_reason.value,
        deduction_steps=result.deduction_count,
        unresolved_characters=len(result.unresolved_character_ids),
        sat_calls=result.total_sat_calls,
        decisions=result.total_decisions,
        propagations=result.total_propagations,
        backtracks=result.total_backtracks,
        solver_runtime=result.total_runtime,
        wall_runtime=float(wall_runtime),
        candidate_analyses=candidate_analyses,
        unknown_analyses=unknown_analyses,
    )


def make_failed_run_metrics(
    *,
    puzzle_file: str,
    run_index: int,
    wall_runtime: float,
    error: BaseException,
    timed_out: bool = False,
    partial_metrics: SATMetrics | None = None,
    deduction_steps: int = 0,
    unresolved_characters: int = 0,
) -> ExperimentRunMetrics:
    """Create a failure/timeout row instead of dropping the run."""
    if not isinstance(puzzle_file, str) or not puzzle_file.strip():
        raise ValueError("puzzle_file must be a non-empty string.")

    if (
        not isinstance(run_index, int)
        or isinstance(run_index, bool)
        or run_index <= 0
    ):
        raise ValueError("run_index must be a positive integer.")

    if wall_runtime < 0.0:
        raise ValueError("wall_runtime must be non-negative.")

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception.")

    metrics = partial_metrics if partial_metrics is not None else SATMetrics()

    candidate_analyses, unknown_analyses = _derive_analysis_counts(
        sat_calls=metrics.sat_calls,
        deduction_steps=deduction_steps,
    )

    return ExperimentRunMetrics(
        puzzle_file=puzzle_file,
        run_index=run_index,
        solved=False,
        stop_reason="TIMEOUT" if timed_out else "ERROR",
        deduction_steps=deduction_steps,
        unresolved_characters=unresolved_characters,
        sat_calls=metrics.sat_calls,
        decisions=metrics.decisions,
        propagations=metrics.propagations,
        backtracks=metrics.backtracks,
        solver_runtime=metrics.runtime,
        wall_runtime=float(wall_runtime),
        candidate_analyses=candidate_analyses,
        unknown_analyses=unknown_analyses,
        timed_out=timed_out,
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _deterministic_signature(
    run: ExperimentRunMetrics,
) -> tuple[int, int, int, int, int, int | None, int | None]:
    return (
        run.deduction_steps,
        run.sat_calls,
        run.decisions,
        run.propagations,
        run.backtracks,
        run.candidate_analyses,
        run.unknown_analyses,
    )


def summarize_runs(
    puzzle_metrics: PuzzleMetrics,
    runs: Sequence[ExperimentRunMetrics],
) -> ExperimentSummary:
    """Aggregate repeated runs into one report-ready puzzle summary.

    Failed/timeout runs remain counted but are not mixed into runtime
    statistics. Deterministic workload counters must be identical across
    successful repetitions; a mismatch is treated as an experiment-
    integrity error rather than averaged away.
    """
    if not isinstance(puzzle_metrics, PuzzleMetrics):
        raise TypeError("puzzle_metrics must be PuzzleMetrics.")

    normalized_runs = tuple(runs)

    if not normalized_runs:
        raise ExperimentMetricsError("At least one experiment run is required.")

    for run in normalized_runs:
        if not isinstance(run, ExperimentRunMetrics):
            raise TypeError(
                "runs must contain only ExperimentRunMetrics."
            )

        if run.puzzle_file != puzzle_metrics.puzzle_file:
            raise MetricsConsistencyError(
                "Run puzzle_file does not match PuzzleMetrics."
            )

    successful_runs = tuple(run for run in normalized_runs if run.successful)

    if not successful_runs:
        raise ExperimentMetricsError(
            "Cannot summarize performance: there are no successful runs."
        )

    expected_signature = _deterministic_signature(successful_runs[0])

    for run in successful_runs[1:]:
        if _deterministic_signature(run) != expected_signature:
            raise MetricsConsistencyError(
                "Deterministic workload counters changed across successful "
                "repetitions. Use fresh GameEngine and LogicAgent instances "
                "for every run and investigate possible nondeterminism before "
                "reporting results."
            )

    representative = successful_runs[0]

    solver_runtime = RuntimeStatistics.from_values(
        tuple(run.solver_runtime for run in successful_runs)
    )
    wall_runtime = RuntimeStatistics.from_values(
        tuple(run.wall_runtime for run in successful_runs)
    )

    return ExperimentSummary(
        puzzle_file=puzzle_metrics.puzzle_file,
        puzzle_name=puzzle_metrics.puzzle_name,
        size=puzzle_metrics.size,
        repetitions=len(normalized_runs),
        successful_runs=len(successful_runs),
        failed_runs=len(normalized_runs) - len(successful_runs),
        primary_variables=puzzle_metrics.primary_variables,
        auxiliary_variables=puzzle_metrics.auxiliary_variables,
        total_variables=puzzle_metrics.total_variables,
        cnf_clauses=puzzle_metrics.cnf_clauses,
        initial_kb_clauses=puzzle_metrics.initial_kb_clauses,
        final_public_kb_clauses=puzzle_metrics.final_public_kb_clauses,
        deduction_steps=representative.deduction_steps,
        sat_calls=representative.sat_calls,
        decisions=representative.decisions,
        propagations=representative.propagations,
        backtracks=representative.backtracks,
        candidate_analyses=representative.candidate_analyses,
        unknown_analyses=representative.unknown_analyses,
        solver_runtime=solver_runtime,
        wall_runtime=wall_runtime,
    )