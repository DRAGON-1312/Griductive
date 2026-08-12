from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence


# ============================================================
# Import path
# ============================================================
#
# The recommended invocation is:
#
#     py -m experiments.run_experiments
#
# from the project Source/ directory.
#
# Adding Source/ explicitly also makes direct execution robust:
#
#     py experiments/run_experiments.py
#

SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


from core.game_engine import GameEngine
from core.puzzle_loader import load_puzzle
from experiments.metrics import (
    ExperimentMetricsError,
    ExperimentRunMetrics,
    ExperimentSummary,
    PuzzleMetrics,
    collect_puzzle_metrics,
    make_failed_run_metrics,
    make_run_metrics,
    summarize_runs,
)
from logic.agent import LogicAgent


# ============================================================
# Benchmark configuration
# ============================================================

PUZZLES_DIR = SOURCE_DIR / "puzzles"
EXPERIMENTS_DIR = SOURCE_DIR / "experiments"
DEFAULT_RESULTS_DIR = EXPERIMENTS_DIR / "results"

BENCHMARK_PUZZLE_FILES: tuple[str, ...] = (
    "puzzle_3x3_01.json",
    "puzzle_3x3_02.json",
    "puzzle_4x4_01.json",
    "puzzle_4x4_02.json",
    "puzzle_5x5_01.json",
    "puzzle_5x5_02.json",
)

DEFAULT_REPETITIONS = 30
DEFAULT_WARMUPS = 3


# ============================================================
# CSV schemas
# ============================================================

PUZZLE_METRIC_FIELDS: tuple[str, ...] = (
    "puzzle_file",
    "puzzle_name",
    "size",
    "total_characters",
    "initial_revealed",
    "primary_variables",
    "auxiliary_variables",
    "total_variables",
    "complete_clue_clauses",
    "initial_kb_clauses",
    "final_public_kb_clauses",
    "cnf_clauses",
)

RUN_METRIC_FIELDS: tuple[str, ...] = (
    "puzzle_file",
    "run_index",
    "solved",
    "stop_reason",
    "deduction_steps",
    "unresolved_characters",
    "sat_calls",
    "decisions",
    "propagations",
    "backtracks",
    "solver_runtime",
    "wall_runtime",
    "candidate_analyses",
    "unknown_analyses",
    "timed_out",
    "error_type",
    "error_message",
    "successful",
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "puzzle_file",
    "puzzle_name",
    "size",
    "repetitions",
    "successful_runs",
    "failed_runs",
    "primary_variables",
    "auxiliary_variables",
    "total_variables",
    "cnf_clauses",
    "initial_kb_clauses",
    "final_public_kb_clauses",
    "deduction_steps",
    "sat_calls",
    "decisions",
    "propagations",
    "backtracks",
    "candidate_analyses",
    "unknown_analyses",
    "solver_runtime_mean",
    "solver_runtime_median",
    "solver_runtime_min",
    "solver_runtime_max",
    "solver_runtime_stddev",
    "wall_runtime_mean",
    "wall_runtime_median",
    "wall_runtime_min",
    "wall_runtime_max",
    "wall_runtime_stddev",
)


# ============================================================
# Validation helpers
# ============================================================

def _positive_int(value: str) -> int:
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )

    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        )

    return parsed


def _validate_puzzle_files(
    puzzle_files: Sequence[str],
) -> tuple[Path, ...]:
    if not puzzle_files:
        raise ValueError(
            "At least one puzzle file is required."
        )

    paths: list[Path] = []

    for filename in puzzle_files:
        if filename not in BENCHMARK_PUZZLE_FILES:
            raise ValueError(
                f"Unknown benchmark puzzle: {filename}"
            )

        path = PUZZLES_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Benchmark puzzle does not exist: {path}"
            )

        paths.append(path)

    return tuple(paths)


# ============================================================
# Reproducibility metadata
# ============================================================

def _git_value(
    *args: str,
) -> str | None:
    repo_root = SOURCE_DIR.parent

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    value = completed.stdout.strip()

    return value or None


def _build_metadata(
    *,
    repetitions: int,
    warmups: int,
    puzzle_files: Sequence[str],
) -> dict[str, Any]:
    return {
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "project_source_directory": str(
            SOURCE_DIR
        ),
        "puzzles_directory": str(
            PUZZLES_DIR
        ),
        "repetitions_per_puzzle": repetitions,
        "warmup_runs_per_puzzle": warmups,
        "timer": "time.perf_counter",
        "gc_collect_before_each_measured_run": True,
        "runtime_reporting": {
            "primary_report_statistic": "median",
            "also_recorded": [
                "mean",
                "minimum",
                "maximum",
                "population_stddev",
            ],
            "solver_runtime_scope": (
                "sum of DPLL runtimes recorded by SAT metrics"
            ),
            "wall_runtime_scope": (
                "end-to-end LogicAgent.auto_solve() elapsed time"
            ),
        },
        "benchmark_puzzles": list(
            puzzle_files
        ),
        "python": {
            "version": sys.version,
            "implementation": (
                platform.python_implementation()
            ),
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": {
            "commit": _git_value(
                "rev-parse",
                "HEAD",
            ),
            "branch": _git_value(
                "branch",
                "--show-current",
            ),
            "working_tree_status": _git_value(
                "status",
                "--porcelain",
            ),
        },
    }


# ============================================================
# Output helpers
# ============================================================

def _create_output_directory(
    requested: Path | None,
) -> Path:
    if requested is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        output_dir = (
            DEFAULT_RESULTS_DIR
            / f"run_{timestamp}"
        )
    else:
        output_dir = requested

        if not output_dir.is_absolute():
            output_dir = (
                SOURCE_DIR
                / output_dir
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_dir


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def _write_json(
    path: Path,
    data: Any,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")


# ============================================================
# One fresh Agent run
# ============================================================

def _run_once(
    *,
    puzzle,
    puzzle_file: str,
    run_index: int,
) -> ExperimentRunMetrics:
    """
    Execute one measured run with fresh mutable state.

    Every repetition constructs a new GameEngine and LogicAgent so
    previous deductions, traces, and SAT counters cannot leak into
    later measurements.
    """
    engine = GameEngine(
        puzzle
    )

    agent = LogicAgent(
        engine
    )

    # Force a GC cycle OUTSIDE the timed region.  This reduces noise
    # from garbage left by the preceding repetition without hiding
    # garbage collection that naturally occurs during auto_solve().
    gc.collect()

    started = perf_counter()

    try:
        result = agent.auto_solve()
    except Exception as error:
        wall_runtime = (
            perf_counter()
            - started
        )

        public_state = (
            engine.get_public_state()
        )

        proved_count = len(
            public_state.proved_statuses
        )

        deduction_steps = max(
            0,
            proved_count
            - len(puzzle.initial_revealed),
        )

        unresolved_characters = (
            len(public_state.characters)
            - proved_count
        )

        return make_failed_run_metrics(
            puzzle_file=puzzle_file,
            run_index=run_index,
            wall_runtime=wall_runtime,
            error=error,
            timed_out=isinstance(
                error,
                TimeoutError,
            ),
            partial_metrics=(
                agent.checker.metrics
            ),
            deduction_steps=(
                deduction_steps
            ),
            unresolved_characters=(
                unresolved_characters
            ),
        )

    wall_runtime = (
        perf_counter()
        - started
    )

    return make_run_metrics(
        puzzle_file=puzzle_file,
        run_index=run_index,
        result=result,
        wall_runtime=wall_runtime,
    )


# ============================================================
# Warm-up
# ============================================================

def _warm_up(
    *,
    puzzle,
    puzzle_file: str,
    warmups: int,
) -> None:
    """
    Perform unrecorded warm-up runs.

    Warm-ups use fresh GameEngine/LogicAgent instances exactly like
    measured runs. They are excluded from all exported measurements.
    """
    for warmup_index in range(
        1,
        warmups + 1,
    ):
        engine = GameEngine(
            puzzle
        )

        agent = LogicAgent(
            engine
        )

        result = agent.auto_solve()

        if not result.solved:
            raise RuntimeError(
                f"Warm-up {warmup_index} for {puzzle_file} "
                f"did not solve the puzzle "
                f"(stop_reason={result.stop_reason.value})."
            )


# ============================================================
# One puzzle benchmark
# ============================================================

def run_puzzle_experiment(
    puzzle_path: Path,
    *,
    repetitions: int,
    warmups: int,
) -> tuple[
    PuzzleMetrics,
    tuple[ExperimentRunMetrics, ...],
    ExperimentSummary | None,
]:
    """
    Run one validated benchmark puzzle repeatedly.

    Static CNF metrics are measured once.
    Each dynamic repetition starts from a fresh GameEngine/LogicAgent.
    """
    puzzle = load_puzzle(
        puzzle_path
    )

    puzzle_file = (
        puzzle_path.name
    )

    static_metrics = (
        collect_puzzle_metrics(
            puzzle,
            puzzle_file=puzzle_file,
        )
    )

    if warmups > 0:
        _warm_up(
            puzzle=puzzle,
            puzzle_file=puzzle_file,
            warmups=warmups,
        )

    runs: list[
        ExperimentRunMetrics
    ] = []

    for run_index in range(
        1,
        repetitions + 1,
    ):
        run = _run_once(
            puzzle=puzzle,
            puzzle_file=puzzle_file,
            run_index=run_index,
        )

        runs.append(
            run
        )

    try:
        summary = summarize_runs(
            static_metrics,
            runs,
        )
    except ExperimentMetricsError:
        # Raw failure rows are still preserved.  A puzzle with zero
        # successful repetitions simply has no performance summary.
        summary = None

    return (
        static_metrics,
        tuple(runs),
        summary,
    )


# ============================================================
# Console reporting
# ============================================================

def _format_seconds(
    value: float,
) -> str:
    if value < 0.001:
        return (
            f"{value * 1_000_000:.3f} us"
        )

    if value < 1.0:
        return (
            f"{value * 1_000:.3f} ms"
        )

    return f"{value:.6f} s"


def _print_puzzle_header(
    *,
    index: int,
    total: int,
    path: Path,
) -> None:
    print()
    print(
        f"[{index}/{total}] {path.name}"
    )


def _print_summary(
    static_metrics: PuzzleMetrics,
    runs: Sequence[ExperimentRunMetrics],
    summary: ExperimentSummary | None,
) -> None:
    successful = sum(
        1
        for run in runs
        if run.successful
    )

    failed = (
        len(runs)
        - successful
    )

    print(
        "  Static: "
        f"vars={static_metrics.primary_variables}"
        f"+{static_metrics.auxiliary_variables} aux, "
        f"clauses={static_metrics.cnf_clauses}"
    )

    print(
        "  Runs:   "
        f"{successful}/{len(runs)} successful, "
        f"{failed} failed"
    )

    if summary is None:
        print(
            "  Summary unavailable: no successful runs."
        )
        return

    print(
        "  Logic:  "
        f"steps={summary.deduction_steps}, "
        f"SAT calls={summary.sat_calls}, "
        f"decisions={summary.decisions}, "
        f"propagations={summary.propagations}, "
        f"backtracks={summary.backtracks}"
    )

    print(
        "  Scan:   "
        f"candidates={summary.candidate_analyses}, "
        f"UNKNOWN={summary.unknown_analyses}"
    )

    print(
        "  Time:   "
        f"solver median="
        f"{_format_seconds(summary.solver_runtime.median)}, "
        f"wall median="
        f"{_format_seconds(summary.wall_runtime.median)}"
    )


# ============================================================
# Full benchmark
# ============================================================

def run_benchmark(
    *,
    puzzle_files: Sequence[str],
    repetitions: int,
    warmups: int,
    output_dir: Path | None = None,
) -> Path:
    """
    Execute the benchmark suite and export reproducible raw/summary data.
    """
    puzzle_paths = (
        _validate_puzzle_files(
            puzzle_files
        )
    )

    resolved_output_dir = (
        _create_output_directory(
            output_dir
        )
    )

    metadata = _build_metadata(
        repetitions=repetitions,
        warmups=warmups,
        puzzle_files=puzzle_files,
    )

    static_rows: list[
        PuzzleMetrics
    ] = []

    raw_runs: list[
        ExperimentRunMetrics
    ] = []

    summaries: list[
        ExperimentSummary
    ] = []

    print(
        "Griductive benchmark"
    )
    print(
        f"  puzzles:      {len(puzzle_paths)}"
    )
    print(
        f"  repetitions:  {repetitions} per puzzle"
    )
    print(
        f"  warm-ups:     {warmups} per puzzle"
    )
    print(
        f"  output:       {resolved_output_dir}"
    )

    for index, puzzle_path in enumerate(
        puzzle_paths,
        start=1,
    ):
        _print_puzzle_header(
            index=index,
            total=len(puzzle_paths),
            path=puzzle_path,
        )

        (
            static_metrics,
            runs,
            summary,
        ) = run_puzzle_experiment(
            puzzle_path,
            repetitions=repetitions,
            warmups=warmups,
        )

        static_rows.append(
            static_metrics
        )

        raw_runs.extend(
            runs
        )

        if summary is not None:
            summaries.append(
                summary
            )

        _print_summary(
            static_metrics,
            runs,
            summary,
        )

    # --------------------------------------------------------
    # Export only AFTER the benchmark has completed.
    #
    # The data structures remain in memory during timing so file I/O
    # cannot contaminate any measured auto_solve() wall-runtime value.
    # --------------------------------------------------------

    _write_csv(
        resolved_output_dir
        / "puzzle_metrics.csv",
        fieldnames=PUZZLE_METRIC_FIELDS,
        rows=(
            item.to_dict()
            for item in static_rows
        ),
    )

    _write_csv(
        resolved_output_dir
        / "raw_runs.csv",
        fieldnames=RUN_METRIC_FIELDS,
        rows=(
            item.to_dict()
            for item in raw_runs
        ),
    )

    _write_csv(
        resolved_output_dir
        / "summary.csv",
        fieldnames=SUMMARY_FIELDS,
        rows=(
            item.to_dict()
            for item in summaries
        ),
    )

    metadata["results"] = {
        "puzzle_count": len(
            static_rows
        ),
        "raw_run_count": len(
            raw_runs
        ),
        "summary_row_count": len(
            summaries
        ),
        "successful_run_count": sum(
            1
            for run in raw_runs
            if run.successful
        ),
        "failed_run_count": sum(
            1
            for run in raw_runs
            if not run.successful
        ),
    }

    _write_json(
        resolved_output_dir
        / "metadata.json",
        metadata,
    )

    print()
    print(
        "Benchmark complete."
    )
    print(
        f"Results written to: "
        f"{resolved_output_dir}"
    )

    return resolved_output_dir


# ============================================================
# CLI
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible experiments on the validated "
            "Griductive benchmark puzzle suite."
        )
    )

    parser.add_argument(
        "-r",
        "--repetitions",
        type=_positive_int,
        default=DEFAULT_REPETITIONS,
        help=(
            "Measured runs per puzzle "
            f"(default: {DEFAULT_REPETITIONS})."
        ),
    )

    parser.add_argument(
        "-w",
        "--warmups",
        type=_non_negative_int,
        default=DEFAULT_WARMUPS,
        help=(
            "Unrecorded warm-up runs per puzzle "
            f"(default: {DEFAULT_WARMUPS})."
        ),
    )

    parser.add_argument(
        "--puzzles",
        nargs="+",
        choices=BENCHMARK_PUZZLE_FILES,
        default=None,
        help=(
            "Optional subset of benchmark puzzle files. "
            "If omitted, all six validated puzzles are used."
        ),
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Relative paths are resolved "
            "from Source/. If omitted, a timestamped directory is "
            "created under experiments/results/."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = (
        build_argument_parser()
    )

    args = parser.parse_args(
        argv
    )

    puzzle_files = tuple(
        args.puzzles
        if args.puzzles is not None
        else BENCHMARK_PUZZLE_FILES
    )

    try:
        run_benchmark(
            puzzle_files=puzzle_files,
            repetitions=args.repetitions,
            warmups=args.warmups,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        print(
            "\nBenchmark interrupted by user.",
            file=sys.stderr,
        )
        return 130
    except Exception as error:
        print(
            f"\nBenchmark failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )