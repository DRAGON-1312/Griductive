from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt


# ============================================================
# Paths / benchmark order
# ============================================================

SOURCE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = SOURCE_DIR / "experiments"
DEFAULT_RESULTS_DIR = EXPERIMENTS_DIR / "results"

BENCHMARK_ORDER: tuple[str, ...] = (
    "puzzle_3x3_01.json",
    "puzzle_3x3_02.json",
    "puzzle_4x4_01.json",
    "puzzle_4x4_02.json",
    "puzzle_5x5_01.json",
    "puzzle_5x5_02.json",
)


# ============================================================
# Required CSV columns
# ============================================================

SUMMARY_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "puzzle_file",
        "puzzle_name",
        "size",
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
        "solver_runtime_median",
        "wall_runtime_median",
    }
)

RAW_RUN_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "puzzle_file",
        "run_index",
        "successful",
        "solver_runtime",
        "wall_runtime",
    }
)


# ============================================================
# Exceptions
# ============================================================

class PlotResultsError(ValueError):
    """
    Raised when experiment output is missing, malformed, or unsuitable
    for report visualization.
    """


# ============================================================
# CSV / value helpers
# ============================================================

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required experiment file does not exist: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise PlotResultsError(
                f"CSV has no header: {path}"
            )

        return list(reader)


def _require_columns(
    rows: Sequence[dict[str, str]],
    required: frozenset[str],
    *,
    csv_name: str,
) -> None:
    if not rows:
        raise PlotResultsError(
            f"{csv_name} contains no data rows."
        )

    available = set(rows[0])

    missing = sorted(
        required - available
    )

    if missing:
        raise PlotResultsError(
            f"{csv_name} is missing required columns: "
            + ", ".join(missing)
        )


def _to_int(
    row: dict[str, str],
    key: str,
) -> int:
    value = row.get(key, "")

    if value is None or value == "":
        raise PlotResultsError(
            f"Missing integer value for column {key!r}."
        )

    try:
        return int(value)
    except ValueError as error:
        raise PlotResultsError(
            f"Invalid integer for {key!r}: {value!r}"
        ) from error


def _to_optional_int(
    row: dict[str, str],
    key: str,
) -> int | None:
    value = row.get(key, "")

    if value is None or value == "":
        return None

    try:
        return int(value)
    except ValueError as error:
        raise PlotResultsError(
            f"Invalid integer for {key!r}: {value!r}"
        ) from error


def _to_float(
    row: dict[str, str],
    key: str,
) -> float:
    value = row.get(key, "")

    if value is None or value == "":
        raise PlotResultsError(
            f"Missing numeric value for column {key!r}."
        )

    try:
        result = float(value)
    except ValueError as error:
        raise PlotResultsError(
            f"Invalid numeric value for {key!r}: {value!r}"
        ) from error

    if not math.isfinite(result):
        raise PlotResultsError(
            f"Non-finite numeric value for {key!r}: {value!r}"
        )

    return result


def _to_bool(
    row: dict[str, str],
    key: str,
) -> bool:
    value = str(
        row.get(key, "")
    ).strip().lower()

    if value in {
        "true",
        "1",
        "yes",
    }:
        return True

    if value in {
        "false",
        "0",
        "no",
    }:
        return False

    raise PlotResultsError(
        f"Invalid boolean for {key!r}: {row.get(key)!r}"
    )


# ============================================================
# Result-directory resolution
# ============================================================

def _latest_result_directory() -> Path:
    if not DEFAULT_RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            "No experiments/results directory exists yet."
        )

    candidates = tuple(
        path
        for path in DEFAULT_RESULTS_DIR.iterdir()
        if path.is_dir()
        and (path / "summary.csv").is_file()
    )

    if not candidates:
        raise FileNotFoundError(
            "No completed experiment result directory was found."
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime_ns,
    )


def _resolve_results_directory(
    requested: Path | None,
) -> Path:
    if requested is None:
        return _latest_result_directory()

    path = requested

    if not path.is_absolute():
        path = SOURCE_DIR / path

    path = path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"Results directory does not exist: {path}"
        )

    return path


# ============================================================
# Ordering / labels
# ============================================================

def _benchmark_rank(
    puzzle_file: str,
) -> tuple[int, str]:
    try:
        index = BENCHMARK_ORDER.index(
            puzzle_file
        )
    except ValueError:
        return (
            len(BENCHMARK_ORDER),
            puzzle_file,
        )

    return (
        index,
        puzzle_file,
    )


def _sort_summary_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: _benchmark_rank(
            row["puzzle_file"]
        ),
    )


def _puzzle_label(
    puzzle_file: str,
) -> str:
    label = puzzle_file

    if label.startswith(
        "puzzle_"
    ):
        label = label[len("puzzle_"):]

    if label.endswith(
        ".json"
    ):
        label = label[:-len(".json")]

    return label


# ============================================================
# Figure output
# ============================================================

def _save_figure(
    figure,
    *,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    generated: list[Path] = []

    for file_format in formats:
        path = output_stem.with_suffix(
            f".{file_format}"
        )

        save_kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
        }

        if file_format == "png":
            save_kwargs["dpi"] = dpi

        figure.savefig(
            path,
            **save_kwargs,
        )

        generated.append(
            path
        )

    plt.close(
        figure
    )

    return tuple(
        generated
    )


def _annotate_bars(
    axis,
    bars,
    *,
    integer: bool = True,
) -> None:
    for bar in bars:
        height = bar.get_height()

        if integer:
            label = f"{int(round(height))}"
        else:
            label = f"{height:.2f}"

        axis.annotate(
            label,
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


# ============================================================
# Main report charts
# ============================================================

def _plot_single_metric(
    rows: Sequence[dict[str, str]],
    *,
    key: str,
    title: str,
    ylabel: str,
    description: str,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    labels = [
        _puzzle_label(
            row["puzzle_file"]
        )
        for row in rows
    ]

    values = [
        _to_int(
            row,
            key,
        )
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(9.5, 5.5)
    )

    bars = axis.bar(
        labels,
        values,
    )

    axis.set_title(
        title
    )
    axis.set_xlabel(
        "Benchmark puzzle"
    )
    axis.set_ylabel(
        ylabel
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    _annotate_bars(
        axis,
        bars,
    )

    figure.text(
        0.5,
        0.01,
        description,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    return _save_figure(
        figure,
        output_stem=output_stem,
        formats=formats,
        dpi=dpi,
    )


def _plot_runtime(
    rows: Sequence[dict[str, str]],
    *,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    labels = [
        _puzzle_label(
            row["puzzle_file"]
        )
        for row in rows
    ]

    solver_ms = [
        _to_float(
            row,
            "solver_runtime_median",
        ) * 1000.0
        for row in rows
    ]

    wall_ms = [
        _to_float(
            row,
            "wall_runtime_median",
        ) * 1000.0
        for row in rows
    ]

    x = list(
        range(len(labels))
    )

    width = 0.38

    figure, axis = plt.subplots(
        figsize=(10, 5.8)
    )

    solver_bars = axis.bar(
        [
            value - width / 2
            for value in x
        ],
        solver_ms,
        width=width,
        label="DPLL solver runtime",
    )

    wall_bars = axis.bar(
        [
            value + width / 2
            for value in x
        ],
        wall_ms,
        width=width,
        label="End-to-end wall runtime",
    )

    axis.set_title(
        "Median runtime by benchmark puzzle"
    )
    axis.set_xlabel(
        "Benchmark puzzle"
    )
    axis.set_ylabel(
        "Median runtime (ms)"
    )
    axis.set_xticks(
        x,
        labels,
        rotation=25,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.legend()

    _annotate_bars(
        axis,
        solver_bars,
        integer=False,
    )
    _annotate_bars(
        axis,
        wall_bars,
        integer=False,
    )

    figure.text(
        0.5,
        0.01,
        (
            "Median over successful measured repetitions; "
            "warm-up runs are excluded."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    return _save_figure(
        figure,
        output_stem=output_stem,
        formats=formats,
        dpi=dpi,
    )


def _plot_runtime_distribution(
    raw_rows: Sequence[dict[str, str]],
    *,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    successful_rows = [
        row
        for row in raw_rows
        if _to_bool(
            row,
            "successful",
        )
    ]

    if not successful_rows:
        raise PlotResultsError(
            "raw_runs.csv contains no successful runs."
        )

    grouped: dict[
        str,
        list[float],
    ] = {}

    for row in successful_rows:
        puzzle_file = row[
            "puzzle_file"
        ]

        grouped.setdefault(
            puzzle_file,
            [],
        ).append(
            _to_float(
                row,
                "wall_runtime",
            ) * 1000.0
        )

    puzzle_files = sorted(
        grouped,
        key=_benchmark_rank,
    )

    labels = [
        _puzzle_label(
            puzzle_file
        )
        for puzzle_file in puzzle_files
    ]

    values = [
        grouped[
            puzzle_file
        ]
        for puzzle_file in puzzle_files
    ]

    figure, axis = plt.subplots(
        figsize=(10, 5.8)
    )

    axis.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
    )

    axis.set_title(
        "End-to-end runtime distribution"
    )
    axis.set_xlabel(
        "Benchmark puzzle"
    )
    axis.set_ylabel(
        "Wall runtime (ms)"
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.tick_params(
        axis="x",
        rotation=25,
    )

    figure.text(
        0.5,
        0.01,
        (
            "Successful measured repetitions only; "
            "the marker inside each box indicates the mean."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    return _save_figure(
        figure,
        output_stem=output_stem,
        formats=formats,
        dpi=dpi,
    )


# ============================================================
# Report-ready table
# ============================================================

REPORT_TABLE_FIELDS: tuple[str, ...] = (
    "puzzle",
    "size",
    "primary_variables",
    "auxiliary_variables",
    "cnf_clauses",
    "deduction_steps",
    "sat_calls",
    "decisions",
    "propagations",
    "backtracks",
    "candidate_analyses",
    "unknown_analyses",
    "solver_runtime_median_ms",
    "wall_runtime_median_ms",
)


def _build_report_table_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[
        dict[str, Any]
    ] = []

    for row in rows:
        candidate_analyses = (
            _to_optional_int(
                row,
                "candidate_analyses",
            )
        )

        unknown_analyses = (
            _to_optional_int(
                row,
                "unknown_analyses",
            )
        )

        output.append(
            {
                "puzzle": _puzzle_label(
                    row["puzzle_file"]
                ),
                "size": _to_int(
                    row,
                    "size",
                ),
                "primary_variables": _to_int(
                    row,
                    "primary_variables",
                ),
                "auxiliary_variables": _to_int(
                    row,
                    "auxiliary_variables",
                ),
                "cnf_clauses": _to_int(
                    row,
                    "cnf_clauses",
                ),
                "deduction_steps": _to_int(
                    row,
                    "deduction_steps",
                ),
                "sat_calls": _to_int(
                    row,
                    "sat_calls",
                ),
                "decisions": _to_int(
                    row,
                    "decisions",
                ),
                "propagations": _to_int(
                    row,
                    "propagations",
                ),
                "backtracks": _to_int(
                    row,
                    "backtracks",
                ),
                "candidate_analyses": (
                    ""
                    if candidate_analyses is None
                    else candidate_analyses
                ),
                "unknown_analyses": (
                    ""
                    if unknown_analyses is None
                    else unknown_analyses
                ),
                "solver_runtime_median_ms": round(
                    _to_float(
                        row,
                        "solver_runtime_median",
                    ) * 1000.0,
                    6,
                ),
                "wall_runtime_median_ms": round(
                    _to_float(
                        row,
                        "wall_runtime_median",
                    ) * 1000.0,
                    6,
                ),
            }
        )

    return output


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
            writer.writerow(
                row
            )


# ============================================================
# Figure manifest
# ============================================================

def _write_manifest(
    path: Path,
    *,
    results_dir: Path,
    generated_files: Sequence[Path],
) -> None:
    data = {
        "source_results_directory": str(
            results_dir
        ),
        "generated_files": [
            file.name
            for file in generated_files
        ],
        "notes": {
            "runtime_primary_statistic": (
                "median of successful measured repetitions"
            ),
            "runtime_units_in_figures": "milliseconds",
            "backtracks_chart": (
                "generated only when at least one puzzle "
                "has a non-zero backtrack count"
            ),
        },
    }

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

        file.write(
            "\n"
        )


# ============================================================
# Full plotting pipeline
# ============================================================

def generate_report_outputs(
    results_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    summary_path = (
        results_dir
        / "summary.csv"
    )

    raw_runs_path = (
        results_dir
        / "raw_runs.csv"
    )

    summary_rows = _read_csv(
        summary_path
    )

    raw_rows = _read_csv(
        raw_runs_path
    )

    _require_columns(
        summary_rows,
        SUMMARY_REQUIRED_COLUMNS,
        csv_name="summary.csv",
    )

    _require_columns(
        raw_rows,
        RAW_RUN_REQUIRED_COLUMNS,
        csv_name="raw_runs.csv",
    )

    summary_rows = (
        _sort_summary_rows(
            summary_rows
        )
    )

    figures_dir = (
        results_dir
        / "figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated: list[
        Path
    ] = []

    generated.extend(
        _plot_single_metric(
            summary_rows,
            key="cnf_clauses",
            title="CNF clause count by benchmark puzzle",
            ylabel="CNF clauses",
            description=(
                "Complete clue-set CNF; hidden statuses are not encoded."
            ),
            output_stem=figures_dir / "cnf_clauses",
            formats=formats,
            dpi=dpi,
        )
    )

    generated.extend(
        _plot_single_metric(
            summary_rows,
            key="sat_calls",
            title="SAT calls by benchmark puzzle",
            ylabel="SAT calls",
            description=(
                "Complete auto-solve workload, including UNKNOWN scans "
                "and GameEngine verdict verification."
            ),
            output_stem=figures_dir / "sat_calls",
            formats=formats,
            dpi=dpi,
        )
    )

    generated.extend(
        _plot_single_metric(
            summary_rows,
            key="unknown_analyses",
            title="UNKNOWN candidate analyses by benchmark puzzle",
            ylabel="UNKNOWN analyses",
            description=(
                "Candidates checked by the no-guess agent but not yet "
                "logically forced."
            ),
            output_stem=figures_dir / "unknown_analyses",
            formats=formats,
            dpi=dpi,
        )
    )

    generated.extend(
        _plot_single_metric(
            summary_rows,
            key="decisions",
            title="DPLL branching decisions by benchmark puzzle",
            ylabel="Decisions",
            description=(
                "Deterministic DPLL branch decisions accumulated across "
                "the complete auto-solve run."
            ),
            output_stem=figures_dir / "decisions",
            formats=formats,
            dpi=dpi,
        )
    )

    generated.extend(
        _plot_single_metric(
            summary_rows,
            key="propagations",
            title="DPLL propagations by benchmark puzzle",
            ylabel="Propagations",
            description=(
                "Unit-propagation workload accumulated across all SAT "
                "calls in the complete auto-solve run."
            ),
            output_stem=figures_dir / "propagations",
            formats=formats,
            dpi=dpi,
        )
    )

    generated.extend(
        _plot_runtime(
            summary_rows,
            output_stem=figures_dir / "runtime",
            formats=formats,
            dpi=dpi,
        )
    )

    generated.extend(
        _plot_runtime_distribution(
            raw_rows,
            output_stem=figures_dir / "wall_runtime_distribution",
            formats=formats,
            dpi=dpi,
        )
    )

    backtrack_values = [
        _to_int(
            row,
            "backtracks",
        )
        for row in summary_rows
    ]

    if any(
        value != 0
        for value in backtrack_values
    ):
        generated.extend(
            _plot_single_metric(
                summary_rows,
                key="backtracks",
                title="DPLL backtracks by benchmark puzzle",
                ylabel="Backtracks",
                description=(
                    "Backtracking workload accumulated across the "
                    "complete auto-solve run."
                ),
                output_stem=figures_dir / "backtracks",
                formats=formats,
                dpi=dpi,
            )
        )
    else:
        print(
            "Skipping backtracks chart: all benchmark values are zero."
        )

    report_table_path = (
        results_dir
        / "report_table.csv"
    )

    _write_csv(
        report_table_path,
        fieldnames=REPORT_TABLE_FIELDS,
        rows=_build_report_table_rows(
            summary_rows
        ),
    )

    generated.append(
        report_table_path
    )

    manifest_path = (
        figures_dir
        / "manifest.json"
    )

    _write_manifest(
        manifest_path,
        results_dir=results_dir,
        generated_files=generated,
    )

    generated.append(
        manifest_path
    )

    return tuple(
        generated
    )


# ============================================================
# CLI
# ============================================================

def _positive_int(
    value: str,
) -> int:
    parsed = int(
        value
    )

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )

    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate report-ready charts and a compact result table "
            "from one Griductive experiment result directory."
        )
    )

    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Experiment result directory. If omitted, the most recent "
            "completed directory under experiments/results/ is used."
        ),
    )

    parser.add_argument(
        "--format",
        choices=(
            "png",
            "svg",
            "both",
        ),
        default="both",
        help=(
            "Figure output format (default: both PNG and SVG)."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=_positive_int,
        default=300,
        help=(
            "PNG resolution in dots per inch (default: 300)."
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

    formats = (
        ("png", "svg")
        if args.format == "both"
        else (args.format,)
    )

    try:
        results_dir = (
            _resolve_results_directory(
                args.results_dir
            )
        )

        generated = (
            generate_report_outputs(
                results_dir,
                formats=formats,
                dpi=args.dpi,
            )
        )
    except KeyboardInterrupt:
        print(
            "\nPlot generation interrupted by user."
        )
        return 130
    except Exception as error:
        print(
            "\nPlot generation failed: "
            f"{type(error).__name__}: {error}"
        )
        return 1

    print()
    print(
        "Report outputs generated successfully."
    )
    print(
        f"Source results: {results_dir}"
    )

    for path in generated:
        print(
            f"  - {path.relative_to(results_dir)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )