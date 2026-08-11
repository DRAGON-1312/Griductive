from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.game_engine import GameEngine
from core.models import (
    Classification,
    ClueType,
    Region,
    RegionType,
    VerdictCode,
)
from core.puzzle_loader import load_puzzle
from logic.agent import AgentStopReason, LogicAgent
from logic.cnf_encoder import CNFEncoder
from logic.dpll import DPLLSolver
from logic.semantic_evaluator import evaluate_clue
from logic.uniqueness import UniquenessChecker


# ============================================================
# Benchmark dataset definition
# ============================================================

PUZZLES_DIR = (
    Path(__file__).resolve().parents[1]
    / "puzzles"
)


@dataclass(frozen=True)
class PuzzleCase:
    """
    Expected deterministic behavior of one benchmark puzzle.

    The deduction sequence is intentionally part of the benchmark
    contract. LogicAgent scans unresolved characters in deterministic
    character order and always chooses the first logically forced
    verdict.
    """

    filename: str
    size: int
    expected_sequence: tuple[str, ...]

    @property
    def path(self) -> Path:
        return PUZZLES_DIR / self.filename


PUZZLE_CASES = (
    PuzzleCase(
        filename="puzzle_3x3_01.json",
        size=3,
        expected_sequence=(
            "A1",
            "B1",
            "C1",
            "A2",
            "B3",
            "C3",
            "C2",
            "A3",
        ),
    ),
    PuzzleCase(
        filename="puzzle_3x3_02.json",
        size=3,
        expected_sequence=(
            "A2",
            "A1",
            "B1",
            "C1",
            "C2",
            "A3",
            "B3",
            "C3",
        ),
    ),
    PuzzleCase(
        filename="puzzle_4x4_01.json",
        size=4,
        expected_sequence=(
            "A2",
            "A1",
            "C1",
            "B1",
            "D1",
            "C2",
            "D2",
            "A3",
            "B3",
            "C3",
            "D3",
            "A4",
            "B4",
            "C4",
            "D4",
        ),
    ),
    PuzzleCase(
        filename="puzzle_4x4_02.json",
        size=4,
        expected_sequence=(
            "D1",
            "A1",
            "C1",
            "D2",
            "A2",
            "B1",
            "C2",
            "A3",
            "D3",
            "B4",
            "C4",
            "A4",
            "B3",
            "C3",
            "D4",
        ),
    ),
    PuzzleCase(
        filename="puzzle_5x5_01.json",
        size=5,
        expected_sequence=(
            "A3",
            "A1",
            "B1",
            "C1",
            "D1",
            "E1",
            "A2",
            "B2",
            "C2",
            "D2",
            "E2",
            "B3",
            "D3",
            "E3",
            "A4",
            "B4",
            "C4",
            "D4",
            "E4",
            "A5",
            "B5",
            "C5",
            "D5",
            "E5",
        ),
    ),
    PuzzleCase(
        filename="puzzle_5x5_02.json",
        size=5,
        expected_sequence=(
            "D5",
            "C5",
            "B5",
            "A5",
            "E4",
            "C4",
            "D4",
            "B4",
            "E3",
            "D3",
            "C3",
            "B3",
            "A3",
            "E2",
            "D2",
            "C2",
            "B2",
            "A2",
            "E1",
            "D1",
            "B1",
            "A1",
            "C1",
        ),
    ),
)


def puzzle_case_id(case: PuzzleCase) -> str:
    return case.filename.removesuffix(".json")


# ============================================================
# Helpers
# ============================================================

def hidden_assignment(puzzle):
    """
    Return the intended complete character-status assignment.

    Hidden data is used only by the validation tests as an oracle.
    It is never passed to LogicAgent.
    """
    return {
        character_id: secret.status
        for character_id, secret
        in puzzle.secrets.items()
    }


def all_clues(puzzle):
    """
    Return every clue in deterministic character order.
    """
    return tuple(
        secret.clue
        for secret
        in puzzle.secrets.values()
    )


# ============================================================
# Gate 1: loader / dataset structure
# ============================================================

@pytest.mark.parametrize(
    "case",
    PUZZLE_CASES,
    ids=puzzle_case_id,
)
def test_benchmark_puzzle_loads_with_expected_shape(case):
    """
    Every benchmark JSON must be accepted by PuzzleLoader and
    represent the intended N x N board.
    """
    assert case.path.is_file(), (
        f"Missing benchmark puzzle: {case.path}"
    )

    puzzle = load_puzzle(case.path)

    assert puzzle.size == case.size
    assert len(puzzle.characters) == case.size * case.size
    assert len(puzzle.secrets) == case.size * case.size

    assert tuple(puzzle.characters) == tuple(puzzle.secrets)

    assert puzzle.initial_revealed
    assert set(puzzle.initial_revealed).issubset(
        puzzle.characters
    )

    clue_ids = [
        secret.clue.id
        for secret in puzzle.secrets.values()
    ]
    assert len(clue_ids) == len(set(clue_ids))


# ============================================================
# Gate 2: direct semantic validation
# ============================================================

@pytest.mark.parametrize(
    "case",
    PUZZLE_CASES,
    ids=puzzle_case_id,
)
def test_every_clue_is_true_under_intended_solution(case):
    """
    Every visible statement in Griductive must be true.

    This check deliberately uses the direct semantic evaluator,
    not CNF encoding or DPLL, so it independently validates the
    intended hidden assignment against all clue semantics.
    """
    puzzle = load_puzzle(case.path)
    assignment = hidden_assignment(puzzle)

    for character_id, secret in puzzle.secrets.items():
        assert evaluate_clue(
            secret.clue,
            assignment,
            puzzle.size,
        ), (
            f"{case.filename}: clue {secret.clue.id} "
            f"owned by {character_id} is false under the "
            f"intended hidden solution."
        )


# ============================================================
# Gate 3: complete CNF is SAT and compatible with hidden model
# ============================================================

@pytest.mark.parametrize(
    "case",
    PUZZLE_CASES,
    ids=puzzle_case_id,
)
def test_complete_clue_cnf_is_sat_and_accepts_hidden_solution(case):
    """
    The complete clue set must encode to a satisfiable CNF.

    A second solve fixes every primary variable to the intended
    hidden status using assumptions. If that solve is SAT, the
    intended solution is also a model of the actual CNF encoding.
    """
    puzzle = load_puzzle(case.path)

    encoder = CNFEncoder(
        characters=puzzle.characters,
        size=puzzle.size,
    )

    cnf = encoder.encode_clues(
        all_clues(puzzle)
    )

    assert encoder.primary_variable_count == (
        puzzle.size * puzzle.size
    )
    assert encoder.auxiliary_variable_count >= 0
    assert encoder.total_variable_count >= (
        encoder.primary_variable_count
    )
    assert len(cnf) > 0

    solver = DPLLSolver()

    result = solver.solve(
        clauses=cnf,
        num_variables=encoder.total_variable_count,
    )

    assert result.satisfiable is True
    assert result.assignment is not None

    intended_assumptions = [
        encoder.literal_for_status(
            character_id,
            secret.status,
        )
        for character_id, secret
        in puzzle.secrets.items()
    ]

    intended_result = solver.solve(
        clauses=cnf,
        num_variables=encoder.total_variable_count,
        assumptions=intended_assumptions,
    )

    assert intended_result.satisfiable is True
    assert intended_result.assignment is not None


# ============================================================
# Gate 4: complete clue set has exactly one primary solution
# ============================================================

@pytest.mark.parametrize(
    "case",
    PUZZLE_CASES,
    ids=puzzle_case_id,
)
def test_complete_clue_set_has_unique_intended_solution(case):
    """
    Validate uniqueness from the complete clue set itself.

    Initial face-up statuses are deliberately excluded here.
    This is stronger than checking uniqueness only after adding
    the initial known verdicts.
    """
    puzzle = load_puzzle(case.path)

    checker = UniquenessChecker(
        size=puzzle.size
    )

    result = checker.check_puzzle(
        puzzle,
        include_initial_statuses=False,
    )

    expected = hidden_assignment(puzzle)

    assert result.satisfiable is True
    assert result.unique is True

    assert result.first_character_assignment == expected
    assert result.second_character_assignment is None

    assert result.second_solve_result is not None
    assert (
        result.second_solve_result.satisfiable
        is False
    )


# ============================================================
# Gate 5: progressive no-guess solve
# ============================================================

@pytest.mark.parametrize(
    "case",
    PUZZLE_CASES,
    ids=puzzle_case_id,
)
def test_logic_agent_solves_benchmark_without_guessing(case):
    """
    Validate the full progressive gameplay pipeline:

        Puzzle
          -> GameEngine
          -> public KB
          -> SAT entailment
          -> forced verdict
          -> clue reveal
          -> repeat

    The test also freezes the deterministic deduction order of
    the benchmark suite so experiments remain reproducible.
    """
    puzzle = load_puzzle(case.path)
    expected_hidden = hidden_assignment(puzzle)

    engine = GameEngine(puzzle)

    initial_state = engine.get_public_state()
    initial_active_clue_ids = tuple(
        clue.id
        for clue in initial_state.revealed_clues
    )

    agent = LogicAgent(engine)
    result = agent.auto_solve()

    # --------------------------------------------------------
    # Terminal behavior
    # --------------------------------------------------------

    assert result.solved is True
    assert result.stop_reason == AgentStopReason.SOLVED
    assert result.unresolved_character_ids == ()

    expected_deduction_count = (
        puzzle.size * puzzle.size
        - len(puzzle.initial_revealed)
    )

    assert len(case.expected_sequence) == (
        expected_deduction_count
    )
    assert result.deduction_count == (
        expected_deduction_count
    )

    # --------------------------------------------------------
    # Deterministic no-guess sequence
    # --------------------------------------------------------

    actual_sequence = tuple(
        step.character_id
        for step in result.steps
    )

    assert actual_sequence == case.expected_sequence

    assert tuple(
        step.step_number
        for step in result.steps
    ) == tuple(
        range(
            1,
            expected_deduction_count + 1,
        )
    )

    for step in result.steps:
        assert step.verdict_code == VerdictCode.ACCEPTED

        assert step.classification in {
            Classification.CRIMINAL,
            Classification.INNOCENT,
        }

        assert (
            step.status
            == expected_hidden[step.character_id]
        )

        assert step.revealed_clue is not None

        # The newly revealed clue is not allowed to justify
        # the same deduction that revealed it.
        assert (
            step.revealed_clue.id
            not in step.active_clue_ids
        )

    # --------------------------------------------------------
    # Deduction-trace reveal protocol
    # --------------------------------------------------------

    assert (
        result.steps[0].active_clue_ids
        == initial_active_clue_ids
    )

    for previous_step, current_step in zip(
        result.steps,
        result.steps[1:],
    ):
        assert previous_step.revealed_clue is not None

        assert (
            previous_step.revealed_clue.id
            in current_step.active_clue_ids
        )

    for step in result.steps:
        expected_active_count = (
            len(puzzle.initial_revealed)
            + step.step_number
            - 1
        )

        assert len(step.active_clue_ids) == (
            expected_active_count
        )

    # --------------------------------------------------------
    # Final state matches the intended hidden solution
    # --------------------------------------------------------

    final_state = engine.get_public_state()

    assert final_state.proved_statuses == expected_hidden
    assert len(final_state.revealed_clues) == (
        puzzle.size * puzzle.size
    )

    # --------------------------------------------------------
    # Workload metrics are populated and valid
    # --------------------------------------------------------

    assert result.total_sat_calls > 0
    assert result.total_decisions >= 0
    assert result.total_propagations >= 0
    assert result.total_backtracks >= 0
    assert result.total_runtime >= 0.0


# ============================================================
# Dataset-level requirement coverage
# ============================================================

def test_benchmark_dataset_covers_required_clues_regions_and_sizes():
    """
    Guard the benchmark suite as a whole.

    The six cases should collectively cover:
        - every core clue type,
        - both implemented extensions,
        - every required region type,
        - two puzzles each at 3x3, 4x4, and 5x5.
    """
    expected_files = {
        case.filename
        for case in PUZZLE_CASES
    }

    actual_files = {
        path.name
        for path in PUZZLES_DIR.glob(
            "puzzle_*.json"
        )
    }

    assert expected_files.issubset(actual_files)

    clue_types: set[ClueType] = set()
    region_types: set[RegionType] = set()
    size_counts: Counter[int] = Counter()

    for case in PUZZLE_CASES:
        puzzle = load_puzzle(case.path)

        size_counts[puzzle.size] += 1

        for secret in puzzle.secrets.values():
            clue = secret.clue

            assert isinstance(
                clue.type,
                ClueType,
            )

            clue_types.add(
                clue.type
            )

            region = clue.params.get(
                "region"
            )

            if region is not None:
                assert isinstance(
                    region,
                    Region,
                )

                region_types.add(
                    region.type
                )

    required_core_clues = {
        ClueType.FACT,
        ClueType.SAME,
        ClueType.DIFFERENT,
        ClueType.EXACTLY,
        ClueType.AT_LEAST,
        ClueType.AT_MOST,
    }

    required_extensions = {
        ClueType.PARITY,
        ClueType.IMPLIES,
    }

    required_regions = {
        RegionType.ROW,
        RegionType.COLUMN,
        RegionType.NEIGHBORS,
        RegionType.EXPLICIT,
    }

    assert required_core_clues.issubset(
        clue_types
    )
    assert required_extensions.issubset(
        clue_types
    )
    assert required_regions.issubset(
        region_types
    )

    assert size_counts == Counter(
        {
            3: 2,
            4: 2,
            5: 2,
        }
    )