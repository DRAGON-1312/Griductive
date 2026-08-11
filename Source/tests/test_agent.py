from dataclasses import replace
from pathlib import Path

import pytest

from core.game_engine import GameEngine
from core.models import (
    Classification,
    Status,
    VerdictCode,
)
from core.puzzle_loader import load_puzzle
from logic.agent import (
    AgentKnowledgeBaseError,
    AgentStopReason,
    InvalidAgentConfigurationError,
    LogicAgent,
)
from logic.entailment import EntailmentChecker


# ============================================================
# Helpers
# ============================================================

def puzzle_3x3_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "puzzles"
        / "puzzle_3x3_01.json"
    )


def create_basic_game():
    """
    Load the basic 3x3 sanity puzzle and construct a fresh
    GameEngine + LogicAgent.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    agent = LogicAgent(
        engine
    )

    return puzzle, engine, agent


def expected_deduction_sequence():
    """
    Expected no-guessing deduction chain for puzzle_3x3_01.

    B2 is already revealed at the beginning and therefore does
    not appear as an AgentStep.
    """
    return [
        ("A1", Status.CRIMINAL),
        ("B1", Status.CRIMINAL),
        ("C1", Status.INNOCENT),
        ("A2", Status.INNOCENT),
        ("B3", Status.INNOCENT),
        ("C3", Status.CRIMINAL),
        ("C2", Status.CRIMINAL),
        ("A3", Status.INNOCENT),
    ]


# ============================================================
# Construction
# ============================================================

def test_agent_infers_correct_puzzle_size():
    _, _, agent = create_basic_game()

    assert agent.size == 3


def test_agent_uses_default_entailment_checker():
    _, _, agent = create_basic_game()

    assert isinstance(
        agent.checker,
        EntailmentChecker,
    )

    assert agent.checker.size == 3


def test_agent_accepts_matching_custom_checker():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    checker = EntailmentChecker(
        size=3
    )

    agent = LogicAgent(
        engine=engine,
        checker=checker,
    )

    assert agent.checker is checker


def test_agent_rejects_checker_with_wrong_size():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    checker = EntailmentChecker(
        size=4
    )

    with pytest.raises(
        InvalidAgentConfigurationError
    ):
        LogicAgent(
            engine=engine,
            checker=checker,
        )


def test_agent_rejects_invalid_engine():
    with pytest.raises(
        TypeError
    ):
        LogicAgent(
            engine="not-an-engine"
        )


def test_agent_rejects_invalid_checker_type():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    engine = GameEngine(
        puzzle
    )

    with pytest.raises(
        TypeError
    ):
        LogicAgent(
            engine=engine,
            checker="not-a-checker",
        )


# ============================================================
# Initial state
# ============================================================

def test_initial_state_has_only_b2_proved():
    _, engine, _ = create_basic_game()

    state = engine.get_public_state()

    assert state.proved_statuses == {
        "B2": Status.CRIMINAL,
    }


def test_initial_agent_trace_is_empty():
    _, _, agent = create_basic_game()

    assert agent.deduction_trace == ()


# ============================================================
# Hint behavior
# ============================================================

def test_first_hint_is_a1_criminal():
    """
    Initial public knowledge contains:

        B2 = CRIMINAL

    and B2's revealed clue proves:

        A1 = CRIMINAL

    Therefore A1 must be the first deterministic hint.
    """
    _, _, agent = create_basic_game()

    hint = agent.find_hint()

    assert hint is not None

    assert hint.character_id == "A1"
    assert hint.status == Status.CRIMINAL

    assert (
        hint.classification
        == Classification.CRIMINAL
    )

    assert (
        hint.analysis.classification
        == Classification.CRIMINAL
    )


def test_find_hint_does_not_mutate_game_state():
    """
    Hint must be read-only.

    It may perform SAT reasoning, but it must not submit a verdict
    or reveal a clue.
    """
    _, engine, agent = create_basic_game()

    before = engine.get_public_state()

    hint = agent.find_hint()

    after = engine.get_public_state()

    assert hint is not None

    assert (
        after.proved_statuses
        == before.proved_statuses
    )

    assert (
        after.revealed_clues
        == before.revealed_clues
    )

    assert agent.deduction_trace == ()


def test_hint_contains_unsat_proof_for_a1():
    _, _, agent = create_basic_game()

    hint = agent.find_hint()

    assert hint is not None

    # KB AND NOT A1 must be UNSAT.
    assert (
        hint.analysis
        .assume_innocent_result
        .satisfiable
        is False
    )

    # KB AND A1 must remain SAT.
    assert (
        hint.analysis
        .assume_criminal_result
        .satisfiable
        is True
    )


# ============================================================
# Single deduction step
# ============================================================

def test_first_step_accepts_a1_criminal():
    _, engine, agent = create_basic_game()

    step = agent.step()

    assert step is not None

    assert step.step_number == 1

    assert step.character_id == "A1"
    assert step.status == Status.CRIMINAL

    assert (
        step.classification
        == Classification.CRIMINAL
    )

    assert (
        step.verdict_code
        == VerdictCode.ACCEPTED
    )

    state = engine.get_public_state()

    assert (
        state.proved_statuses["A1"]
        == Status.CRIMINAL
    )


def test_successful_step_reveals_new_clue():
    _, engine, agent = create_basic_game()

    before = engine.get_public_state()

    step = agent.step()

    after = engine.get_public_state()

    assert step is not None

    assert step.revealed_clue is not None

    assert (
        len(after.proved_statuses)
        == len(before.proved_statuses) + 1
    )

    assert (
        len(after.revealed_clues)
        == len(before.revealed_clues) + 1
    )


def test_step_is_recorded_in_deduction_trace():
    _, _, agent = create_basic_game()

    step = agent.step()

    assert step is not None

    assert agent.deduction_trace == (
        step,
    )


def test_second_step_is_b1_criminal():
    """
    After A1 is accepted, A1's clue becomes public:

        SAME(A1, B1)

    Since A1 is Criminal, B1 becomes provably Criminal.
    """
    _, _, agent = create_basic_game()

    first = agent.step()
    second = agent.step()

    assert first is not None
    assert second is not None

    assert first.character_id == "A1"

    assert second.character_id == "B1"
    assert second.status == Status.CRIMINAL

    assert (
        second.verdict_code
        == VerdictCode.ACCEPTED
    )


# ============================================================
# No guessing
# ============================================================

def test_agent_stops_when_nothing_is_publicly_provable():
    """
    Remove the initial reveal from the same puzzle.

    The hidden puzzle still contains all statuses and clues, but
    none of them are public.

    Therefore every character is UNKNOWN and the agent must stop
    instead of guessing.
    """
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    no_initial_reveal_puzzle = replace(
        puzzle,
        initial_revealed=(),
    )

    engine = GameEngine(
        no_initial_reveal_puzzle
    )

    agent = LogicAgent(
        engine
    )

    before = engine.get_public_state()

    result = agent.auto_solve()

    after = engine.get_public_state()

    assert result.solved is False

    assert (
        result.stop_reason
        == AgentStopReason.NO_PROVABLE_MOVE
    )

    assert result.steps == ()

    assert len(
        result.unresolved_character_ids
    ) == 9

    # No guessed verdict.
    assert not after.proved_statuses

    # No hidden clue was leaked.
    assert not after.revealed_clues

    # Game state remains unchanged.
    assert (
        after.proved_statuses
        == before.proved_statuses
    )

    assert (
        after.revealed_clues
        == before.revealed_clues
    )


def test_find_hint_returns_none_when_everything_is_unknown():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    puzzle = replace(
        puzzle,
        initial_revealed=(),
    )

    engine = GameEngine(
        puzzle
    )

    agent = LogicAgent(
        engine
    )

    assert agent.find_hint() is None


def test_step_returns_none_when_no_move_is_provable():
    puzzle = load_puzzle(
        puzzle_3x3_path()
    )

    puzzle = replace(
        puzzle,
        initial_revealed=(),
    )

    engine = GameEngine(
        puzzle
    )

    agent = LogicAgent(
        engine
    )

    assert agent.step() is None

    assert agent.deduction_trace == ()


# ============================================================
# End-to-end auto solve
# ============================================================

def test_auto_solve_completes_basic_3x3_puzzle():
    """
    End-to-end integration test.

    The Logic Agent must solve every unresolved character without
    guessing.
    """
    _, engine, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.solved is True

    assert (
        result.stop_reason
        == AgentStopReason.SOLVED
    )

    assert result.unresolved_character_ids == ()

    state = engine.get_public_state()

    assert (
        len(state.proved_statuses)
        == len(state.characters)
        == 9
    )


def test_auto_solve_performs_exactly_eight_deductions():
    """
    B2 is already proved initially.

    A 3x3 puzzle has 9 characters, therefore the agent must make
    exactly 8 additional deductions.
    """
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.deduction_count == 8

    assert len(result.steps) == 8

    assert len(
        agent.deduction_trace
    ) == 8


def test_auto_solve_uses_expected_deterministic_order():
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    actual = [
        (
            step.character_id,
            step.status,
        )
        for step in result.steps
    ]

    assert actual == (
        expected_deduction_sequence()
    )


def test_every_auto_solve_verdict_is_accepted():
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.solved is True

    for step in result.steps:
        assert (
            step.verdict_code
            == VerdictCode.ACCEPTED
        )


def test_agent_never_submits_unknown_classification():
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    for step in result.steps:
        assert step.classification in {
            Classification.CRIMINAL,
            Classification.INNOCENT,
        }

        assert (
            step.classification
            != Classification.UNKNOWN
        )

        assert (
            step.classification
            != Classification.INCONSISTENT
        )


def test_final_public_statuses_match_expected_hidden_solution():
    """
    Hidden statuses are used here only as a test oracle.

    The LogicAgent itself never receives Puzzle.secrets.
    """
    puzzle, engine, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.solved is True

    final_state = engine.get_public_state()

    expected = {
        character_id:
            secret.status
        for character_id, secret
        in puzzle.secrets.items()
    }

    assert (
        final_state.proved_statuses
        == expected
    )


def test_all_character_clues_are_revealed_after_solution():
    _, engine, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.solved is True

    state = engine.get_public_state()

    assert (
        len(state.revealed_clues)
        == len(state.characters)
        == 9
    )


# ============================================================
# Exact deduction chain
# ============================================================

def test_expected_character_deduction_chain():
    """
    Verify the intended reasoning chain of puzzle_3x3_01:

        B2 initially known
          ↓
        A1
          ↓
        B1
          ↓
        C1
          ↓
        A2
          ↓
        B3
          ↓
        C3
          ↓
        C2
          ↓
        A3
    """
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    expected_ids = [
        "A1",
        "B1",
        "C1",
        "A2",
        "B3",
        "C3",
        "C2",
        "A3",
    ]

    actual_ids = [
        step.character_id
        for step in result.steps
    ]

    assert actual_ids == expected_ids


def test_step_numbers_are_sequential():
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    assert [
        step.step_number
        for step in result.steps
    ] == list(
        range(1, 9)
    )


# ============================================================
# Step limit
# ============================================================

def test_auto_solve_zero_step_limit_does_nothing():
    _, engine, agent = create_basic_game()

    before = engine.get_public_state()

    result = agent.auto_solve(
        max_steps=0
    )

    after = engine.get_public_state()

    assert result.solved is False

    assert (
        result.stop_reason
        == AgentStopReason.STEP_LIMIT
    )

    assert result.steps == ()

    assert len(
        result.unresolved_character_ids
    ) == 8

    assert (
        after.proved_statuses
        == before.proved_statuses
    )

    assert (
        after.revealed_clues
        == before.revealed_clues
    )


def test_auto_solve_respects_step_limit():
    _, engine, agent = create_basic_game()

    result = agent.auto_solve(
        max_steps=3
    )

    assert result.solved is False

    assert (
        result.stop_reason
        == AgentStopReason.STEP_LIMIT
    )

    assert result.deduction_count == 3

    assert [
        step.character_id
        for step in result.steps
    ] == [
        "A1",
        "B1",
        "C1",
    ]

    state = engine.get_public_state()

    # Initial B2 + 3 agent deductions.
    assert len(
        state.proved_statuses
    ) == 4

    assert len(
        result.unresolved_character_ids
    ) == 5


def test_auto_solve_can_continue_after_step_limited_run():
    _, _, agent = create_basic_game()

    first_run = agent.auto_solve(
        max_steps=3
    )

    assert (
        first_run.stop_reason
        == AgentStopReason.STEP_LIMIT
    )

    second_run = agent.auto_solve()

    assert second_run.solved is True

    assert (
        second_run.stop_reason
        == AgentStopReason.SOLVED
    )

    # Three deductions were already made, so five remain.
    assert second_run.deduction_count == 5

    # Persistent trace contains all eight deductions.
    assert len(
        agent.deduction_trace
    ) == 8


def test_invalid_negative_step_limit():
    _, _, agent = create_basic_game()

    with pytest.raises(
        ValueError
    ):
        agent.auto_solve(
            max_steps=-1
        )


def test_invalid_non_integer_step_limit():
    _, _, agent = create_basic_game()

    with pytest.raises(
        TypeError
    ):
        agent.auto_solve(
            max_steps=1.5
        )


def test_boolean_step_limit_is_rejected():
    _, _, agent = create_basic_game()

    with pytest.raises(
        TypeError
    ):
        agent.auto_solve(
            max_steps=True
        )


# ============================================================
# Solved-state behavior
# ============================================================

def test_step_returns_none_after_puzzle_is_solved():
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.solved is True

    assert agent.step() is None


def test_find_hint_returns_none_after_puzzle_is_solved():
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.solved is True

    assert agent.find_hint() is None


def test_auto_solve_on_already_solved_game_returns_no_new_steps():
    _, _, agent = create_basic_game()

    first_run = agent.auto_solve()

    assert first_run.solved is True

    second_run = agent.auto_solve()

    assert second_run.solved is True

    assert (
        second_run.stop_reason
        == AgentStopReason.SOLVED
    )

    assert second_run.steps == ()

    assert (
        second_run.unresolved_character_ids
        == ()
    )

    # Existing historical trace remains available.
    assert len(
        agent.deduction_trace
    ) == 8


# ============================================================
# Deduction trace
# ============================================================
def test_first_step_records_initial_active_clues():
    """
    Before the first deduction, only the initially revealed
    character B2 contributes a clue to the public KB.

    AgentStep must snapshot exactly that clue.
    """
    _, engine, agent = create_basic_game()

    before = engine.get_public_state()

    expected_active_clue_ids = tuple(
        clue.id
        for clue in before.revealed_clues
    )

    step = agent.step()

    assert step is not None

    assert (
        step.active_clue_ids
        == expected_active_clue_ids
    )

    assert len(
        step.active_clue_ids
    ) == 1


def test_newly_revealed_clue_is_not_active_in_same_step():
    """
    The clue revealed by an accepted verdict becomes active only
    for future deductions.

    It must not appear in the active-clue snapshot used to justify
    the verdict that revealed it.
    """
    _, _, agent = create_basic_game()

    step = agent.step()

    assert step is not None
    assert step.revealed_clue is not None

    assert (
        step.revealed_clue.id
        not in step.active_clue_ids
    )


def test_newly_revealed_clue_becomes_active_on_next_step():
    """
    A clue revealed by step 1 must become part of the public KB
    used by step 2.
    """
    _, engine, agent = create_basic_game()

    first = agent.step()

    assert first is not None
    assert first.revealed_clue is not None

    state_before_second = (
        engine.get_public_state()
    )

    expected_active_clue_ids = tuple(
        clue.id
        for clue
        in state_before_second.revealed_clues
    )

    second = agent.step()

    assert second is not None

    assert (
        second.active_clue_ids
        == expected_active_clue_ids
    )

    assert (
        first.revealed_clue.id
        in second.active_clue_ids
    )


def test_auto_solve_active_clue_snapshots_grow_by_one_per_step():
    """
    puzzle_3x3_01 starts with one revealed clue.

    Every accepted deduction reveals exactly one additional clue.

    Therefore, before deduction step k:

        number of active clues = k

    for steps 1..8.
    """
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.solved is True

    assert len(
        result.steps
    ) == 8

    for expected_count, step in enumerate(
        result.steps,
        start=1,
    ):
        assert len(
            step.active_clue_ids
        ) == expected_count


def test_each_revealed_clue_appears_in_following_step_snapshot():
    """
    For every non-final deduction, the clue revealed by that
    deduction must be active in the next deduction.
    """
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    for current, following in zip(
        result.steps,
        result.steps[1:],
    ):
        assert (
            current.revealed_clue
            is not None
        )

        assert (
            current.revealed_clue.id
            in following.active_clue_ids
        )


def test_active_clue_ids_are_stored_as_immutable_tuple():
    _, _, agent = create_basic_game()

    step = agent.step()

    assert step is not None

    assert isinstance(
        step.active_clue_ids,
        tuple,
    )


def test_clear_trace_removes_history_without_changing_game():
    _, engine, agent = create_basic_game()

    agent.auto_solve(
        max_steps=3
    )

    assert len(
        agent.deduction_trace
    ) == 3

    before = engine.get_public_state()

    agent.clear_trace()

    after = engine.get_public_state()

    assert agent.deduction_trace == ()

    # Clearing trace must not restart or mutate the game.
    assert (
        after.proved_statuses
        == before.proved_statuses
    )

    assert (
        after.revealed_clues
        == before.revealed_clues
    )


# ============================================================
# Metrics
# ============================================================

def test_agent_step_exposes_non_negative_solver_metrics():
    _, _, agent = create_basic_game()

    step = agent.step()

    assert step is not None

    assert step.decisions >= 0
    assert step.propagations >= 0
    assert step.backtracks >= 0
    assert step.runtime >= 0.0


def test_agent_run_aggregates_solver_metrics():
    _, _, agent = create_basic_game()

    result = agent.auto_solve()

    assert result.total_decisions >= 0
    assert result.total_propagations >= 0
    assert result.total_backtracks >= 0
    assert result.total_runtime >= 0.0

    assert result.total_decisions == sum(
        step.decisions
        for step in result.steps
    )

    assert result.total_propagations == sum(
        step.propagations
        for step in result.steps
    )

    assert result.total_backtracks == sum(
        step.backtracks
        for step in result.steps
    )

    assert result.total_runtime == pytest.approx(
        sum(
            step.runtime
            for step in result.steps
        )
    )