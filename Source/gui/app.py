from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any


# ============================================================
# Import path
# ============================================================
#
# Recommended:
#
#     py -m gui.app
#
# from the project Source/ directory.
#
# Direct execution is also supported:
#
#     py gui/app.py
#

SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


from core.game_engine import (
    GameEngine,
    GameEngineError,
    InconsistentKnowledgeBaseError,
)
from core.models import (
    Clue,
    ClueType,
    Region,
    RegionType,
    Status,
    VerdictCode,
)
from core.puzzle_loader import (
    PuzzleFormatError,
    load_puzzle,
    resolve_region_cells,
)
from logic.agent import (
    AgentKnowledgeBaseError,
    AgentStep,
    LogicAgent,
    LogicAgentError,
)


# ============================================================
# Paths / visual constants
# ============================================================

PUZZLES_DIR = SOURCE_DIR / "puzzles"

APP_TITLE = "Griductive Solver"

WINDOW_WIDTH = 1320
WINDOW_HEIGHT = 860

# Automatic replay pacing.  The replay itself still uses LogicAgent.step()
# so every visible transition is a real no-guess deduction from public state.
REPLAY_DELAY_MS = 800

COLOR_WINDOW = "#f7f7f5"
COLOR_PANEL = "#ffffff"

COLOR_UNKNOWN = "#f3f4f6"
COLOR_CRIMINAL = "#fee2e2"
COLOR_INNOCENT = "#dcfce7"

COLOR_BORDER = "#9ca3af"
COLOR_SELECTED = "#2563eb"
COLOR_HIGHLIGHT = "#f59e0b"
COLOR_SELECTED_HIGHLIGHT = "#7c3aed"

COLOR_TEXT = "#111827"
COLOR_MUTED = "#6b7280"


# ============================================================
# Small helpers
# ============================================================

def _enum_value(value: Any) -> str:
    """
    Return a readable value for Enum/string-like objects.
    """
    raw = getattr(value, "value", value)
    return str(raw)


def _sat_label(result: Any) -> str:
    return (
        "SAT"
        if bool(result.satisfiable)
        else "UNSAT"
    )


def _status_short(status: Status) -> str:
    return (
        "C"
        if status == Status.CRIMINAL
        else "I"
    )


# ============================================================
# Main application
# ============================================================

class GriductiveApp:
    """
    Thin GUI/controller layer for Griductive.

    Architecture rule:
        - GameEngine owns hidden statuses and unrevealed clues.
        - LogicAgent receives only PublicState.
        - GUI renders only public information returned by GameEngine.

    The GUI never reads Puzzle.secrets and never calls CNF/DPLL directly.
    """

    def __init__(
        self,
        root: tk.Tk,
    ):
        self.root = root

        self.engine: GameEngine | None = None
        self.agent: LogicAgent | None = None

        self.current_puzzle_path: Path | None = None

        # Replay Solution state.  Replay is enabled only after Auto Solve
        # has successfully reached a complete solution.  When replaying,
        # the puzzle is reloaded to its initial public state and the
        # deterministic LogicAgent is executed one real step at a time.
        self.replay_available = False
        self.replay_active = False
        self.replay_paused = False
        self.replay_after_id: str | None = None
        self.replay_step_index = 0
        self.replay_total_steps = 0

        self.selected_character_id: str | None = None
        self.selected_clue_id: str | None = None

        # Cells highlighted because the currently selected revealed clue
        # references or counts them, or because Hint identified a target.
        self.highlighted_cells: set[str] = set()

        # Character-id -> outer frame / card button.
        self.card_frames: dict[str, tk.Frame] = {}
        self.card_buttons: dict[str, tk.Button] = {}

        # Revealed clue rows shown in the listbox:
        #     (owner_character_id, Clue)
        self.revealed_entries: list[
            tuple[str, Clue]
        ] = []

        self.puzzle_paths: dict[
            str,
            Path,
        ] = {}

        self.puzzle_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value="Ready"
        )
        self.progress_var = tk.StringVar(
            value=""
        )

        self._configure_window()
        self._build_layout()
        self._load_puzzle_list()
        self._load_first_puzzle()

    # ========================================================
    # Window / layout
    # ========================================================

    def _configure_window(
        self,
    ) -> None:
        self.root.title(
            APP_TITLE
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.minsize(
            1080,
            700,
        )

        self.root.configure(
            bg=COLOR_WINDOW
        )

    def _build_layout(
        self,
    ) -> None:
        self.root.columnconfigure(
            0,
            weight=1,
        )
        self.root.rowconfigure(
            1,
            weight=1,
        )
        self.root.rowconfigure(
            3,
            weight=1,
        )

        self._build_toolbar()
        self._build_main_area()
        self._build_controls()
        self._build_trace_panel()
        self._build_status_bar()

    def _build_toolbar(
        self,
    ) -> None:
        toolbar = ttk.Frame(
            self.root,
            padding=(12, 10),
        )
        toolbar.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        toolbar.columnconfigure(
            5,
            weight=1,
        )

        ttk.Label(
            toolbar,
            text=APP_TITLE,
            font=("Segoe UI", 17, "bold"),
        ).grid(
            row=0,
            column=0,
            padx=(0, 24),
            sticky="w",
        )

        ttk.Label(
            toolbar,
            text="Puzzle:",
        ).grid(
            row=0,
            column=1,
            padx=(0, 6),
        )

        self.puzzle_combo = ttk.Combobox(
            toolbar,
            textvariable=self.puzzle_var,
            state="readonly",
            width=28,
        )
        self.puzzle_combo.grid(
            row=0,
            column=2,
            padx=(0, 8),
        )

        self.load_button = ttk.Button(
            toolbar,
            text="Load",
            command=self.on_load,
        )
        self.load_button.grid(
            row=0,
            column=3,
            padx=4,
        )

        self.restart_button = ttk.Button(
            toolbar,
            text="Restart",
            command=self.on_restart,
        )
        self.restart_button.grid(
            row=0,
            column=4,
            padx=4,
        )

        self.puzzle_title_label = ttk.Label(
            toolbar,
            text="",
            font=("Segoe UI", 10, "bold"),
        )
        self.puzzle_title_label.grid(
            row=0,
            column=5,
            padx=(18, 0),
            sticky="e",
        )

    def _build_main_area(
        self,
    ) -> None:
        main = ttk.Frame(
            self.root,
            padding=(12, 4, 12, 4),
        )
        main.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        main.columnconfigure(
            0,
            weight=3,
        )
        main.columnconfigure(
            1,
            weight=2,
        )
        main.rowconfigure(
            0,
            weight=1,
        )

        # ----------------------------------------------------
        # Board
        # ----------------------------------------------------

        board_box = ttk.LabelFrame(
            main,
            text="Board",
            padding=12,
        )
        board_box.grid(
            row=0,
            column=0,
            padx=(0, 8),
            sticky="nsew",
        )

        board_box.columnconfigure(
            0,
            weight=1,
        )
        board_box.rowconfigure(
            0,
            weight=1,
        )

        self.board_frame = tk.Frame(
            board_box,
            bg=COLOR_PANEL,
        )
        self.board_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        # ----------------------------------------------------
        # Right side
        # ----------------------------------------------------

        right = ttk.Frame(
            main,
        )
        right.grid(
            row=0,
            column=1,
            padx=(8, 0),
            sticky="nsew",
        )

        right.columnconfigure(
            0,
            weight=1,
        )
        right.rowconfigure(
            1,
            weight=1,
        )

        self._build_selected_character_panel(
            right
        )
        self._build_clue_panel(
            right
        )

    def _build_selected_character_panel(
        self,
        parent,
    ) -> None:
        box = ttk.LabelFrame(
            parent,
            text="Selected Character",
            padding=10,
        )
        box.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )

        box.columnconfigure(
            0,
            weight=1,
        )

        self.character_detail = tk.Text(
            box,
            height=8,
            wrap="word",
            relief="flat",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            padx=6,
            pady=6,
            state="disabled",
        )
        self.character_detail.grid(
            row=0,
            column=0,
            sticky="ew",
        )

    def _build_clue_panel(
        self,
        parent,
    ) -> None:
        box = ttk.LabelFrame(
            parent,
            text="Revealed Clues",
            padding=10,
        )
        box.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        box.columnconfigure(
            0,
            weight=1,
        )
        box.rowconfigure(
            0,
            weight=1,
        )
        box.rowconfigure(
            1,
            weight=1,
        )

        list_frame = ttk.Frame(
            box
        )
        list_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        list_frame.columnconfigure(
            0,
            weight=1,
        )
        list_frame.rowconfigure(
            0,
            weight=1,
        )

        self.clue_listbox = tk.Listbox(
            list_frame,
            exportselection=False,
            activestyle="dotbox",
            height=9,
        )
        self.clue_listbox.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        clue_scroll = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.clue_listbox.yview,
        )
        clue_scroll.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        self.clue_listbox.configure(
            yscrollcommand=clue_scroll.set
        )

        self.clue_listbox.bind(
            "<<ListboxSelect>>",
            self.on_clue_selected,
        )

        detail_frame = ttk.LabelFrame(
            box,
            text="Selected Clue",
            padding=6,
        )
        detail_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            pady=(8, 0),
        )

        detail_frame.columnconfigure(
            0,
            weight=1,
        )
        detail_frame.rowconfigure(
            0,
            weight=1,
        )

        self.clue_detail = tk.Text(
            detail_frame,
            height=8,
            wrap="word",
            relief="flat",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            padx=6,
            pady=6,
            state="disabled",
        )
        self.clue_detail.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

    def _build_controls(
        self,
    ) -> None:
        controls = ttk.Frame(
            self.root,
            padding=(12, 8),
        )
        controls.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        controls.columnconfigure(
            8,
            weight=1,
        )

        ttk.Label(
            controls,
            text="Manual verdict:",
            font=("Segoe UI", 10, "bold"),
        ).grid(
            row=0,
            column=0,
            padx=(0, 8),
        )

        self.criminal_button = ttk.Button(
            controls,
            text="Criminal",
            command=lambda: self.on_manual_verdict(
                Status.CRIMINAL
            ),
        )
        self.criminal_button.grid(
            row=0,
            column=1,
            padx=4,
        )

        self.innocent_button = ttk.Button(
            controls,
            text="Innocent",
            command=lambda: self.on_manual_verdict(
                Status.INNOCENT
            ),
        )
        self.innocent_button.grid(
            row=0,
            column=2,
            padx=4,
        )

        ttk.Separator(
            controls,
            orient="vertical",
        ).grid(
            row=0,
            column=3,
            sticky="ns",
            padx=12,
        )

        self.hint_button = ttk.Button(
            controls,
            text="Hint",
            command=self.on_hint,
        )
        self.hint_button.grid(
            row=0,
            column=4,
            padx=4,
        )

        self.step_button = ttk.Button(
            controls,
            text="Agent Step",
            command=self.on_agent_step,
        )
        self.step_button.grid(
            row=0,
            column=5,
            padx=4,
        )

        self.auto_button = ttk.Button(
            controls,
            text="Auto Solve",
            command=self.on_auto_solve,
        )
        self.auto_button.grid(
            row=0,
            column=6,
            padx=4,
        )

        self.replay_button = ttk.Button(
            controls,
            text="Replay Solution",
            command=self.on_replay_solution,
            state="disabled",
        )
        self.replay_button.grid(
            row=0,
            column=7,
            padx=(12, 4),
        )

        self.progress_label = ttk.Label(
            controls,
            textvariable=self.progress_var,
        )
        self.progress_label.grid(
            row=0,
            column=8,
            sticky="e",
        )

    def _build_trace_panel(
        self,
    ) -> None:
        box = ttk.LabelFrame(
            self.root,
            text="Deduction Trace / Feedback",
            padding=(10, 8),
        )
        box.grid(
            row=3,
            column=0,
            padx=12,
            pady=(0, 8),
            sticky="nsew",
        )

        box.columnconfigure(
            0,
            weight=1,
        )
        box.rowconfigure(
            0,
            weight=1,
        )

        self.trace_text = tk.Text(
            box,
            height=14,
            wrap="word",
            bg="#fbfbfb",
            fg=COLOR_TEXT,
            padx=8,
            pady=8,
            state="disabled",
        )
        self.trace_text.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        trace_scroll = ttk.Scrollbar(
            box,
            orient="vertical",
            command=self.trace_text.yview,
        )
        trace_scroll.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        self.trace_text.configure(
            yscrollcommand=trace_scroll.set
        )

    def _build_status_bar(
        self,
    ) -> None:
        status = ttk.Frame(
            self.root,
            padding=(12, 3, 12, 7),
        )
        status.grid(
            row=4,
            column=0,
            sticky="ew",
        )

        status.columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            status,
            textvariable=self.status_var,
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

    # ========================================================
    # Puzzle lifecycle
    # ========================================================

    def _load_puzzle_list(
        self,
    ) -> None:
        paths = sorted(
            PUZZLES_DIR.glob(
                "puzzle_*.json"
            ),
            key=lambda path: path.name,
        )

        self.puzzle_paths = {
            path.name: path
            for path in paths
        }

        self.puzzle_combo[
            "values"
        ] = tuple(
            self.puzzle_paths
        )

        if paths:
            self.puzzle_var.set(
                paths[0].name
            )

    def _load_first_puzzle(
        self,
    ) -> None:
        if not self.puzzle_paths:
            self._set_status(
                f"No puzzle files found in {PUZZLES_DIR}."
            )
            return

        self.on_load()

    def on_load(
        self,
    ) -> None:
        filename = (
            self.puzzle_var.get().strip()
        )

        path = self.puzzle_paths.get(
            filename
        )

        if path is None:
            messagebox.showerror(
                "Load Puzzle",
                "Please select a valid puzzle.",
            )
            return

        self._create_game_from_file(
            path
        )

    def on_restart(
        self,
    ) -> None:
        if self.current_puzzle_path is None:
            return

        self._create_game_from_file(
            self.current_puzzle_path
        )

    def _create_game_from_file(
        self,
        path: Path,
    ) -> None:
        """
        Load one puzzle and immediately hand the complete Puzzle object
        to GameEngine.

        The GUI does not keep or inspect the Puzzle object afterward.
        Rendering uses GameEngine public APIs only.
        """
        try:
            puzzle = load_puzzle(
                path
            )

            engine = GameEngine(
                puzzle
            )

            agent = LogicAgent(
                engine
            )

        except (
            FileNotFoundError,
            PuzzleFormatError,
            GameEngineError,
            LogicAgentError,
        ) as error:
            messagebox.showerror(
                "Load Puzzle",
                str(error),
            )
            return

        self._cancel_replay_callback()

        self.engine = engine
        self.agent = agent
        self.current_puzzle_path = path

        self.replay_available = False
        self.replay_active = False
        self.replay_paused = False
        self.replay_step_index = 0
        self.replay_total_steps = 0
        self.replay_button.configure(
            text="Replay Solution",
        )

        self.selected_character_id = None
        self.selected_clue_id = None
        self.highlighted_cells.clear()

        self._clear_trace()
        self._build_board()
        self.refresh_from_public_state()

        initial_ids = (
            self.engine
            .get_revealed_character_ids()
        )

        self._append_trace(
            "Loaded puzzle: "
            f"{self.engine.puzzle_name}\n"
            f"Board: {self.engine.size}x{self.engine.size}\n"
            "Initially revealed: "
            + (
                ", ".join(initial_ids)
                if initial_ids
                else "(none)"
            )
            + "\n"
        )

        self._set_status(
            "Puzzle loaded. Select a character or a revealed clue."
        )

    # ========================================================
    # Board
    # ========================================================

    def _build_board(
        self,
    ) -> None:
        for child in (
            self.board_frame.winfo_children()
        ):
            child.destroy()

        self.card_frames.clear()
        self.card_buttons.clear()

        if self.engine is None:
            return

        size = self.engine.size

        # Smaller cards for 5x5; larger cards for 3x3.
        if size >= 5:
            card_width = 17
            card_height = 7
            wraplength = 125
        elif size == 4:
            card_width = 19
            card_height = 8
            wraplength = 145
        else:
            card_width = 22
            card_height = 9
            wraplength = 175

        # Top-left empty coordinate cell.
        ttk.Label(
            self.board_frame,
            text="",
        ).grid(
            row=0,
            column=0,
            padx=3,
            pady=3,
        )

        for column in range(
            size
        ):
            column_name = chr(
                ord("A") + column
            )

            ttk.Label(
                self.board_frame,
                text=column_name,
                anchor="center",
                font=("Segoe UI", 10, "bold"),
            ).grid(
                row=0,
                column=column + 1,
                padx=4,
                pady=(0, 5),
                sticky="ew",
            )

            self.board_frame.columnconfigure(
                column + 1,
                weight=1,
            )

        characters = (
            self.engine.get_characters()
        )

        for row in range(
            1,
            size + 1,
        ):
            ttk.Label(
                self.board_frame,
                text=str(row),
                anchor="center",
                font=("Segoe UI", 10, "bold"),
            ).grid(
                row=row,
                column=0,
                padx=(0, 6),
                pady=4,
                sticky="ns",
            )

            self.board_frame.rowconfigure(
                row,
                weight=1,
            )

            for column in range(
                size
            ):
                character_id = (
                    f"{chr(ord('A') + column)}{row}"
                )

                if character_id not in characters:
                    continue

                frame = tk.Frame(
                    self.board_frame,
                    bg=COLOR_BORDER,
                    padx=3,
                    pady=3,
                )
                frame.grid(
                    row=row,
                    column=column + 1,
                    padx=4,
                    pady=4,
                    sticky="nsew",
                )

                button = tk.Button(
                    frame,
                    text="",
                    width=card_width,
                    height=card_height,
                    wraplength=wraplength,
                    justify="center",
                    anchor="center",
                    relief="flat",
                    bg=COLOR_UNKNOWN,
                    fg=COLOR_TEXT,
                    activebackground=COLOR_UNKNOWN,
                    command=lambda cid=character_id:
                        self.on_character_selected(
                            cid
                        ),
                )
                button.pack(
                    fill="both",
                    expand=True,
                )

                self.card_frames[
                    character_id
                ] = frame

                self.card_buttons[
                    character_id
                ] = button

    def refresh_from_public_state(
        self,
    ) -> None:
        if (
            self.engine is None
            or self.agent is None
        ):
            return

        state = (
            self.engine.get_public_state()
        )

        self.puzzle_title_label.configure(
            text=(
                f"{self.engine.puzzle_name}  "
                f"({self.engine.size}x{self.engine.size})"
            )
        )

        for (
            character_id,
            character,
        ) in state.characters.items():
            status = state.proved_statuses.get(
                character_id
            )

            clue = (
                self.engine.get_revealed_clue(
                    character_id
                )
            )

            text = self._format_card_text(
                character_id=character_id,
                name=character.name,
                profession=character.profession,
                status=status,
                clue=clue,
            )

            button = self.card_buttons.get(
                character_id
            )

            if button is None:
                continue

            background = (
                COLOR_UNKNOWN
                if status is None
                else (
                    COLOR_CRIMINAL
                    if status == Status.CRIMINAL
                    else COLOR_INNOCENT
                )
            )

            button.configure(
                text=text,
                bg=background,
                activebackground=background,
            )

            frame = self.card_frames[
                character_id
            ]

            is_selected = (
                character_id
                == self.selected_character_id
            )

            is_highlighted = (
                character_id
                in self.highlighted_cells
            )

            if (
                is_selected
                and is_highlighted
            ):
                border = (
                    COLOR_SELECTED_HIGHLIGHT
                )
            elif is_selected:
                border = COLOR_SELECTED
            elif is_highlighted:
                border = COLOR_HIGHLIGHT
            else:
                border = COLOR_BORDER

            frame.configure(
                bg=border
            )

        self._refresh_character_detail(
            state
        )
        self._refresh_revealed_clues(
            state
        )
        self._refresh_progress(
            state
        )
        self._refresh_control_states(
            state
        )

    def _format_card_text(
        self,
        *,
        character_id: str,
        name: str,
        profession: str,
        status: Status | None,
        clue: Clue | None,
    ) -> str:
        status_text = (
            "UNKNOWN"
            if status is None
            else status.value
        )

        clue_text = (
            "Clue: HIDDEN"
            if clue is None
            else self.format_clue(
                clue,
                compact=True,
            )
        )

        return (
            f"{character_id}\n"
            f"{name}\n"
            f"{profession}\n\n"
            f"{status_text}\n"
            f"{clue_text}"
        )

    # ========================================================
    # Character selection / manual play
    # ========================================================

    def on_character_selected(
        self,
        character_id: str,
    ) -> None:
        self.selected_character_id = (
            character_id
        )

        self.refresh_from_public_state()

        self._set_status(
            f"Selected {character_id}."
        )

    def on_manual_verdict(
        self,
        status: Status,
    ) -> None:
        if (
            self.engine is None
            or self.agent is None
        ):
            return

        if self.selected_character_id is None:
            messagebox.showinfo(
                "Manual Verdict",
                "Select a character first.",
            )
            return

        character_id = (
            self.selected_character_id
        )

        self._set_status(
            f"Checking {character_id} as {status.value}..."
        )
        self.root.update_idletasks()

        try:
            result = (
                self.engine.submit_verdict(
                    character_id=character_id,
                    submitted_status=status,
                    classifier=(
                        self.agent.checker
                        .classify_character
                    ),
                )
            )

        except InconsistentKnowledgeBaseError as error:
            self._append_trace(
                f"MANUAL {character_id} = {status.value}\n"
                f"Result: INCONSISTENT\n"
                f"{error}\n"
            )
            messagebox.showerror(
                "Inconsistent Knowledge Base",
                str(error),
            )
            self._set_status(
                "Knowledge base is inconsistent."
            )
            return

        except (
            GameEngineError,
            LogicAgentError,
        ) as error:
            self._show_runtime_error(
                "Manual Verdict",
                error,
            )
            return

        if (
            result.code
            == VerdictCode.ACCEPTED
        ):
            self._append_trace(
                f"MANUAL {character_id} = {status.value}\n"
                "Result: ACCEPTED\n"
                + (
                    "Revealed: "
                    f"{result.revealed_clue.id}\n"
                    if result.revealed_clue
                    is not None
                    else "Already resolved; no new clue.\n"
                )
            )

            if result.revealed_clue is not None:
                self.selected_clue_id = (
                    result.revealed_clue.id
                )

            self._set_status(
                f"ACCEPTED: {character_id} is {status.value}."
            )

        elif (
            result.code
            == VerdictCode.NOT_PROVABLE
        ):
            self._append_trace(
                f"MANUAL {character_id} = {status.value}\n"
                "Result: NOT_PROVABLE\n"
                "Neither status is currently forced. "
                "The game state is unchanged.\n"
            )

            self._set_status(
                "NOT_PROVABLE: neither verdict is currently forced."
            )

        elif (
            result.code
            == VerdictCode.CONTRADICTED
        ):
            self._append_trace(
                f"MANUAL {character_id} = {status.value}\n"
                "Result: CONTRADICTED\n"
                "The opposite status is logically forced. "
                "The game state is unchanged.\n"
            )

            self._set_status(
                "CONTRADICTED: the opposite verdict is forced."
            )

        self.highlighted_cells.clear()
        self.refresh_from_public_state()

        if (
            result.code
            == VerdictCode.ACCEPTED
            and result.revealed_clue
            is not None
        ):
            self._select_clue_by_id(
                result.revealed_clue.id
            )

    # ========================================================
    # Hint
    # ========================================================

    def on_hint(
        self,
    ) -> None:
        if self.agent is None:
            return

        self._set_status(
            "Searching for a logically forced hint..."
        )
        self.root.update_idletasks()

        try:
            hint = (
                self.agent.find_hint()
            )

        except AgentKnowledgeBaseError as error:
            self._show_runtime_error(
                "Hint",
                error,
            )
            return

        if hint is None:
            self._append_trace(
                "HINT\n"
                "No currently unresolved character can be proved "
                "without guessing.\n"
            )

            self._set_status(
                "No provable hint is currently available."
            )
            return

        self.selected_character_id = (
            hint.character_id
        )
        self.selected_clue_id = None
        self.highlighted_cells = {
            hint.character_id
        }

        self._append_trace(
            self._format_hint_trace(
                hint
            )
        )

        self.refresh_from_public_state()
        self.clue_listbox.selection_clear(
            0,
            tk.END,
        )
        self._set_text(
            self.clue_detail,
            (
                "Hint target\n\n"
                f"{hint.character_id} = {hint.status.value}\n\n"
                "The hint is derived only from the current public "
                "knowledge base; no hidden clue is revealed."
            ),
        )

        self._set_status(
            f"Hint: {hint.character_id} is provably {hint.status.value}."
        )

    # ========================================================
    # Agent step
    # ========================================================

    def on_agent_step(
        self,
    ) -> None:
        if (
            self.engine is None
            or self.agent is None
        ):
            return

        if self.engine.is_solved():
            self._set_status(
                "Puzzle is already solved."
            )
            return

        self._set_status(
            "Agent is searching for the next forced verdict..."
        )
        self.root.update_idletasks()

        try:
            step = self.agent.step()

        except (
            AgentKnowledgeBaseError,
            LogicAgentError,
            GameEngineError,
        ) as error:
            self._show_runtime_error(
                "Agent Step",
                error,
            )
            return

        if step is None:
            self._append_trace(
                "AGENT STEP\n"
                "No provable move exists. The agent stops "
                "instead of guessing.\n"
            )
            self._set_status(
                "NO_PROVABLE_MOVE."
            )
            return

        self.selected_character_id = (
            step.character_id
        )

        self.highlighted_cells = set()

        if step.revealed_clue is not None:
            self.selected_clue_id = (
                step.revealed_clue.id
            )

        self._append_trace(
            self._format_step_trace(
                step
            )
        )

        self.refresh_from_public_state()

        if step.revealed_clue is not None:
            self._select_clue_by_id(
                step.revealed_clue.id
            )

        self._set_status(
            f"Agent proved {step.character_id} = {step.status.value}."
        )

    # ========================================================
    # Auto solve
    # ========================================================

    def on_auto_solve(
        self,
    ) -> None:
        if (
            self.engine is None
            or self.agent is None
        ):
            return

        if self.engine.is_solved():
            self._set_status(
                "Puzzle is already solved."
            )
            return

        self._set_controls_enabled(
            False
        )
        self._set_status(
            "Auto Solve is running..."
        )
        self.root.update_idletasks()

        try:
            result = (
                self.agent.auto_solve()
            )

        except (
            AgentKnowledgeBaseError,
            LogicAgentError,
            GameEngineError,
        ) as error:
            self._show_runtime_error(
                "Auto Solve",
                error,
            )
            return

        finally:
            self._set_controls_enabled(
                True
            )

        # A successful complete solve unlocks Replay Solution.  Replay
        # does not expose hidden labels; it later reloads the puzzle and
        # runs the deterministic LogicAgent one step at a time.
        self.replay_available = bool(
            result.solved
        )

        for step in result.steps:
            self._append_trace(
                self._format_step_trace(
                    step
                )
            )

        if result.steps:
            last_step = (
                result.steps[-1]
            )

            self.selected_character_id = (
                last_step.character_id
            )

            if (
                last_step.revealed_clue
                is not None
            ):
                self.selected_clue_id = (
                    last_step.revealed_clue.id
                )

        self.highlighted_cells.clear()
        self.refresh_from_public_state()

        if (
            result.steps
            and result.steps[-1].revealed_clue
            is not None
        ):
            self._select_clue_by_id(
                result.steps[-1]
                .revealed_clue.id
            )

        self._append_trace(
            "AUTO SOLVE SUMMARY\n"
            f"Stop reason: {result.stop_reason.value}\n"
            f"New deductions: {result.deduction_count}\n"
            f"SAT calls: {result.total_sat_calls}\n"
            f"Decisions: {result.total_decisions}\n"
            f"Propagations: {result.total_propagations}\n"
            f"Backtracks: {result.total_backtracks}\n"
            f"DPLL runtime: {result.total_runtime * 1000.0:.3f} ms\n"
        )

        if result.solved:
            self._set_status(
                "SOLVED: the agent proved every character without guessing."
            )
        else:
            self._set_status(
                f"Auto Solve stopped: {result.stop_reason.value}."
            )

    # ========================================================
    # Replay solution
    # ========================================================

    def on_replay_solution(
        self,
    ) -> None:
        """
        Start, pause, or resume the step-by-step solution replay.

        Replay deliberately does NOT paint recorded hidden answers onto
        the board.  Instead it reloads the same puzzle to its initial
        public state and calls LogicAgent.step() repeatedly.  Therefore
        every replayed verdict is again proved from the currently public
        knowledge base and the architecture remains no-guess.
        """
        if self.replay_active:
            if self.replay_paused:
                self._resume_replay()
            else:
                self._pause_replay()
            return

        if not self.replay_available:
            messagebox.showinfo(
                "Replay Solution",
                (
                    "Run Auto Solve successfully first. "
                    "Replay becomes available after a complete solution "
                    "has been found."
                ),
            )
            return

        if self.current_puzzle_path is None:
            return

        self._start_replay()

    def _start_replay(
        self,
    ) -> None:
        """
        Recreate a fresh game at the initial public state, then schedule
        deterministic LogicAgent steps with Tk.after().
        """
        if self.current_puzzle_path is None:
            return

        try:
            puzzle = load_puzzle(
                self.current_puzzle_path
            )

            engine = GameEngine(
                puzzle
            )

            agent = LogicAgent(
                engine
            )

        except (
            FileNotFoundError,
            PuzzleFormatError,
            GameEngineError,
            LogicAgentError,
        ) as error:
            self._show_runtime_error(
                "Replay Solution",
                error,
            )
            return

        self._cancel_replay_callback()

        self.engine = engine
        self.agent = agent

        state = (
            self.engine.get_public_state()
        )

        self.replay_active = True
        self.replay_paused = False
        self.replay_step_index = 0
        self.replay_total_steps = (
            len(state.characters)
            - len(state.proved_statuses)
        )

        self.selected_character_id = None
        self.selected_clue_id = None
        self.highlighted_cells.clear()

        self._clear_trace()
        self._build_board()
        self.refresh_from_public_state()

        initial_ids = (
            self.engine
            .get_revealed_character_ids()
        )

        self._append_trace(
            "REPLAY SOLUTION\n"
            f"Puzzle: {self.engine.puzzle_name}\n"
            "Reset to the initial public knowledge state.\n"
            "Initially revealed: "
            + (
                ", ".join(initial_ids)
                if initial_ids
                else "(none)"
            )
            + "\n"
            "Each transition below is recomputed by LogicAgent.step(); "
            "no hidden status is read by the GUI.\n"
        )

        self.replay_button.configure(
            text="Pause Replay",
        )

        self._set_status(
            "Replay ready. Starting step-by-step deduction..."
        )

        # Give the initial state a short moment on screen before the
        # first deduction is executed.
        self._schedule_replay_step(
            REPLAY_DELAY_MS
        )

    def _schedule_replay_step(
        self,
        delay_ms: int = REPLAY_DELAY_MS,
    ) -> None:
        self._cancel_replay_callback()

        if (
            not self.replay_active
            or self.replay_paused
        ):
            return

        self.replay_after_id = (
            self.root.after(
                delay_ms,
                self._run_next_replay_step,
            )
        )

    def _run_next_replay_step(
        self,
    ) -> None:
        self.replay_after_id = None

        if (
            not self.replay_active
            or self.replay_paused
        ):
            return

        if (
            self.engine is None
            or self.agent is None
        ):
            self._finish_replay(
                success=False,
                message="Replay state is unavailable.",
            )
            return

        if self.engine.is_solved():
            self._finish_replay(
                success=True,
            )
            return

        try:
            step = self.agent.step()

        except (
            AgentKnowledgeBaseError,
            LogicAgentError,
            GameEngineError,
        ) as error:
            self._append_trace(
                "REPLAY ERROR\n"
                f"{type(error).__name__}: {error}\n"
            )
            self._finish_replay(
                success=False,
                message=str(error),
            )
            return

        if step is None:
            self._append_trace(
                "REPLAY STOPPED\n"
                "No provable move exists. Replay refuses to guess.\n"
            )
            self._finish_replay(
                success=False,
                message="NO_PROVABLE_MOVE during replay.",
            )
            return

        self.replay_step_index += 1
        self.selected_character_id = (
            step.character_id
        )

        self.highlighted_cells.clear()

        if step.revealed_clue is not None:
            self.selected_clue_id = (
                step.revealed_clue.id
            )

        self._append_trace(
            (
                f"REPLAY {self.replay_step_index}"
                f"/{self.replay_total_steps}\n"
                + self._format_step_trace(
                    step
                )
            )
        )

        self.refresh_from_public_state()

        if step.revealed_clue is not None:
            self._select_clue_by_id(
                step.revealed_clue.id
            )

        self._set_status(
            (
                f"Replay step {self.replay_step_index}/"
                f"{self.replay_total_steps}: "
                f"{step.character_id} = {step.status.value}."
            )
        )

        if self.engine.is_solved():
            self._finish_replay(
                success=True,
            )
            return

        self._schedule_replay_step(
            REPLAY_DELAY_MS
        )

    def _pause_replay(
        self,
    ) -> None:
        if not self.replay_active:
            return

        self.replay_paused = True
        self._cancel_replay_callback()

        self.replay_button.configure(
            text="Resume Replay",
        )

        self._set_status(
            (
                f"Replay paused at step {self.replay_step_index}/"
                f"{self.replay_total_steps}."
            )
        )

    def _resume_replay(
        self,
    ) -> None:
        if not self.replay_active:
            return

        self.replay_paused = False

        self.replay_button.configure(
            text="Pause Replay",
        )

        self._set_status(
            (
                f"Replay resumed at step {self.replay_step_index}/"
                f"{self.replay_total_steps}."
            )
        )

        self._schedule_replay_step(
            250
        )

    def _finish_replay(
        self,
        *,
        success: bool,
        message: str | None = None,
    ) -> None:
        self._cancel_replay_callback()

        self.replay_active = False
        self.replay_paused = False

        self.replay_button.configure(
            text="Replay Solution",
        )

        self.refresh_from_public_state()

        if success:
            self._append_trace(
                "REPLAY COMPLETE\n"
                f"Replayed {self.replay_step_index} deduction steps.\n"
                "Every replayed verdict was recomputed by the "
                "LogicAgent without guessing.\n"
            )

            self._set_status(
                "REPLAY COMPLETE: full solution shown step-by-step."
            )

        else:
            self._set_status(
                (
                    "Replay stopped."
                    if not message
                    else f"Replay stopped: {message}"
                )
            )

    def _cancel_replay_callback(
        self,
    ) -> None:
        if self.replay_after_id is None:
            return

        try:
            self.root.after_cancel(
                self.replay_after_id
            )
        except tk.TclError:
            pass

        self.replay_after_id = None

    # ========================================================
    # Revealed clues / highlighting
    # ========================================================

    def _refresh_revealed_clues(
        self,
        state,
    ) -> None:
        previous_id = (
            self.selected_clue_id
        )

        self.clue_listbox.delete(
            0,
            tk.END,
        )

        self.revealed_entries.clear()

        if self.engine is None:
            return

        for character_id in (
            self.engine
            .get_revealed_character_ids()
        ):
            clue = (
                self.engine
                .get_revealed_clue(
                    character_id
                )
            )

            if clue is None:
                continue

            self.revealed_entries.append(
                (
                    character_id,
                    clue,
                )
            )

            self.clue_listbox.insert(
                tk.END,
                (
                    f"{clue.id}  [{character_id}]  "
                    f"{_enum_value(clue.type)}"
                ),
            )

        available_clue_ids = {
            clue.id
            for _owner_id, clue
            in self.revealed_entries
        }

        if (
            previous_id is not None
            and previous_id in available_clue_ids
        ):
            self._select_clue_by_id(
                previous_id,
                trigger=False,
            )
        else:
            self.selected_clue_id = None

            self.clue_listbox.selection_clear(
                0,
                tk.END,
            )

            self._set_text(
                self.clue_detail,
                (
                    "Select a revealed clue to inspect it and "
                    "highlight the referenced cells."
                    if self.revealed_entries
                    else "No clues have been revealed."
                ),
            )

    def on_clue_selected(
        self,
        _event=None,
    ) -> None:
        selection = (
            self.clue_listbox.curselection()
        )

        if not selection:
            return

        index = int(
            selection[0]
        )

        if not (
            0 <= index
            < len(self.revealed_entries)
        ):
            return

        owner_id, clue = (
            self.revealed_entries[
                index
            ]
        )

        self.selected_clue_id = (
            clue.id
        )

        self.highlighted_cells = set(
            self.referenced_cells(
                clue
            )
        )

        self._set_text(
            self.clue_detail,
            (
                f"{clue.id}\n"
                f"Owner: {owner_id}\n"
                f"Type: {_enum_value(clue.type)}\n\n"
                f"{self.format_clue(clue)}\n\n"
                "Highlighted cells: "
                + (
                    ", ".join(
                        sorted(
                            self.highlighted_cells,
                            key=self._cell_sort_key,
                        )
                    )
                    if self.highlighted_cells
                    else "(none)"
                )
            ),
        )

        self.refresh_from_public_state()

        self._set_status(
            f"Selected {clue.id}; referenced cells are highlighted."
        )

    def _select_clue_by_id(
        self,
        clue_id: str,
        *,
        trigger: bool = True,
    ) -> None:
        for index, (
            _owner_id,
            clue,
        ) in enumerate(
            self.revealed_entries
        ):
            if clue.id != clue_id:
                continue

            self.clue_listbox.selection_clear(
                0,
                tk.END,
            )
            self.clue_listbox.selection_set(
                index
            )
            self.clue_listbox.see(
                index
            )

            self.selected_clue_id = (
                clue_id
            )

            if trigger:
                self.on_clue_selected()
            return

    # ========================================================
    # Clue formatting / references
    # ========================================================

    def format_clue(
        self,
        clue: Clue,
        *,
        compact: bool = False,
    ) -> str:
        clue_type = _enum_value(
            clue.type
        ).upper()

        params = clue.params

        if clue_type == "FACT":
            person = params[
                "person"
            ]
            status = params[
                "status"
            ]

            if compact:
                return (
                    f"{person} = "
                    f"{_status_short(status)}"
                )

            return (
                f"{person} is "
                f"{status.value}."
            )

        if clue_type in {
            "SAME",
            "DIFFERENT",
        }:
            first, second = (
                params["people"]
            )

            if compact:
                operator = (
                    "="
                    if clue_type == "SAME"
                    else "!="
                )

                return (
                    f"{first} {operator} {second}"
                )

            relation = (
                "the same status"
                if clue_type == "SAME"
                else "different statuses"
            )

            return (
                f"{first} and {second} have "
                f"{relation}."
            )

        if clue_type in {
            "EXACTLY",
            "AT_LEAST",
            "AT_MOST",
        }:
            k = params[
                "k"
            ]
            region = params[
                "region"
            ]
            region_text = (
                self._format_region(
                    region,
                    compact=compact,
                )
            )

            if compact:
                keyword = {
                    "EXACTLY": "Exactly",
                    "AT_LEAST": "At least",
                    "AT_MOST": "At most",
                }[
                    clue_type
                ]

                return (
                    f"{keyword} {k} in "
                    f"{region_text}"
                )

            phrase = {
                "EXACTLY": "Exactly",
                "AT_LEAST": "At least",
                "AT_MOST": "At most",
            }[
                clue_type
            ]

            return (
                f"{phrase} {k} Criminal(s) "
                f"in {region_text}."
            )

        if clue_type == "PARITY":
            parity = params[
                "parity"
            ]

            region = params[
                "region"
            ]

            region_text = (
                self._format_region(
                    region,
                    compact=compact,
                )
            )

            if compact:
                return (
                    f"{parity} parity in "
                    f"{region_text}"
                )

            return (
                f"The number of Criminals in "
                f"{region_text} is {parity}."
            )

        if clue_type == "IMPLIES":
            antecedent = params[
                "antecedent"
            ]
            consequent = params[
                "consequent"
            ]

            a_person = antecedent[
                "person"
            ]
            a_status = antecedent[
                "status"
            ]
            c_person = consequent[
                "person"
            ]
            c_status = consequent[
                "status"
            ]

            if compact:
                return (
                    f"{a_person}={_status_short(a_status)} "
                    f"-> "
                    f"{c_person}={_status_short(c_status)}"
                )

            return (
                f"If {a_person} is {a_status.value}, "
                f"then {c_person} is {c_status.value}."
            )

        # Defensive fallback for future extension clues.
        return (
            f"{clue_type}: "
            f"{params}"
        )

    def _format_region(
        self,
        region: Region,
        *,
        compact: bool = False,
    ) -> str:
        region_type = (
            _enum_value(
                region.type
            ).upper()
        )

        value = region.value

        if region_type == "ROW":
            return (
                f"row {value}"
                if not compact
                else f"R{value}"
            )

        if region_type == "COLUMN":
            return (
                f"column {value}"
                if not compact
                else f"C{value}"
            )

        if region_type == "NEIGHBORS":
            return (
                f"neighbors of {value}"
                if not compact
                else f"N({value})"
            )

        if region_type == "EXPLICIT":
            cells = ", ".join(
                value
            )

            return (
                f"cells {{{cells}}}"
                if not compact
                else f"{{{cells}}}"
            )

        return str(
            value
        )

    def referenced_cells(
        self,
        clue: Clue,
    ) -> tuple[str, ...]:
        """
        Return exactly the characters/cells referenced or counted by a
        revealed clue for GUI highlighting.
        """
        if self.engine is None:
            return ()

        clue_type = (
            _enum_value(
                clue.type
            ).upper()
        )

        params = clue.params

        if clue_type == "FACT":
            return (
                params["person"],
            )

        if clue_type in {
            "SAME",
            "DIFFERENT",
        }:
            return tuple(
                params["people"]
            )

        if clue_type in {
            "EXACTLY",
            "AT_LEAST",
            "AT_MOST",
            "PARITY",
        }:
            region = params[
                "region"
            ]

            return (
                resolve_region_cells(
                    region,
                    self.engine.size,
                )
            )

        if clue_type == "IMPLIES":
            return (
                params[
                    "antecedent"
                ]["person"],
                params[
                    "consequent"
                ]["person"],
            )

        return ()

    # ========================================================
    # Detail / progress rendering
    # ========================================================

    def _refresh_character_detail(
        self,
        state,
    ) -> None:
        character_id = (
            self.selected_character_id
        )

        if (
            character_id is None
            or character_id
            not in state.characters
        ):
            self._set_text(
                self.character_detail,
                (
                    "Select a card to inspect its public information "
                    "or submit a manual verdict."
                ),
            )
            return

        character = (
            state.characters[
                character_id
            ]
        )

        status = (
            state.proved_statuses.get(
                character_id
            )
        )

        clue = (
            self.engine.get_revealed_clue(
                character_id
            )
            if self.engine is not None
            else None
        )

        text = (
            f"Cell: {character_id}\n"
            f"Name: {character.name}\n"
            f"Profession: {character.profession}\n"
            "Status: "
            f"{status.value if status is not None else 'UNKNOWN'}\n"
            "Card: "
            f"{'FACE-UP' if clue is not None else 'FACE-DOWN'}\n"
        )

        if clue is not None:
            text += (
                f"\nClue {clue.id}:\n"
                f"{self.format_clue(clue)}"
            )
        else:
            text += (
                "\nClue: HIDDEN"
            )

        self._set_text(
            self.character_detail,
            text,
        )

    def _refresh_progress(
        self,
        state,
    ) -> None:
        solved = len(
            state.proved_statuses
        )
        total = len(
            state.characters
        )
        clues = len(
            state.revealed_clues
        )

        sat_calls = (
            self.agent.checker
            .metrics.sat_calls
            if self.agent is not None
            else 0
        )

        self.progress_var.set(
            f"Solved {solved}/{total}  |  "
            f"Revealed {clues}/{total}  |  "
            f"Session SAT calls {sat_calls}"
        )

    def _refresh_control_states(
        self,
        state,
    ) -> None:
        # Replay is intentionally isolated from manual interaction so
        # the public state cannot be changed while timed playback is in
        # progress.  The replay button remains available as Pause/Resume.
        if self.replay_active:
            self.load_button.configure(
                state="disabled"
            )
            self.restart_button.configure(
                state="disabled"
            )
            self.puzzle_combo.configure(
                state="disabled"
            )

            for button in (
                self.criminal_button,
                self.innocent_button,
                self.hint_button,
                self.step_button,
                self.auto_button,
            ):
                button.configure(
                    state="disabled"
                )

            self.replay_button.configure(
                state="normal"
            )
            return

        self.load_button.configure(
            state="normal"
        )
        self.restart_button.configure(
            state="normal"
        )
        self.puzzle_combo.configure(
            state="readonly"
        )

        solved = (
            len(state.proved_statuses)
            == len(state.characters)
        )

        selected_unresolved = (
            self.selected_character_id
            is not None
            and self.selected_character_id
            not in state.proved_statuses
        )

        manual_state = (
            "normal"
            if selected_unresolved
            else "disabled"
        )

        agent_state = (
            "disabled"
            if solved
            else "normal"
        )

        self.criminal_button.configure(
            state=manual_state
        )
        self.innocent_button.configure(
            state=manual_state
        )

        self.hint_button.configure(
            state=agent_state
        )
        self.step_button.configure(
            state=agent_state
        )
        self.auto_button.configure(
            state=agent_state
        )

        self.replay_button.configure(
            state=(
                "normal"
                if self.replay_available
                else "disabled"
            )
        )

    # ========================================================
    # Trace formatting
    # ========================================================

    def _format_hint_trace(
        self,
        hint,
    ) -> str:
        analysis = (
            hint.analysis
        )

        return (
            "HINT\n"
            f"Target: {hint.character_id}\n"
            f"Forced verdict: {hint.status.value}\n"
            "SAT queries:\n"
            f"  KB + ({hint.character_id}=INNOCENT): "
            f"{_sat_label(analysis.assume_innocent_result)}\n"
            f"  KB + ({hint.character_id}=CRIMINAL): "
            f"{_sat_label(analysis.assume_criminal_result)}\n"
            "Hint does not modify the game state.\n"
        )

    def _format_step_trace(
        self,
        step: AgentStep,
    ) -> str:
        analysis = (
            step.analysis
        )

        active = (
            ", ".join(
                step.active_clue_ids
            )
            if step.active_clue_ids
            else "(none)"
        )

        revealed = (
            step.revealed_clue.id
            if step.revealed_clue
            is not None
            else "(none)"
        )

        return (
            f"STEP {step.step_number}\n"
            f"Active clues before deduction: {active}\n"
            f"Character: {step.character_id}\n"
            "SAT queries:\n"
            f"  KB + ({step.character_id}=INNOCENT): "
            f"{_sat_label(analysis.assume_innocent_result)}\n"
            f"  KB + ({step.character_id}=CRIMINAL): "
            f"{_sat_label(analysis.assume_criminal_result)}\n"
            f"Verdict: {step.status.value} "
            f"[{step.verdict_code.value}]\n"
            f"Newly revealed clue: {revealed}\n"
            "Step workload:\n"
            f"  SAT calls: {step.sat_calls}\n"
            f"  Decisions: {step.decisions}\n"
            f"  Propagations: {step.propagations}\n"
            f"  Backtracks: {step.backtracks}\n"
            f"  DPLL runtime: {step.runtime * 1000.0:.3f} ms\n"
        )

    # ========================================================
    # Generic UI helpers
    # ========================================================

    def _append_trace(
        self,
        text: str,
    ) -> None:
        self.trace_text.configure(
            state="normal"
        )

        if (
            self.trace_text.index(
                "end-1c"
            )
            != "1.0"
        ):
            self.trace_text.insert(
                tk.END,
                "\n"
                + "=" * 72
                + "\n\n",
            )

        self.trace_text.insert(
            tk.END,
            text.rstrip()
            + "\n",
        )

        self.trace_text.see(
            tk.END
        )

        self.trace_text.configure(
            state="disabled"
        )

    def _clear_trace(
        self,
    ) -> None:
        self.trace_text.configure(
            state="normal"
        )
        self.trace_text.delete(
            "1.0",
            tk.END,
        )
        self.trace_text.configure(
            state="disabled"
        )

    @staticmethod
    def _set_text(
        widget: tk.Text,
        text: str,
    ) -> None:
        widget.configure(
            state="normal"
        )
        widget.delete(
            "1.0",
            tk.END,
        )
        widget.insert(
            "1.0",
            text,
        )
        widget.configure(
            state="disabled"
        )

    def _set_status(
        self,
        text: str,
    ) -> None:
        self.status_var.set(
            text
        )

    def _set_controls_enabled(
        self,
        enabled: bool,
    ) -> None:
        state = (
            "normal"
            if enabled
            else "disabled"
        )

        for button in (
            self.load_button,
            self.restart_button,
            self.criminal_button,
            self.innocent_button,
            self.hint_button,
            self.step_button,
            self.auto_button,
        ):
            button.configure(
                state=state
            )

        self.replay_button.configure(
            state=(
                "normal"
                if enabled
                and self.replay_available
                else "disabled"
            )
        )

        self.puzzle_combo.configure(
            state=(
                "readonly"
                if enabled
                else "disabled"
            )
        )

    def _show_runtime_error(
        self,
        title: str,
        error: Exception,
    ) -> None:
        self._append_trace(
            f"{title.upper()}\n"
            f"ERROR: {type(error).__name__}: {error}\n"
        )

        self._set_status(
            f"{title} failed: {error}"
        )

        messagebox.showerror(
            title,
            str(error),
        )

    @staticmethod
    def _cell_sort_key(
        cell_id: str,
    ) -> tuple[int, int]:
        try:
            column = (
                ord(cell_id[0].upper())
                - ord("A")
            )
            row = (
                int(cell_id[1:])
                - 1
            )
        except (
            IndexError,
            ValueError,
        ):
            return (
                10_000,
                10_000,
            )

        return (
            row,
            column,
        )


# ============================================================
# Entrypoint
# ============================================================

def main() -> None:
    root = tk.Tk()

    try:
        ttk.Style(
            root
        ).theme_use(
            "clam"
        )
    except tk.TclError:
        # Keep the platform default theme if "clam" is unavailable.
        pass

    GriductiveApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()