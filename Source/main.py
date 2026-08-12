from __future__ import annotations

"""
Griductive Solver
=================

Top-level application entry point.

Recommended usage from the Source/ directory:

    py main.py

The actual GUI implementation lives in gui/app.py.  Keeping this file
small makes main.py responsible only for launching the application,
while GameEngine, LogicAgent, SAT solving, and GUI behavior remain in
their dedicated modules.
"""

import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent

# Make imports robust when main.py is executed directly from another
# working directory, for example:
#
#     py "C:\...\Griductive\Source\main.py"
#
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def _validate_project_layout() -> None:
    """
    Fail early with a clear message if the submitted source package is
    incomplete.

    This validation checks only public project files/directories. It
    does not load puzzle secrets or interact with the reasoning layer.
    """
    required_paths = (
        SOURCE_DIR / "core",
        SOURCE_DIR / "logic",
        SOURCE_DIR / "gui" / "app.py",
        SOURCE_DIR / "puzzles",
    )

    missing = tuple(
        path
        for path in required_paths
        if not path.exists()
    )

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise RuntimeError(
            "Griductive source package is incomplete. "
            "Missing required path(s):\n"
            f"{formatted}"
        )

    puzzle_files = tuple(
        (SOURCE_DIR / "puzzles").glob(
            "puzzle_*.json"
        )
    )

    if not puzzle_files:
        raise RuntimeError(
            "No puzzle_*.json files were found in "
            f"{SOURCE_DIR / 'puzzles'}."
        )


def main() -> int:
    """
    Validate the packaged project and launch the Griductive GUI.

    Returns:
        0 on normal application shutdown.
        1 if startup fails before the Tkinter event loop is entered.
    """
    try:
        _validate_project_layout()

        # Import only after validating the source layout so startup
        # errors are easier to diagnose in a packaged submission.
        from gui.app import main as run_gui

        run_gui()

    except KeyboardInterrupt:
        print(
            "\nGriductive Solver interrupted by user.",
            file=sys.stderr,
        )
        return 130

    except Exception as error:
        print(
            "Failed to start Griductive Solver: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )