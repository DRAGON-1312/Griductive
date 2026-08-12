# Griductive Solver

A no-guess Griductive solver for **CSC14003 – Introduction to Artificial Intelligence**.

The project models Griductive with propositional logic, automatically converts revealed clues into CNF, and uses a custom DPLL SAT solver to prove whether each unresolved character is **CRIMINAL**, **INNOCENT**, or still **UNKNOWN**.  
The GUI supports manual play, hints, single-step deduction, full auto solving, clue highlighting, and step-by-step solution replay.

---

# 1. How to Run

## 1.1. Requirements

- Python 3
- Windows, macOS, or Linux
- Required Python packages listed in `requirements.txt`

> On Windows, the commands below use the Python launcher `py`.  
> If `py` is unavailable on your system, replace it with `python`.

---

## 1.2. Step-by-step Setup

### Step 1 — Extract or clone the project

If you received the submitted archive, extract it first.

The project root should contain:

```text
Griductive/
├── Source/
├── README.md
└── requirements.txt
```

If using Git:

```powershell
git clone https://github.com/DRAGON-1312/Griductive.git
cd Griductive
```

---

### Step 2 — Install dependencies

From the project root:

```powershell
py -m pip install -r requirements.txt
```

---

### Step 3 — Enter the source directory

```powershell
cd Source
```

You should now be inside:

```text
Griductive/Source/
```

---

### Step 4 — Run the application

```powershell
py main.py
```

The Griductive desktop GUI should open.

`main.py` is the official application entry point. It validates the required project structure and then launches `gui/app.py`.

---

## 1.3. Basic GUI Usage

After the GUI opens:

1. Select a puzzle from the **Puzzle** dropdown.
2. Click **Load**.
3. Click a character card to inspect its public information.
4. Use **Criminal** or **Innocent** to submit a manual verdict.
5. Use **Hint** to find a currently provable character without changing the game state.
6. Use **Agent Step** to execute exactly one logical deduction.
7. Use **Auto Solve** to let the Logic Agent solve the remaining puzzle automatically.
8. After a solution is available, use **Replay Solution** to replay the deductions step-by-step.
9. During replay, use **Pause Replay / Resume Replay** when needed.
10. Click a revealed clue in the clue list to highlight the cells referenced or counted by that clue.
11. Use **Restart** to return the current puzzle to its initial public state.

### Manual verdict outcomes

The GUI distinguishes three important outcomes:

- **ACCEPTED** — the submitted verdict is logically entailed by the current public knowledge base. The character is marked and its clue is revealed.
- **NOT_PROVABLE** — neither status is currently forced. The game state is unchanged.
- **CONTRADICTED** — the opposite verdict is logically forced. The game state is unchanged.

The solver never guesses.

---

# 2. Optional Validation Commands

These commands are useful for checking the project before submission.

From `Griductive/Source/`:

### Syntax-check the main application

```powershell
py -m py_compile main.py
py -m py_compile gui\app.py
```

### Run the complete automated test suite

```powershell
py -m pytest -v
```

All tests should pass before the project is packaged or demonstrated.

---

# 3. Optional Experiment Commands

The project also contains a reproducible benchmark pipeline.

Run experiments from `Griductive/Source/`.

### Quick smoke test

```powershell
py -m experiments.run_experiments --repetitions 2 --warmups 1
```

### Full benchmark

```powershell
py -m experiments.run_experiments --repetitions 30 --warmups 3
```

The runner records:

- primary and auxiliary variables;
- CNF clause counts;
- SAT calls;
- DPLL decisions;
- propagations;
- backtracks;
- deduction steps;
- solver runtime;
- end-to-end runtime;
- successful/failed runs.

### Generate report-ready plots

After an experiment run:

```powershell
py -m experiments.plot_results
```

The plotting script reads the generated CSV files and creates report-ready PNG/SVG figures. It does not rerun the solver.

---

# 4. Project Structure

```text
Griductive/
│
├── README.md
├── requirements.txt
│
└── Source/
    │
    ├── main.py
    │
    ├── core/
    │   ├── models.py
    │   ├── puzzle_loader.py
    │   └── game_engine.py
    │
    ├── logic/
    │   ├── __init__.py
    │   ├── semantic_evaluator.py
    │   ├── cnf_encoder.py
    │   ├── dpll.py
    │   ├── entailment.py
    │   ├── uniqueness.py
    │   └── agent.py
    │
    ├── gui/
    │   └── app.py
    │
    ├── puzzles/
    │   ├── puzzle_3x3_01.json
    │   ├── puzzle_3x3_02.json
    │   ├── puzzle_4x4_01.json
    │   ├── puzzle_4x4_02.json
    │   ├── puzzle_5x5_01.json
    │   └── puzzle_5x5_02.json
    │
    ├── experiments/
    │   ├── metrics.py
    │   ├── run_experiments.py
    │   └── plot_results.py
    │
    └── tests/
        ├── test_puzzle_loader.py
        ├── test_semantic_evaluator.py
        ├── test_cnf_encoder.py
        ├── test_dpll.py
        ├── test_entailment.py
        ├── test_uniqueness.py
        ├── test_agent.py
        └── test_puzzle_dataset.py
```

Generated folders such as `__pycache__/`, `.pytest_cache/`, and experiment result folders are runtime artifacts and are not part of the core source architecture.

---

# 5. File Responsibilities

## 5.1. Project root

| File | Responsibility |
|---|---|
| `README.md` | Project run instructions, structure, and module overview. |
| `requirements.txt` | Python dependencies required to run, test, and evaluate the project. |

---

## 5.2. `Source/main.py`

| File | Responsibility |
|---|---|
| `main.py` | Official application entry point. Validates the packaged project structure, configures the import path when necessary, and launches the GUI through `gui.app`. |

`main.py` intentionally contains no SAT, deduction, or game logic. Those responsibilities remain in their dedicated modules.

---

## 5.3. `Source/core/`

The `core` package owns puzzle data and game-state management.

| File | Responsibility |
|---|---|
| `models.py` | Defines the main domain models and enums used across the project, including characters, clues, regions, statuses, classifications, public state, and verdict codes. |
| `puzzle_loader.py` | Loads and validates puzzle JSON files. It validates board structure, clue parameters, referenced characters/regions, and resolves row/column/neighborhood/explicit regions. |
| `game_engine.py` | Owns the complete puzzle, hidden statuses, and unrevealed clues. Maintains the public game state, validates submitted verdicts, reveals clues after accepted verdicts, and exposes only public information to the GUI and Logic Agent. |

### Important rule

`GameEngine` is the only layer that owns the hidden solution and unrevealed clue content during gameplay.

---

## 5.4. `Source/logic/`

The `logic` package implements the logical representation, SAT solver, entailment checks, uniqueness checking, and the automatic agent.

| File | Responsibility |
|---|---|
| `__init__.py` | Marks and initializes the `logic` package. |
| `semantic_evaluator.py` | Directly evaluates clue semantics under a complete assignment without using CNF. Used for independent semantic validation of clue implementations. |
| `cnf_encoder.py` | Deterministically maps characters to propositional variables and automatically converts active clues into CNF. Also adds proved statuses as unit clauses when building the current knowledge base. |
| `dpll.py` | Custom deterministic DPLL SAT solver implementing unit propagation, conflict detection, branching, backtracking, pure-literal elimination, complete SAT assignments, and solver workload metrics. |
| `entailment.py` | Uses SAT queries to classify an unresolved character as `CRIMINAL`, `INNOCENT`, `UNKNOWN`, or detect an inconsistent KB. Also aggregates SAT workload metrics. |
| `uniqueness.py` | Checks whether the complete clue set has exactly one primary-variable solution by solving once, blocking the first primary assignment, and solving again. |
| `agent.py` | Implements the no-guess Logic Agent. Scans unresolved characters deterministically, chooses only logically forced verdicts, requests clue reveals through `GameEngine`, supports Hint / Agent Step / Auto Solve, and records deduction traces and workload metrics. |

---

# 6. Supported Logic

## 6.1. Core clue types

The project supports all required core clue templates:

- `FACT`
- `SAME`
- `DIFFERENT`
- `EXACTLY`
- `AT_LEAST`
- `AT_MOST`

## 6.2. Implemented extensions

The project also implements two additional clue types:

- `PARITY`
- `IMPLIES`

## 6.3. Supported regions

Counting and parity clues can use:

- `ROW`
- `COLUMN`
- `NEIGHBORS`
- `EXPLICIT`

---

# 7. Knowledge-Base and No-Guess Architecture

At deduction step `t`, the public knowledge base contains only:

```text
KB_t =
    CNF(revealed clues)
    +
    unit clauses for previously proved statuses
```

Hidden statuses and unrevealed clues are never inserted into the Logic Agent's knowledge base.

For an unresolved character `i`:

```text
KB_t ∧ ¬C_i  is UNSAT  ->  CRIMINAL
KB_t ∧  C_i  is UNSAT  ->  INNOCENT
both are SAT            ->  UNKNOWN
KB_t itself is UNSAT    ->  INCONSISTENT
```

Therefore, the automatic solver derives only logically entailed verdicts and never guesses.

---

# 8. `Source/gui/app.py`

`gui/app.py` implements the playable desktop interface.

Main GUI functionality:

- displays the `N x N` board;
- shows row and column coordinates;
- shows character names and professions;
- distinguishes `UNKNOWN`, `CRIMINAL`, and `INNOCENT`;
- distinguishes face-up and face-down clues;
- supports manual Criminal/Innocent verdicts;
- reports `ACCEPTED`, `NOT_PROVABLE`, and `CONTRADICTED`;
- supports puzzle Load and Restart;
- supports SAT-based Hint;
- supports one-step Agent deduction;
- supports full Auto Solve;
- displays the deduction trace;
- allows revealed clues to be selected;
- highlights cells referenced or counted by a selected clue;
- supports step-by-step solution replay with pause/resume.

The GUI does not directly inspect hidden statuses or unrevealed clues and does not call the CNF encoder or DPLL solver directly.

---

# 9. `Source/puzzles/`

The project includes six validated benchmark puzzles.

| Puzzle | Purpose |
|---|---|
| `puzzle_3x3_01.json` | Basic sanity case for simple binary clue reasoning. |
| `puzzle_3x3_02.json` | Counting-clue and region-coverage case. |
| `puzzle_4x4_01.json` | Full integration of the six required core clue types with longer deduction dependencies. |
| `puzzle_4x4_02.json` | Extension integration using `PARITY`, `IMPLIES`, and core clues. |
| `puzzle_5x5_01.json` | Moderate scalability stress with larger counting regions and a larger CNF. |
| `puzzle_5x5_02.json` | Hard scalability stress with dense counting, extensions, many UNKNOWN scans, and higher DPLL branching pressure. |

Each benchmark puzzle is validated for:

- valid JSON/puzzle structure;
- all clues being true under the intended complete assignment;
- satisfiable complete clue CNF;
- a unique primary-variable solution;
- progressive no-guess solvability by `LogicAgent`.

---

# 10. `Source/experiments/`

The `experiments` package evaluates the solver on the validated benchmark suite.

| File | Responsibility |
|---|---|
| `metrics.py` | Defines static puzzle/CNF metrics, per-run solver metrics, runtime statistics, failure records, and repeated-run summaries. |
| `run_experiments.py` | Runs benchmark puzzles repeatedly using fresh `GameEngine` and `LogicAgent` instances. Records raw runs, summaries, static puzzle metrics, runtime data, failures, and environment metadata. |
| `plot_results.py` | Reads frozen experiment CSV results and generates report-ready charts and a compact report table without rerunning the solver. |

Typical experiment outputs include:

```text
summary.csv
raw_runs.csv
puzzle_metrics.csv
metadata.json
report_table.csv
figures/
```

---

# 11. `Source/tests/`

The automated test suite validates individual modules and the complete benchmark pipeline.

| File | Responsibility |
|---|---|
| `test_puzzle_loader.py` | Tests valid and invalid puzzle loading, clue validation, and region resolution. |
| `test_semantic_evaluator.py` | Tests direct semantic evaluation for supported clue types and validation errors. |
| `test_cnf_encoder.py` | Tests deterministic variable mapping, clue-to-CNF encoding, KB construction, and encoding validation. |
| `test_dpll.py` | Tests SAT/UNSAT behavior, unit propagation, branching, backtracking, deterministic solving, assumptions, and DPLL metrics. |
| `test_entailment.py` | Tests SAT-based character classification, inconsistent KB detection, and standardized SAT workload metrics. |
| `test_uniqueness.py` | Tests the two-solve blocking-clause uniqueness procedure and primary-assignment extraction. |
| `test_agent.py` | Tests Hint, single-step deduction, no-guess behavior, Auto Solve, reveal protocol, deduction trace, and full workload aggregation. |
| `test_puzzle_dataset.py` | End-to-end validation of all benchmark puzzles: loading, clue semantics, CNF satisfiability, uniqueness, progressive no-guess solving, deterministic deduction order, and dataset coverage. |

---

# 12. High-Level Architecture

```text
                        main.py
                           |
                           v
                       GUI / app.py
                           |
              +------------+------------+
              |                         |
              v                         v
          GameEngine                LogicAgent
              |                         |
              |                         v
              |                 EntailmentChecker
              |                         |
              |                         v
              |                    CNFEncoder
              |                         |
              |                         v
              +----------------------> DPLL
```

Responsibility separation:

```text
GameEngine
    owns hidden truth and unrevealed clues

LogicAgent
    sees only public state
    never guesses

CNFEncoder
    builds the current logical KB automatically

DPLL
    answers SAT / UNSAT queries

GUI
    displays public state and sends user/agent actions
```

This separation ensures that hidden information is never used by the Logic Agent to choose a verdict or hint.

---

# 13. Recommended Final Run Check

Before packaging the submission:

```powershell
cd Source

py -m pytest -v
py main.py
```

Confirm that:

- all tests pass;
- all six puzzles load;
- manual verdicts work;
- Hint works;
- Agent Step works;
- Auto Solve completes validated puzzles;
- clue highlighting works;
- Replay Solution works step-by-step;
- Restart returns the puzzle to its initial public state.

