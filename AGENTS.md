# Claude Code Development Guide

**IMPORTANT NOTE TO CLAUDE:** ALWAYS Run all three checks whenever you write any new code. See commands below.

## Commands

### Formatting

To format the codebase according to project standards:

```bash
ruff format .
```

### Linting

To check and automatically fix linting issues:

```bash
ruff check . --fix
```

To only check without fixing:

```bash
ruff check .
```

### Type Checking

To run static type analysis:

```bash
mypy .
```

### All Checks

To run all three tools in sequence:

```bash
ruff format . && ruff check . && mypy .
```

### Run Algorithm Command

To run a bouncer algorithm on a game:

```bash
python manage.py run_algorithm --game-id <GAME_ID> --algorithm <ALGORITHM_NAME>
```

- You can see the available `ALGORITHMS` in `algorithms.py`
- You can start a new game by opening Django shell with `python manage.py shell` and calling `.start_new_game(scenario_number)` on `RemoteGame` or `LocalGame` class, then you can pass this GAME_ID into the command.

**Notes:**

- The command can resume games from where they left off as long as the last person is pending (both locally and on the server)
- Games must be in "running" status (completed/failed/errored games cannot be restarted)
- The command shows real-time decision making and running totals

### Compare Policies

Compare a trained PPO model's actions to an exported expert dataset and report agreement plus neutral, need-overlap–conditioned metrics.

```bash
python manage.py compare_policies --dataset path/to/export.npz --model-path path/to/model.zip [--limit N]
```

- --dataset: Path to an .npz file produced by `export_dataset` (must contain `obs` and `actions`).
- --model-path: Path to a Stable-Baselines3 PPO `.zip` model to evaluate.
- --limit: Optional, only compare the first N steps after filtering.

**Output:**

- Agreement/Disagreement rates between expert `actions` and model predictions (deterministic).
- Needed-overlap thresholds: for each k = 1..max, shows accept rates when a person overlaps with at least k currently-needed attributes (derived from `obs`).
- Zero-overlap steps: reject rates when no currently-needed attributes are present.

**Notes:**

- "Overlap" is when the person has an attribute that is still needed (we haven't hit the minima for). So when we say it accepted 90% of people with k >=4 overlap, it means it accepted 90% of people it say who had 4 or more traits we still needed.
