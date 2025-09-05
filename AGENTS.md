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

Run a bouncer algorithm on existing or new game(s) and run to completion. Algorithms are defined in `bouncer/algorithms.py` and exposed via the `ALGORITHMS` map.

Common usage patterns:

```bash
# Run on an existing game (prompts for algorithm if omitted)
python manage.py run_algorithm --game-id <GAME_ID> --algorithm <ALGO>

# Start a new local game (LocalGame) for a scenario and run
python manage.py run_algorithm --scenario 2 --algorithm <ALGO>

# Start a new server-side (RemoteGame) episode and run
python manage.py run_algorithm --scenario 2 --server --algorithm <ALGO>

# Run multiple sequential episodes
python manage.py run_algorithm --scenario 2 --n-games 5 --algorithm <ALGO>

# Add a small delay between decisions for readability
python manage.py run_algorithm --scenario 1 --algorithm <ALGO> --delay 0.1

# Use a model-backed algorithm with a model path
python manage.py run_algorithm --scenario 3 --algorithm ppo --model-path path/to/model.zip
```

Flags:

- `--game-id <str>`: Run on an existing game (must be in `running` status). If omitted, the command lists the latest running games and prompts for an ID.
- `--scenario {1,2,3}`: Create a new episode for the given scenario when `--game-id` is not provided.
- `--server`: With `--scenario`, create a server-side `RemoteGame` instead of a local `LocalGame`. Ignored when `--game-id` is used.
- `--n-games <int>`: Number of new episodes to run (requires `--scenario`). Episodes are created sequentially and run to completion one-by-one.
- `--algorithm <str>`: Algorithm to use (see `bouncer/algorithms.py`). If omitted, you will be prompted to choose.
- `--delay <float>`: Seconds to sleep between decisions (default `0.0`).
- `--model-path <str>`: Optional model file path for model-based algorithms (e.g., PPO).

**Notes:**

- You can inspect available algorithms by viewing `bouncer/algorithms.py` (`ALGORITHMS` dictionary). If you omit `--algorithm`, the command will list available names and prompt.
- Remote rate limiting: When using `--server`, the command enforces a rolling rate limit of at most 10 `RemoteGame` creations per 15 minutes. If the limit is reached, it prints a status message and sleeps until the limit would be passed, rechecking on wake (to gaurd against race conditions where this command is running in other terminals that are waiting in an attempt to do the same). This allows continuous operation without exhausting the server quota.
- Restrictions: Completed/failed/error games cannot be restarted. If a running game has no pending people, the command marks the game status as `error` and exits.

### Compare Policies

Compare a trained PPO model's actions to an exported expert dataset and report agreement plus neutral, need-overlap–conditioned metrics.

```bash
python manage.py compare_policies --dataset path/to/export.npz --model-path path/to/model.zip [--limit N]
```

- `--dataset <str>`: Path to an .npz file produced by `export_dataset` (must contain `obs` and `actions`).
- `--model-path <str>`: Path to a Stable-Baselines3 PPO `.zip` model to evaluate.
- `--limit int`: Optional, only compare the first N steps after filtering.

**Output:**

- Agreement/Disagreement rates between expert `actions` and model predictions (deterministic).
- Needed-overlap thresholds: for each k = 1..max, shows accept rates when a person overlaps with at least k currently-needed attributes (derived from `obs`).
- Zero-overlap steps: reject rates when no currently-needed attributes are present.

**Notes:**

- "Overlap" is when the person has an attribute that is still needed (we haven't hit the minima for). So when we say it accepted 90% of people with k >=4 overlap, it means it accepted 90% of people it say who had 4 or more traits we still needed.
