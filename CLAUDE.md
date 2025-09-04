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
