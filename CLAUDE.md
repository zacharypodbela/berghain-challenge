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
