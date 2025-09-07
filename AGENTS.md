# Claude Code Development Guide

**IMPORTANT NOTE TO CLAUDE:** ALWAYS Run all three checks whenever you write any new code. See commands below.

# CLI Tools

## Development Tools

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

# Django App Commands

All Django app commands are run with `python manage.py <command_name> <options>`.

## Playing Games

### Run Algorithm (`run_algorithm`)

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

**Params:**

- `--game-id <str>`: Run on an existing game (must be in `running` status`). If omitted, the command lists the latest running games and prompts for an ID.
- `--scenario <int>`: Create a new episode for the given scenario (choices: 1, 2, 3) when `--game-id` is not provided.
- `--server`: With `--scenario`, create a server-side `RemoteGame` instead of a local `LocalGame`. Ignored when `--game-id` is used.
- `--n-games <int>`: Number of new episodes to run (requires `--scenario`). Episodes are created sequentially and run to completion one-by-one.
- `--algorithm <str>`: Algorithm to use (see `bouncer/algorithms.py`). If omitted, you will be prompted to choose.
- `--delay <float>`: Seconds to sleep between decisions (default `0.0`).
- `--model-path <str>`: Optional model file path for model-based algorithms (e.g., PPO).
- `--quiet`: Optional flag to silence output with bouncer decisions (the most noise of the logs). You still get notified dof game start, end, run count, errors, etc.

**Notes:**

- You can inspect available algorithms by viewing `bouncer/algorithms.py` (`ALGORITHMS` dictionary). If you omit `--algorithm`, the command will list available names and prompt.
- Remote rate limiting: When using `--server`, the command enforces a rolling rate limit of at most 10 `RemoteGame` creations per 15 minutes. If the limit is reached, it prints a status message and sleeps until the limit would be passed, rechecking on wake (to gaurd against race conditions where this command is running in other terminals that are waiting in an attempt to do the same). This allows continuous operation without exhausting the server quota.
- Restrictions: Completed/failed/error games cannot be restarted. If a running game has no pending people, the command marks the game status as `error` and exits.

## Model Training + Eval

### Train PPO (`train_ppo`)

Train a PPO policy on the simulated environment.

```bash
python manage.py train_ppo --scenario 1 --total-timesteps 200000 --n-envs 8 --log-dir runs/ppo_sim --save-path models/ppo_sim.zip [--init-from models/bc_init.zip]
```

**Required Params:**

- `--scenario <int>`: Scenario to train on (choices: 1, 2, 3).
- `--total-timesteps <int>`: Timesteps to train when not using curriculum.
- `--n-envs <int>`: Number of parallel envs (vectorized training).
- `--log-dir <str>`: Directory for eval logs and checkpoints.
- `--save-path <str>`: Output PPO `.zip` path.

**Optional Params:**

- `--init-from <str>`: Optional PPO `.zip` to initialize from (e.g., BC pretrain or continue training).
- `--seed <int>`: Random seed.
- `--curriculum <str>`: Comma-separated capacities (e.g., `200,400,700,1000`) to stage training.
- `--stage-steps <int>`: Timesteps per curriculum stage.
- `--no-vecnorm`: Disable reward normalization (VecNormalize).
- `--gamma <float>` / `--gae-lambda <float>` / `--n-steps <int>` / `--ent-coef <float>`: PPO hyperparameters.
- Reward shaping (training only; eval uses true rewards):
  - `--shape-coef <float>`: Dense reward for reducing total deficits between steps.
  - `--nonhelp-penalty <float>`: Extra penalty when an accept does not reduce any deficits while deficits remain.
  - `--success-bonus <float>`: Fixed bonus added on successful completion (at capacity with all minima met).
  - `--minmeet-bonus <float>`: Per-attribute bonus when a minimum is first met on that step.
  - `--fail-penalty-scale <float>`: Scale the terminal penalty for failing at capacity due to unmet minima; effective penalty is `-s * REJECTION_LIMIT` (default `1.0`).
  - `--success-bonus-per-saved <float>`: Adds `k * (REJECTION_LIMIT - rejected)` at success to reward saving rejections.
  - `--late-reject-weight <float>`: Extra penalty on reject steps weighted by low slack: subtracts `w * (1 - slack_frac)`.
- Eval selection:
  - `--eval-freq <int>` / `--eval-episodes <int>`: Eval cadence and episodes per eval.
  - `--eval-percentile <float>`: If > 0 (e.g., `90` or `95`), selects the best checkpoint by that reward percentile instead of mean.

**Notes:**

- Curriculum scales minimum counts proportionally to the staged capacity.
- When `--init-from` is provided, weights are loaded into a fresh PPO to match current rollout shape/hyperparams.
- Training reward shaping applies only to the training envs. Evaluation envs are unshaped (true task rewards). If `--eval-percentile` is set, percentile is computed over true rewards.

### Pretrain BC (`pretrain_bc`)

Pretrain PPO model on existing games that have been exported with `export_dataset`. (Behavioral cloning pretrain for PPO’s policy network.)

```bash
python manage.py pretrain_bc --datasets ds1.npz,ds2.npz --out models/bc_init.zip [--epochs 5 --batch-size 1024 --lr 3e-4 --val-split 0.1]
```

**Params:**

- `--datasets <str>`: Comma-separated NPZ files containing `obs` and `actions`.
- `--out <str>`: Output PPO `.zip` with initialized policy weights.
- `--epochs <int>` / `--batch-size <int>` / `--lr <float>` / `--val-split <float>`: Training options and validation split.

**Notes:**

- Initializes PPO policy via cross-entropy on expert actions; value network is frozen during pretrain.
- Output can be fed to `train_ppo` via `--init-from`.

### Eval PPO (`eval_ppo`)

Evaluate a saved PPO model quickly in the in-memory simulator (no database I/O). Runs several episodes in `SimBerghainEnv` and reports summary stats.

```bash
python manage.py eval_ppo --model-path models/ppo_model.zip --scenario 2 [--episodes 100] [--deterministic] [--seed 123]
```

**Params:**

- `--model-path <str>`: Path to a Stable-Baselines3 PPO `.zip` model.
- `--scenario <int>`: Scenario to evaluate (choices: 1, 2, 3).
- `--episodes <int>`: Number of episodes to roll out (default `50`).
- `--seed <int>`: Base RNG seed; each episode uses `seed + ep` (default `123`).
- `--deterministic`: If set, select greedy actions; otherwise sample stochastically.

**Notes:**

- Evaluation uses true task rewards (no shaping wrapper), matching `train_ppo` eval.
- Summary includes mean/std of reward and episode length, mean/std admitted/rejected, and counts of outcomes (`success`, `constraints_unmet_at_capacity`, `rejection_limit`).
- Deterministic: Picks the most probable action (argmax of the policy’s categorical distribution). For PPO with Discrete(2), it always chooses the action with the higher logit. Produces stable, repeatable behavior and is what you typically want for deployment or head‑to‑head comparisons.
- Stochastic: Samples from the policy’s action distribution (softmax over logits). Adds variability across runs/steps, giving an unbiased estimate of the policy’s expected return and revealing how “confident” or sharp the policy is. Results with stochastic depend on RNG state, so use more episodes for stable averages.
- Deterministic is good for final evaluation, reproducible metrics, and live play (our ppo_bouncer uses deterministic=True). Stochastic is good for diagnostics and robustness checks; estimating expected return over many episodes;seeing whether the policy relies on probabilistic choices.
- If the policy is confident (peaked distribution), deterministic and stochastic behave similarly. With exploration/entropy during training, the learned policy can remain somewhat stochastic; deterministic eval can be slightly better (no unlucky samples), but stochastic gives the true expected performance.

### Compare Policies (`compare_policies`)

Compare a trained PPO model’s decisions to an exported expert dataset and report agreement on decisions.

```bash
python manage.py compare_policies --dataset path/to/export.npz --model-path path/to/model.zip [--limit N]
```

**Params:**

- `--dataset <str>`: Path to `.npz` produced by `export_dataset` (must contain `obs` and `actions`).
- `--model-path <str>`: Path to a Stable-Baselines3 PPO `.zip` model.
- `--limit <int>`: Optional, only compare the first N steps.

**Output:**

- Agreement/Disagreement rates between expert `actions` and model predictions.
- Needed-overlap thresholds: for each k ≥ 1, accept rates when a person overlaps with at least k currently-needed attributes (derived from `obs`).
- Zero-overlap steps: reject rates when no currently-needed attributes are present.

**Notes:**

- "Overlap" is when the person has an attribute that is still needed (we haven't hit the minima for). So when we say it accepted 90% of people with k >=4 overlap, it means it accepted 90% of people it say who had 4 or more traits we still needed.

### Export Dataset (`export_dataset`)

Export imitation-learning dataset from games stored in the DB.

```bash
python manage.py export_dataset --out data/export.npz [--scenarios 1,2] [--statuses running,completed] [--games <id1>,<id2>]
```

**Params:**

- `--out <str>`: Output `.npz` path (writes `obs`, `actions`, `episodes`).
- `--scenarios <str>`: Comma-separated scenarios to include; empty=all.
- `--statuses <str>`: Comma-separated statuses to include; empty=all.
- `--games <str>`: Comma-separated specific `game_id`s to include; empty=all.
- `--tags <str>`: Comma-separated tags to filter games by (matches games whose tags contain all provided). empty=all.

Notes:

- Stops each episode at the first pending person to avoid overlapping future state.
- Observations match the RL environment’s feature layout.

## Utils

### Test Correlations (`test_correlations`)

Test whether a population of Persons matches a given set of target population distribution metrics (`relativeFrequency` and `correlations` between traits). Used for validating that the attribute generation utility is functioning accurately. Also can be used to look at populations from Games in the database to validate that those games were played with valid Person distribution.

```bash
python manage.py test_correlations [--sample-size 200000] [--test-real-games]
```

**Params:**

- `--sample-size <int>`: Number of synthetic people to generate (default 200000).
- `--test-real-games`: Validate against real data stored in the DB.

### Export Game CSV (`export_game_csv`)

Export a game's people to CSV for analysis. Columns include person index, decision, decision text, created_at, and one column per attribute present in the game.

```bash
python manage.py export_game_csv <GAME_ID> [--output game_<GAME_ID>.csv]
```

**Params:**

- `<GAME_ID>`: Required game UUID.
- `--output <str>`: Output CSV path (default `game_<GAME_ID>.csv`).

### Oracle Baseline (`oracle_baseline`)

Estimate the best-achievable (lower-bound) rejections with perfect foresight for a scenario by finding the earliest stream prefix where it is possible to select exactly `CAPACITY` people that satisfy all minima.

```bash
python manage.py oracle_baseline --scenario 2 [--episodes 200] [--seed 123]
```

**Params:**

- `--scenario <int>`: Scenario to evaluate (choices: 1, 2, 3).
- `--episodes <int>`: Number of independent trials to simulate (default `200`).
- `--seed <int>`: Base RNG seed; each episode uses an offset of this seed (default `123`).

**Behavior:**

- For each episode, samples a stream of people via the in-memory generator and binary-searches the minimum prefix length `t*` such that there exists a subset of exactly `CAPACITY` people within the first `t*` that meets all per-attribute minima.
- Feasibility is checked via OR-Tools CP-SAT. It pre-filters obvious infeasible prefixes and uses a constructive greedy to short-circuit easy feasible cases.
- Outputs per-episode progress and a summary of lower-bound rejections `t* - CAPACITY` (mean/std, p90/p95/p99).

**Notes:**

- Requires `ortools` (already listed in `requirements.txt`). The command raises an error if CP-SAT returns an unknown/timeout status rather than silently treating it as infeasible.
- Feasibility is modeled on “useful” people (those with at least one constrained attribute); remaining slots can always be filled with unconstrained “filler” people to reach exactly `CAPACITY` if minima are satisfied.
