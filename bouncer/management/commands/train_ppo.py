from __future__ import annotations

import os
from typing import Any, Callable

from django.core.management.base import BaseCommand

# Third-party RL libs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from bouncer.constants import CAPACITY, SCENARIO_CONFIGS
from bouncer.rl.callbacks import PercentileEvalCallback
from bouncer.rl.env import ATTRIBUTE_ORDER, DeficitRewardWrapper, SimBerghainEnv


def _validated_n_steps(n_envs: int, target: int = 2048) -> int:
    # Ensure n_steps is divisible by n_envs; keep it >= 64 per-env
    per_env = max(64, target // max(1, n_envs))
    return per_env * max(1, n_envs)


class Command(BaseCommand):
    help = "Train a PPO policy on SimBerghainEnv (vectorized)"

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--scenario", type=int, choices=[1, 2, 3], default=1)
        parser.add_argument("--total-timesteps", type=int, default=200_000)
        parser.add_argument("--n-envs", type=int, default=8)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--log-dir", type=str, default="runs/ppo_sim")
        parser.add_argument("--save-path", type=str, default="models/ppo_sim.zip")
        parser.add_argument(
            "--init-from",
            type=str,
            default="",
            help="Optional PPO model path to initialize from (e.g., BC pretrain).",
        )
        parser.add_argument("--eval-freq", type=int, default=10_000)
        parser.add_argument("--eval-episodes", type=int, default=5)
        parser.add_argument(
            "--eval-percentile",
            type=float,
            default=0.0,
            help=(
                "If > 0, select best model by this reward percentile (e.g., 90 or 95) instead of mean."
            ),
        )
        parser.add_argument(
            "--curriculum",
            type=str,
            default="",
            help="Comma-separated capacities to stage training (e.g., '200,400,700,1000').",
        )
        parser.add_argument(
            "--stage-steps",
            type=int,
            default=300_000,
            help="Timesteps to train per curriculum stage.",
        )
        parser.add_argument(
            "--no-vecnorm",
            action="store_true",
            help="Disable VecNormalize reward normalization.",
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=0.999,
            help="Discount factor; use a high value to account for long horizon (capacity=1000).",
        )
        parser.add_argument(
            "--gae-lambda",
            dest="gae_lambda",
            type=float,
            default=0.99,
            help="GAE lambda. Use high value to propagate terminal penalty further.",
        )
        parser.add_argument(
            "--n-steps",
            type=int,
            default=None,
            help="Explicit per-env rollout steps (n_steps). Overrides other rollout length options if set.",
        )
        parser.add_argument(
            "--shape-coef",
            type=float,
            default=0.0,
            help="Reward shaping coefficient for deficit reduction (0 disables).",
        )
        parser.add_argument(
            "--nonhelp-penalty",
            type=float,
            default=0.0,
            help="Extra shaping penalty when an accept does not reduce deficits while deficits remain.",
        )
        parser.add_argument(
            "--success-bonus",
            type=float,
            default=0.0,
            help="Shaping bonus applied on success (training only).",
        )
        parser.add_argument(
            "--minmeet-bonus",
            type=float,
            default=0.0,
            help="Per-attribute shaping bonus when a deficit reaches zero (training only).",
        )
        parser.add_argument(
            "--ent-coef",
            dest="ent_coef",
            type=float,
            default=0.0,
            help="Entropy regularization coefficient for PPO (exploration).",
        )
        parser.add_argument(
            "--fail-penalty-scale",
            type=float,
            default=1.0,
            help=(
                "Scale terminal penalty on unmet minima at capacity: effective penalty becomes -s*REJECTION_LIMIT (training only)."
            ),
        )
        parser.add_argument(
            "--success-bonus-per-saved",
            type=float,
            default=0.0,
            help=("Add k*(REJECTION_LIMIT - rejected) upon success (training only)."),
        )
        parser.add_argument(
            "--late-reject-weight",
            type=float,
            default=0.0,
            help=(
                "Extra per-reject penalty scaled by endgame slack: -w*(1 - slack_frac) (training only)."
            ),
        )

    def handle(self, *args: Any, **options: Any) -> None:
        scenario: int = int(options["scenario"])
        total_timesteps: int = int(options["total_timesteps"])
        n_envs: int = int(options["n_envs"])
        seed: int = int(options["seed"])
        log_dir: str = str(options["log_dir"])  # TensorBoard + eval + ckpts
        save_path: str = str(options["save_path"])  # Final model path
        eval_freq: int = int(options["eval_freq"])
        eval_episodes: int = int(options["eval_episodes"])
        use_vecnorm: bool = not bool(options.get("no_vecnorm", False))
        gamma: float = float(options["gamma"])  # long horizon
        gae_lambda: float = float(options["gae_lambda"])  # advantage smoothing
        n_steps_override: int | None = options.get("n_steps")
        shape_coef: float = float(options["shape_coef"])
        nonhelp_penalty: float = float(options["nonhelp_penalty"])
        success_bonus: float = float(options["success_bonus"])
        minmeet_bonus: float = float(options["minmeet_bonus"])
        ent_coef: float = float(options["ent_coef"])
        fail_penalty_scale: float = float(options.get("fail_penalty_scale", 1.0))
        success_bonus_per_saved: float = float(
            options.get("success_bonus_per_saved", 0.0)
        )
        late_reject_weight: float = float(options.get("late_reject_weight", 0.0))
        cur_str: str = str(options.get("curriculum") or "").strip()
        stage_steps: int = int(options.get("stage_steps") or 300_000)

        if (
            bool(total_timesteps)
            and bool(options.get("curriculum"))
            and not bool(options.get("stage_steps"))
        ):
            self.stdout.write(
                self.style.WARNING(
                    "--total_timesteps is ignored when using --curriculum. To control total timesteps, adjust --stage-steps."
                )
            )

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"PPO training | scenario={scenario} n_envs={n_envs} seed={seed}"
            )
        )

        # Helper to build envs with optional capacity/minima overrides
        def _make_env(
            cap_override: int | None = None, min_override: dict[str, int] | None = None
        ) -> SimBerghainEnv | DeficitRewardWrapper:
            env = SimBerghainEnv(
                scenario=scenario,
                capacity=cap_override,
                min_counts=min_override,
            )
            if shape_coef > 0.0:
                return DeficitRewardWrapper(
                    env,
                    coef=shape_coef,
                    nonhelp_penalty=nonhelp_penalty,
                    success_bonus=success_bonus,
                    minmeet_bonus=minmeet_bonus,
                    fail_penalty_scale=fail_penalty_scale,
                    success_bonus_per_saved=success_bonus_per_saved,
                    late_reject_weight=late_reject_weight,
                )
            return env

        # Default training env (no curriculum overrides)
        venv: VecEnv = make_vec_env(lambda: _make_env(), n_envs=n_envs, seed=seed)

        # Optional reward normalization (obs are already 0..1 scaled)
        vec_norm_path = os.path.join(log_dir, "vecnormalize.pkl")
        if use_vecnorm:
            venv = VecNormalize(
                venv,
                norm_obs=False,
                norm_reward=True,
                clip_reward=10.0,
                gamma=0.99,
            )

        # Evaluation env (single)
        # Evaluation WITHOUT shaping to reflect real reward
        def _make_eval_env() -> SimBerghainEnv:
            return SimBerghainEnv(scenario=scenario)

        eval_env: VecEnv = make_vec_env(_make_eval_env, n_envs=1, seed=seed + 100)
        if use_vecnorm:
            eval_env = VecNormalize(
                eval_env, training=False, norm_obs=False, norm_reward=False
            )

        # Model
        # Rollout length
        if n_steps_override is not None and n_steps_override > 0:
            n_steps = int(n_steps_override)
            self.stdout.write(
                self.style.WARNING(f"Using explicit n_steps override: {n_steps}")
            )
        else:
            n_steps = _validated_n_steps(n_envs)

        # Disable TensorBoard logging by default to avoid dependency
        tb_log_dir: str | None = None
        init_from: str = str(options.get("init_from", "") or "").strip()
        if init_from:
            self.stdout.write(self.style.WARNING(f"Initializing PPO from {init_from}"))
            loaded = PPO.load(init_from)
            # Fresh model with desired rollout shape/hyperparams
            model = PPO(
                policy="MlpPolicy",
                env=venv,
                verbose=1,
                tensorboard_log=tb_log_dir,
                n_steps=n_steps,
                batch_size=64,
                learning_rate=3e-4,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                device="auto",
            )
            model.policy.load_state_dict(loaded.policy.state_dict(), strict=False)
        else:
            model = PPO(
                policy="MlpPolicy",
                env=venv,
                verbose=1,
                tensorboard_log=tb_log_dir,
                n_steps=n_steps,
                batch_size=64,
                learning_rate=3e-4,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                device="auto",
            )

        # Callbacks: eval + periodic checkpoints
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq=max(10_000 // n_envs, 1), save_path=ckpt_dir, name_prefix="ppo"
        )
        eval_percentile: float = float(options.get("eval_percentile", 0.0) or 0.0)
        eval_cb: EvalCallback | PercentileEvalCallback
        if eval_percentile > 0.0:
            eval_cb = PercentileEvalCallback(
                eval_env,
                eval_freq=max(eval_freq // max(1, n_envs), 1),
                n_eval_episodes=eval_episodes,
                percentile=eval_percentile,
                deterministic=True,
                best_model_save_path=os.path.join(log_dir, "best"),
                log_path=os.path.join(log_dir, "eval"),
                verbose=1,
            )
        else:
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(log_dir, "best"),
                log_path=os.path.join(log_dir, "eval"),
                eval_freq=max(eval_freq // max(1, n_envs), 1),
                n_eval_episodes=eval_episodes,
                deterministic=True,
            )

        # Train (with optional curriculum)
        if cur_str:
            caps = [int(x) for x in cur_str.split(",") if x.strip()]
            # Build min-counts override from scenario config scaled by capacity
            base_constraints = {
                c["attribute"]: int(c["minCount"])
                for c in SCENARIO_CONFIGS[scenario]["constraints"]
            }
            self.stdout.write(
                self.style.MIGRATE_HEADING(
                    f"Curriculum: {caps} (stage_steps={stage_steps})"
                )
            )

            # Start from the already-initialized model and loop stages
            for cap in caps:
                scale = float(cap) / float(CAPACITY)
                min_override = {
                    a: int(round(base_constraints.get(a, 0) * scale))
                    for a in ATTRIBUTE_ORDER
                }

                # Rebuild training env for this stage
                def _stage_env_factory(
                    capacity: int, minov: dict[str, int]
                ) -> Callable[[], SimBerghainEnv | DeficitRewardWrapper]:
                    return lambda: _make_env(cap_override=capacity, min_override=minov)

                venv = make_vec_env(
                    _stage_env_factory(cap, min_override), n_envs=n_envs, seed=seed
                )
                if use_vecnorm:
                    venv = VecNormalize(
                        venv,
                        norm_obs=False,
                        norm_reward=True,
                        clip_reward=10.0,
                        gamma=0.99,
                    )
                # Swap env on model
                model.set_env(venv)
                self.stdout.write(self.style.WARNING(f"Training stage capacity={cap}"))
                model.learn(
                    total_timesteps=stage_steps,
                    callback=[checkpoint_cb, eval_cb],
                    reset_num_timesteps=False,
                )
        else:
            model.learn(
                total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb]
            )

        # Save model and VecNormalize stats
        model.save(save_path)
        if use_vecnorm:
            assert isinstance(venv, VecNormalize | DummyVecEnv)
            if isinstance(venv, VecNormalize):
                venv.save(vec_norm_path)

        self.stdout.write(self.style.SUCCESS(f"Saved model -> {save_path}"))
        if use_vecnorm:
            self.stdout.write(
                self.style.SUCCESS(f"Saved VecNormalize -> {vec_norm_path}")
            )
