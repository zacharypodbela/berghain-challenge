from __future__ import annotations

import os
from typing import Any

from django.core.management.base import BaseCommand

# Third-party RL libs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from bouncer.rl.env import DeficitRewardWrapper, SimBerghainEnv


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

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"PPO training | scenario={scenario} n_envs={n_envs} seed={seed}"
            )
        )

        # Build vectorized training envs
        def _make_env() -> SimBerghainEnv | DeficitRewardWrapper:
            env = SimBerghainEnv(scenario=scenario)
            if shape_coef > 0.0:
                return DeficitRewardWrapper(
                    env,
                    coef=shape_coef,
                    nonhelp_penalty=nonhelp_penalty,
                    success_bonus=success_bonus,
                    minmeet_bonus=minmeet_bonus,
                )
            return env

        venv: VecEnv = make_vec_env(_make_env, n_envs=n_envs, seed=seed)

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
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=max(eval_freq // max(1, n_envs), 1),
            n_eval_episodes=eval_episodes,
            deterministic=True,
        )

        # Train
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])

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
