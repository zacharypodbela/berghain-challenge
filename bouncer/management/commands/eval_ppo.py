from __future__ import annotations

from typing import Any

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from stable_baselines3 import PPO

from bouncer.rl.env import EpisodeResult, SimBerghainEnv


class Command(BaseCommand):
    help = "Evaluate a PPO model quickly in the in-memory SimBerghainEnv (no DB)."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--model-path", type=str, required=True, help="Path to PPO .zip model"
        )
        parser.add_argument(
            "--scenario",
            type=int,
            choices=[1, 2, 3],
            required=True,
            help="Scenario to evaluate",
        )
        parser.add_argument(
            "--episodes", type=int, default=50, help="Number of episodes to evaluate"
        )
        parser.add_argument(
            "--seed", type=int, default=123, help="Random seed for people generator"
        )
        parser.add_argument(
            "--deterministic",
            action="store_true",
            help="Use deterministic actions (default: stochastic)",
        )

    def handle(self, *args: Any, **opts: Any) -> None:
        model_path = str(opts["model_path"]).strip()
        scenario = int(opts["scenario"])
        n_episodes = int(opts["episodes"]) or 1
        seed = int(opts["seed"]) or 0
        deterministic = bool(opts.get("deterministic", False))

        if not model_path.endswith(".zip"):
            raise CommandError("--model-path must be a Stable-Baselines3 PPO .zip file")

        try:
            model: PPO = PPO.load(model_path)
        except Exception as e:  # pragma: no cover - defensive
            raise CommandError(f"Failed to load model: {e}") from e

        rewards: list[float] = []
        admits: list[int] = []
        rejects: list[int] = []
        lengths: list[int] = []
        reasons: dict[str, int] = {
            EpisodeResult.SUCCESS.value: 0,
            EpisodeResult.CONSTRAINTS_UNMET_AT_CAPACITY.value: 0,
            EpisodeResult.REJECTION_LIMIT.value: 0,
        }

        # Run episodes sequentially on the in-memory env (fast, no DB)
        for ep in range(1, n_episodes + 1):
            env = SimBerghainEnv(scenario=scenario)
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, r, terminated, truncated, info = env.step(int(action))
                ep_reward += float(r)
                steps += 1
                done = bool(terminated or truncated)

            rewards.append(ep_reward)
            admits.append(int(info.get("admitted", env.admitted)))
            rejects.append(int(info.get("rejected", env.rejected)))
            lengths.append(steps)
            reason = str(info.get("reason", EpisodeResult.FAILED.value))
            if reason in reasons:
                reasons[reason] += 1

            self.stdout.write(
                f"Episode {ep}/{n_episodes}: reward={ep_reward:.2f} admitted={admits[-1]} rejected={rejects[-1]} reason={reason} length={steps}"
            )

        # Summaries
        def _mean_std(xs: list[float]) -> tuple[float, float]:
            if not xs:
                return 0.0, 0.0
            arr = np.asarray(xs, dtype=np.float64)
            return float(arr.mean()), float(arr.std(ddof=0))

        mean_r, std_r = _mean_std(rewards)
        mean_len, std_len = _mean_std([float(x) for x in lengths])
        mean_adm, std_adm = _mean_std([float(x) for x in admits])
        mean_rej, std_rej = _mean_std([float(x) for x in rejects])

        self.stdout.write("\nSummary")
        self.stdout.write("-------")
        self.stdout.write(f"Model:     {model_path}")
        self.stdout.write(f"Scenario:  {scenario}")
        self.stdout.write(f"Episodes:  {n_episodes}")
        self.stdout.write(
            f"Reward:    mean={mean_r:.2f} std={std_r:.2f} | length: mean={mean_len:.1f} std={std_len:.1f}"
        )
        self.stdout.write(
            f"Admitted:  mean={mean_adm:.1f} std={std_adm:.1f} | Rejected: mean={mean_rej:.1f} std={std_rej:.1f}"
        )
        self.stdout.write(
            f"Outcomes:  success={reasons[EpisodeResult.SUCCESS.value]} unmet_at_capacity={reasons[EpisodeResult.CONSTRAINTS_UNMET_AT_CAPACITY.value]} rejection_limit={reasons[EpisodeResult.REJECTION_LIMIT.value]}"
        )
