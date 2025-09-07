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
        seed = int(opts["seed"]) or 12345
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

        def _pctl(xs: list[float], q: float) -> float:
            if not xs:
                return 0.0
            return float(np.percentile(np.asarray(xs, dtype=np.float64), q))

        mean_r, std_r = _mean_std(rewards)
        mean_len, std_len = _mean_std([float(x) for x in lengths])
        mean_adm, std_adm = _mean_std([float(x) for x in admits])
        mean_rej, std_rej = _mean_std([float(x) for x in rejects])

        self.stdout.write("\nSummary")
        self.stdout.write("-------")
        self.stdout.write(f"Model: {model_path}")
        self.stdout.write(f"Scenario: {scenario}")
        self.stdout.write(f"Episodes: {n_episodes}")
        # Percentiles (lower and upper tails)
        rewards_arr = [float(x) for x in rewards]
        lengths_arr = [float(x) for x in lengths]
        admits_arr = [float(x) for x in admits]
        rejects_arr = [float(x) for x in rejects]

        p01_r = _pctl(rewards_arr, 1.0)
        p05_r = _pctl(rewards_arr, 5.0)
        p10_r = _pctl(rewards_arr, 10.0)
        p90_r = _pctl(rewards_arr, 90.0)
        p95_r = _pctl(rewards_arr, 95.0)
        p99_r = _pctl(rewards_arr, 99.0)
        min_r = min(rewards_arr) if rewards_arr else 0.0
        max_r = max(rewards_arr) if rewards_arr else 0.0

        p01_len = _pctl(lengths_arr, 1.0)
        p05_len = _pctl(lengths_arr, 5.0)
        p10_len = _pctl(lengths_arr, 10.0)
        p90_len = _pctl(lengths_arr, 90.0)
        p95_len = _pctl(lengths_arr, 95.0)
        p99_len = _pctl(lengths_arr, 99.0)
        min_len = min(lengths_arr) if lengths_arr else 0.0
        max_len = max(lengths_arr) if lengths_arr else 0.0

        p01_adm = _pctl(admits_arr, 1.0)
        p05_adm = _pctl(admits_arr, 5.0)
        p10_adm = _pctl(admits_arr, 10.0)
        p90_adm = _pctl(admits_arr, 90.0)
        p95_adm = _pctl(admits_arr, 95.0)
        p99_adm = _pctl(admits_arr, 99.0)
        min_adm = min(admits_arr) if admits_arr else 0.0
        max_adm = max(admits_arr) if admits_arr else 0.0

        p01_rej = _pctl(rejects_arr, 1.0)
        p05_rej = _pctl(rejects_arr, 5.0)
        p10_rej = _pctl(rejects_arr, 10.0)
        p90_rej = _pctl(rejects_arr, 90.0)
        p95_rej = _pctl(rejects_arr, 95.0)
        p99_rej = _pctl(rejects_arr, 99.0)
        min_rej = min(rejects_arr) if rejects_arr else 0.0
        max_rej = max(rejects_arr) if rejects_arr else 0.0

        self.stdout.write()
        self.stdout.write(
            "|   | Mean | Std | P01 | P05 | P10 | P90 | P95 | P99 | Min | Max |"
        )
        self.stdout.write(
            "|---|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|"
        )
        self.stdout.write(
            f"| Reward | {mean_r:.2f} | {std_r:.2f} | {p01_r:.2f} | {p05_r:.2f} | {p10_r:.2f} | "
            f"{p90_r:.2f} | {p95_r:.2f} | {p99_r:.2f} | {min_r:.2f} | {max_r:.2f} |"
        )
        self.stdout.write(
            f"| length | {mean_len:.1f} | {std_len:.1f} | {p01_len:.1f} | {p05_len:.1f} | {p10_len:.1f} | "
            f"{p90_len:.1f} | {p95_len:.1f} | {p99_len:.1f} | {min_len:.1f} | {max_len:.1f} |"
        )
        self.stdout.write(
            f"| Admitted | {mean_adm:.1f} | {std_adm:.1f} | {p01_adm:.1f} | {p05_adm:.1f} | {p10_adm:.1f} | "
            f"{p90_adm:.1f} | {p95_adm:.1f} | {p99_adm:.1f} | {min_adm:.1f} | {max_adm:.1f} |"
        )
        self.stdout.write(
            f"| Rejected | {mean_rej:.1f} | {std_rej:.1f} | {p01_rej:.1f} | {p05_rej:.1f} | {p10_rej:.1f} | "
            f"{p90_rej:.1f} | {p95_rej:.1f} | {p99_rej:.1f} | {min_rej:.1f} | {max_rej:.1f} |"
        )
        self.stdout.write(
            f"| Outcomes | success={reasons[EpisodeResult.SUCCESS.value]} | unmet_at_capacity={reasons[EpisodeResult.CONSTRAINTS_UNMET_AT_CAPACITY.value]} | rejection_limit={reasons[EpisodeResult.REJECTION_LIMIT.value]} |"
        )
