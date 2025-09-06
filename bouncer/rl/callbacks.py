from __future__ import annotations

import os

import numpy as np
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv


class PercentileEvalCallback(BaseCallback):
    """Evaluate the policy every ``eval_freq`` steps and save the best model by percentile reward.

    Metric: p-th percentile of episode rewards over ``n_eval_episodes`` (deterministic or stochastic).
    """

    def __init__(
        self,
        eval_env: VecEnv,
        *,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        percentile: float = 90.0,
        deterministic: bool = True,
        best_model_save_path: str | None = None,
        log_path: str | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.percentile = float(percentile)
        self.deterministic = bool(deterministic)
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.best_metric: float | None = None
        self.timesteps: list[int] = []
        self.metrics: list[float] = []

        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq == 0:
            assert isinstance(self.model, base_class.BaseAlgorithm)
            rewards_list, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
            # evaluate_policy with return_episode_rewards returns (rewards, lengths)
            episode_rewards = np.asarray(rewards_list, dtype=np.float64)
            metric = float(np.percentile(episode_rewards, self.percentile))
            self.timesteps.append(self.num_timesteps)
            self.metrics.append(metric)

            if self.verbose > 0:
                print(
                    f"Eval at {self.num_timesteps} steps | p{self.percentile:.0f} reward = {metric:.2f}"
                )

            if self.best_metric is None or metric > self.best_metric:
                self.best_metric = metric
                if self.best_model_save_path is not None:
                    path = os.path.join(self.best_model_save_path, "best_model.zip")
                    self.model.save(path)
                    if self.verbose > 0:
                        print(
                            f"New best model saved to {path} (p{self.percentile:.0f}={metric:.2f})"
                        )

            # Persist evaluations to npz similar to SB3's EvalCallback
            if self.log_path is not None:
                np.savez(
                    os.path.join(self.log_path, "evaluations_percentile.npz"),
                    timesteps=np.array(self.timesteps, dtype=np.int64),
                    metrics=np.array(self.metrics, dtype=np.float64),
                    percentile=np.array([self.percentile], dtype=np.float64),
                )

        return True
