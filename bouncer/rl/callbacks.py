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


class RiskSeekingWeightCallback(BaseCallback):
    """Episode-weighted advantages for risk-seeking PPO.

    For each rollout, compute per-episode returns (within the rollout window) per env
    using episode_starts boundaries. Convert each episode return R into a weight
    w = exp(beta * (R - baseline)). Clip to [1e-6, w_max]. Multiply PPO's rollout
    advantages by these weights before the training update.

    Notes:
    - baseline is an exponential moving average of observed episode returns.
    - Episodes that span multiple rollouts are approximated by the partial return
      within the current rollout window.
    - This callback assumes the algorithm exposes `rollout_buffer` with
      `.rewards` and `.episode_starts` shaped (n_steps, n_envs), and `.advantages`
      of the same shape after GAE computation.
    """

    def __init__(
        self, beta: float = 1e-4, w_max: float = 20.0, ema_decay: float = 0.99
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.w_max = float(max(1.0, w_max))
        self.ema_decay = float(ema_decay)
        self.baseline: float | None = None
        # Running return per env across rollouts for episodes that started in a previous buffer
        from numpy.typing import NDArray

        self._carry: NDArray[np.float64] | None = None

    def _on_rollout_end(self) -> None:
        # Access rollout buffer
        algo = self.model
        # Defensive: only operate if buffer and fields exist
        buf = getattr(algo, "rollout_buffer", None)
        if buf is None:
            return
        rewards = getattr(buf, "rewards", None)
        starts = getattr(buf, "episode_starts", None)
        adv = getattr(buf, "advantages", None)
        if rewards is None or starts is None or adv is None:
            return

        # Expect shapes (n_steps, n_envs)
        try:
            n_steps, n_envs = rewards.shape
        except Exception:
            return

        # Initialize carry if first run or env count changed
        if self._carry is None or self._carry.shape != (n_envs):
            self._carry = np.zeros((n_envs,), dtype=np.float64)

        # Build weight matrix; apply weights only to steps that belong to episodes
        # that end within this rollout, using full episode return (carry + segment sum).
        weights = np.ones_like(rewards, dtype=np.float32)
        completed_returns: list[float] = []

        for e in range(n_envs):
            seg_start = 0
            # If a new episode starts at t=0, previous episode ended before; reset carry
            if bool(starts[0, e]):
                self._carry[e] = 0.0
            # Iterate and detect boundaries where a new episode begins (so previous ended)
            for t in range(1, n_steps):
                if bool(starts[t, e]):
                    seg_sum = float(np.sum(rewards[seg_start:t, e]))
                    full_R = float(self._carry[e] + seg_sum)
                    completed_returns.append(full_R)
                    # Compute weight from baseline
                    if self.baseline is None:
                        w = 1.0
                    else:
                        w = float(np.exp(self.beta * (full_R - self.baseline)))
                    w = float(np.clip(w, 1e-6, self.w_max))
                    weights[seg_start:t, e] = w
                    # Reset carry and start next segment at t
                    self._carry[e] = 0.0
                    seg_start = t
            # Tail segment (possibly incomplete episode): accumulate into carry only
            tail_sum = float(np.sum(rewards[seg_start:n_steps, e]))
            self._carry[e] += tail_sum

        # Update baseline EMA using full returns of completed episodes this rollout
        if completed_returns:
            mean_R = float(np.mean(completed_returns))
            if self.baseline is None:
                self.baseline = mean_R
            else:
                self.baseline = (
                    self.ema_decay * self.baseline + (1.0 - self.ema_decay) * mean_R
                )

        # Scale advantages in-place
        buf.advantages *= weights

    def _on_step(self) -> bool:
        # Not used; weights are applied at rollout end.
        return True
