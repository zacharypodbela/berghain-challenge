from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from django.core.management.base import BaseCommand, CommandError
from gymnasium import Env, spaces
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class _DummyObsEnv(Env[NDArray[np.float32], int]):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(31,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # type: ignore[assignment]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        return np.zeros((31,), dtype=np.float32), {}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        return np.zeros((31,), dtype=np.float32), 0.0, True, False, {}


def _load_npz(paths: list[str]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    obs_chunks: list[NDArray[np.float32]] = []
    act_chunks: list[NDArray[np.int64]] = []
    for p in paths:
        data = np.load(p)
        obs = data["obs"]
        actions = data["actions"]
        obs_chunks.append(obs)
        act_chunks.append(actions)
    X: NDArray[np.float32] = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    y: NDArray[np.int64] = np.concatenate(act_chunks, axis=0).astype(np.int64)
    return X, y


class Command(BaseCommand):
    help = "Behavioral cloning pretrain for PPO policy from exported datasets (NPZ)."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--datasets", type=str, required=True, help="Comma-separated NPZ files."
        )
        parser.add_argument(
            "--out", type=str, required=True, help="Output model path (zip)."
        )
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--val-split", type=float, default=0.1)

    def handle(self, *args: Any, **opts: Any) -> None:
        paths = [p.strip() for p in str(opts["datasets"]).split(",") if p.strip()]
        out_path = str(opts["out"]).strip()
        if not out_path.endswith(".zip"):
            raise CommandError("--out must end with .zip")
        if not paths:
            raise CommandError("No dataset paths provided")

        X, y = _load_npz(paths)
        n = X.shape[0]
        if n == 0:
            raise CommandError("Empty dataset")
        val_split = float(opts["val_split"]) or 0.1
        m = int(n * (1.0 - val_split))
        perm = np.random.permutation(n)
        train_idx, val_idx = perm[:m], perm[m:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Instantiate PPO policy with a dummy env
        vec_env = DummyVecEnv([lambda: _DummyObsEnv()])
        model = PPO(policy="MlpPolicy", env=vec_env, verbose=0)
        policy = model.policy
        device = policy.device

        # Collect policy parameters for actor (exclude value_net)
        actor_params = []
        # features_extractor + policy_net + action_net
        actor_params += list(policy.features_extractor.parameters())
        actor_params += list(policy.mlp_extractor.policy_net.parameters())
        actor_params += list(policy.action_net.parameters())
        for p in policy.mlp_extractor.value_net.parameters():
            p.requires_grad_(False)
        for p in policy.value_net.parameters():
            p.requires_grad_(False)

        optim = torch.optim.Adam(actor_params, lr=float(opts["lr"]))
        batch = int(opts["batch_size"]) or 1024
        epochs = int(opts["epochs"]) or 5

        def _ce_loss(obs: NDArray[np.float32], acts: NDArray[np.int64]) -> float:
            policy.train()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist = policy.get_distribution(obs_t)
            assert dist.distribution is not None
            logits = dist.distribution.logits
            a_t = torch.as_tensor(acts, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, a_t)
            optim.zero_grad()
            loss.backward()
            optim.step()
            return float(loss.detach().cpu().item())

        def _accuracy(obs: NDArray[np.float32], acts: NDArray[np.int64]) -> float:
            policy.eval()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                dist = policy.get_distribution(obs_t)
                assert dist.distribution is not None
                logits = dist.distribution.logits
                pred = torch.argmax(logits, dim=1)
                a_t = torch.as_tensor(acts, dtype=torch.long, device=device)
                acc = (pred == a_t).float().mean().cpu().item()
                return float(acc)

        for ep in range(1, epochs + 1):
            # Shuffle each epoch
            order = np.random.permutation(X_train.shape[0])
            X_train = X_train[order]
            y_train = y_train[order]
            # Mini-batch loop
            losses = []
            for i in range(0, X_train.shape[0], batch):
                j = min(i + batch, X_train.shape[0])
                loss = _ce_loss(X_train[i:j], y_train[i:j])
                losses.append(loss)
            train_acc = _accuracy(
                X_train[: min(10000, X_train.shape[0])],
                y_train[: min(10000, y_train.shape[0])],
            )
            val_acc = _accuracy(X_val, y_val) if X_val.shape[0] > 0 else float("nan")
            self.stdout.write(
                f"Epoch {ep}/{epochs} | train CE: {np.mean(losses):.4f} | train acc: {train_acc:.3f} | val acc: {val_acc:.3f}"
            )

        # Save the initialized policy for PPO to fine-tune
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        model.save(out_path)
        self.stdout.write(self.style.SUCCESS(f"Saved BC-init model -> {out_path}"))
