from __future__ import annotations

from typing import Any

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from numpy.typing import NDArray
from stable_baselines3 import PPO


class Command(BaseCommand):
    help = "Compare a PPO model's decisions against an expert dataset (NPZ) and report disagreement metrics."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--dataset", required=True, help="Path to NPZ exported via export_dataset."
        )
        parser.add_argument(
            "--model-path", required=True, help="Path to a PPO .zip model."
        )
        parser.add_argument(
            "--limit", type=int, default=0, help="Limit number of steps (0 = all)."
        )

    def handle(self, *args: Any, **opts: Any) -> None:
        ds_path = str(opts["dataset"]).strip()
        model_path = str(opts["model_path"]).strip()
        if not ds_path.endswith(".npz"):
            raise CommandError(
                "--dataset must be an .npz file produced by export_dataset"
            )

        data = np.load(ds_path, allow_pickle=True)
        obs: NDArray[np.float32] = data["obs"].astype(np.float32)
        actions: NDArray[np.int64] = data["actions"].astype(np.int64)
        # episodes index available if needed for future per-episode grouping
        # episodes: NDArray[np.int64] = data["episodes"].astype(np.int64)
        keep_mask = np.ones(obs.shape[0], dtype=bool)

        # Limit steps if requested
        if int(opts["limit"]) > 0:
            n = int(opts["limit"])
            idx = np.nonzero(keep_mask)[0][:n]
            keep_mask = np.zeros_like(keep_mask)
            keep_mask[idx] = True

        X = obs[keep_mask]
        y = actions[keep_mask]
        n_steps = X.shape[0]
        if n_steps == 0:
            raise CommandError("No steps after filtering to compare.")

        # Load model once
        model: PPO = PPO.load(model_path)

        # Predict in simple mini-batches to avoid memory spikes
        batch_size = 4096
        pred_list: list[int] = []
        for i in range(0, n_steps, batch_size):
            j = min(i + batch_size, n_steps)
            for k in range(i, j):
                a, _ = model.predict(X[k], deterministic=True)
                pred_list.append(int(a))

        y_pred = np.array(pred_list, dtype=np.int64)

        # Derive needed-attribute overlap from observation:
        # Pairs start at index 7: for each i, [curr_bit, remain_need_frac].
        # We consider an "overlap" when the person has an attribute (curr_bit==1)
        # that is still needed (remain_need_frac>0). We summarize for >=1 and >=2 overlaps.
        def _overlap_needed_count(vec: NDArray[np.float32]) -> int:
            cnt = 0
            # obs length 31 => (31-7)/2 = 12 attribute pairs
            for i in range((vec.shape[0] - 7) // 2):
                curr = int(vec[7 + 2 * i] > 0.5)
                need = float(vec[7 + 2 * i + 1]) > 0.0
                if curr and need:
                    cnt += 1
            return cnt

        overlaps = np.array([_overlap_needed_count(row) for row in X], dtype=np.int32)

        # Metrics
        disagree = y_pred != y
        disagree_rate = float(disagree.mean())
        agree_rate = 1.0 - disagree_rate
        # Accept rates conditioned on overlap thresholds k = 1..max_overlap
        max_overlap = int(overlaps.max()) if overlaps.size > 0 else 0
        threshold_metrics: list[tuple[int, float, float]] = []
        for k in range(1, max_overlap + 1):
            mask = overlaps >= k
            if mask.any():
                expert_acc = float((y[mask] == 1).mean())
                model_acc = float((y_pred[mask] == 1).mean())
                threshold_metrics.append((k, expert_acc, model_acc))
        # Zero-overlap reject rate (overlaps == 0)
        zero_olap_mask = overlaps == 0
        if zero_olap_mask.any():
            expert_zero_olap_reject_rate = float((y[zero_olap_mask] == 0).mean())
            model_zero_olap_reject_rate = float((y_pred[zero_olap_mask] == 0).mean())
        else:
            expert_zero_olap_reject_rate = float("nan")
            model_zero_olap_reject_rate = float("nan")

        self.stdout.write(self.style.MIGRATE_HEADING("Disagreement Summary"))
        self.stdout.write(f"Dataset: {ds_path}")
        self.stdout.write(f"Model:   {model_path}")
        self.stdout.write(f"Steps compared: {n_steps}")
        self.stdout.write(f"Agreement rate:   {agree_rate:.3f}")
        self.stdout.write(f"Disagreement rate:{disagree_rate:.3f}")
        self.stdout.write("")
        if threshold_metrics:
            self.stdout.write("Needed-overlap thresholds:")
            for k, e_rate, m_rate in threshold_metrics:
                self.stdout.write(
                    f"  >= {k} attribute(s): Expert accept {e_rate:.3f} | Model accept {m_rate:.3f}"
                )
        else:
            self.stdout.write("Needed-overlap thresholds: none present in dataset")
        self.stdout.write("Zero-overlap steps:")
        self.stdout.write(
            f"  Expert reject rate: {expert_zero_olap_reject_rate:.3f} | Model reject rate: {model_zero_olap_reject_rate:.3f}"
        )
