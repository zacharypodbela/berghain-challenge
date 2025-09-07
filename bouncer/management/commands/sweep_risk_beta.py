from __future__ import annotations

import os
from typing import Any

import numpy as np
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Sweep --risk-beta values for train_ppo, selecting best by eval percentile."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True)
        parser.add_argument("--init-from", dest="init_from", type=str, required=True)
        parser.add_argument(
            "--betas",
            type=str,
            required=True,
            help="Comma-separated risk-beta values (e.g., '0,0.0001,0.0002,0.0005')",
        )

        # Common PPO/train flags (subset; extend as needed)
        parser.add_argument("--total-timesteps", type=int, default=1_000_000)
        parser.add_argument("--n-envs", type=int, default=4)
        parser.add_argument("--gamma", type=float, default=0.9995)
        parser.add_argument(
            "--gae-lambda", dest="gae_lambda", type=float, default=0.995
        )
        parser.add_argument("--n-steps", type=int, default=8192)
        parser.add_argument("--ent-coef", type=float, default=0.02)
        parser.add_argument("--shape-coef", type=float, default=0.0)
        parser.add_argument("--nonhelp-penalty", type=float, default=0.0)
        parser.add_argument("--success-bonus", type=float, default=0.0)
        parser.add_argument("--minmeet-bonus", type=float, default=0.0)
        parser.add_argument("--fail-penalty-scale", type=float, default=1.0)
        parser.add_argument("--success-bonus-per-saved", type=float, default=0.0)
        parser.add_argument("--late-reject-weight", type=float, default=0.0)

        # Curriculum / eval
        parser.add_argument("--curriculum", type=str, default="")
        parser.add_argument("--stage-steps", type=int, default=300_000)
        parser.add_argument("--eval-freq", type=int, default=50_000)
        parser.add_argument("--eval-episodes", type=int, default=40)
        parser.add_argument(
            "--eval-percentile",
            type=float,
            default=95.0,
            help="Percentile used to pick best model (default 95).",
        )
        parser.add_argument("--no-vecnorm", action="store_true")

        # Paths
        parser.add_argument(
            "--log-root",
            type=str,
            default="runs/sweeps/risk_beta",
            help="Root directory to place per-beta log dirs.",
        )
        parser.add_argument(
            "--save-root",
            type=str,
            default="models/sweeps/risk_beta",
            help="Root directory to save final models per beta.",
        )

    def handle(self, *args: Any, **opts: Any) -> None:
        scenario = int(opts["scenario"])
        init_from = str(opts["init_from"]).strip()
        betas = [float(x) for x in str(opts["betas"]).split(",") if x.strip()]
        if not betas:
            raise CommandError("--betas must contain at least one value")

        log_root = str(opts["log_root"]).rstrip("/")
        save_root = str(opts["save_root"]).rstrip("/")
        os.makedirs(log_root, exist_ok=True)
        os.makedirs(save_root, exist_ok=True)

        # Collect results per beta
        records: list[
            tuple[float, float, str, str]
        ] = []  # (beta, best_metric, best_model_path, log_dir)

        for beta in betas:
            beta_tag = f"{beta:.6f}".rstrip("0").rstrip(".")
            log_dir = os.path.join(log_root, f"beta_{beta_tag}")
            save_path = os.path.join(save_root, f"ppo_s{scenario}_beta_{beta_tag}.zip")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

            self.stdout.write(
                self.style.MIGRATE_HEADING(f"\nTraining with risk_beta={beta_tag}")
            )

            # Build kwargs for train_ppo
            train_kwargs: dict[str, Any] = {
                "scenario": scenario,
                "init_from": init_from,
                "total_timesteps": int(opts["total_timesteps"]),
                "n_envs": int(opts["n_envs"]),
                "gamma": float(opts["gamma"]),
                "gae_lambda": float(opts["gae_lambda"]),
                "n_steps": int(opts["n_steps"]),
                "ent_coef": float(opts["ent_coef"]),
                "shape_coef": float(opts["shape_coef"]),
                "nonhelp_penalty": float(opts["nonhelp_penalty"]),
                "success_bonus": float(opts["success_bonus"]),
                "minmeet_bonus": float(opts["minmeet_bonus"]),
                "fail_penalty_scale": float(opts["fail_penalty_scale"]),
                "success_bonus_per_saved": float(opts["success_bonus_per_saved"]),
                "late_reject_weight": float(opts["late_reject_weight"]),
                "curriculum": str(opts["curriculum"]),
                "stage_steps": int(opts["stage_steps"]),
                "eval_freq": int(opts["eval_freq"]),
                "eval_episodes": int(opts["eval_episodes"]),
                "eval_percentile": float(opts["eval_percentile"]),
                "log_dir": log_dir,
                "save_path": save_path,
                "risk_beta": float(beta),
            }
            if bool(opts.get("no_vecnorm", False)):
                train_kwargs["no_vecnorm"] = True

            # Call train_ppo synchronously
            call_command("train_ppo", **train_kwargs)

            # Read percentile evals and select best metric
            eval_npz = os.path.join(log_dir, "eval", "evaluations_percentile.npz")
            best_metric = float("-inf")
            if os.path.exists(eval_npz):
                data = np.load(eval_npz)
                metrics = data.get("metrics")
                if metrics is not None and metrics.size > 0:
                    best_metric = float(np.max(metrics))

            best_model_path = os.path.join(log_dir, "best", "best_model.zip")
            self.stdout.write(
                f"risk_beta={beta_tag} | best p{int(opts['eval_percentile'])} reward: {best_metric:.2f} | best_model: {best_model_path}"
            )
            records.append((beta, best_metric, best_model_path, log_dir))

        # Pick overall best by metric
        records_sorted = sorted(records, key=lambda r: r[1], reverse=True)
        best_beta, best_metric, best_model_path, best_log_dir = records_sorted[0]

        self.stdout.write(self.style.MIGRATE_HEADING("\nSweep Summary"))
        for beta, metric, model_path, _log_dir in records_sorted:
            beta_tag = f"{beta:.6f}".rstrip("0").rstrip(".")
            self.stdout.write(
                f"  beta={beta_tag:>8}  metric={metric:>10.2f}  model={os.path.relpath(model_path)}"
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Best: beta={best_beta} | metric={best_metric:.2f} | model={best_model_path} | logs={best_log_dir}"
            )
        )
