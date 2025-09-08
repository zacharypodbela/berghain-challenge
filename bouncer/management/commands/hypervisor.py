from __future__ import annotations

import asyncio
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Q

from bouncer.algorithms import ALGORITHMS, AlgorithmFunc, get_algorithm
from bouncer.constants import REJECTION_LIMIT
from bouncer.models import RemoteGame
from bouncer.runner import run_game_until


class Command(BaseCommand):
    help = "Async hypervisor: manage multiple RemoteGames concurrently per scenario."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--scenarios",
            type=str,
            default="1,2,3",
            help=(
                "Comma-separated scenario list; repeat to control concurrency per scenario "
                "(e.g., '1,1,2,2,3,3' for two slots each). Default: 1,2,3"
            ),
        )
        parser.add_argument(
            "--algorithm-map",
            type=str,
            default="",
            help=(
                "Per-scenario mapping: '1=algo[@model],2=algo[@model],3=algo[@model]'. "
                "Example: 1=ppo_bouncer@models/s1.zip,2=deficit_weighted_bouncer,3=ppo_bouncer@models/s3.zip"
            ),
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Enable per-decision logging"
        )

    def handle(self, *args: Any, **opts: Any) -> None:
        scenario_list = [int(s) for s in str(opts["scenarios"]).split(",") if s.strip()]
        verbose = bool(opts.get("verbose", False))

        # Parse optional per-scenario algorithm map
        algo_map_raw = str(opts.get("algorithm_map") or "").strip()
        resolved: dict[int, tuple[AlgorithmFunc, str | None]] = {}
        MODEL_REQUIRED = {"ppo_bouncer"}
        if algo_map_raw:
            for entry in [e.strip() for e in algo_map_raw.split(",") if e.strip()]:
                if "=" not in entry:
                    raise CommandError(
                        f"Invalid --algorithm-map entry '{entry}'. Expected 'S=algo[@model]'"
                    )
                s_str, spec = entry.split("=", 1)
                try:
                    s_val = int(s_str)
                except ValueError as e:
                    raise CommandError(
                        f"Invalid scenario '{s_str}' in --algorithm-map"
                    ) from e
                if s_val not in (1, 2, 3):
                    raise CommandError(
                        "Scenario must be one of 1,2,3 in --algorithm-map"
                    )
                if "@" in spec:
                    algo_name, model_spec = spec.split("@", 1)
                    model_spec = model_spec.strip()
                else:
                    algo_name, model_spec = spec, None
                algo_name = algo_name.strip()
                if algo_name not in ALGORITHMS:
                    raise CommandError(
                        f"Unknown algorithm '{algo_name}' in --algorithm-map. Available: {list(ALGORITHMS.keys())}"
                    )
                if algo_name in MODEL_REQUIRED and not model_spec:
                    raise ValueError(
                        f"Algorithm '{algo_name}' requires a model path in --algorithm-map (use 'algo@path')."
                    )
                self.stdout.write(
                    f"Scenario {s_val}: algo={algo_name} model={model_spec if model_spec else '-'}"
                )
                resolved[s_val] = (get_algorithm(algo_name), model_spec)
        else:
            raise CommandError(
                "--algorithm-map is required for hypervisor to know which algorithms to run"
            )

        # Determine current best (lowest rejects) per scenario from completed RemoteGames
        unique_scenarios = sorted(set(scenario_list))
        per_game_rows = (
            RemoteGame.objects.filter(status="completed")
            .values("scenario", "id")
            .annotate(rej_count=Count("people", filter=Q(people__decision=False)))
        )
        best_by_s = dict.fromkeys(unique_scenarios)
        for row in per_game_rows:
            scen = int(row.get("scenario") or 0)
            rej = int(row.get("rej_count") or REJECTION_LIMIT)
            curr = best_by_s.get(scen)
            if curr is None or rej < curr:
                best_by_s[scen] = rej
        for s in unique_scenarios:
            self.stdout.write(
                f"Scenario {s}: current best rejects = {best_by_s[s] if best_by_s[s] is not None else 'N/A'}"
            )

        async def run_one_slot(s: int) -> None:
            def stop_condition(rejected_count: int) -> bool:
                best = best_by_s.get(s)
                return best is not None and rejected_count > best

            await run_game_until(
                algorithm=resolved[s][0],
                stdout=self.stdout,
                scenario=s,
                use_server=True,
                model_path=resolved[s][1],
                verbose=verbose,
                stop_condition=stop_condition,
            )

            # Recompute best after this slot finishes a run
            row = (
                RemoteGame.objects.filter(status="completed", scenario=s)
                .values("scenario")
                .annotate(best=Count("people", filter=Q(people__decision=False)))
                .order_by("best")
                .first()
            )
            if row is not None:
                new_best = int(row.get("best", REJECTION_LIMIT))
                prev = best_by_s.get(s)
                if prev is None or new_best < prev:
                    best_by_s[s] = new_best
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"Updated best for scenario {s}: {new_best} rejects (prev={prev})"
                        )
                    )

        async def manage_scenario(s: int, slots: int) -> None:
            # Maintain up to `slots` concurrent runners for this scenario
            active: set[asyncio.Task[None]] = set()

            def launch() -> None:
                t = asyncio.create_task(run_one_slot(s))
                active.add(t)
                t.add_done_callback(lambda _t: active.discard(_t))

            while True:
                if not active:
                    for _ in range(slots):
                        launch()
                done, _ = await asyncio.wait(
                    active, return_when=asyncio.FIRST_COMPLETED
                )
                for _t in done:
                    launch()

        async def main() -> None:
            # Build counts by scenario value
            counts: dict[int, int] = {}
            for s in scenario_list:
                counts[s] = counts.get(s, 0) + 1
            await asyncio.gather(
                *(manage_scenario(s, slots) for s, slots in counts.items())
            )

        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            self.stdout.write("\nHypervisor stopped (KeyboardInterrupt).")
