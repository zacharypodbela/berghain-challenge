from __future__ import annotations

import os
from typing import Any

import numpy as np
from django.core.management.base import BaseCommand, CommandError
from numpy.typing import NDArray

from bouncer.models import Game, Person
from bouncer.rl.env import ATTRIBUTE_ORDER, build_observation_vector


def _iter_games(
    scenarios: set[int] | None,
    statuses: set[str] | None,
    game_ids: set[str] | None,
) -> Any:
    qs = Game.objects.all().order_by("created_at")
    if scenarios:
        qs = qs.filter(scenario__in=list(scenarios))
    if statuses:
        qs = qs.filter(status__in=list(statuses))
    if game_ids:
        qs = qs.filter(game_id__in=list(game_ids))
    return qs


class Command(BaseCommand):
    help = "Export imitation dataset from DB games to NPZ (obs/actions/meta)."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--out", required=True, help="Output NPZ filepath.")
        parser.add_argument(
            "--scenarios",
            type=str,
            default="",
            help="Comma-separated scenarios to include (e.g., '1,2'). Empty=all.",
        )
        parser.add_argument(
            "--statuses",
            type=str,
            default="",
            help="Comma-separated statuses to include (running,completed,failed). Empty=all.",
        )
        parser.add_argument(
            "--games",
            type=str,
            default="",
            help="Comma-separated specific game_ids to include. Empty=all.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        out_path = str(options["out"]).strip()
        if not out_path.endswith(".npz"):
            raise CommandError("--out must end with .npz")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        scenarios: set[int] | None = None
        if options["scenarios"]:
            scenarios = {int(s) for s in options["scenarios"].split(",") if s.strip()}

        statuses: set[str] | None = None
        if options["statuses"]:
            statuses = {s.strip() for s in options["statuses"].split(",") if s.strip()}

        game_ids: set[str] | None = None
        if options["games"]:
            game_ids = {s.strip() for s in options["games"].split(",") if s.strip()}

        obs_list: list[NDArray[np.float32]] = []
        actions_list: list[int] = []
        scenarios_list: list[int] = []
        episodes_list: list[int] = []
        episode_game_ids: list[str] = []
        helpful_list: list[int] = []  # 1/0
        episode_statuses: list[str] = []

        episode_idx = 0
        total_steps = 0

        for game in _iter_games(scenarios, statuses, game_ids):
            # Prepare per-episode counters/state
            constraints = {c["attribute"]: int(c["minCount"]) for c in game.constraints}
            min_counts = {a: int(constraints.get(a, 0)) for a in ATTRIBUTE_ORDER}
            accepted_attr_counts: dict[str, int] = dict.fromkeys(ATTRIBUTE_ORDER, 0)
            admitted = 0
            rejected = 0

            people = list(
                Person.objects.filter(game=game)
                .order_by("person_index")
                .values("attributes", "decision")
            )
            if not people:
                continue

            for row in people:
                decision = row["decision"]
                if decision is None:
                    break  # stop at first pending
                attrs: dict[str, bool] = row["attributes"]

                # Build observation BEFORE applying this decision
                obs = build_observation_vector(
                    scenario=int(game.scenario),
                    admitted=admitted,
                    rejected=rejected,
                    min_counts=min_counts,
                    accepted_attr_counts=accepted_attr_counts,
                    current_attrs=attrs,
                )
                obs_list.append(obs)
                act = 1 if bool(decision) else 0
                actions_list.append(act)
                scenarios_list.append(int(game.scenario))
                episodes_list.append(episode_idx)

                # Helpful label: deficit strictly decreases on accept
                helpful = 0
                if act == 1:
                    before = sum(
                        max(0, int(min_counts[a]) - int(accepted_attr_counts[a]))
                        for a in ATTRIBUTE_ORDER
                    )
                    # Apply accept updates
                    admitted += 1
                    for a in ATTRIBUTE_ORDER:
                        if bool(attrs.get(a, False)):
                            accepted_attr_counts[a] += 1
                    after = sum(
                        max(0, int(min_counts[a]) - int(accepted_attr_counts[a]))
                        for a in ATTRIBUTE_ORDER
                    )
                    helpful = 1 if after < before else 0
                else:
                    rejected += 1

                helpful_list.append(helpful)
                total_steps += 1

            episode_game_ids.append(str(game.game_id))
            episode_statuses.append(str(game.status))
            episode_idx += 1

        if not obs_list:
            raise CommandError("No matching data to export.")

        obs_arr = np.stack(obs_list).astype(np.float32)
        actions_arr = np.array(actions_list, dtype=np.int64)
        scenarios_arr = np.array(scenarios_list, dtype=np.int64)
        episodes_arr = np.array(episodes_list, dtype=np.int64)
        episode_game_ids_arr = np.array(episode_game_ids)
        episode_statuses_arr = np.array(episode_statuses)
        helpful_arr = np.array(helpful_list, dtype=np.int8)

        np.savez_compressed(
            out_path,
            obs=obs_arr,
            actions=actions_arr,
            scenarios=scenarios_arr,
            episodes=episodes_arr,
            episode_game_ids=episode_game_ids_arr,
            episode_statuses=episode_statuses_arr,
            helpful=helpful_arr,
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Exported {obs_arr.shape[0]} steps from {episode_idx} episodes -> {out_path}"
            )
        )
