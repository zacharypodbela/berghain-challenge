"""
Django management command to run bouncer algorithms on games
"""

import asyncio
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from bouncer.algorithms import ALGORITHMS, get_algorithm
from bouncer.models import Game
from bouncer.runner import run_game_until


class Command(BaseCommand):
    help = "Run a bouncer algorithm on a game"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--game-id",
            type=str,
            help="Game ID to run algorithm on (if not provided, will ask for input)",
        )
        parser.add_argument(
            "--scenario",
            type=int,
            choices=[1, 2, 3],
            help="If provided (and no --game-id), start a new LocalGame for this scenario and run bouncer algorithm on that.",
        )
        parser.add_argument(
            "--n-games",
            type=int,
            default=1,
            help="Run algorithm on N new Games (requires --scenario). Will use LocalGame unless --server flag is passed.",
        )
        parser.add_argument(
            "--server",
            action="store_true",
            help=(
                "Use RemoteGame (server API) when starting new game with --scenario."
            ),
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            help="Algorithm to use (if not provided, will ask for input)",
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default="",
            help="Optional path to a model file for model-based algorithms (e.g., PPO).",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help=("Enable per-decision logging (default is quiet)."),
        )

    def handle(self, *args: Any, **options: Any) -> None:
        game_id = options["game_id"]
        scenario = options.get("scenario")
        use_server = bool(options.get("server"))
        n_games = int(options.get("n_games") or 1)
        algorithm_name = options["algorithm"]
        model_path = options.get("model_path")
        verbose = bool(options.get("verbose"))

        # Validate scenario/game-id combinations
        if game_id and scenario:
            raise CommandError("Provide either --game-id or --scenario, not both.")
        if use_server and not scenario:
            raise CommandError("--server can only be used with --scenario.")
        if n_games != 1 and not scenario:
            raise CommandError("--n-games can only be used with --scenario.")

        # Get algorithm from user if not provided
        if not algorithm_name:
            self.stdout.write("\nAvailable algorithms:")
            for name in ALGORITHMS.keys():
                self.stdout.write(f"  {name}")

            algorithm_name = input("\nEnter algorithm to use: ").strip()
            if not algorithm_name:
                raise CommandError("No algorithm provided")

        # Get game ID from user if not provided and scenario not given
        if not game_id and not scenario:
            self.stdout.write("\nAvailable games:")
            games = Game.objects.filter(status="running").all().order_by("-created_at")
            for game in games[:10]:  # Show latest 10 games
                self.stdout.write(
                    f"{game.game_id} (Scenario {game.scenario}, "
                    f"Status: {game.status}, "
                    f"People: {game.people.count()})"
                )

            game_id = input("\nEnter game ID to run algorithm on: ").strip()
            if not game_id:
                raise CommandError("No game ID provided")

        # Get algorithm
        try:
            algorithm = get_algorithm(algorithm_name)
        except ValueError as e:
            raise CommandError(str(e)) from e

        # Loop over one or more games
        for idx in range(1, n_games + 1):
            if n_games > 1:
                self.stdout.write(self.style.MIGRATE_HEADING(f"\nRun {idx}/{n_games}"))

            # Run the algorithm
            asyncio.run(
                run_game_until(
                    algorithm,
                    self.stdout,
                    game_id=game_id,
                    scenario=scenario,
                    use_server=use_server,
                    model_path=model_path,
                    verbose=verbose,
                )
            )
