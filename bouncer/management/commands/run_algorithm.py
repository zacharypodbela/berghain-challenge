"""
Django management command to run bouncer algorithms on games
"""

import time
from datetime import timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.utils import timezone

from bouncer.algorithms import ALGORITHMS, AlgorithmFunc, get_algorithm
from bouncer.models import Game, LocalGame, RemoteGame
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
            "--delay",
            type=float,
            default=0.0,
            help="Delay in seconds between decisions (default: 0.0)",
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
        delay = options["delay"]
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

        # Get algorithm
        try:
            algorithm = get_algorithm(algorithm_name)
        except ValueError as e:
            raise CommandError(str(e)) from e

        # Loop over one or more games
        for idx in range(1, n_games + 1):
            if n_games > 1:
                self.stdout.write(self.style.MIGRATE_HEADING(f"\nRun {idx}/{n_games}"))

            if scenario:
                # Start one or more new LocalGame/RemoteGame episodes for the given scenario
                if use_server:
                    # Respect remote creation limit: max 10 per rolling 15 minutes
                    self._wait_for_remote_game_capacity()
                    game = RemoteGame.start_new_game(int(scenario))
                else:
                    game = LocalGame.start_new_game(int(scenario))
                game.tags.append(f"algorithm:{algorithm_name}")
                if model_path:
                    game.tags.append(f"model:{model_path}")
                game.save()
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Created {'RemoteGame' if use_server else 'LocalGame'} {game.game_id} for scenario {game.scenario}"
                    )
                )
            else:
                # Get game ID from user if not provided
                if not game_id:
                    self.stdout.write("\nAvailable games:")
                    games = (
                        Game.objects.filter(status="running")
                        .all()
                        .order_by("-created_at")
                    )
                    for game in games[:10]:  # Show latest 10 games
                        status_color = self.get_status_color(game.status)
                        if status_color:
                            status_text = status_color(game.status)
                        else:
                            status_text = game.status
                        self.stdout.write(
                            f"  {game.game_id} (Scenario {game.scenario}, "
                            f"Status: {status_text}, "
                            f"People: {game.people.count()})"
                        )

                    game_id = input("\nEnter game ID to run algorithm on: ").strip()
                    if not game_id:
                        raise CommandError("No game ID provided")

                # Get and validate single game
                try:
                    game = Game.objects.get(game_id=game_id)
                    if game.status != "running":
                        raise CommandError(
                            f'Game "{game_id}" has status "{game.status}" - can only run on "running" games'
                        )
                except Game.DoesNotExist as e:
                    raise CommandError(f'Game "{game_id}" does not exist') from e

            self.stdout.write(
                f"\n{self.style.SUCCESS('Starting algorithm run:')}\n"
                f"Game: {game.game_id} (Scenario {game.scenario})\n"
                f"Algorithm: {algorithm_name}\n"
                f"Current status: {game.status}\n"
                f"Current counts - Admitted: {game.admitted_count}, "
                f"Rejected: {game.rejected_count}, Pending: {game.pending_count}\n"
            )

            # Check if we can run on this game
            if game.status == "completed":
                raise CommandError("Game is already completed")
            elif game.status == "failed":
                raise CommandError("Game has failed - cannot restart")
            elif game.status == "error":
                raise CommandError("Game has an error status - cannot restart")

            # Handle restart logic
            self.handle_game_restart(game)

            # Run the algorithm
            self.run_algorithm(game, algorithm, delay, model_path, verbose)

    def get_status_color(self, status: str) -> Any | None:
        """Get color styling for game status"""
        colors = {
            "running": self.style.WARNING,
            "completed": self.style.SUCCESS,
            "failed": self.style.ERROR,
            "error": self.style.ERROR,
        }
        return colors.get(status, "")

    def handle_game_restart(self, game: Game) -> None:
        """Handle restarting a game from where we left off"""
        pending_people = game.people.filter(decision__isnull=True).order_by(
            "person_index"
        )

        if not pending_people.exists():
            # No pending people - something is wrong if game isn't completed
            if game.status == "running":
                self.stdout.write(
                    f"{self.style.ERROR('ERROR:')} Game is running but has no pending people. "
                    f"Setting status to 'error'."
                )
                game.status = "error"
                game.save()
                raise CommandError("Game state is inconsistent - marked as error")
        else:
            first_pending = pending_people.first()
            self.stdout.write(
                f"Resuming from Person #{first_pending.person_index} "
                f"(found {pending_people.count()} pending people)"
            )

    def run_algorithm(
        self,
        game: Game,
        algorithm: AlgorithmFunc,
        delay: float,
        model_path: str | None,
        verbose: bool,
    ) -> None:
        """Run the algorithm on the game until completion or error"""
        decisions_made = 0

        while game.status == "running":
            # Get next pending person
            pending_person = (
                game.people.filter(decision__isnull=True)
                .order_by("person_index")
                .first()
            )

            # Make decision using algorithm
            decision = algorithm(pending_person, game, self.stdout, model_path)
            decision_text = "ACCEPT" if decision else "REJECT"

            if verbose:
                self.stdout.write(
                    f"Person #{pending_person.person_index}: {decision_text} "
                    f"(Attributes: {pending_person.attributes})"
                )

            try:
                # Make the API call and update database
                pending_person.make_decision(accept=decision)
                decisions_made += 1

                # Refresh game status
                game.refresh_from_db()

                # Show updated counts
                if verbose:
                    self.stdout.write(
                        f"  → Counts: Admitted={game.admitted_count}, "
                        f"Rejected={game.rejected_count}, Pending={game.pending_count}"
                    )

                # Check if game ended
                if game.status in ["completed", "failed"]:
                    status = game.status
                    self.stdout.write(
                        f"\n{self.style.SUCCESS(f'Game {status.upper()}!')}\n"
                        f"Final score (rejections): {game.rejected_count}\n"
                        f"Total decisions made this run: {decisions_made}"
                    )
                    # Refresh game to get the updated completion_reason
                    game.refresh_from_db()
                    if game.completion_reason:
                        self.stdout.write(f"Reason: {game.completion_reason}")
                    break

                # Add delay between decisions
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                self.stdout.write(
                    f"{self.style.ERROR('ERROR making decision:')} {str(e)}\n"
                    f"Stopping algorithm run. You can restart with the same command."
                )
                break

        self.stdout.write(
            f"\nAlgorithm run completed. Decisions made: {decisions_made}"
        )

    def _wait_for_remote_game_capacity(self) -> None:
        """
        Ensure fewer than 10 RemoteGames exist in the last 15 minutes.
        If the cap is reached, sleep until capacity frees up, rechecking on wake.
        """
        while True:
            now = timezone.now()
            window_start = now - timedelta(minutes=15, seconds=30)  # Add 30s buffer
            recent_qs = RemoteGame.objects.filter(created_at__gte=window_start)
            count = int(recent_qs.count())
            if count < 10:
                return

            oldest = recent_qs.order_by("created_at").first()
            if not oldest:
                # Defensive: if query returns no row unexpectedly, do not block
                return

            target_time = oldest.created_at + timedelta(minutes=15)
            wait_seconds = (target_time - now).total_seconds()
            # Guard against race/clock edge cases
            wait_seconds = max(1.0, float(wait_seconds))

            self.stdout.write(
                self.style.WARNING(
                    "Remote game creation limit reached: "
                    f"Sleeping {int(wait_seconds)}s until {target_time.isoformat()}"
                )
            )
            time.sleep(wait_seconds)
