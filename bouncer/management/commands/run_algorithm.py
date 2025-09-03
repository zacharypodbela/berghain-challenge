"""
Django management command to run bouncer algorithms on games
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from bouncer.models import Game, Person
from bouncer.algorithms import get_algorithm, ALGORITHMS
import time


class Command(BaseCommand):
    help = "Run a bouncer algorithm on a game"

    def add_arguments(self, parser):
        parser.add_argument(
            "--game-id",
            type=str,
            help="Game ID to run algorithm on (if not provided, will ask for input)",
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

    def handle(self, *args, **options):
        game_id = options["game_id"]
        algorithm_name = options["algorithm"]
        delay = options["delay"]

        # Get game ID from user if not provided
        if not game_id:
            self.stdout.write("\nAvailable games:")
            games = Game.objects.all().order_by("-created_at")
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

        # Get algorithm from user if not provided
        if not algorithm_name:
            self.stdout.write("\nAvailable algorithms:")
            for name in ALGORITHMS.keys():
                self.stdout.write(f"  {name}")
            
            algorithm_name = input("\nEnter algorithm to use: ").strip()
            if not algorithm_name:
                raise CommandError("No algorithm provided")

        # Get and validate game
        try:
            game = Game.objects.get(game_id=game_id)
        except Game.DoesNotExist:
            raise CommandError(f'Game "{game_id}" does not exist')

        # Get algorithm
        try:
            algorithm = get_algorithm(algorithm_name)
        except ValueError as e:
            raise CommandError(str(e))

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
        self.run_algorithm(game, algorithm, delay)

    def get_status_color(self, status):
        """Get color styling for game status"""
        colors = {
            "running": self.style.WARNING,
            "completed": self.style.SUCCESS,
            "failed": self.style.ERROR,
            "error": self.style.ERROR,
        }
        return colors.get(status, "")

    def handle_game_restart(self, game):
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

    def run_algorithm(self, game, algorithm, delay):
        """Run the algorithm on the game until completion or error"""
        decisions_made = 0

        while game.status == "running":
            # Get next pending person
            pending_person = (
                game.people.filter(decision__isnull=True)
                .order_by("person_index")
                .first()
            )

            if not pending_person:
                self.stdout.write(
                    f"{self.style.ERROR('No pending people found - updating game status')}"
                )
                game.refresh_from_db()
                break

            # Make decision using algorithm
            decision = algorithm(pending_person, game)
            decision_text = "ACCEPT" if decision else "REJECT"

            self.stdout.write(
                f"Person #{pending_person.person_index}: {decision_text} "
                f"(Attributes: {pending_person.attributes})"
            )

            try:
                # Make the API call and update database
                api_response = pending_person.make_decision(accept=decision)
                decisions_made += 1

                # Refresh game status
                game.refresh_from_db()

                # Show updated counts
                self.stdout.write(
                    f"  → Counts: Admitted={game.admitted_count}, "
                    f"Rejected={game.rejected_count}, Pending={game.pending_count}"
                )

                # Check if game ended
                if api_response.get("status") in ["completed", "failed"]:
                    status = api_response.get("status")
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
