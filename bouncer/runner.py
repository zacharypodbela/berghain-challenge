from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta

from django.utils import timezone

from bouncer.algorithms import AlgorithmFunc
from bouncer.models import Game, LocalGame, RemoteGame


async def wait_for_remote_game_capacity(log: Callable[[str], None]) -> None:
    """Respect remote creation limit: max 10 RemoteGames per rolling 15 minutes.

    Blocks until capacity is available. Mirrors logic from run_algorithm.
    """
    from bouncer.models import RemoteGame  # late import to avoid cycles

    while True:
        now = timezone.now()
        window_start = now - timedelta(minutes=15, seconds=30)  # Add 30s buffer
        recent_qs = RemoteGame.objects.filter(created_at__gte=window_start)
        count = await recent_qs.acount()
        if count < 10:
            return

        oldest = await recent_qs.order_by("created_at").afirst()
        if not oldest:
            # Defensive: if query returns no row unexpectedly, do not block
            return

        target_time = oldest.created_at + timedelta(minutes=15, seconds=30)
        wait_seconds = max(1.0, float((target_time - now).total_seconds()))
        log(
            f"Remote game creation limit reached: Sleeping {int(wait_seconds)}s until {target_time.isoformat()}\n"
        )
        await asyncio.sleep(wait_seconds)


async def run_game_until(
    algorithm: AlgorithmFunc,
    log: Callable[[str], None],
    game_id: str | None = None,
    scenario: int | None = None,
    use_server: bool = False,
    model_path: str | None = None,
    verbose: bool = False,
    stop_condition: Callable[[int], bool] | None = None,
) -> None:
    """Run algorithm on a game until completion, error, or stop_condition.

    Reuses the decision + DB update loop used by the run_algorithm command.
    """
    if scenario:
        # Start one or more new LocalGame/RemoteGame episodes for the given scenario
        if use_server:
            # Respect remote creation limit: max 10 per rolling 15 minutes
            await wait_for_remote_game_capacity(log)
            game = await RemoteGame.astart_new_game(int(scenario))
        else:
            game = await LocalGame.astart_new_game(int(scenario))
        game.tags.append(f"algorithm:{algorithm.__name__}")
        if model_path:
            game.tags.append(f"model:{model_path}")
        await game.asave()
        log(
            f"Created {'RemoteGame' if use_server else 'LocalGame'} {game.game_id} for scenario {game.scenario}"
        )
    else:
        # Get and validate single game
        try:
            game = await Game.objects.aget(game_id=game_id)
            if game.status != "running":
                raise ValueError(
                    f'Game "{game_id}" has status "{game.status}" - can only run on "running" games'
                )
        except Game.DoesNotExist as e:
            raise ValueError(f'Game "{game_id}" does not exist') from e

    stats = await game.attribute_and_top_of_house_counts

    log(
        f"Starting algorithm run:\n"
        f"Game: {game.game_id} (Scenario {game.scenario})\n"
        f"Algorithm: {algorithm.__name__}\n"
        f"Current status: {game.status}\n"
        f"Current counts - Admitted: {stats['admitted']}, "
        f"Rejected: {stats['rejected']}, Pending: {stats['pending']}\n"
    )

    # Check if we can run on this game
    if game.status == "completed":
        raise ValueError("Game is already completed")
    elif game.status == "failed":
        raise ValueError("Game has failed - cannot restart")
    elif game.status == "error":
        raise ValueError("Game has an error status - cannot restart")

    decisions_made = 0

    while game.status == "running":
        if stop_condition is not None and stop_condition(stats["rejected"]):
            log(
                f"Stopping early due to stop condition.\n"
                f"Current rejections: {stats['rejected']}\n"
                f"Decisions made this run: {decisions_made}"
            )
            break

        # Get next pending person
        pending_person = (
            await game.people.filter(decision__isnull=True)
            .order_by("person_index")
            .afirst()
        )

        # Make decision using algorithm
        decision = await algorithm(pending_person, game, log, model_path)
        decision_text = "ACCEPT" if decision else "REJECT"

        if verbose:
            log(
                f"Person #{pending_person.person_index}: {decision_text} "
                f"(Attributes: {pending_person.attributes})"
            )

        try:
            # Make the API call and update database
            await game.amake_decision_and_get_next(
                person=pending_person, accept=decision
            )
            decisions_made += 1

            # Refresh game status
            stats = await game.attribute_and_top_of_house_counts

            # Show updated counts
            if verbose:
                log(
                    f"  → Counts: Admitted={stats['admitted']}, "
                    f"Rejected={stats['rejected']}, Pending={stats['pending']}"
                )

            # Check if game ended
            if game.status in ["completed", "failed"]:
                status = game.status
                log(
                    f"Game {status.upper()}!\n"
                    f"Final score (rejections): {stats['rejected']}\n"
                    f"Total decisions made this run: {decisions_made}"
                )
                break

        except Exception as e:
            log(
                f"ERROR making decision: {str(e)}\n"
                f"Stopping algorithm run. You can restart with the same command."
            )
            return

    log("Algorithm run completed.")
