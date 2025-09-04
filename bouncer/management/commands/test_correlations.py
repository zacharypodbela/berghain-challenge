"""
Django management command to test the correlation generation implementation
"""

from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from bouncer.test_correlations import (
    test_all_scenarios,
    validate_test_script_with_real_games,
)


class Command(BaseCommand):
    help = "Test the correlation generation implementation"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--sample-size",
            type=int,
            default=200000,
            help="Number of people to generate for testing (default: 200000)",
        )
        parser.add_argument(
            "--test-real-games",
            action="store_true",
            help="Also test against real game data from database",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        sample_size = options["sample_size"]
        test_real_games = options["test_real_games"]

        if test_real_games:
            validate_test_script_with_real_games()
        else:
            test_all_scenarios(sample_size)
