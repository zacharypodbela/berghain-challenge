"""
Remote API client for Berghain Challenge API
"""

from typing import Any, cast

import requests

API_BASE_URL = "https://berghain.challenges.listenlabs.ai"
PLAYER_ID = "5465bb54-f27c-48b7-9655-db22dc55a78b"


def create_new_game(scenario: int, retries: int = 4) -> dict[str, Any]:
    """
    Create a new game via the API.

    Args:
        scenario (int): Scenario number (1, 2, or 3)

    Returns:
        dict: API response data

    Raises:
        ValueError: If scenario is invalid
        requests.RequestException: If API call fails
    """
    if scenario not in [1, 2, 3]:
        raise ValueError("Scenario must be 1, 2, or 3")

    url = f"{API_BASE_URL}/new-game"
    params: dict[str, int | str] = {"scenario": scenario, "playerId": PLAYER_ID}

    # Try once, then retry on 500 error
    for attempt in range(retries + 1):
        response = requests.get(url, params=params)
        if response.status_code == 500 and attempt != retries:
            continue  # Retry on first 500 error
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    # This should never be reached due to raise_for_status, but mypy needs it
    raise requests.RequestException("Maximum retries exceeded")


def make_decision_and_get_next(
    game_id: str, person_index: int, accept: bool | None = None, retries: int = 4
) -> dict[str, Any]:
    """
    Make a decision on a person and get the next person.

    Args:
        game_id (str): The game ID
        person_index (int): Index of the person to make decision on
        accept (bool): True to accept, False to reject

    Returns:
        dict: API response data

    Raises:
        requests.RequestException: If API call fails
    """
    url = f"{API_BASE_URL}/decide-and-next"
    params: dict[str, str | int] = {
        "gameId": game_id,
        "personIndex": person_index,
    }
    if accept is not None:
        params["accept"] = str(accept).lower()

    # Try once, then retry on 500 error
    for attempt in range(retries + 1):
        response = requests.get(url, params=params)
        if response.status_code == 500 and attempt != retries:
            continue  # Retry on first 500 error
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    # This should never be reached due to raise_for_status, but mypy needs it
    raise requests.RequestException("Maximum retries exceeded")
