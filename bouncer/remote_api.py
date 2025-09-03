"""
Remote API client for Berghain Challenge API
"""

import requests

API_BASE_URL = "https://berghain.challenges.listenlabs.ai"
PLAYER_ID = "5465bb54-f27c-48b7-9655-db22dc55a78b"


def create_new_game(scenario, retries=4):
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
    params = {"scenario": scenario, "playerId": PLAYER_ID}

    # Try once, then retry on 500 error
    for attempt in range(retries + 1):
        response = requests.get(url, params=params)
        if response.status_code == 500 and attempt != retries:
            continue  # Retry on first 500 error
        response.raise_for_status()
        return response.json()


def make_decision_and_get_next(game_id, person_index, accept=None, retries=4):
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
    params = {
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
        return response.json()
