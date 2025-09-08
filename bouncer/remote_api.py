"""
Remote API client for Berghain Challenge API (async, httpx)
"""

from typing import Any, cast

import httpx

API_BASE_URL = "https://berghain.challenges.listenlabs.ai"
PLAYER_ID = "5465bb54-f27c-48b7-9655-db22dc55a78b"


async def create_new_game(scenario: int, retries: int = 4) -> dict[str, Any]:
    """
    Create a new game via the API.

    Args:
        scenario: Scenario number (1, 2, or 3)
        retries: Number of retries on HTTP 500 before giving up

    Returns:
        JSON response as a dict.
    """
    if scenario not in [1, 2, 3]:
        raise ValueError("Scenario must be 1, 2, or 3")

    url = f"{API_BASE_URL}/new-game"
    params: dict[str, int | str] = {"scenario": scenario, "playerId": PLAYER_ID}

    async with httpx.AsyncClient() as client:
        for attempt in range(retries + 1):
            resp = await client.get(url, params=params, timeout=15.0)
            if resp.status_code == 500 and attempt != retries:
                continue
            resp.raise_for_status()
            return cast(dict[str, Any], resp.json())

    raise httpx.HTTPError("Maximum retries exceeded")


async def make_decision_and_get_next(
    game_id: str, person_index: int, accept: bool | None = None, retries: int = 4
) -> dict[str, Any]:
    """
    Make a decision on a person and get the next person.

    Args:
        game_id: The game ID
        person_index: Index of the person to make decision on
        accept: True to accept, False to reject, None to just fetch next
        retries: Number of retries on HTTP 500 before giving up

    Returns:
        JSON response as a dict.
    """
    url = f"{API_BASE_URL}/decide-and-next"
    params: dict[str, str | int] = {
        "gameId": game_id,
        "personIndex": person_index,
    }
    if accept is not None:
        params["accept"] = str(accept).lower()

    async with httpx.AsyncClient() as client:
        for attempt in range(retries + 1):
            resp = await client.get(url, params=params, timeout=15.0)
            if resp.status_code == 500 and attempt != retries:
                continue
            resp.raise_for_status()
            return cast(dict[str, Any], resp.json())

    raise httpx.HTTPError("Maximum retries exceeded")
