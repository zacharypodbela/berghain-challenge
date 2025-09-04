from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces

from bouncer.constants import CAPACITY, REJECTION_LIMIT
from bouncer.models import LocalGame, Person

# Fixed union attribute order across all scenarios (see RL.md)
ATTRIBUTE_ORDER: list[str] = [
    "young",
    "well_dressed",
    "techno_lover",
    "well_connected",
    "creative",
    "berlin_local",
    "underground_veteran",
    "international",
    "fashion_forward",
    "queer_friendly",
    "vinyl_collector",
    "german_speaker",
]


def _one_hot_scenario(scenario: int) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float32)
    if 1 <= scenario <= 3:
        vec[scenario - 1] = 1.0
    return vec


class BerghainEnv(Env[np.ndarray, int]):
    """Gymnasium-compatible environment wrapping LocalGame.

    Notes:
        - Requires Django to be initialized (e.g., via manage.py or django.setup()).
        - Uses in-memory counters for speed; relies on LocalGame for transitions.
    """

    def __init__(self, scenario: int, seed: int | None = None) -> None:
        if scenario not in (1, 2, 3):
            raise ValueError("scenario must be one of {1,2,3}")
        self.scenario = scenario
        self._rng = np.random.default_rng(seed)

        # Spaces
        self.action_space = spaces.Discrete(2)  # type: ignore[assignment]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(31,), dtype=np.float32
        )

        # Episode state
        self.game: LocalGame | None = None
        self.current_person: Person | None = None
        self.admitted: int = 0
        self.rejected: int = 0
        # Per-attribute accepted counters
        self.accepted_attr_counts: dict[str, int] = dict.fromkeys(ATTRIBUTE_ORDER, 0)
        # Per-attribute minima for this scenario
        self.min_counts: dict[str, int] = dict.fromkeys(ATTRIBUTE_ORDER, 0)

    # Gymnasium interface -------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Start a fresh local game; first person is created pending
        self.game = LocalGame.start_new_game(self.scenario)

        # Initialize counters
        self.admitted = 0
        self.rejected = 0
        self.accepted_attr_counts = dict.fromkeys(ATTRIBUTE_ORDER, 0)

        # Build per-attribute minima map from game constraints
        assert self.game is not None
        constraints = {
            c["attribute"]: int(c["minCount"]) for c in self.game.constraints
        }
        self.min_counts = {a: constraints.get(a, 0) for a in ATTRIBUTE_ORDER}

        # Fetch current pending person
        self.current_person = Person.objects.get(game=self.game, decision__isnull=True)

        obs = self._build_observation()
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.game is None or self.current_person is None:
            raise RuntimeError("Environment must be reset() before step().")

        # Map action to decision
        accept = bool(action == 1)

        # Reward per step
        reward: float = 0.0 if accept else -1.0

        # Update in-memory counters before committing to DB (for speed)
        if accept:
            self.admitted += 1
            # Update per-attribute accepted counts
            for attr, has_attr in self.current_person.attributes.items():
                if has_attr:
                    # Only count known attributes from the fixed order
                    if attr in self.accepted_attr_counts:
                        self.accepted_attr_counts[attr] += 1
        else:
            self.rejected += 1

        # Apply decision via LocalGame (updates status and spawns next person if running)
        _resp = self.current_person.make_decision(accept)

        terminated = False
        truncated = False
        info: dict[str, Any] = {
            "status": self.game.status,
            "admitted": self.admitted,
            "rejected": self.rejected,
        }

        # Terminal handling and bonus/penalty
        if self.game.status != "running":
            terminated = True
            # Add terminal penalty only if capacity reached without meeting constraints
            if self.game.status == "failed" and self.admitted >= CAPACITY:
                reward += -1000.0

            # No next observation when done; return zeros for shape compatibility
            obs = np.zeros((31,), dtype=np.float32)
            return obs, reward, terminated, truncated, info

        # Still running: fetch next pending person and build observation
        self.current_person = Person.objects.get(game=self.game, decision__isnull=True)
        obs = self._build_observation()
        return obs, reward, terminated, truncated, info

    # Helpers -------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        assert self.game is not None and self.current_person is not None

        # Scenario one-hot
        s = _one_hot_scenario(self.scenario)  # (3,)

        # Global features
        admitted_frac = float(self.admitted) / float(CAPACITY)
        remaining_abs = CAPACITY - self.admitted
        remaining_frac = float(remaining_abs) / float(CAPACITY)
        rejection_pressure = float(self.rejected) / float(REJECTION_LIMIT)

        # Per-attribute deficits (absolute and fractional)
        remain_need_fracs: list[float] = []
        deficits_abs_total = 0
        for attr in ATTRIBUTE_ORDER:
            min_c = self.min_counts.get(attr, 0)
            accepted_c = self.accepted_attr_counts.get(attr, 0)
            deficit_abs = max(0, min_c - accepted_c)
            deficits_abs_total += deficit_abs
            remain_need_fracs.append(float(deficit_abs) / float(CAPACITY))

        slack_frac = float(max(0, remaining_abs - deficits_abs_total)) / float(CAPACITY)

        # Per-attribute current bits
        curr_bits: list[float] = []
        attrs: dict[str, Any] = self.current_person.attributes
        for attr in ATTRIBUTE_ORDER:
            bit = 1.0 if bool(attrs.get(attr, False)) else 0.0
            curr_bits.append(bit)

        # Interleave per-attribute pairs (curr_bit, remain_need_frac)
        pairs: list[float] = []
        for i in range(len(ATTRIBUTE_ORDER)):
            pairs.append(curr_bits[i])
            pairs.append(remain_need_fracs[i])

        obs = np.array(
            [
                *list(s),
                admitted_frac,
                remaining_frac,
                rejection_pressure,
                slack_frac,
                *pairs,
            ],
            dtype=np.float32,
        )
        return obs
