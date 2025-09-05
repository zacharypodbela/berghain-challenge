from __future__ import annotations

from enum import Enum
from typing import Any, cast

import numpy as np
from gymnasium import Env, spaces
from gymnasium.spaces import Space
from numpy.typing import NDArray

from bouncer.constants import CAPACITY, REJECTION_LIMIT, SCENARIO_CONFIGS
from bouncer.generate_attributes import CorrelatedAttributeGenerator
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


def _one_hot_scenario(scenario: int) -> NDArray[np.float32]:
    vec = np.zeros(3, dtype=np.float32)
    if 1 <= scenario <= 3:
        vec[scenario - 1] = 1.0
    return vec


def build_observation_vector(
    *,
    scenario: int,
    admitted: int,
    rejected: int,
    min_counts: dict[str, int],
    accepted_attr_counts: dict[str, int],
    current_attrs: dict[str, bool],
) -> NDArray[np.float32]:
    """Construct the 31-dim observation vector used by both env and wrappers.

    Fields: [one-hot scenario(3), admitted_frac, remaining_frac, rejection_pressure,
    slack_frac, for each attr in ATTRIBUTE_ORDER: (curr_bit, remain_need_frac)].
    """
    s = _one_hot_scenario(scenario)
    admitted_frac = float(admitted) / float(CAPACITY)
    remaining_abs = CAPACITY - admitted
    remaining_frac = float(remaining_abs) / float(CAPACITY)
    rejection_pressure = float(rejected) / float(REJECTION_LIMIT)

    remain_need_fracs: list[float] = []
    deficits_abs_total = 0
    for attr in ATTRIBUTE_ORDER:
        min_c = int(min_counts.get(attr, 0))
        accepted_c = int(accepted_attr_counts.get(attr, 0))
        deficit_abs = max(0, min_c - accepted_c)
        deficits_abs_total += deficit_abs
        remain_need_fracs.append(float(deficit_abs) / float(CAPACITY))

    slack_frac = float(max(0, remaining_abs - deficits_abs_total)) / float(CAPACITY)

    curr_bits: list[float] = []
    for attr in ATTRIBUTE_ORDER:
        curr_bits.append(1.0 if bool(current_attrs.get(attr, False)) else 0.0)

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


class EpisodeResult(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    CONSTRAINTS_UNMET_AT_CAPACITY = "constraints_unmet_at_capacity"
    REJECTION_LIMIT = "rejection_limit"
    FAILED = "failed"


class AbstractBerghainEnv(Env[NDArray[np.float32], int]):
    """Template-method Env sharing all logic; backends provide person sourcing and status."""

    action_space: Space[int]
    observation_space: Space[Any]

    # Shared episode state
    scenario: int
    admitted: int
    rejected: int
    accepted_attr_counts: dict[str, int]
    min_counts: dict[str, int]
    _current_attrs: dict[str, bool] | None

    def __init__(self, scenario: int, seed: int | None = None) -> None:
        if type(self) is AbstractBerghainEnv:
            raise NotImplementedError(
                "You should not instantiate AbstractBerghainEnv directly"
            )

        if scenario not in (1, 2, 3):
            raise ValueError("scenario must be one of {1,2,3}")
        self.scenario = scenario

        # Spaces
        self.action_space = spaces.Discrete(2)  # type: ignore[assignment]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(31,), dtype=np.float32
        )

        self.admitted = 0
        self.rejected = 0
        self.accepted_attr_counts = dict.fromkeys(ATTRIBUTE_ORDER, 0)
        self.min_counts = dict.fromkeys(ATTRIBUTE_ORDER, 0)
        self._current_attrs = None

        # Cached constraints for scenario
        cfg = SCENARIO_CONFIGS[self.scenario]
        constraints = {c["attribute"]: int(c["minCount"]) for c in cfg["constraints"]}
        self.min_counts = {a: constraints.get(a, 0) for a in ATTRIBUTE_ORDER}

    # Hooks to be implemented by subclasses --------------------------------
    def get_curr_person_attrs(self) -> dict[str, bool] | None:
        raise NotImplementedError

    # Gymnasium interface -------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        # Clear counters
        self.admitted = 0
        self.rejected = 0
        self.accepted_attr_counts = dict.fromkeys(ATTRIBUTE_ORDER, 0)
        self._current_attrs = None

        # First person
        self._current_attrs = self.get_curr_person_attrs()
        assert self._current_attrs is not None, (
            "No people available at start of episode"
        )

        obs = build_observation_vector(
            scenario=self.scenario,
            admitted=self.admitted,
            rejected=self.rejected,
            min_counts=self.min_counts,
            accepted_attr_counts=self.accepted_attr_counts,
            current_attrs=self._current_attrs,
        )
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._current_attrs is None:
            raise RuntimeError("Environment must be reset() before step().")

        accept = bool(action == 1)
        reward: float = 0.0 if accept else -1.0

        # Update counters
        if accept:
            self.admitted += 1
            for attr, has_attr in self._current_attrs.items():
                if has_attr and attr in self.accepted_attr_counts:
                    self.accepted_attr_counts[attr] += 1
        else:
            self.rejected += 1

        # Check if episode is done
        result = EpisodeResult.RUNNING
        if self.admitted >= CAPACITY:
            deficits = 0
            for attr in ATTRIBUTE_ORDER:
                min_c = self.min_counts.get(attr, 0)
                accepted_c = self.accepted_attr_counts.get(attr, 0)
                deficits += max(0, min_c - accepted_c)
            if deficits == 0:
                result = EpisodeResult.SUCCESS
            else:
                result = EpisodeResult.CONSTRAINTS_UNMET_AT_CAPACITY
        if self.rejected >= REJECTION_LIMIT:
            result = EpisodeResult.REJECTION_LIMIT

        terminated = result is not EpisodeResult.RUNNING
        truncated = False
        # Derive status string for info
        status_str = (
            "running"
            if result is EpisodeResult.RUNNING
            else ("completed" if result is EpisodeResult.SUCCESS else "failed")
        )
        info: dict[str, Any] = {
            "status": status_str,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "reason": result.value,
        }

        if terminated:
            if result is EpisodeResult.CONSTRAINTS_UNMET_AT_CAPACITY:
                reward += -1000.0
            return np.zeros((31,), dtype=np.float32), reward, True, truncated, info

        # Next person
        self._current_attrs = self.get_curr_person_attrs()
        assert self._current_attrs is not None, (
            "No more people available but episode not terminated"
        )
        obs = build_observation_vector(
            scenario=self.scenario,
            admitted=self.admitted,
            rejected=self.rejected,
            min_counts=self.min_counts,
            accepted_attr_counts=self.accepted_attr_counts,
            current_attrs=self._current_attrs,
        )
        return obs, reward, False, truncated, info


class DbBerghainEnv(AbstractBerghainEnv):
    """DB-backed env using LocalGame and Person models."""

    def __init__(self, scenario: int, seed: int | None = None) -> None:
        super().__init__(scenario, seed)
        self.game: LocalGame | None = None
        self._db_person: Person | None = None

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        self.game = LocalGame.start_new_game(self.scenario)
        self._db_person = Person.objects.get(game=self.game, decision__isnull=True)
        return super().reset(seed=seed, options=options)

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        assert self.game is not None and self._db_person is not None, (
            "Environment must be reset() before step()."
        )
        self._db_person.make_decision(bool(action))
        return super().step(action)

    def get_curr_person_attrs(self) -> dict[str, bool] | None:
        self._db_person = Person.objects.get(game=self.game, decision__isnull=True)
        return cast(dict[str, bool], self._db_person.attributes)


class SimBerghainEnv(AbstractBerghainEnv):
    """In-memory env with seeded correlated attribute generator (no DB)."""

    def __init__(self, scenario: int, seed: int | None = None) -> None:
        super().__init__(scenario, seed)
        cfg = SCENARIO_CONFIGS[self.scenario]
        self._person_attr_generator: CorrelatedAttributeGenerator = (
            CorrelatedAttributeGenerator(cfg["attribute_statistics"])
        )
        self._precomputed_people: list[dict[str, bool]] | None = None

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        # Pre-generate a large pool of people for this episode
        self._precomputed_people = self._person_attr_generator.sample(
            CAPACITY + REJECTION_LIMIT, seed=seed
        )
        return super().reset(seed=seed, options=options)

    def get_curr_person_attrs(self) -> dict[str, bool] | None:
        assert self._precomputed_people is not None, (
            "Environment must be reset() before fetching current person."
        )
        return self._precomputed_people.pop(0)
