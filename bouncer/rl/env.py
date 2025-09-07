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
    capacity: int,
) -> NDArray[np.float32]:
    """Construct the 31-dim observation vector used by both env and wrappers.

    Fields: [one-hot scenario(3), admitted_frac, remaining_frac, rejection_pressure,
    slack_frac, for each attr in ATTRIBUTE_ORDER: (curr_bit, remain_need_frac)].
    """
    s = _one_hot_scenario(scenario)
    admitted_frac = float(admitted) / float(capacity)
    remaining_abs = capacity - admitted
    remaining_frac = float(remaining_abs) / float(capacity)
    rejection_pressure = float(rejected) / float(REJECTION_LIMIT)

    remain_need_fracs: list[float] = []
    deficits_abs_total = 0
    for attr in ATTRIBUTE_ORDER:
        min_c = int(min_counts.get(attr, 0))
        accepted_c = int(accepted_attr_counts.get(attr, 0))
        deficit_abs = max(0, min_c - accepted_c)
        deficits_abs_total += deficit_abs
        remain_need_fracs.append(float(deficit_abs) / float(capacity))

    slack_frac = float(max(0, remaining_abs - deficits_abs_total)) / float(capacity)

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
    capacity: int

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
        self.capacity = CAPACITY

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
            capacity=self.capacity,
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
        if self.admitted >= self.capacity:
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
                # Make failing at capacity strictly worse than any plausible
                # successful run, by penalizing with the full rejection limit.
                reward += -float(REJECTION_LIMIT)
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
            capacity=self.capacity,
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
        self.game.tags.append("db-berghain-env")
        self.game.save()
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

    def __init__(
        self,
        scenario: int,
        seed: int | None = None,
        *,
        capacity: int | None = None,
        min_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__(scenario, seed)
        cfg = SCENARIO_CONFIGS[self.scenario]
        self._person_attr_generator: CorrelatedAttributeGenerator = (
            CorrelatedAttributeGenerator(cfg["attribute_statistics"])
        )
        self._precomputed_people: list[dict[str, bool]] | None = None
        if capacity is not None:
            self.capacity = int(capacity)
        if min_counts is not None:
            self.min_counts = {a: int(min_counts.get(a, 0)) for a in ATTRIBUTE_ORDER}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        # Pre-generate a large pool of people for this episode
        self._precomputed_people = self._person_attr_generator.sample(
            self.capacity + REJECTION_LIMIT, seed=seed
        )
        return super().reset(seed=seed, options=options)

    def get_curr_person_attrs(self) -> dict[str, bool] | None:
        assert self._precomputed_people is not None, (
            "Environment must be reset() before fetching current person."
        )
        return self._precomputed_people.pop(0)


class DeficitRewardWrapper(Env[NDArray[np.float32], int]):
    """Reward shaping: adds positive reward when deficits decrease after an accept.

    Shaping term at step t: coef * (deficits_before - deficits_after), where
    deficits = sum(max(0, min_count[attr] - accepted_attr_counts[attr])) over attributes.
    This is potential-based (monotone w.r.t deficits), preserving optimal policy while
    providing dense feedback.
    """

    def __init__(
        self,
        env: AbstractBerghainEnv,
        coef: float = 1.0,
        nonhelp_penalty: float = 0.0,
        success_bonus: float = 0.0,
        minmeet_bonus: float = 0.0,
        fail_penalty_scale: float = 1.0,
        success_bonus_per_saved: float = 0.0,
        late_reject_weight: float = 0.0,
    ) -> None:
        self.env = env
        self.coef = float(coef)
        self.nonhelp_penalty = float(nonhelp_penalty)
        self.success_bonus = float(success_bonus)
        self.minmeet_bonus = float(minmeet_bonus)
        self.fail_penalty_scale = float(fail_penalty_scale)
        self.success_bonus_per_saved = float(success_bonus_per_saved)
        self.late_reject_weight = float(late_reject_weight)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(
        self, *args: Any, **kwargs: Any
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        base_env: AbstractBerghainEnv = self.env

        # Compute deficits before (by-attr and total)
        def _attr_deficit(a: str) -> int:
            return max(
                0,
                int(base_env.min_counts.get(a, 0))
                - int(base_env.accepted_attr_counts.get(a, 0)),
            )

        deficits_before_total = 0
        deficits_before_attr: dict[str, int] = {}
        for attr in ATTRIBUTE_ORDER:
            d = _attr_deficit(attr)
            deficits_before_attr[attr] = d
            deficits_before_total += d

        obs, reward, terminated, truncated, info = base_env.step(action)

        # Compute deficits after (by-attr and total)
        deficits_after_total = 0
        deficits_after_attr: dict[str, int] = {}
        for attr in ATTRIBUTE_ORDER:
            d = _attr_deficit(attr)
            deficits_after_attr[attr] = d
            deficits_after_total += d

        shaped = reward + self.coef * float(
            deficits_before_total - deficits_after_total
        )
        # Penalize non-helpful accepts while deficits remain
        if (
            action == 1
            and deficits_before_total > 0
            and deficits_after_total == deficits_before_total
        ):
            shaped -= self.nonhelp_penalty
        # Additional penalty for late rejections (when slack is low)
        if self.late_reject_weight != 0.0 and action == 0:
            remaining_abs = int(base_env.capacity) - int(base_env.admitted)
            slack_after = max(0, remaining_abs - int(deficits_after_total))
            slack_after_frac = float(slack_after) / float(max(1, base_env.capacity))
            shaped -= self.late_reject_weight * (1.0 - slack_after_frac)
        # Bonus for meeting any minimum exactly at this step
        if self.minmeet_bonus != 0.0:
            met_now = 0
            for attr in ATTRIBUTE_ORDER:
                if deficits_before_attr[attr] > 0 and deficits_after_attr[attr] == 0:
                    met_now += 1
            if met_now > 0:
                shaped += self.minmeet_bonus * float(met_now)
        # Terminal shaping
        if terminated:
            reason = info.get("reason")
            if reason == EpisodeResult.SUCCESS.value:
                # Fixed success bonus
                shaped += self.success_bonus
                # Bonus proportional to rejections saved in this episode
                if self.success_bonus_per_saved != 0.0:
                    saved = float(REJECTION_LIMIT - int(base_env.rejected))
                    shaped += self.success_bonus_per_saved * saved
            elif reason == EpisodeResult.CONSTRAINTS_UNMET_AT_CAPACITY.value:
                # Base env already applied -REJECTION_LIMIT; scale it by adding back a fraction
                # Effective penalty becomes -fail_penalty_scale * REJECTION_LIMIT
                if self.fail_penalty_scale != 1.0:
                    shaped += (1.0 - self.fail_penalty_scale) * float(REJECTION_LIMIT)
        return obs, shaped, terminated, truncated, info
