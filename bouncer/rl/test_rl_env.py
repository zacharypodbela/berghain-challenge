from __future__ import annotations

from typing import Any

import numpy as np
from django.test import TestCase

from bouncer.constants import CAPACITY
from bouncer.rl.env import ATTRIBUTE_ORDER, DbBerghainEnv, SimBerghainEnv


class BaseEnvMixin:
    ENV_CLS: type[Any]

    def _basic_env_checks(self, scenario: int) -> None:
        env = self.ENV_CLS(scenario=scenario, seed=123)
        assert int(env.action_space.sample()) in (0, 1)

        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (31,)
        assert isinstance(info, dict)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

        for _ in range(10):
            action = int(env.action_space.sample())
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            assert next_obs.shape == (31,)
            assert reward in (-1.0, 0.0)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            if terminated:
                break

    def test_basic_all_scenarios(self) -> None:
        for scenario in (1, 2, 3):
            self._basic_env_checks(scenario)

    def test_reward_sign_and_counters(self) -> None:
        env = self.ENV_CLS(scenario=1, seed=999)
        env.reset(seed=999)
        _, r1, _, _, _ = env.step(0)
        assert r1 == -1.0
        _, r2, _, _, _ = env.step(1)
        assert r2 == 0.0
        assert env.admitted == 1
        assert env.rejected == 1

    def test_capacity_unmet_terminal_penalty(self) -> None:
        env = self.ENV_CLS(scenario=1, seed=123)
        env.reset(seed=123)
        last_reward = 0.0
        for _ in range(CAPACITY + 5):
            _, reward, terminated, _, _ = env.step(1)
            if terminated:
                last_reward = float(reward)
                break
        assert env.admitted == CAPACITY
        assert last_reward <= -1000.0

    def test_success_outcome_possible(self) -> None:
        """Use a greedy policy to achieve success at capacity.

        Policy: accept a person only if they cover all still-needed attributes
        (i.e., their current bits are 1 where remain_need_frac > 0). Once all
        deficits are zero, accept everyone until capacity is filled.
        """
        env = self.ENV_CLS(scenario=1, seed=7)
        obs, _ = env.reset(seed=7)

        last_info: dict[str, Any] = {}
        while True:
            # Parse observation: pairs start at index 7
            curr_bits = [int(obs[7 + 2 * i]) for i in range(len(ATTRIBUTE_ORDER))]
            need_fracs = [
                float(obs[7 + 2 * i + 1]) for i in range(len(ATTRIBUTE_ORDER))
            ]

            needed_idxs = [i for i, f in enumerate(need_fracs) if f > 0.0]
            if needed_idxs:
                accept = int(all(curr_bits[i] == 1 for i in needed_idxs))
            else:
                accept = 1  # no deficits remaining; admit rest

            obs, _, terminated, _, info = env.step(accept)
            if terminated:
                last_info = info
                break

        assert last_info.get("status") == "completed"
        assert last_info.get("reason") == "success"


class DbEnvTests(BaseEnvMixin, TestCase):
    ENV_CLS = DbBerghainEnv

    def test_db_consistency_counts_and_attributes(self) -> None:
        """Ensure env counters match DB state for LocalGame/Persons.

        Verifies:
        - env.admitted == game.admitted_count
        - env.rejected == game.rejected_count
        - info["status"] == game.status
        - env.accepted_attr_counts[attr] == DB count of accepted persons with that attr
        """
        env = self.ENV_CLS(scenario=1, seed=123)
        obs, _ = env.reset(seed=123)
        assert obs.shape == (31,)

        assert env.game is not None
        game = env.game

        # Take a deterministic sequence of actions (accept on even steps, reject on odd)
        # and periodically assert DB <-> env consistency.
        for t in range(20):
            action = 1 if (t % 2 == 0) else 0
            _, _, terminated, _, info = env.step(action)

            # Refresh game from DB to ensure up-to-date counts
            game.refresh_from_db()

            # Check admitted/rejected counters
            assert env.admitted == game.admitted_count
            assert env.rejected == game.rejected_count
            assert info["status"] == game.status

            # Check per-attribute accepted counts
            for attr in ATTRIBUTE_ORDER:
                db_count = game.people.filter(
                    decision=True, **{f"attributes__{attr}": True}
                ).count()
                assert env.accepted_attr_counts[attr] == db_count

            if terminated:
                break


class SimEnvTests(BaseEnvMixin, TestCase):
    ENV_CLS = SimBerghainEnv

    def test_determinism_seed(self) -> None:
        def collect_bits(env_seed: int, steps: int) -> list[tuple[int, ...]]:
            env = self.ENV_CLS(scenario=2, seed=env_seed)
            obs, _ = env.reset(seed=env_seed)
            bits: list[tuple[int, ...]] = []
            for t in range(steps):
                curr = tuple(int(obs[7 + 2 * i]) for i in range(len(ATTRIBUTE_ORDER)))
                bits.append(curr)
                action = 1 if (t % 2 == 0) else 0
                obs, _, term, _, _ = env.step(action)
                if term:
                    break
            return bits

        a = collect_bits(2024, 50)
        b = collect_bits(2024, 50)
        assert a == b
