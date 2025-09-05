from __future__ import annotations

from typing import Any

import numpy as np
from django.test import TestCase

import bouncer.models as models_mod
import bouncer.rl.env as env_mod
from bouncer.constants import CAPACITY, SCENARIO_CONFIGS
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

    def test_rejection_limit_termination(self) -> None:
        # Monkeypatch small rejection limit in both env and models
        old_env_limit = env_mod.REJECTION_LIMIT
        old_models_limit = models_mod.REJECTION_LIMIT
        try:
            env_mod.REJECTION_LIMIT = 5
            models_mod.REJECTION_LIMIT = 5

            env = self.ENV_CLS(scenario=1, seed=321)
            env.reset(seed=321)
            last_reward = None
            last_info: dict[str, Any] = {}
            for _ in range(10):
                _, reward, terminated, _, info = env.step(0)  # always reject
                if terminated:
                    last_reward = reward
                    last_info = info
                    break
            assert last_info.get("status") == "failed"
            assert last_info.get("reason") == "rejection_limit"
            assert last_reward == -1.0
        finally:
            # Restore
            env_mod.REJECTION_LIMIT = old_env_limit
            models_mod.REJECTION_LIMIT = old_models_limit

    def test_observation_initial_deficits_and_one_hot(self) -> None:
        for scenario in (1, 2, 3):
            env = self.ENV_CLS(scenario=scenario, seed=42)
            obs, _ = env.reset(seed=42)
            # One-hot
            assert int(obs[0] + obs[1] + obs[2]) == 1
            assert int(obs[scenario - 1]) == 1
            # Initial deficits per attribute
            cfg = SCENARIO_CONFIGS[scenario]
            require = {c["attribute"]: int(c["minCount"]) for c in cfg["constraints"]}
            for i, attr in enumerate(ATTRIBUTE_ORDER):
                expected = float(require.get(attr, 0)) / float(CAPACITY)
                got = float(obs[7 + 2 * i + 1])
                assert abs(got - expected) < 1e-6

    def test_observation_monotonicity_metrics(self) -> None:
        env = self.ENV_CLS(scenario=1, seed=100)
        obs, _ = env.reset(seed=100)

        admitted_fracs: list[float] = []
        remaining_fracs: list[float] = []
        rejection_pressures: list[float] = []

        # 5 accepts
        for _ in range(5):
            admitted_fracs.append(float(obs[3]))
            remaining_fracs.append(float(obs[4]))
            rejection_pressures.append(float(obs[5]))
            obs, _, term, _, _ = env.step(1)
            if term:
                break
        if len(admitted_fracs) >= 2:
            assert all(
                x2 > x1
                for x1, x2 in zip(admitted_fracs, admitted_fracs[1:], strict=False)
            )
            assert all(
                y2 < y1
                for y1, y2 in zip(remaining_fracs, remaining_fracs[1:], strict=False)
            )

        # 5 rejects
        for _ in range(5):
            rejection_pressures.append(float(obs[5]))
            obs, _, term, _, _ = env.step(0)
            if term:
                break
        if len(rejection_pressures) >= 2:
            assert all(
                z2 >= z1
                for z1, z2 in zip(
                    rejection_pressures, rejection_pressures[1:], strict=False
                )
            )

    def test_deficit_decreases_on_accept_of_needed_attr(self) -> None:
        # Focus on scenario 1 with attributes 'young' and 'well_dressed'
        target_attr = "young"
        idx = ATTRIBUTE_ORDER.index(target_attr)

        env = self.ENV_CLS(scenario=1, seed=777)
        obs, _ = env.reset(seed=777)

        confirmations = 0
        for _ in range(500):
            need_before = float(obs[7 + 2 * idx + 1])
            curr_bit = int(obs[7 + 2 * idx])

            if need_before > 0.0 and curr_bit == 1:
                # Accept and assert the deficit strictly decreases
                obs, _, term, _, _ = env.step(1)
                need_after = float(obs[7 + 2 * idx + 1]) if not term else 0.0
                assert need_after < need_before or term
                confirmations += 1
            else:
                obs, _, term, _, _ = env.step(0)

            if term or confirmations >= 3:
                break

        assert confirmations >= 1

    def test_current_bits_match_internal_attrs(self) -> None:
        env = self.ENV_CLS(scenario=2, seed=135)
        obs, _ = env.reset(seed=135)

        # Verify current bits reflect _current_attrs (internal but stable for tests)
        curr_attrs = env._current_attrs
        assert isinstance(curr_attrs, dict)
        for i, attr in enumerate(ATTRIBUTE_ORDER):
            bit = int(obs[7 + 2 * i])
            assert bit in (0, 1)
            assert bit == (1 if bool(curr_attrs.get(attr, False)) else 0)

        # Step once and re-check
        obs, _, term, _, _ = env.step(1)
        if not term:
            curr_attrs = env._current_attrs
            assert isinstance(curr_attrs, dict)
            for i, attr in enumerate(ATTRIBUTE_ORDER):
                bit = int(obs[7 + 2 * i])
                assert bit == (1 if bool(curr_attrs.get(attr, False)) else 0)

    def test_terminated_observation_zero_vector(self) -> None:
        # Use small rejection limit to terminate quickly and check terminal obs
        old_env_limit = env_mod.REJECTION_LIMIT
        old_models_limit = models_mod.REJECTION_LIMIT
        try:
            env_mod.REJECTION_LIMIT = 2
            models_mod.REJECTION_LIMIT = 2

            env = self.ENV_CLS(scenario=1, seed=246)
            env.reset(seed=246)
            last_obs: np.ndarray | None = None
            for _ in range(5):
                obs, _, term, _, _ = env.step(0)
                if term:
                    last_obs = obs
                    break
            assert last_obs is not None
            assert np.all(last_obs == 0.0)
        finally:
            env_mod.REJECTION_LIMIT = old_env_limit
            models_mod.REJECTION_LIMIT = old_models_limit

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

    def test_db_pending_invariant(self) -> None:
        env = self.ENV_CLS(scenario=1, seed=11)
        env.reset(seed=11)
        assert env.game is not None
        game = env.game

        # Initial pending person
        pending = list(game.people.filter(decision__isnull=True))
        assert len(pending) == 1
        last_idx = pending[0].person_index

        # Advance a few steps, checking pending invariant and monotonic index
        for t in range(10):
            action = 1 if (t % 2 == 0) else 0
            _, _, terminated, _, _ = env.step(action)
            game.refresh_from_db()
            if terminated:
                # At termination, there may be zero pending
                assert game.status in ("completed", "failed")
                break
            pending = list(game.people.filter(decision__isnull=True))
            assert len(pending) == 1
            assert pending[0].person_index == last_idx + 1
            last_idx = pending[0].person_index

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
