from __future__ import annotations

import numpy as np
from django.test import TestCase

from bouncer.rl.env import BerghainEnv


class BerghainEnvSmokeTest(TestCase):
    def test_env_reset_and_step(self) -> None:
        for scenario in (1, 2, 3):
            env = BerghainEnv(scenario=scenario, seed=123)

            # Check spaces
            # Action space is Discrete(2); check via sampling bounds
            self.assertIn(int(env.action_space.sample()), (0, 1))

            obs, info = env.reset(seed=42)
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.shape, (31,))
            self.assertIsInstance(info, dict)

            # Take up to 10 random steps; ensure shapes and reward bounds
            for _ in range(10):
                action = int(env.action_space.sample())
                next_obs, reward, terminated, truncated, step_info = env.step(action)

                self.assertIsInstance(next_obs, np.ndarray)
                self.assertEqual(next_obs.shape, (31,))
                self.assertIn(reward, (-1.0, 0.0))
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                self.assertIsInstance(step_info, dict)

                if terminated or truncated:
                    break
