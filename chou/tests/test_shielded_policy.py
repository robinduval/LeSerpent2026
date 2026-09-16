"""Semantic tests of the explicitly optional local safety filter."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_rl.env import Env, RIGHT, UP
from snake_rl.shielded_policy import ShieldedPolicy


class FixedQ:
    def __init__(self, values):
        self.values = torch.tensor([values], dtype=torch.float32)

    def __call__(self, state):
        return self.values


def learner(values):
    return SimpleNamespace(online=FixedQ(values), config=SimpleNamespace(encoder="ordered"),
                           metadata={}, env_steps=0, updates=0)


class ShieldTests(unittest.TestCase):
    def test_observation_and_rng_are_not_mutated(self):
        env = Env(4321)
        before = copy.deepcopy(vars(env))
        before_rng = env.rng.getstate()
        policy = ShieldedPolicy(learner([0, 2, 1, 3]), "tail2")
        action = policy.action(env)
        self.assertEqual(before_rng, env.rng.getstate())
        for name, value in before.items():
            if name != "rng":
                self.assertEqual(value, getattr(env, name))
        self.assertFalse(env.would_collide(action))
        self.assertNotEqual(action, (env.direction + 2) % 4)

    def test_network_selects_among_safe_actions(self):
        env = Env(543)
        env.apple = (9, 9)
        self.assertEqual(ShieldedPolicy(learner([3, 2, 1, 0]), "tail2").action(env), UP)
        self.assertEqual(ShieldedPolicy(learner([1, 3, 2, 0]), "tail2").action(env), RIGHT)

    def test_apple_with_no_next_exit_is_rejected(self):
        env = Env(0)
        env.body = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0),
                    (2, 14), (1, 14), (0, 14), (14, 14)]
        env.direction = UP
        env.apple = (1, 0)
        self.assertFalse(env.would_collide(RIGHT))
        policy = ShieldedPolicy(learner([0, 5, 1, 2]), "tail2")
        self.assertNotEqual(policy.action(env), RIGHT)
        self.assertNotIn(RIGHT, policy.last["allowed"])

    def test_final_full_board_apple_is_allowed(self):
        # Cycle order has toroidal horizontal rows connected by vertical wraps.
        cycle = [((-y + x) % 15, y) for y in range(15) for x in range(15)]
        env = Env(0)
        env.body = list(reversed(cycle[:-1]))
        env.direction = RIGHT
        env.grow_pending = True
        env.score = 222
        env.apple = cycle[-1]
        policy = ShieldedPolicy(learner([0, 3, 2, 1]), "tail2")
        action = policy.action(env)
        env.step(action)
        self.assertTrue(env.completed)
        self.assertEqual(env.score, 223)


if __name__ == "__main__":
    unittest.main()
