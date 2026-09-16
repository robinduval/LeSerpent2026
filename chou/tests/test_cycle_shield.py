"""Structural and rollout verification, not timed competition evaluation."""
import random
import unittest

from snake_rl.cycle_shield import (
    BOARD_CELLS, CYCLE, INDEX, RewiredCycleShield, RewiredGreedyPolicy,
)
from snake_rl.env import Env, DIRECTIONS, GRID_SIZE


class CycleShieldTests(unittest.TestCase):
    def assert_cycle(self, shield):
        self.assertEqual(len(set(shield.cycle)), BOARD_CELLS)
        self.assertEqual(set(shield.cycle), set(CYCLE))
        for a, b in zip(shield.cycle, shield.cycle[1:] + shield.cycle[:1]):
            self.assertTrue(any(((a[0] + dx) % GRID_SIZE,
                                 (a[1] + dy) % GRID_SIZE) == b
                                for dx, dy in DIRECTIONS))

    def test_reset_cycle_is_hamiltonian_and_aligned(self):
        env = Env(0)
        shield = RewiredCycleShield()
        self.assert_cycle(shield)
        self.assertTrue(shield.is_aligned(env))
        self.assertEqual([INDEX[cell] for cell in env.body], [115, 114, 113])

    def test_enumeration_is_pure_and_prospective_index_matches_commit(self):
        for seed in range(20):
            env = Env(seed)
            shield = RewiredCycleShield()
            state = (list(env.body), env.rng.getstate(), env.apple, env.steps)
            original_cycle = list(shield.cycle)
            actions = shield.allowed_actions(env)
            self.assertTrue(actions)
            for action in actions:
                candidate = shield.candidate_index(env, action)
                trial = RewiredCycleShield()
                trial.commit(env, action)
                self.assertEqual(candidate, trial.index)
                self.assert_cycle(trial)
                self.assertFalse(env.would_collide(action))
            self.assertEqual(state, (env.body, env.rng.getstate(), env.apple, env.steps))
            self.assertEqual(original_cycle, shield.cycle)

    def test_random_admissible_choices_complete_without_collision(self):
        for seed in (0, 3, 73, 107, 243, 285, 497):
            env = Env(seed)
            shield = RewiredCycleShield()
            choices = random.Random(20000 + seed)
            while not env.terminated and env.steps <= 223 * 224:
                self.assertTrue(shield.is_aligned(env))
                before_distance = shield.distance(env.body[0], env.apple)
                action = choices.choice(shield.allowed_actions(env))
                expected_distance = shield.next_food_distance(env, action)
                self.assertLess(expected_distance, before_distance)
                self.assertFalse(env.would_collide(action))
                shield.commit(env, action)
                _, _, info = env.step(action)
                if not info['ate_apple']:
                    self.assertEqual(shield.distance(env.body[0], env.apple), expected_distance)
            self.assertEqual(env.score, 223)
            self.assertTrue(env.completed)
            self.assertEqual(len(env.body), 225)
            self.assert_cycle(shield)

    def test_adversarial_consecutive_apples_respect_delayed_growth(self):
        env = Env(0)
        shield = RewiredCycleShield()
        while not env.terminated:
            h = shield.index[env.body[0]]
            # A deliberately adversarial training fixture schedules growth on
            # every step. The production shield never controls apple placement.
            env.apple = shield.cycle[(h + 1) % BOARD_CELLS]
            action = min(shield.allowed_actions(env),
                         key=lambda a: shield.next_food_distance(env, a))
            shield.commit(env, action)
            env.step(action)
            self.assertTrue(shield.is_aligned(env))
        self.assertEqual((env.score, env.steps, len(env.body)), (223, 223, 225))
        self.assertTrue(env.completed)
        self.assertEqual(shield.allowed_actions(env), ())

    def test_greedy_comparator_is_explicitly_unlearned(self):
        self.assertFalse(RewiredGreedyPolicy.learned)
        self.assertTrue(RewiredGreedyPolicy.model_id.startswith('algorithmic_'))


if __name__ == '__main__':
    unittest.main()


class FreeArcOptimizerTests(unittest.TestCase):
    def test_free_arc_search_preserves_body_cycle_and_food_progress(self):
        from snake_rl.cycle_shield import explore_free_arc, optimize_free_arc
        for seed in (17, 83):
            env = Env(seed)
            shield = RewiredCycleShield()
            choices = random.Random(seed)
            while not env.terminated:
                before = shield.distance(env.body[0], env.apple)
                rng_state = env.rng.getstate()
                if env.steps == 0 or env.grow_pending:
                    explore_free_arc(shield, env, attempts=64)
                else:
                    optimize_free_arc(shield, env)
                self.assertEqual(rng_state, env.rng.getstate())
                self.assertTrue(shield.is_aligned(env))
                self.assertLessEqual(shield.distance(env.body[0], env.apple), before)
                if env.grow_pending:
                    CycleShieldTests.assert_cycle(self, shield)
                action = choices.choice(shield.allowed_actions(env))
                shield.commit(env, action)
                env.step(action)
            self.assertTrue(env.completed)
            self.assertEqual(env.score, 223)
