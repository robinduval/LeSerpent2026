"""Semantic parity tests; these are not accelerated training experiments."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

CHOU = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHOU))
from snake_rl.env import (DIRECTIONS, DOWN, Env, GRID_SIZE, INITIAL_BODY,
                          LEFT, RIGHT, THEORETICAL_MAX_SCORE, UP)


def toroidal_cycle():
    """An explicit adjacent 225-cell cycle used only for boundary fixtures."""
    return [((-y + x) % GRID_SIZE, y)
            for y in range(GRID_SIZE) for x in range(GRID_SIZE)]


class EngineTests(unittest.TestCase):
    def test_reset_seed_and_independent_random_stream(self):
        env = Env(13)
        apple = env.apple
        self.assertEqual(env.body, list(INITIAL_BODY))
        self.assertEqual(env.direction, RIGHT)
        env.step(UP)
        env.reset(13)
        self.assertEqual(env.apple, apple)
        self.assertEqual((env.score, env.steps, env.grow_pending), (0, 0, False))
        self.assertFalse(env.terminated)
        self.assertFalse(env.completed)
        self.assertIsNone(env.termination_reason)
        random.seed(102)
        self.assertEqual(Env(13).apple, apple)

    def test_actions_reverse_and_wrap(self):
        for action, expected in [(UP, (3, 6)), (RIGHT, (4, 7)),
                                 (DOWN, (3, 8)), (LEFT, (4, 7))]:
            env = Env(0)
            env.step(action)
            self.assertEqual(env.body[0], expected)
        env = Env(0)
        env.body = [(14, 5), (13, 5), (12, 5)]
        env.step(RIGHT)
        self.assertEqual(env.body[0], (0, 5))
        env.body = [(0, 0), (1, 0), (2, 0)]
        env.direction = UP
        env.step(UP)
        self.assertEqual(env.body[0], (0, 14))
        for invalid in [-1, 4, 1.5, "UP"]:
            with self.assertRaises(ValueError):
                Env(0).step(invalid)

    def test_delayed_growth_and_official_score(self):
        env = Env(0)
        env.apple = (4, 7)
        reward, done, info = env.step(RIGHT)
        self.assertEqual((reward, done, env.score, len(env.body)), (10., False, 1, 3))
        self.assertTrue(env.grow_pending)
        self.assertTrue(info["ate_apple"])
        env.apple = (0, 0)
        reward, _, info = env.step(RIGHT)
        self.assertEqual((reward, env.score, len(env.body)), (0.1, 1, 4))
        self.assertFalse(env.grow_pending)
        self.assertTrue(info["grew"])
        self.assertEqual(env.steps, 2)
        zero = Env(0, move_reward=0)
        zero.apple = (0, 0)
        self.assertEqual(zero.step(RIGHT)[0], 0.)

    def test_tail_release_depends_on_pending_growth(self):
        env = Env(0)
        env.body = [(1, 1), (1, 2), (0, 2), (0, 1)]
        env.direction = UP
        env.apple = (10, 10)
        self.assertFalse(env.would_collide(LEFT))
        self.assertFalse(env.step(LEFT)[1])
        env = Env(0)
        env.body = [(1, 1), (1, 2), (0, 2), (0, 1)]
        env.direction = UP
        env.grow_pending = True
        self.assertTrue(env.would_collide(LEFT))
        reward, terminated, _ = env.step(LEFT)
        self.assertEqual((reward, terminated, env.termination_reason),
                         (-10., True, "self_collision"))
        with self.assertRaises(RuntimeError):
            env.step(LEFT)

    def test_collision_precedes_apple_and_preserves_mutated_body(self):
        env = Env(0)
        env.body = [(1, 1), (1, 2), (2, 2), (2, 1), (3, 1)]
        env.direction = UP
        env.apple = (2, 1)  # Artificial overlap tests operation priority.
        reward, done, info = env.step(RIGHT)
        self.assertTrue(done)
        self.assertEqual(reward, -10.)
        self.assertEqual(env.score, 0)
        self.assertFalse(info["ate_apple"])
        self.assertEqual(env.body[0], env.body[4])

    def test_final_free_cell_can_yield_score_223_victory(self):
        cycle = toroidal_cycle()
        self.assertEqual(len(set(cycle)), 225)
        env = Env(0)
        env.body = [cycle[0]] + list(reversed(cycle[2:]))
        env.direction = DOWN  # Last cycle edge wrapped from (0,14) to (0,0).
        env.grow_pending = True
        env.score = 222
        env.apple = cycle[1]
        reward, done, _ = env.step(RIGHT)
        self.assertEqual((reward, done, env.score, len(env.body)), (100., True, 223, 225))
        self.assertEqual(env.score, THEORETICAL_MAX_SCORE)
        self.assertTrue(env.completed)
        self.assertEqual(env.termination_reason, "completed")
        self.assertEqual(env.apple, cycle[1])

    def test_score_222_does_not_automatically_win(self):
        cycle = toroidal_cycle()
        env = Env(0)
        env.body = [cycle[0]] + list(reversed(cycle[2:]))
        env.direction = DOWN
        env.score = 221
        env.apple = cycle[1]
        reward, done, _ = env.step(RIGHT)
        self.assertEqual((reward, done, env.score, len(env.body)), (10., False, 222, 224))
        self.assertTrue(env.grow_pending)
        self.assertEqual(env.apple, cycle[2])
        self.assertTrue(env.step(RIGHT)[1])
        self.assertTrue(env.completed)

    def test_late_growth_requires_the_only_free_cell(self):
        cycle = toroidal_cycle()
        env = Env(0)
        env.body = [cycle[0]] + list(reversed(cycle[2:]))
        env.direction = DOWN
        env.score = 222
        env.grow_pending = True
        env.apple = cycle[1]
        self.assertTrue(env.would_collide(DOWN))
        reward, done, _ = env.step(DOWN)
        self.assertEqual((reward, done, env.score), (-10., True, 222))
        self.assertFalse(env.completed)
        self.assertEqual(len(env.body), 225)  # Duplicates do not mean full occupancy.

    def test_queries_and_copies_do_not_mutate_real_state(self):
        env = Env(1)
        before = (env.body[:], env.apple, env.direction, env.score, env.steps,
                  env.grow_pending, env.rng.getstate())
        for action in range(4):
            env.would_collide(action)
        clone = env.copy()
        clone.apple = (4, 7)
        clone.step(RIGHT)
        after = (env.body[:], env.apple, env.direction, env.score, env.steps,
                 env.grow_pending, env.rng.getstate())
        self.assertEqual(before, after)
        self.assertIsNot(clone.body, env.body)
        self.assertIsNot(clone.rng, env.rng)

    def test_seeded_transition_parity_with_unmodified_source(self):
        spec = importlib.util.spec_from_file_location(
            "original_snake_parity", CHOU / "baseline" / "serpent_original.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        prior_global_rng = random.getstate()
        total_steps = 0
        try:
            for seed in range(1000):
                random.seed(seed)
                snake = original.Snake()
                apple = original.Apple(snake.body)
                env = Env(seed)
                actions = random.Random(seed + 40000)
                self.assertEqual(apple.position, env.apple)
                for _ in range(50):
                    action = actions.randrange(4)
                    snake.set_direction(DIRECTIONS[action])
                    snake.move()
                    done = snake.is_game_over()
                    completed = False
                    if not done and snake.head_pos == list(apple.position):
                        snake.grow()
                        if not apple.relocate(snake.body):
                            done = completed = True
                    _, actual_done, _ = env.step(action)
                    total_steps += 1
                    self.assertEqual(env.body, [tuple(p) for p in snake.body])
                    self.assertEqual(DIRECTIONS[env.direction], snake.direction)
                    self.assertEqual(env.grow_pending, snake.grow_pending)
                    self.assertEqual(env.score, snake.score)
                    self.assertEqual(env.apple, apple.position)
                    self.assertEqual(actual_done, done)
                    self.assertEqual(env.completed, completed)
                    if done:
                        break
            self.assertGreater(total_steps, 10000)
        finally:
            random.setstate(prior_global_rng)


if __name__ == "__main__":
    unittest.main()
