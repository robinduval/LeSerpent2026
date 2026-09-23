"""Régressions du moteur différé et du cycle, sans affichage ni réseau."""

import ast
from pathlib import Path
import random
import unittest

from test_serpent_algo import game


def initial_rank(cell):
    x, y = cell
    return game.GRID_SIZE * y + (x + y) % game.GRID_SIZE


def fixed_successor(cell):
    rank = (initial_rank(cell) + 1) % game.BOARD_CELLS
    y, offset = divmod(rank, game.GRID_SIZE)
    return ((offset - y) % game.GRID_SIZE, y)


def action_between(start, end):
    for action, (dx, dy) in enumerate(game.DIRECTIONS):
        if (
            (start[0] + dx) % game.GRID_SIZE,
            (start[1] + dy) % game.GRID_SIZE,
        ) == end:
            return action
    raise AssertionError(f"Arête non torique : {start} -> {end}")


class DeferredEngineTests(unittest.TestCase):
    def test_initial_state_is_exact_and_snapshot_has_no_random_generator(self):
        engine = game.Game(seed=20)
        state = engine.state()
        self.assertEqual(game.GRID_SIZE, 15)
        self.assertEqual(game.BOARD_CELLS, 225)
        self.assertEqual(game.GAME_SPEED, 5)
        self.assertEqual(state.body, ((3, 7), (2, 7), (1, 7)))
        self.assertEqual([initial_rank(cell) for cell in state.body], [115, 114, 113])
        self.assertEqual(state.direction, game.DIRECTIONS.index(game.RIGHT))
        self.assertEqual((state.score, state.steps, state.grow_pending, state.terminated),
                         (0, 0, False, False))
        self.assertNotIn(state.apple, state.body)
        self.assertEqual(set(state._fields), {
            "body", "direction", "grow_pending", "score", "steps", "apple", "terminated"
        })
        self.assertIsInstance(state.body, tuple)
        self.assertTrue(all(isinstance(cell, tuple) for cell in state.body))

    def test_growth_occurs_after_eating_and_consecutive_apples_keep_it_pending(self):
        engine = game.Game(seed=1)
        right = game.DIRECTIONS.index(game.RIGHT)
        engine.apple.position = (4, 7)
        first = engine.step(right)
        self.assertTrue(first["ate_apple"])
        self.assertFalse(first["grew"])
        self.assertFalse(first["collision"])
        self.assertEqual((len(engine.snake.body), engine.snake.score), (3, 1))
        self.assertTrue(engine.snake.grow_pending)

        engine.apple.position = (5, 7)
        second = engine.step(right)
        self.assertTrue(second["ate_apple"])
        self.assertTrue(second["grew"])
        self.assertEqual((len(engine.snake.body), engine.snake.score), (4, 2))
        self.assertTrue(engine.snake.grow_pending)

        engine.apple.position = (0, 0)
        third = engine.step(right)
        self.assertFalse(third["ate_apple"])
        self.assertTrue(third["grew"])
        self.assertEqual((len(engine.snake.body), engine.snake.score), (5, 2))
        self.assertFalse(engine.snake.grow_pending)

    def test_pending_growth_makes_tail_occupied_during_collision_check(self):
        for pending, collision in ((False, False), (True, True)):
            with self.subTest(pending=pending):
                engine = game.Game(seed=1)
                engine.snake.body = [[1, 1], [1, 2], [2, 2], [2, 1]]
                engine.snake.head_pos = engine.snake.body[0]
                engine.snake.direction = game.RIGHT
                engine.snake.grow_pending = pending
                engine.apple.position = (0, 0)
                event = engine.step(game.DIRECTIONS.index(game.RIGHT))
                self.assertEqual(event["collision"], collision)
                self.assertEqual(engine.terminated, collision)

    def test_collision_is_checked_before_apple_consumption(self):
        engine = game.Game(seed=1)
        engine.snake.body = [[1, 1], [1, 2], [2, 2], [2, 1], [3, 1]]
        engine.snake.head_pos = engine.snake.body[0]
        engine.snake.direction = game.RIGHT
        # Une pomme artificiellement placée sur le corps teste l'ordre moteur.
        engine.apple.position = (2, 1)
        event = engine.step(game.DIRECTIONS.index(game.RIGHT))
        self.assertTrue(event["collision"])
        self.assertFalse(event["ate_apple"])
        self.assertEqual(engine.snake.score, 0)
        self.assertFalse(engine.snake.grow_pending)

    def test_223_consecutive_apples_fill_board_and_terminal_state_is_stable(self):
        engine = game.Game(seed=20)
        for score in range(1, 224):
            head = tuple(engine.snake.head_pos)
            apple = fixed_successor(head)
            self.assertNotIn(list(apple), engine.snake.body)
            engine.apple.position = apple
            event = engine.step(action_between(head, apple))
            self.assertTrue(event["ate_apple"])
            self.assertEqual(event["grew"], score > 1)
            self.assertFalse(event["collision"])
            self.assertEqual(engine.snake.score, score)
            self.assertEqual(len(engine.snake.body), score + 2)
            self.assertEqual(engine.completed, score == 223)
            self.assertEqual(engine.terminated, score == 223)
            self.assertEqual(event["completed"], score == 223)
        self.assertEqual(engine.steps, 223)
        self.assertIsNone(engine.apple.position)
        self.assertEqual(len({tuple(cell) for cell in engine.snake.body}), 225)
        terminal = engine.state()
        with self.assertRaises(RuntimeError):
            engine.step(0)
        self.assertEqual(engine.state(), terminal)

    def test_seeded_restart_restores_initial_state_and_sequence(self):
        engine = game.Game(seed=20)
        initial = engine.state()
        engine.apple.position = (4, 7)
        engine.step(game.DIRECTIONS.index(game.RIGHT))
        self.assertNotEqual(engine.state(), initial)
        engine.reset(seed=20)
        self.assertEqual(engine.state(), initial)
        self.assertFalse(engine.completed)
        other = game.Game(seed=20)
        for _ in range(20):
            apple = fixed_successor(tuple(engine.snake.head_pos))
            engine.apple.position = other.apple.position = apple
            action = action_between(tuple(engine.snake.head_pos), apple)
            self.assertEqual(engine.step(action), other.step(action))
            self.assertEqual(engine.state(), other.state())

    def test_invalid_action_is_rejected_without_moving(self):
        engine = game.Game(seed=20)
        initial = engine.state()
        for action in (-1, 4, 100):
            with self.subTest(action=action), self.assertRaises(ValueError):
                engine.step(action)
            self.assertEqual(engine.state(), initial)


class HamiltonianPolicyTests(unittest.TestCase):
    def assert_full_cycle(self, shield, state):
        cycle = shield.cycle
        self.assertEqual(len(cycle), game.BOARD_CELLS)
        self.assertEqual(set(cycle), {
            (x, y) for x in range(game.GRID_SIZE) for y in range(game.GRID_SIZE)
        })
        index = {cell: rank for rank, cell in enumerate(cycle)}
        self.assertEqual(shield.index, index)
        for start, end in zip(cycle, cycle[1:] + cycle[:1]):
            action_between(start, end)
        for headward, tailward in zip(state.body, state.body[1:]):
            self.assertEqual((index[headward] - index[tailward]) % game.BOARD_CELLS, 1)
        self.assertTrue(shield.is_aligned(state))
        shield.validate(state)

    def test_initial_cycle_matches_rank_and_body(self):
        shield = game.RewiredCycleShield()
        state = game.Game(seed=20).state()
        self.assert_full_cycle(shield, state)
        for cell in shield.cycle:
            self.assertEqual(shield.index[cell], initial_rank(cell))

    def test_rejects_cycle_corruption_and_broken_body_alignment(self):
        shield = game.RewiredCycleShield()
        state = game.Game(seed=20).state()
        bad = state._replace(body=(state.body[0], (0, 0), state.body[2]))
        self.assertFalse(shield.is_aligned(bad))
        with self.assertRaises(ValueError):
            shield.validate(bad)
        shield.cycle[0] = shield.cycle[1]
        with self.assertRaises(ValueError):
            shield.validate(state)

    def test_decision_is_deterministic_and_does_not_consume_apple_rng(self):
        original_random = random.getstate()
        self.addCleanup(random.setstate, original_random)
        random.seed(9102)
        before_random = random.getstate()
        first = game.Game(seed=20)
        second = game.Game(seed=20)
        first_policy = game.ExploredRewiredGreedyPolicy()
        second_policy = game.ExploredRewiredGreedyPolicy()
        for _ in range(100):
            before = first.state()
            action = first_policy.select_action(before)
            self.assertEqual(first.state(), before)
            other_action = second_policy.select_action(second.state())
            self.assertEqual(action, other_action)
            self.assertEqual(first.step(action), second.step(other_action))
            self.assertEqual(first.state(), second.state())
        self.assertEqual(random.getstate(), before_random)

    def test_reconfigured_agent_eats_223_consecutive_apples_with_growth_pending(self):
        engine = game.Game(seed=20)
        policy = game.ExploredRewiredGreedyPolicy()
        for score in range(1, 224):
            head = tuple(engine.snake.head_pos)
            successor_rank = (policy.shield.index[head] + 1) % game.BOARD_CELLS
            engine.apple.position = policy.shield.cycle[successor_rank]
            state = engine.state()
            self.assertNotIn(state.apple, state.body)
            action = policy.select_action(state)
            self.assert_full_cycle(policy.shield, state)
            self.assertEqual(action, action_between(head, state.apple))
            event = engine.step(action)
            self.assertTrue(event["ate_apple"])
            self.assertFalse(event["collision"])
            self.assertEqual(event["grew"], score > 1)
            self.assertTrue(engine.snake.grow_pending)
            self.assertEqual(engine.snake.score, score)
            self.assertEqual(len(engine.snake.body), score + 2)
            self.assert_full_cycle(policy.shield, engine.state())
        self.assertTrue(engine.completed)
        self.assertEqual(engine.steps, 223)
        self.assertIsNone(engine.apple.position)

    def test_reused_policy_resets_like_fresh_policy_after_game_restart(self):
        engine = game.Game(seed=11)
        policy = game.ExploredRewiredGreedyPolicy()
        for _ in range(40):
            engine.step(policy.select_action(engine.state()))
        self.assertGreater(engine.steps, 0)
        engine.reset(seed=20)
        fresh = game.ExploredRewiredGreedyPolicy()
        initial = engine.state()
        self.assertEqual(policy.select_action(initial), fresh.select_action(initial))
        self.assertEqual(policy.shield.cycle, fresh.shield.cycle)
        self.assertEqual(policy.shield.index, fresh.shield.index)
        self.assert_full_cycle(policy.shield, initial)

    def test_exploration_budget_best_distance_and_invalid_commit_rollback(self):
        state = game.Game(seed=20).state()
        for attempts in (5, 1000):
            with self.subTest(attempts=attempts):
                shield = game.RewiredCycleShield()
                initial_distance = shield.distance(state.body[0], state.apple)
                game.explore_free_arc(shield, state, attempts=attempts)
                self.assertGreaterEqual(shield.last_exploration_attempts, 0)
                self.assertLessEqual(shield.last_exploration_attempts, min(attempts, 64))
                self.assertLessEqual(shield.distance(state.body[0], state.apple),
                                     initial_distance)
                self.assert_full_cycle(shield, state)
                before_cycle, before_index = list(shield.cycle), dict(shield.index)
                reverse_action = (state.direction + 2) % len(game.DIRECTIONS)
                for operation in (shield.candidate_index, shield.commit):
                    with self.assertRaises(ValueError):
                        operation(state, reverse_action)
                    self.assertEqual(shield.cycle, before_cycle)
                    self.assertEqual(shield.index, before_index)

    def test_complete_games_preserve_cycle_and_make_strict_progress(self):
        for policy_type, seed in (
            (game.FixedCyclePolicy, 20),
            (game.ExploredRewiredGreedyPolicy, 2),
            (game.ExploredRewiredGreedyPolicy, 11),
        ):
            with self.subTest(policy=policy_type.__name__, seed=seed):
                engine = game.Game(seed=seed)
                policy = policy_type()
                for _ in range(game.BOARD_CELLS * 223):
                    state = engine.state()
                    before_distance = policy.shield.distance(state.body[0], state.apple)
                    action = policy.select_action(state)
                    # Vérification indépendante après préparation, avant exécution.
                    self.assert_full_cycle(policy.shield, state)
                    next_cell = policy.shield.cycle[
                        (policy.shield.index[state.body[0]] + 1) % game.BOARD_CELLS
                    ]
                    self.assertEqual(action, action_between(state.body[0], next_cell))
                    event = engine.step(action)
                    self.assertFalse(event["collision"])
                    self.assert_full_cycle(policy.shield, engine.state())
                    if not event["ate_apple"]:
                        after_distance = policy.shield.distance(
                            tuple(engine.snake.head_pos), engine.apple.position
                        )
                        self.assertLess(after_distance, before_distance)
                    if engine.terminated:
                        break
                self.assertTrue(engine.completed)
                self.assertEqual(engine.snake.score, 223)
                self.assertEqual(len(engine.snake.body), 225)
                with self.assertRaises(ValueError):
                    policy.select_action(engine.state())

    def test_runtime_is_standalone_without_learning_frameworks(self):
        source = Path(__file__).with_name("serpent-algo.py").read_text()
        imports = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
        self.assertTrue(imports.isdisjoint({"torch", "tensorflow", "snake_rl", "ia"}))


class BenchmarkAccountingTests(unittest.TestCase):
    def test_interruption_is_distinct_from_collision_and_has_no_completion_time(self):
        for policy in ("fixed", "explored64"):
            with self.subTest(policy=policy):
                samples = []
                result = game.simulate(policy, seed=20, max_steps=1, timings=samples)
                self.assertTrue(result["interrupted"])
                self.assertFalse(result["collision"])
                self.assertFalse(result["completed"])
                self.assertIsNone(result["steps_to_223"])
                self.assertEqual(result["steps"], 1)
                self.assertEqual(result["seconds_x1"], 0.2)
                self.assertEqual(result["decision"]["count"], 1)
                self.assertEqual(len(samples), 1)
                self.assertGreaterEqual(result["decision"]["mean_ms"], 0)


if __name__ == "__main__":
    unittest.main()
