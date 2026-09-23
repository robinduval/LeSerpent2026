"""Tests sans fenêtre : .venv/bin/python -m unittest -v test_snake_algo.py"""
import importlib.util
import os
from pathlib import Path
import random
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
spec = importlib.util.spec_from_file_location(
    'snake_algo', Path(__file__).with_name('snake-algo.py'))
game = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = game
spec.loader.exec_module(game)


def snake_at(body, direction=game.RIGHT, pending=False):
    snake = game.Snake()
    snake.body = [list(position) for position in body]
    snake.head_pos = snake.body[0]
    snake.direction = direction
    snake.grow_pending = pending
    return snake


def full_body():
    # Cycle torique : fin de chaque ligne sous le début de la suivante.
    return [((x - y) % 15, y) for y in range(15) for x in range(15)]


class AlgorithmTests(unittest.TestCase):
    def test_wrap_and_shortest_path(self):
        snake = snake_at([(14, 7), (13, 7), (12, 7)])
        self.assertEqual(game.toroidal_manhattan((14, 7), (0, 7)), 1)
        self.assertEqual(game.astar(snake.head_pos, (0, 7), snake.body), [(0, 7)])
        self.assertEqual(game.choose_ai_direction(snake, (0, 7)), game.RIGHT)
        self.assertEqual(game.direction_to((0, 5), (14, 5)), game.LEFT)
        self.assertEqual(game.direction_to((5, 0), (5, 14)), game.UP)

    def test_astar_avoids_body_and_returns_no_path(self):
        body = [(0, 0), (1, 0), (2, 0)]
        path = game.astar((0, 0), (3, 0), body)
        self.assertEqual(len(path), 5)
        self.assertTrue(set(path).isdisjoint(body))
        blocked = [(0, 0)] + game.get_toroidal_neighbors((0, 0))
        self.assertIsNone(game.astar((0, 0), (5, 5), blocked))
        self.assertIsNone(game.astar((0, 0), None, body))

    def test_simulation_matches_real_delayed_growth(self):
        for pending in (False, True):
            snake = snake_at([(14, 7), (13, 7), (12, 7)], pending=pending)
            original = [position[:] for position in snake.body]
            path = [(0, 7), (1, 7), (1, 8)]
            simulated = game.simulate_path(snake, path, [0, 7])
            self.assertEqual(snake.body, original)
            self.assertEqual(snake.grow_pending, pending)
            for position in path:
                snake.set_direction(game.direction_to(snake.head_pos, position))
                snake.move()
                if position == (0, 7):
                    snake.grow()
            self.assertEqual(simulated.body, tuple(tuple(p) for p in snake.body))
            self.assertEqual(simulated.grow_pending, snake.grow_pending)
            self.assertEqual(simulated.direction, snake.direction)
            self.assertEqual(snake.score, 1)

    def test_tail_is_legal_only_when_it_moves(self):
        snake = snake_at([(1, 1), (1, 2), (0, 2), (0, 1)], game.UP)
        self.assertIn(game.LEFT, game.get_legal_moves(snake))
        snake.grow_pending = True
        self.assertNotIn(game.LEFT, game.get_legal_moves(snake))
        self.assertIsNone(game.simulate_path(snake, [(1, 2)], None))
        self.assertIsNone(game.simulate_path(snake, [(8, 8)], None))

    def test_flood_fill_uses_borders_and_excludes_head(self):
        body = [(0, 0), (1, 0), (0, 1)]
        self.assertEqual(game.flood_fill_space([0, 0], body), 222)
        sealed = [(0, 0)] + game.get_toroidal_neighbors((0, 0))
        self.assertEqual(game.flood_fill_space((0, 0), sealed), 0)
        self.assertEqual(game.flood_fill_space((0, 0), full_body()), 0)

    def test_safety_threshold_and_traps(self):
        snake = game.Snake()
        self.assertTrue(game.is_safe_path(snake))
        with patch.object(game, 'flood_fill_space', return_value=166):
            self.assertFalse(game.is_safe_path(snake))
        with patch.object(game, 'flood_fill_space', return_value=167):
            self.assertTrue(game.is_safe_path(snake))
        trapped = snake_at([(0, 0), (1, 0), (14, 0), (0, 1), (0, 14)],
                           pending=True)
        self.assertFalse(game.is_safe_path(trapped))
        self.assertIsNone(game.choose_ai_direction(trapped, (5, 5)))
        one_exit = snake_at([(0, 0), (1, 0), (0, 1), (0, 14)],
                            game.LEFT, True)
        self.assertEqual(list(game.get_legal_moves(one_exit)), [game.LEFT])
        self.assertEqual(game.choose_ai_direction(one_exit, (5, 5)), game.LEFT)
        # Une sortie forcée peut être acceptée si les coups suivants libèrent le corps.
        self.assertGreaterEqual(game.assess_safety(one_exit)['future_survival'], 8)

    def test_fallback_priorities(self):
        snake = game.Snake()
        with patch.object(game, 'flood_fill_space', return_value=10):
            self.assertEqual(game.choose_safest_move(snake, (4, 7)), game.RIGHT)
        def space(head, body):
            return 20 if head == (3, 6) else 10
        with patch.object(game, 'flood_fill_space', side_effect=space):
            self.assertEqual(game.choose_safest_move(snake, (4, 7)), game.UP)

    def test_small_snake_and_missing_apple(self):
        snake = snake_at([(0, 0)])
        self.assertNotIn(game.LEFT, game.get_legal_moves(snake))
        self.assertIn(game.choose_ai_direction(snake, None), game.get_legal_moves(snake))

    def test_last_cell_and_victory(self):
        snake = snake_at(full_body()[:-1], game.LEFT, True)
        target = full_body()[-1]
        direction = game.choose_ai_direction(snake, target)
        self.assertEqual(direction, game.UP)
        simulated = game.simulate_path(snake, [target], target)
        self.assertEqual(len(simulated.body), 225)
        self.assertTrue(game.is_safe_path(simulated))
        snake.set_direction(direction)
        snake.move()
        self.assertFalse(snake.is_game_over())
        self.assertIsNone(game.choose_ai_direction(snake, None))
        apple = game.Apple([])
        self.assertFalse(apple.relocate(snake.body))
        self.assertIsNone(apple.position)

    def test_autonomous_games(self):
        # De vraies transitions de jeu, sans attente ni affichage.
        for seed in range(3):
            random.seed(seed)
            snake = game.Snake()
            apple = game.Apple(snake.body)
            for _ in range(60):
                direction = game.choose_ai_direction(snake, apple.position)
                if direction is None:
                    self.assertFalse(game.get_legal_moves(snake, apple.position))
                    break
                self.assertIn(direction, game.get_legal_moves(snake, apple.position))
                snake.set_direction(direction)
                snake.move()
                self.assertFalse(snake.is_game_over())
                if tuple(snake.head_pos) == apple.position:
                    snake.grow()
                    if not apple.relocate(snake.body):
                        break
            self.assertGreater(snake.score, 0)


class FoodPlannerTests(unittest.TestCase):
    def trap(self):
        # La pomme à droite enferme immédiatement la tête entre quatre segments.
        return snake_at([(0, 0), (0, 1), (1, 1), (2, 1), (2, 0),
                         (2, 14), (1, 14), (0, 14), (14, 14)], game.UP)

    def replay_plan(self, snake, path, apple):
        initial_score = snake.score
        for position in path:
            snake.set_direction(game.direction_to(snake.head_pos, position))
            snake.move()
            self.assertFalse(snake.is_game_over())
            if tuple(snake.head_pos) == apple:
                snake.grow()
                apple = None
        self.assertEqual(snake.score, initial_score + 1)
        return snake

    def test_safe_direct_path_remains_first_choice(self):
        snake = game.Snake()
        with patch.object(game, 'food_lookahead', wraps=game.food_lookahead) as beam:
            self.assertEqual(game.choose_ai_direction(snake, (4, 7)), game.RIGHT)
            beam.assert_not_called()
        self.assertEqual(snake.last_mode, 'ASTAR_DIRECT')
        self.assertTrue(snake.last_decision['astar']['accepted'])
        self.assertEqual(snake.last_decision['astar']['future_survival'], 8)

    def test_long_safe_food_route_beats_short_suicide(self):
        snake = self.trap()
        original = [p[:] for p in snake.body]
        short = game.astar(snake.head_pos, (1, 0), snake.body, snake.direction)
        self.assertEqual(len(short), 1)
        dead = game.simulate_path(snake, short, (1, 0))
        self.assertEqual(game.get_legal_moves(dead), {})
        self.assertEqual(game.choose_ai_direction(snake, (1, 0)), game.LEFT)
        self.assertEqual(snake.body, original)
        self.assertEqual(snake.last_mode, 'FOOD_LOOKAHEAD')
        self.assertGreater(len(snake.last_plan), len(short))
        report = snake.last_decision
        self.assertEqual(report['astar']['rejection_reason'], 'post_apple_trap')
        self.assertTrue(report['candidates']['LEFT']['viable'])
        self.assertFalse(report['candidates']['RIGHT']['viable'])
        # Horizon local : plusieurs recalculs doivent mener à la pomme.
        for _ in range(20):
            direction = game.choose_ai_direction(snake, (1, 0))
            self.assertIsNotNone(direction)
            snake.set_direction(direction)
            snake.move()
            self.assertFalse(snake.is_game_over())
            if tuple(snake.head_pos) == (1, 0):
                snake.grow()
                break
        self.assertEqual(snake.score, 1)

    def test_moving_tail_opens_route_missing_from_static_astar(self):
        snake = snake_at([(0, 0), (1, 0), (2, 0), (2, 1),
                          (2, 2), (1, 2), (0, 2), (0, 1)], game.LEFT)
        self.assertIsNone(game.astar(snake.head_pos, (1, 1), snake.body, snake.direction))
        game.choose_ai_direction(snake, (1, 1))
        self.assertEqual(snake.last_mode, 'FOOD_LOOKAHEAD')
        self.assertTrue(snake.last_decision['search']['apple_reached'])
        self.assertTrue(set(snake.last_plan).intersection(map(tuple, snake.body[1:])))
        self.replay_plan(snake, snake.last_plan, (1, 1))

    def test_no_new_random_apple_during_search(self):
        with patch.object(game.random, 'choice', side_effect=AssertionError('random simulation')):
            snake = self.trap()
            game.choose_ai_direction(snake, (1, 0))
            state = game.simulate_path(snake, snake.last_plan, (1, 0))
            result = game.probe_survival(state)
            self.assertEqual(result['future_survival'], game.POST_APPLE_SURVIVAL_DEPTH)

    def test_food_is_not_eaten_twice_in_simulation(self):
        snake = snake_at([(0, 0)])
        loop = [(1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]
        result = game.simulate_path(snake, loop, (1, 0))
        self.assertEqual(len(result.body), 2)
        self.assertFalse(result.grow_pending)
        self.assertTrue(result.apple_eaten)

    def test_temporary_survival_retries_astar_next_tick(self):
        snake = self.trap()
        with patch.object(game, 'FOOD_LOOKAHEAD_DEPTH', 1):
            direction = game.choose_ai_direction(snake, (1, 0))
            self.assertEqual(snake.last_mode, 'TEMP_SURVIVAL')
        snake.set_direction(direction)
        snake.move()
        with patch.object(game, 'astar', wraps=game.astar) as astar:
            game.choose_ai_direction(snake, (1, 0))
            self.assertEqual(astar.call_args_list[0].args[0], snake.body[0])
        self.assertIn(snake.last_mode, ('ASTAR_DIRECT', 'FOOD_LOOKAHEAD'))

    def test_maneuver_opens_food_route_beyond_horizon(self):
        snake = self.trap()
        with patch.object(game, 'FOOD_LOOKAHEAD_DEPTH', 3):
            game.choose_ai_direction(snake, (1, 0))
        self.assertEqual(snake.last_mode, 'FOOD_LOOKAHEAD')
        report = snake.last_decision['search']
        self.assertFalse(report['apple_reached'])
        self.assertTrue(report['astar_available_after_simulation'])
        self.assertEqual(report['depth_reached'], report['depth_requested'])

    def test_stagnation_prefers_unvisited_equally_safe_branch(self):
        state = game.simulated_copy(game.Snake())
        revisited = game.SearchBranch(state, revisits=3)
        fresh = game.SearchBranch(state)
        self.assertGreater(game.food_branch_rank(fresh, (4, 7), stagnant=True),
                           game.food_branch_rank(revisited, (4, 7), stagnant=True))

    def test_history_is_bounded_and_reset_when_eating(self):
        snake = game.Snake()
        for _ in range(game.RECENT_STATES_LIMIT + 5):
            snake.recent_states.append(game.state_key(snake))
        self.assertEqual(len(snake.recent_states), game.RECENT_STATES_LIMIT)
        snake.move()
        self.assertEqual(snake.ticks_since_last_apple, 1)
        snake.grow()
        self.assertEqual(snake.ticks_since_last_apple, 0)
        self.assertEqual(len(snake.recent_states), 0)

    def test_dedup_includes_direction_growth_and_food(self):
        state = game.simulated_copy(game.Snake())
        states = [state,
                  game.SimulatedSnake(state.body, game.UP, False),
                  game.SimulatedSnake(state.body, state.direction, True),
                  game.SimulatedSnake(state.body, state.direction, False, True)]
        self.assertEqual(len(set(map(game.state_key, states))), 4)
        duplicate = game.simulated_copy(game.Snake())
        self.assertEqual(game.state_key(state), game.state_key(duplicate))

    def test_expansion_budget_and_large_snakes(self):
        data = json.loads(Path(__file__).with_name('benchmark_results.json').read_text())
        for row in data['baseline_states']:
            snake = snake_at(row['body'], tuple(row['direction']), row['grow_pending'])
            direction = game.choose_ai_direction(snake, row['apple'])
            self.assertIn(direction, game.get_legal_moves(snake, row['apple']))
            if 'search' in snake.last_decision:
                report = snake.last_decision['search']
                bound = 3 + (game.FOOD_LOOKAHEAD_DEPTH - 1) * game.FOOD_BEAM_WIDTH * 3
                self.assertLessEqual(report['states_expanded'], bound)
                self.assertLessEqual(report['food_evaluations'], game.FOOD_GOAL_LIMIT)


class PocketTests(unittest.TestCase):
    def case(self, name):
        data = json.loads(Path(__file__).with_name('pocket_cases.json').read_text())[name]
        snake = snake_at(data['body'], tuple(data['direction']), data.get('grow_pending', False))
        return snake, data

    def test_single_cell_and_toroidal_components(self):
        free = {(14, 7), (0, 7)}
        body = game.GRID_CELLS - free
        report = game.analyze_free_space(body, (1, 7))
        self.assertEqual(report['number_of_free_components'], 1)
        self.assertEqual(report['accessible_from_head'], 2)
        isolated = game.analyze_free_space(game.GRID_CELLS - {(1, 1)}, (5, 5))
        self.assertEqual(isolated['lost_space'], 1)
        self.assertEqual(isolated['isolated_components_sizes'], (1,))
        self.assertEqual(isolated['largest_isolated_component'], 1)

    def test_new_singleton_remains_closed_within_horizon(self):
        snake, case = self.case('persistent')
        before = game.analyze_free_space(snake.body, snake.head_pos)
        child = game.get_legal_moves(snake)[tuple(case['move'])]
        after = game.analyze_free_space(child.body, child.body[0])
        self.assertEqual(before['lost_space'], 0)
        self.assertEqual(after['isolated_components_sizes'], (1,))
        result = game.check_dynamic_pocket(before, child)
        self.assertEqual(result['new_lost_space'], 1)
        self.assertTrue(result['pocket_persistent'])
        self.assertEqual(result['persistent_singletons'], 1)

    def test_temporary_pocket_opens_with_tail(self):
        snake, case = self.case('temporary')
        before = game.analyze_free_space(snake.body, snake.head_pos)
        child = game.get_legal_moves(snake)[tuple(case['move'])]
        result = game.check_dynamic_pocket(before, child)
        self.assertEqual(result['new_lost_space'], 1)
        self.assertFalse(result['pocket_persistent'])
        self.assertEqual(result['pocket_depth'], 3)
        self.assertEqual(result['effective_lost_space'], 0)

    def test_same_lost_count_does_not_hide_a_different_pocket(self):
        snake, case = self.case('persistent')
        before = dict(game.analyze_free_space(snake.body, snake.head_pos),
                      lost_space=1, isolated_cells=frozenset({(5, 5)}))
        child = game.get_legal_moves(snake)[tuple(case['move'])]
        result = game.check_dynamic_pocket(before, child)
        self.assertEqual(result['new_lost_space'], 0)
        self.assertTrue(result['pocket_persistent'])

    def test_short_detour_avoids_persistent_pocket(self):
        snake, case = self.case('detour')
        apple = tuple(case['apple'])
        self.assertEqual(len(game.astar(snake.head_pos, apple, snake.body, snake.direction)), 1)
        chosen = game.choose_ai_direction(snake, apple)
        self.assertNotEqual(chosen, tuple(case['bad_move']))
        self.assertEqual(snake.last_plan[-1], apple)
        self.assertGreater(len(snake.last_plan), 1)
        self.assertLessEqual(len(snake.last_plan), game.FOOD_LOOKAHEAD_DEPTH)
        self.assertFalse(snake.last_decision['pocket_persistent'])
        self.assertEqual(snake.last_decision['astar']['rejection_reason'], 'persistent_pocket')
        self.assertTrue(snake.last_decision['candidates'][game.direction_name(tuple(case['bad_move']))]['pocket_persistent'])

    def test_stagnation_detected_once_and_resolved_on_food(self):
        snake = game.Snake()
        snake.ticks_since_last_apple = game.STAGNATION_LIMIT + 1
        with self.assertLogs(game.decision_logger, level='INFO') as logs:
            game.update_stagnation(snake, (6, 7))
            game.update_stagnation(snake, (6, 7))
            snake.grow()
        self.assertEqual(sum('STAGNATION_DETECTED' in s for s in logs.output), 1)
        self.assertEqual(sum('STAGNATION_RESOLVED' in s for s in logs.output), 1)
        self.assertFalse(snake.anti_stagnation_mode)
        self.assertEqual(snake.ticks_since_last_apple, 0)

    def test_repeated_states_trigger_detection_with_apple_in_key(self):
        snake = game.Snake()
        key = game.history_key(snake, (6, 7))
        snake.recent_state_hashes.extend([key, key])
        self.assertEqual(game.update_stagnation(snake, (6, 7)), 2)
        self.assertTrue(snake.anti_stagnation_mode)
        self.assertNotEqual(key, game.history_key(snake, (6, 8)))

    def test_anti_loop_can_override_repeated_direct_astar_move(self):
        snake = game.Snake()
        apple = (6, 7)
        repeated_child = game.get_legal_moves(snake, apple)[game.RIGHT]
        snake.recent_state_hashes.extend([game.history_key(repeated_child, apple)] * 3)
        snake.recent_geometries.extend([game.geometry_key(repeated_child, apple)] * 3)
        snake.ticks_since_last_apple = game.STAGNATION_LIMIT + 1
        direction = game.choose_ai_direction(snake, apple)
        self.assertNotEqual(direction, game.RIGHT)
        self.assertIn(direction, game.get_legal_moves(snake, apple))
        self.assertFalse(snake.last_decision['pocket_persistent'])
        self.assertTrue(snake.last_decision['anti_stagnation'])
        self.assertEqual(snake.last_decision['candidates']['RIGHT']['recent_state_penalty'], 3)

    def test_anti_loop_does_not_prefer_fresh_persistent_pocket(self):
        snake, case = self.case('detour')
        apple = tuple(case['apple'])
        bad = tuple(case['bad_move'])
        for direction, child in game.get_legal_moves(snake, apple).items():
            if direction != bad:
                snake.recent_state_hashes.extend([game.history_key(child, apple)] * 4)
        snake.ticks_since_last_apple = game.STAGNATION_LIMIT + 1
        chosen = game.choose_ai_direction(snake, apple)
        self.assertNotEqual(chosen, bad)
        self.assertFalse(snake.last_decision['pocket_persistent'])

    def test_logged_stagnation_and_length_198_are_legal(self):
        snake, case = self.case('logged_stagnation')
        snake.ticks_since_last_apple = case['ticks_since_last_apple']
        chosen = game.choose_ai_direction(snake, case['apple'])
        self.assertIn(chosen, game.get_legal_moves(snake, case['apple']))
        self.assertTrue(snake.anti_stagnation_mode)
        long_snake = snake_at(full_body()[:198], game.LEFT)
        apple = full_body()[198]
        chosen = game.choose_ai_direction(long_snake, apple)
        self.assertIn(chosen, game.get_legal_moves(long_snake, apple))


class MainLoopTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        configure = game.configure_decision_logging
        self.log_path = Path(self.directory.name) / 'decisions.log'
        patcher = patch.object(game, 'configure_decision_logging',
                               side_effect=lambda: configure(self.log_path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_frames(self, snakes, apple_position, restart=False, real_clock=False):
        factory = game.Snake
        states = iter(snakes)
        frames, messages, decisions = [], [], []
        choose = game.choose_ai_direction
        display = game.display_message

        def display_and_record(surface, font, message, *args, **kwargs):
            messages.append(message)
            display(surface, font, message, *args, **kwargs)

        def events():
            frames.append(1)
            frames_per_move = max(1, game.GAME_SPEED // 10)
            if len(frames) >= 10 * frames_per_move:
                return [game.pygame.event.Event(game.pygame.QUIT)]
            if restart and len(frames) == 4 * frames_per_move:
                return [game.pygame.event.Event(game.pygame.KEYDOWN, key=game.pygame.K_SPACE)]
            return []

        def choose_direction(snake, apple):
            decisions.append(tuple(snake.head_pos))
            return choose(snake, apple)

        def make_apple(body):
            apple = original_apple(body)
            apple.position = apple_position
            return apple

        original_apple = game.Apple
        with patch.object(game, 'Snake', side_effect=lambda: next(states, factory())), \
                patch.object(game, 'Apple', side_effect=make_apple), \
                patch.object(game.pygame.event, 'get', side_effect=events), \
                patch.object(game, 'choose_ai_direction', side_effect=choose_direction), \
                patch.object(game, 'display_message', side_effect=display_and_record):
            if real_clock:
                game.main()
            else:
                with patch.object(game.pygame.time, 'Clock'):
                    game.main()
        return messages, decisions

    def test_launch_movement_food_and_recalculation_at_real_speed(self):
        snake = game.Snake()
        _, decisions = self.run_frames([snake], (4, 7), real_clock=True)
        self.assertEqual(len(decisions), 9)
        self.assertGreater(len(set(decisions)), 1)
        self.assertGreaterEqual(snake.score, 1)

    def test_game_over_and_space_restart(self):
        trapped = snake_at([(0, 0), (1, 0), (14, 0), (0, 1), (0, 14)], pending=True)
        restarted = game.Snake()
        messages, _ = self.run_frames([trapped, restarted], (4, 7), restart=True)
        self.assertIn('GAME OVER', messages)
        self.assertGreater(restarted.score, 0)
        death_line = next(line for line in self.log_path.read_text().splitlines() if 'DEATH ' in line)
        death = json.loads(death_line.split('DEATH ', 1)[1])
        self.assertEqual(death['body'], trapped.body)
        self.assertTrue(death['grow_pending'])
        self.assertEqual(death['reason'], 'no_legal_move')

    def test_victory_and_space_restart(self):
        almost_full = snake_at(full_body()[:-1], game.LEFT, True)
        restarted = game.Snake()
        messages, decisions = self.run_frames(
            [almost_full, restarted], full_body()[-1], restart=True)
        self.assertIn('VICTOIRE !', messages)
        self.assertEqual(self.log_path.read_text().count('VICTORY '), 1)
        self.assertEqual(len(almost_full.body), 225)
        self.assertIn((3, 7), decisions)

    def test_missing_apple_in_main(self):
        snake = game.Snake()
        _, decisions = self.run_frames([snake], None)
        self.assertEqual(len(decisions), 9)

    def test_collision_death_log(self):
        snake = snake_at([(0, 0), (1, 0), (1, 1), (0, 1)], game.UP, True)
        with patch.object(game, 'choose_ai_direction', return_value=game.RIGHT):
            messages, _ = self.run_frames([snake], (5, 5))
        self.assertIn('GAME OVER', messages)
        death_line = next(line for line in self.log_path.read_text().splitlines() if 'DEATH ' in line)
        death = json.loads(death_line.split('DEATH ', 1)[1])
        self.assertEqual(death['reason'], 'collision')
        self.assertEqual(death['head'], [1, 0])


if __name__ == '__main__':
    unittest.main()
