"""Régressions de la logique Snake, exécutables sans écran ni pygame installé."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_game():
    spec = importlib.util.spec_from_file_location(
        "serpent_algo_test_subject", Path(__file__).with_name("serpent-algo.py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"pygame": ModuleType("pygame")}):
        spec.loader.exec_module(module)
    return module


game = load_game()


def almost_full_snake():
    snake = game.Snake()
    snake.body = [
        [x, y]
        for y in range(game.GRID_SIZE)
        for x in (
            range(game.GRID_SIZE)
            if y % 2 == 0
            else range(game.GRID_SIZE - 1, -1, -1)
        )
        if (x, y) != (0, 0)
    ]
    snake.head_pos = snake.body[0]
    snake.direction = game.LEFT
    snake.score = len(snake.body) - 2
    snake.grow_pending = True
    return snake


class SnakeLogicTests(unittest.TestCase):
    def test_second_turn_waits_for_next_movement(self):
        snake = game.Snake()
        start = snake.head_pos[:]
        snake.set_direction(game.UP)
        snake.set_direction(game.LEFT)
        snake.move()
        self.assertEqual(snake.head_pos, [start[0], start[1] - 1])
        self.assertFalse(snake.is_game_over())
        snake.set_direction(game.LEFT)
        snake.move()
        self.assertEqual(snake.head_pos, [start[0] - 1, start[1] - 1])

    def test_repeated_or_opposite_key_does_not_consume_turn(self):
        for ignored_direction in (game.RIGHT, game.LEFT):
            with self.subTest(direction=ignored_direction):
                snake = game.Snake()
                snake.set_direction(ignored_direction)
                snake.set_direction(game.UP)
                snake.move()
                self.assertEqual(snake.direction, game.UP)
                self.assertFalse(snake.is_game_over())

    def test_all_four_edges_wrap_without_ending_game(self):
        last = game.GRID_SIZE - 1
        cases = (
            ([0, 7], game.LEFT, [last, 7]),
            ([last, 7], game.RIGHT, [0, 7]),
            ([7, 0], game.UP, [7, last]),
            ([7, last], game.DOWN, [7, 0]),
        )
        for head, direction, expected in cases:
            with self.subTest(direction=direction):
                snake = game.Snake()
                snake.head_pos = head
                snake.body = [head]
                snake.direction = direction
                self.assertEqual(snake.next_head_position(), expected)
                self.assertEqual(snake.head_pos, head)
                snake.move()
                self.assertEqual(snake.head_pos, expected)
                self.assertFalse(snake.is_game_over())

    def test_tail_cell_is_free_when_tail_moves(self):
        snake = game.Snake()
        snake.body = [[1, 1], [1, 2], [2, 2], [2, 1]]
        snake.head_pos = snake.body[0]
        snake.direction = game.RIGHT
        snake.move()
        self.assertEqual(snake.head_pos, [2, 1])
        self.assertEqual(len(snake.body), 4)
        self.assertFalse(snake.is_game_over())

    def test_entering_body_ends_game(self):
        snake = game.Snake()
        snake.body = [[1, 1], [1, 2], [2, 2], [2, 1], [3, 1]]
        snake.head_pos = snake.body[0]
        snake.direction = game.RIGHT
        snake.move()
        self.assertTrue(snake.is_game_over())

    def test_pending_growth_fills_board_on_last_apple(self):
        snake = almost_full_snake()
        apple = game.Apple(snake.body)
        self.assertEqual(len(snake.body), 224)
        self.assertEqual(apple.position, (0, 0))
        self.assertEqual(snake.next_head_position(), list(apple.position))
        snake.move()
        snake.grow()
        self.assertEqual(len(snake.body), 225)
        self.assertEqual(snake.score, 223)
        self.assertFalse(snake.is_game_over())
        self.assertFalse(apple.relocate(snake.body))
        self.assertIsNone(apple.position)

    def test_relocate_never_overlaps_snake(self):
        snake = game.Snake()
        apple = game.Apple(snake.body)
        for _ in range(20):
            self.assertTrue(apple.relocate(snake.body))
            self.assertNotIn(list(apple.position), snake.body)


class MainLoopTests(unittest.TestCase):
    def run_game(self, frames, snake_factory, apple_factory, speed=5):
        clock = Mock()
        pygame = SimpleNamespace(
            init=Mock(),
            quit=Mock(),
            display=Mock(),
            time=SimpleNamespace(Clock=Mock(return_value=clock)),
            font=Mock(),
            draw=Mock(),
            Rect=Mock(),
            event=SimpleNamespace(get=Mock(side_effect=frames)),
            QUIT="quit",
            KEYDOWN="keydown",
            K_SPACE="space",
            K_UP="up",
            K_DOWN="down",
            K_LEFT="left",
            K_RIGHT="right",
        )
        info = Mock()
        messages = Mock()
        with (
            patch.object(game, "pygame", pygame),
            patch.object(game, "Snake", side_effect=snake_factory),
            patch.object(game, "Apple", side_effect=apple_factory),
            patch.object(game, "GAME_SPEED", speed),
            patch.object(game, "draw_grid"),
            patch.object(game, "display_info", info),
            patch.object(game, "display_message", messages),
        ):
            game.main(manual=True)
        pygame.quit.assert_called_once_with()
        return pygame, clock, info, messages

    def test_one_movement_per_frame_at_twenty_steps_per_second(self):
        snake = game.Snake()
        snake.move = Mock(wraps=snake.move)
        snake.draw = Mock()
        apple = game.Apple(snake.body)
        apple.position = (0, 0)
        apple.draw = Mock()
        _, clock, _, _ = self.run_game(
            [[], [], [], [], [SimpleNamespace(type="quit")]],
            lambda: snake,
            lambda body, rng=None: apple,
            speed=20,
        )
        self.assertEqual(snake.move.call_count, 4)
        self.assertEqual(clock.tick.call_count, 4)
        for call in clock.tick.call_args_list:
            self.assertEqual(call.args, (20,))

    def test_last_apple_triggers_victory_in_main_and_freezes_timer(self):
        snake = almost_full_snake()
        snake.move = Mock(wraps=snake.move)
        snake.draw = Mock()
        apple = game.Apple(snake.body)
        apple.draw = Mock()
        _, _, info, messages = self.run_game(
            [[], [], [], [SimpleNamespace(type="quit")]],
            lambda: snake,
            lambda body, rng=None: apple,
        )
        self.assertEqual(snake.move.call_count, 1)
        self.assertEqual(len(snake.body), 225)
        self.assertEqual(snake.score, 223)
        self.assertIsNone(apple.position)
        self.assertIn("VICTOIRE !", [call.args[2] for call in messages.call_args_list])
        self.assertEqual(info.call_count, 3)
        elapsed_values = [call.args[3] for call in info.call_args_list]
        self.assertEqual(elapsed_values, [elapsed_values[0]] * 3)

    def test_space_restarts_in_same_main_loop(self):
        original_snake = game.Snake
        original_apple = game.Apple
        snakes = []
        apples = []

        def create_snake():
            snake = original_snake()
            snake.body = [[1, 1], [1, 2], [2, 2], [2, 1], [3, 1]]
            snake.head_pos = snake.body[0]
            snake.direction = game.RIGHT
            snake.draw = Mock()
            snakes.append(snake)
            return snake

        def create_apple(body, rng=None):
            apple = original_apple(body, rng)
            apple.position = (0, 0)
            apple.draw = Mock()
            apples.append(apple)
            return apple

        pygame, _, _, _ = self.run_game(
            [
                [],
                [SimpleNamespace(type="keydown", key="space")],
                [SimpleNamespace(type="keydown", key="space")],
                [SimpleNamespace(type="quit")],
            ],
            create_snake,
            create_apple,
        )
        self.assertEqual(len(snakes), 3)
        self.assertEqual(len(apples), 3)
        pygame.init.assert_called_once_with()
        pygame.display.set_mode.assert_called_once()

    def test_display_info_formats_elapsed_duration(self):
        font = Mock()
        font.render.return_value.get_width.return_value = 100
        pygame = SimpleNamespace(draw=Mock())
        with patch.object(game, "pygame", pygame):
            game.display_info(Mock(), font, game.Snake(), 62)
        self.assertIn("Temps: 01:02", [call.args[0] for call in font.render.call_args_list])


if __name__ == "__main__":
    unittest.main()
