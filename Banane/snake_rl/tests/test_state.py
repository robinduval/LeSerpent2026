"""Tests du vecteur d'état : chaque bit est vérifié isolément."""

import numpy as np

from snake_rl import rules
from snake_rl.game import SnakeGame
from snake_rl.state import STATE_LABELS, STATE_SIZE, build_state, turn_left, turn_right


def make_game(head, direction, food, body=None):
    game = SnakeGame(seed=0)
    game.body = [tuple(head)] + [tuple(p) for p in (body or [])]
    game.direction = direction
    game.food = tuple(food) if food is not None else None
    return game


def bit(state, label):
    return state[STATE_LABELS.index(label)]


def test_state_has_exactly_eleven_float32_values():
    state = build_state(SnakeGame(seed=0))
    assert state.shape == (STATE_SIZE,) == (11,)
    assert state.dtype == np.float32
    assert set(np.unique(state)).issubset({0.0, 1.0})


def test_label_order_is_frozen():
    """Garde-fou : l'ordre ne doit jamais bouger sans décision explicite."""
    assert STATE_LABELS == (
        "danger_straight",
        "danger_right",
        "danger_left",
        "direction_left",
        "direction_right",
        "direction_up",
        "direction_down",
        "food_left",
        "food_right",
        "food_up",
        "food_down",
    )


# ----------------------------------------------------------------------
# Rotations
# ----------------------------------------------------------------------


def test_turn_right_cycles_clockwise_on_screen():
    assert turn_right(rules.RIGHT) == rules.DOWN
    assert turn_right(rules.DOWN) == rules.LEFT
    assert turn_right(rules.LEFT) == rules.UP
    assert turn_right(rules.UP) == rules.RIGHT


def test_turn_left_is_the_inverse_of_turn_right():
    for direction in rules.ACTIONS:
        assert turn_left(turn_right(direction)) == direction


# ----------------------------------------------------------------------
# Bits de direction : un seul actif à la fois
# ----------------------------------------------------------------------


def test_exactly_one_direction_bit_is_set():
    for direction, label in [
        (rules.LEFT, "direction_left"),
        (rules.RIGHT, "direction_right"),
        (rules.UP, "direction_up"),
        (rules.DOWN, "direction_down"),
    ]:
        state = build_state(make_game((7, 7), direction, (7, 7)))
        direction_bits = state[3:7]
        assert direction_bits.sum() == 1.0
        assert bit(state, label) == 1.0


# ----------------------------------------------------------------------
# Bits de nourriture
# ----------------------------------------------------------------------


def test_food_strictly_to_the_left_only():
    state = build_state(make_game((7, 7), rules.UP, (3, 7)))
    assert bit(state, "food_left") == 1.0
    assert bit(state, "food_right") == 0.0
    assert bit(state, "food_up") == 0.0
    assert bit(state, "food_down") == 0.0


def test_food_strictly_above_only():
    state = build_state(make_game((7, 7), rules.UP, (7, 2)))
    assert bit(state, "food_up") == 1.0
    assert bit(state, "food_down") == 0.0
    assert bit(state, "food_left") == 0.0
    assert bit(state, "food_right") == 0.0


def test_food_in_diagonal_sets_two_bits():
    state = build_state(make_game((7, 7), rules.UP, (9, 4)))
    assert bit(state, "food_right") == 1.0
    assert bit(state, "food_up") == 1.0
    assert bit(state, "food_left") == 0.0
    assert bit(state, "food_down") == 0.0


def test_food_aligned_sets_no_bit_on_that_axis():
    state = build_state(make_game((7, 7), rules.UP, (7, 3)))
    assert bit(state, "food_left") == 0.0
    assert bit(state, "food_right") == 0.0


def test_no_food_neutralises_food_bits():
    """Grille pleine : plus de pomme, les quatre bits tombent à zéro."""
    state = build_state(make_game((7, 7), rules.UP, None))
    assert state[7:].sum() == 0.0


# ----------------------------------------------------------------------
# Les bordures libres ne sont pas dangereuses
# ----------------------------------------------------------------------


def test_no_danger_straight_at_top_edge():
    state = build_state(make_game((7, 0), rules.UP, (7, 7)))
    assert bit(state, "danger_straight") == 0.0
    assert bit(state, "danger_right") == 0.0
    assert bit(state, "danger_left") == 0.0


def test_no_danger_right_at_right_edge():
    state = build_state(make_game((14, 7), rules.UP, (0, 0)))
    assert bit(state, "danger_right") == 0.0
    assert bit(state, "danger_straight") == 0.0
    assert bit(state, "danger_left") == 0.0


def test_no_danger_left_at_left_edge():
    state = build_state(make_game((0, 7), rules.UP, (14, 14)))
    assert bit(state, "danger_left") == 0.0
    assert bit(state, "danger_straight") == 0.0
    assert bit(state, "danger_right") == 0.0


def test_free_corner_has_no_danger():
    state = build_state(make_game((0, 0), rules.UP, (7, 7)))
    assert bit(state, "danger_straight") == 0.0
    assert bit(state, "danger_left") == 0.0
    assert bit(state, "danger_right") == 0.0


def test_open_field_has_no_danger():
    state = build_state(make_game((7, 7), rules.UP, (1, 1)))
    assert state[0:3].sum() == 0.0


# ----------------------------------------------------------------------
# Bits de danger : corps
# ----------------------------------------------------------------------


def test_danger_straight_from_own_body():
    game = make_game((7, 7), rules.UP, (0, 0), body=[(8, 7), (8, 6), (7, 6), (6, 6)])
    state = build_state(game)
    assert bit(state, "danger_straight") == 1.0


def test_danger_relative_to_direction_not_to_the_screen():
    """Même corps, direction inversée : le danger change de côté.

    C'est le point subtil de cette représentation : droite et gauche sont
    relatives au serpent, pas à l'écran.
    """
    body = [(8, 7), (8, 8)]
    facing_up = build_state(make_game((7, 7), rules.UP, (0, 0), body=body))
    facing_down = build_state(make_game((7, 7), rules.DOWN, (0, 0), body=body))
    assert bit(facing_up, "danger_right") == 1.0
    assert bit(facing_up, "danger_left") == 0.0
    assert bit(facing_down, "danger_left") == 1.0
    assert bit(facing_down, "danger_right") == 0.0


def test_state_of_a_fresh_game_is_all_zero_except_direction_and_food():
    """La partie de départ : direction droite, pomme quelque part, pas de danger."""
    game = SnakeGame(seed=0)
    state = build_state(game)
    assert bit(state, "direction_right") == 1.0
    assert state[0:3].sum() == 0.0, "le serpent démarre loin des murs"
    assert state[7:].sum() >= 1.0, "la pomme est forcément dans une direction"
