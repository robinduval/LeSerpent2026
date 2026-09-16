"""Régressions du plateau torique : moteur, dangers et trajectoires."""

import random

import numpy as np
import pytest

from snake_rl import rules
from snake_rl.game import COURSE_REWARDS, SnakeGame
from snake_rl.state import build_state, turn_left, turn_right


@pytest.mark.parametrize("direction", rules.ACTIONS)
@pytest.mark.parametrize("relative", ["straight", "right", "left"])
@pytest.mark.parametrize("occupied", [False, True])
def test_wraparound_danger_matches_actual_step(direction, relative, occupied):
    movement = {"straight": direction, "right": turn_right(direction),
                "left": turn_left(direction)}[relative]
    n = rules.GRID_SIZE
    head = (n - 1 if movement[0] == 1 else 0 if movement[0] == -1 else 7,
            n - 1 if movement[1] == 1 else 0 if movement[1] == -1 else 7)
    target = ((head[0] + movement[0]) % n, (head[1] + movement[1]) % n)
    game = SnakeGame(seed=17)
    # Le segment cible n'est pas la queue, qui se libère au prochain pas.
    game.body = [head, target if occupied else (5, 5), (6, 5)]
    game.direction = direction
    game.food = (2, 2)
    index = {"straight": 0, "right": 1, "left": 2}[relative]
    assert build_state(game)[index] == float(occupied)
    assert game.is_collision((head[0] + movement[0], head[1] + movement[1])) == occupied
    action = rules.ACTIONS.index(movement)
    assert game.legal_action_mask()[action] is True
    result = game.step(action)
    assert game.head == target
    assert result.done == occupied
    assert result.reward == (COURSE_REWARDS.death if occupied else COURSE_REWARDS.step)
    assert result.info == ({"cause": "self"} if occupied else {})


@pytest.mark.parametrize("grow_pending", [False, True])
def test_wraparound_tail_is_dangerous_only_when_not_vacated(grow_pending):
    game = SnakeGame(seed=0)
    game.body = [(14, 7), (14, 8), (0, 8), (0, 7)]
    game.direction = rules.UP
    game.food = (2, 2)
    game._grow_pending = grow_pending
    assert build_state(game)[1] == float(grow_pending)
    result = game.step(3)
    assert result.done == grow_pending


@pytest.mark.parametrize("direction", rules.ACTIONS)
def test_wraparound_apple_and_body_at_edge(direction):
    n = rules.GRID_SIZE
    head = (14 if direction[0] == 1 else 0 if direction[0] == -1 else 7,
            14 if direction[1] == 1 else 0 if direction[1] == -1 else 7)
    game = SnakeGame(seed=10)
    game.body = [((head[0] - i * direction[0]) % n,
                  (head[1] - i * direction[1]) % n) for i in range(3)]
    game.direction = direction
    game.food = ((head[0] + direction[0]) % n, (head[1] + direction[1]) % n)
    assert np.array_equal(build_state(game)[:3], [0, 0, 0])
    result = game.step(rules.ACTIONS.index(direction))
    assert result.ate and not result.done
    assert result.reward == COURSE_REWARDS.apple and result.score == 1
    assert game.food not in game.body
    game.step(rules.ACTIONS.index(direction))
    assert len(game.body) == 4
    assert all(0 <= x < n and 0 <= y < n for x, y in game.body)


def trace(seed):
    game = SnakeGame(seed=seed)
    rng = random.Random(seed)
    frames = [game.snapshot()]
    # D'abord deux tours complets, garantissant plusieurs franchissements.
    game.food = (2, 2)
    for _ in range(2 * rules.GRID_SIZE):
        result = game.step(3)
        assert not result.done
        frames.append(game.snapshot(action=3, reward=result.reward))
    for _ in range(3000):
        action = rng.choice([i for i, legal in enumerate(game.legal_action_mask()) if legal])
        result = game.step(action)
        frames.append(game.snapshot(action=action, reward=result.reward))
        if result.done:
            break
    return frames


def test_wraparound_seed_reproduces_full_trajectory():
    assert trace(123) == trace(123)


def test_wraparound_other_seed_changes_trajectory():
    assert trace(123) != trace(124)


def test_wraparound_does_not_add_implicit_time_limit():
    game = SnakeGame(seed=0)
    game.food = (2, 2)
    for _ in range(10000):
        assert not game.step(3).done
    assert game.steps == 10000
