"""Experimental cutoffs stay distinct from official Snake endings."""

import numpy as np
import pytest

from snake_rl import rules
from snake_rl.config import Config
from snake_rl.game import COURSE_REWARDS, SnakeGame, StepResult
from snake_rl import evaluate as evaluate_module
from snake_rl import train as train_module


RIGHT = rules.ACTIONS.index(rules.RIGHT)
UP = rules.ACTIONS.index(rules.UP)


def straight_game(limit=500):
    game = SnakeGame(seed=123, max_steps_without_food=limit)
    # A different row makes a perpetual straight trajectory food-free.
    game.food = (0, (game.head[1] + 1) % rules.GRID_SIZE)
    return game


def assert_flags(game, result, *, terminated, truncated):
    assert result.terminated is terminated
    assert result.truncated is truncated
    assert result.done is (terminated or truncated)
    assert game.terminated is terminated
    assert game.truncated is truncated
    assert game.done is (terminated or truncated)
    snapshot = game.snapshot()
    assert snapshot["terminated"] is terminated
    assert snapshot["truncated"] is truncated
    assert snapshot["done"] is (terminated or truncated)


def test_default_config_retains_explicit_500_step_experimental_limit():
    assert Config().max_steps_without_food == 500
    assert SnakeGame().max_steps_without_food is None


def test_500_food_free_steps_truncate_without_death_or_collision():
    game = straight_game()
    total_reward = 0.0
    for _ in range(499):
        result = game.step(RIGHT)
        total_reward += result.reward
        assert_flags(game, result, terminated=False, truncated=False)

    result = game.step(RIGHT)
    total_reward += result.reward
    assert_flags(game, result, terminated=False, truncated=True)
    assert result.reward == COURSE_REWARDS.step
    assert result.reward != COURSE_REWARDS.death
    assert result.info["cause"] == "truncated"
    assert result.won is False
    assert game.is_collision() is False
    assert game.score == 0
    assert game.steps == game.steps_since_food == 500
    assert total_reward == pytest.approx(50.0)
    with pytest.raises(RuntimeError):
        game.step(RIGHT)


@pytest.mark.parametrize("limit", [None, 0])
def test_disabled_limit_never_truncates_a_food_free_toric_loop(limit):
    game = straight_game(limit)
    for _ in range(501):
        result = game.step(RIGHT)
    assert_flags(game, result, terminated=False, truncated=False)
    assert game.steps_since_food == 501


@pytest.mark.parametrize("limit", [-1, -500, 1.5, "500", True, False])
def test_invalid_experimental_limit_is_rejected(limit):
    with pytest.raises((ValueError, TypeError)):
        SnakeGame(max_steps_without_food=limit)


def test_step_result_cannot_be_both_terminated_and_truncated():
    with pytest.raises(ValueError):
        StepResult(reward=0.1, score=0, terminated=True, truncated=True)


def test_apple_on_limit_step_is_accepted_and_resets_the_budget():
    game = straight_game(3)
    game.food = ((game.head[0] + 3) % rules.GRID_SIZE, game.head[1])
    game.step(RIGHT)
    game.step(RIGHT)
    result = game.step(RIGHT)
    assert_flags(game, result, terminated=False, truncated=False)
    assert result.ate is True
    assert result.reward == COURSE_REWARDS.apple
    assert game.score == 1
    assert game.steps_since_food == 0

    game.food = (0, (game.head[1] + 1) % rules.GRID_SIZE)
    for _ in range(2):
        assert game.step(RIGHT).done is False
    result = game.step(RIGHT)
    assert_flags(game, result, terminated=False, truncated=True)
    assert game.steps == 6
    assert game.steps_since_food == 3
    assert result.reward == COURSE_REWARDS.step


def colliding_game(limit=1):
    game = SnakeGame(seed=123, max_steps_without_food=limit)
    game.body = [(6, 5), (5, 5), (5, 4), (6, 4), (7, 4)]
    game.direction = rules.UP
    game.food = (0, 0)
    return game


def test_real_collision_takes_precedence_over_simultaneous_limit():
    game = colliding_game()
    result = game.step(UP)
    assert_flags(game, result, terminated=True, truncated=False)
    assert result.info["cause"] == "self"
    assert result.reward == COURSE_REWARDS.death
    assert result.won is False


def test_victory_is_terminated_not_truncated_at_limit():
    game = SnakeGame(seed=123, max_steps_without_food=1)
    cells = []
    for y in range(rules.GRID_SIZE):
        row = range(rules.GRID_SIZE) if y % 2 == 0 else range(rules.GRID_SIZE - 1, -1, -1)
        cells.extend((x, y) for x in row)
    game.body = list(reversed(cells[:-1]))
    game.food = cells[-1]
    game.direction = (
        game.food[0] - game.head[0], game.food[1] - game.head[1]
    )
    result = game.step(rules.ACTIONS.index(game.direction))
    assert_flags(game, result, terminated=True, truncated=False)
    assert result.info["cause"] == "victory"
    assert result.won is True
    assert result.ate is True
    assert result.reward == COURSE_REWARDS.apple + COURSE_REWARDS.victory


@pytest.mark.parametrize("ending", ["truncation", "collision"])
def test_reset_clears_both_end_flags(ending):
    game = straight_game(1) if ending == "truncation" else colliding_game()
    game.step(RIGHT if ending == "truncation" else UP)
    assert game.done is True
    game.reset()
    assert game.terminated is False
    assert game.truncated is False
    assert game.done is False
    assert game.steps == game.steps_since_food == 0
    assert game.snapshot()["terminated"] is False
    assert game.snapshot()["truncated"] is False


def test_done_is_a_read_only_compatibility_view():
    game = SnakeGame(seed=123)
    result = StepResult(reward=0.1, score=0)
    assert result.done is False
    with pytest.raises(AttributeError):
        game.done = True
    with pytest.raises(AttributeError):
        result.done = True


class RecordingAgent:
    """No network updates: record only the replay terminal mask."""

    def __init__(self, config):
        self.episodes_done = 0
        self.epsilon = 0.0
        self.transitions = []
        self.action = RIGHT

    def act(self, state, mask=None, greedy=False):
        return self.action

    def q_values(self, state, mask=None):
        return np.zeros(rules.N_ACTIONS, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done, next_mask=None):
        self.transitions.append({"reward": reward, "terminal": done})

    def learn(self):
        return None


@pytest.mark.parametrize("ending", ["truncation", "collision"])
def test_trainer_bootstraps_only_truncated_transitions(monkeypatch, tmp_path, ending):
    monkeypatch.setattr(train_module, "Agent", RecordingAgent)
    game = straight_game(1) if ending == "truncation" else colliding_game()
    monkeypatch.setattr(train_module, "SnakeGame", lambda **kwargs: game)
    trainer = train_module.Trainer(
        Config(run_id="end_flags", device="cpu", max_steps_without_food=1),
        run_dir=str(tmp_path / "run"),
        show_window=False,
    )
    trainer.agent.action = RIGHT if ending == "truncation" else UP
    stats = trainer.run_training_episode(1)
    assert len(trainer.agent.transitions) == 1
    transition = trainer.agent.transitions[0]
    assert transition["terminal"] is (ending == "collision")
    assert stats["terminated"] is (ending == "collision")
    assert stats["truncated"] is (ending == "truncation")
    assert stats["cause"] == ("self" if ending == "collision" else "truncated")
    assert transition["reward"] == (
        COURSE_REWARDS.death if ending == "collision" else COURSE_REWARDS.step
    )
    assert trainer.agent.episodes_done == 1


def test_evaluation_never_counts_truncation_as_an_official_termination(monkeypatch):
    monkeypatch.setattr(
        evaluate_module, "SnakeGame",
        lambda **kwargs: straight_game(kwargs["max_steps_without_food"]),
    )
    agent = RecordingAgent(Config())
    episode = evaluate_module.play_episode(
        agent, seed=123, record_frames=True, max_steps_without_food=3
    )
    assert episode["terminated"] is False
    assert episode["truncated"] is True
    assert episode["frames"][-1]["terminated"] is False
    assert episode["frames"][-1]["truncated"] is True
    block = evaluate_module.evaluate(
        agent, seeds=[123, 124], record_best_frames=False, max_steps_without_food=3
    )
    assert block["termination_counts"] == {}
    assert block["truncation_rate"] == 1.0
    assert block["win_rate"] == 0.0
    assert agent.transitions == []
    assert agent.episodes_done == 0
