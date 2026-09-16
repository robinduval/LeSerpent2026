"""Deterministic tests for non-terminal analytical stagnation observations."""

import json

import pytest

from snake_rl import rules
from snake_rl.diagnostics import EpisodeDiagnostics, aggregate_episode_metrics
from snake_rl.game import SnakeGame


RIGHT = rules.ACTIONS.index(rules.RIGHT)


def straight_game(limit=None):
    game = SnakeGame(seed=12, max_steps_without_food=limit)
    game.food = (0, 0)  # Outside the horizontal trajectory.
    return game


def move(game, diagnostics, count, **kwargs):
    for _ in range(count):
        result = game.step(RIGHT)
        diagnostics.observe(game, result, **kwargs)


def test_exact_toric_cycle_is_analytical_only():
    game = straight_game()
    observer = EpisodeDiagnostics(game)
    move(game, observer, rules.GRID_SIZE * 2)
    metrics = observer.summary()
    assert metrics["first_cycle_step"] == rules.GRID_SIZE
    assert metrics["first_cycle_period"] == rules.GRID_SIZE
    assert metrics["cycle_repetitions"] == rules.GRID_SIZE + 1
    assert metrics["head_direction_repetitions"] == rules.GRID_SIZE + 1
    assert metrics["max_repeated_head_direction_count"] == 3
    assert metrics["probable_cycle"]
    assert not metrics["terminated"] and not metrics["truncated"]
    assert not game.done


def test_compressed_observation_repetition_is_not_cycle_proof():
    game = straight_game()
    observer = EpisodeDiagnostics(game)
    move(game, observer, 5, state=[0] * 11)
    metrics = observer.summary()
    assert metrics["compressed_state_repetitions"] == 4
    assert metrics["cycle_repetitions"] == 0
    assert not metrics["probable_cycle"]


def test_food_and_pending_growth_are_part_of_exact_physical_state():
    game = straight_game()
    observer = EpisodeDiagnostics(game)
    move(game, observer, rules.GRID_SIZE - 1)
    game.food = (1, 0)
    move(game, observer, 1)
    metrics = observer.summary()
    assert metrics["head_direction_repetitions"] == 1
    assert metrics["cycle_repetitions"] == 0
    before = observer._physical_state(game)
    game._grow_pending = True
    assert observer._physical_state(game) != before


def test_actual_self_collision_is_counted_as_death():
    game = straight_game()
    game.body = [(6, 5), (5, 5), (5, 4), (6, 4), (7, 4)]
    game.direction = rules.UP
    observer = EpisodeDiagnostics(game)
    result = game.step(rules.ACTIONS.index(rules.UP))
    observer.observe(game, result)
    metrics = observer.summary()
    assert metrics["terminated"] and not metrics["truncated"]
    assert metrics["death_count"] == 1
    assert metrics["cause"] == "self"
    assert metrics["reward"] == game.reward_profile.death


def test_foodless_threshold_counted_once_per_segment_and_eating_resets():
    game = straight_game()
    observer = EpisodeDiagnostics(game, long_without_food_threshold=3)
    game.food = ((game.head[0] + 4) % rules.GRID_SIZE, game.head[1])
    move(game, observer, 4)
    first = observer.summary()
    assert first["long_without_food_sequences"] == 1
    assert first["longest_without_food"] == 4
    assert first["steps_since_food"] == 0
    assert first["apple_step_intervals"] == [4]
    assert first["mean_inter_apple_steps"] is None
    assert first["mean_inter_apple_toric_distance"] is None
    assert first["inter_apple_toric_distances"] == []
    assert first["head_direction_repetitions"] == 0
    game.food = ((game.head[0] + 5) % rules.GRID_SIZE, game.head[1])
    move(game, observer, 5)
    game.food = (0, 0)
    move(game, observer, 4)
    metrics = observer.summary()
    assert metrics["long_without_food_sequences"] == 3
    assert metrics["longest_without_food"] == 5
    assert metrics["steps_since_food"] == 4
    assert metrics["apple_step_intervals"] == [4, 5]
    assert metrics["inter_apple_distances"] == [5]
    assert metrics["mean_inter_apple_steps"] == 5
    assert metrics["mean_inter_apple_toric_distance"] == 5
    assert metrics["inter_apple_toric_distances"] == [5]
    assert metrics["steps_per_apple"] == 6.5
    assert metrics["apples"] == metrics["score"] == 2


def test_successive_apple_spatial_distance_wraps_at_edge():
    game = straight_game()
    game.body = [(13, 7), (12, 7), (11, 7)]
    game.food = (14, 7)
    observer = EpisodeDiagnostics(game)
    move(game, observer, 1)
    game.food = (0, 7)
    move(game, observer, 1)
    metrics = observer.summary()
    assert metrics["inter_apple_toric_distances"] == [1]
    assert metrics["mean_inter_apple_toric_distance"] == 1
    assert metrics["inter_apple_distances"] == [1]
    pooled = aggregate_episode_metrics([metrics, metrics])
    assert pooled["mean_inter_apple_toric_distance"] == 1
    assert pooled["inter_apple_toric_distance_count"] == 2
    assert metrics["won"] is False


def test_undefined_nofood_metrics_and_truncation_not_death():
    game = straight_game(limit=4)
    observer = EpisodeDiagnostics(game, long_without_food_threshold=3)
    move(game, observer, 4)
    metrics = observer.summary()
    assert metrics["truncated"] and not metrics["terminated"]
    assert metrics["truncation_count"] == 1
    assert metrics["death_count"] == 0
    assert metrics["reward"] == pytest.approx(0.4)
    assert metrics["steps_per_apple"] is None
    assert metrics["mean_inter_apple_steps"] is None
    assert metrics["mean_inter_apple_toric_distance"] is None
    assert metrics["q_mean"] is None and metrics["loss_mean"] is None
    json.dumps(metrics, allow_nan=False)


def test_observer_preserves_game_rng_reward_and_rules():
    plain = straight_game()
    observed = straight_game()
    observer = EpisodeDiagnostics(observed, long_without_food_threshold=2)
    for _ in range(40):
        expected = plain.step(RIGHT)
        actual = observed.step(RIGHT)
        before = observed.snapshot()
        rng_before = observed._rng.getstate()
        observer.observe(observed, actual, state=[0] * 11)
        assert actual == expected
        assert observed.snapshot() == before == plain.snapshot()
        assert observed._rng.getstate() == rng_before
    assert not observed.done


def test_q_and_loss_are_decision_update_averages_and_legal_maxima():
    game = straight_game()
    observer = EpisodeDiagnostics(game)
    move(game, observer, 1, q_values=[1, 3, 5], loss=2)
    move(game, observer, 1, q_values=[2, 4, 6], loss=None)
    move(game, observer, 1, q_values=[], loss=4)
    metrics = observer.summary()
    assert metrics["q_mean"] == 3.5
    assert metrics["q_max"] == 6
    assert metrics["q_count"] == 2
    assert metrics["loss_mean"] == 3
    assert metrics["loss_count"] == 2


def test_aggregate_distinguishes_weighted_ratio_and_defined_episode_macro():
    game = straight_game(limit=4)
    no_food = EpisodeDiagnostics(game)
    move(game, no_food, 4)
    game = straight_game()
    game.food = ((game.head[0] + 2) % rules.GRID_SIZE, game.head[1])
    apples = EpisodeDiagnostics(game)
    move(game, apples, 2, q_values=[4, 6, 8], loss=2)
    game.food = (0, 0)
    move(game, apples, 2)
    metrics = aggregate_episode_metrics([no_food.summary(), apples.summary()])
    assert metrics["episodes"] == 2
    assert metrics["apples_total"] == 1
    assert metrics["steps_per_apple"] == 8
    assert metrics["mean_steps_per_apple"] == 4
    assert metrics["truncation_count"] == 1
    assert metrics["truncation_rate"] == 0.5
    assert metrics["death_count"] == 0
    assert metrics["mean_score"] == metrics["median_score"] == 0.5
    assert metrics["record"] == 1
    assert metrics["mean_inter_apple_steps"] is None
    assert metrics["q_mean"] == 6 and metrics["q_max"] == 8
    assert metrics["loss_mean"] == 2
    json.dumps(metrics, allow_nan=False)


def test_empty_aggregate_is_json_safe():
    metrics = aggregate_episode_metrics([])
    assert metrics["episodes"] == metrics["death_count"] == 0
    assert metrics["mean_score"] is None
    assert metrics["truncation_rate"] is None
    assert metrics["steps_per_apple"] is None
    assert metrics["mean_inter_apple_toric_distance"] is None
    assert metrics["inter_apple_toric_distance_count"] == 0
    json.dumps(metrics, allow_nan=False)


def test_threshold_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        EpisodeDiagnostics(straight_game(), long_without_food_threshold=0)
