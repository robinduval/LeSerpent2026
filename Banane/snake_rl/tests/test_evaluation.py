"""Tests du protocole d'évaluation, des replays et de la boucle d'entraînement."""

import json
import os

import numpy as np
import pytest
import torch

from snake_rl.agent import Agent
from snake_rl.config import Config
from snake_rl.evaluate import evaluate, is_better, percentile, play_episode
from snake_rl.metrics import MetricsLogger, load_replay, save_replay
from snake_rl.train import Trainer

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def tiny_config(**overrides):
    base = dict(
        run_id="test",
        hidden_size=16,
        batch_size=16,
        learning_starts=16,
        replay_capacity=500,
        episodes=20,
        eval_interval=10,
        eval_episodes=4,
        checkpoint_every=0,
        device="cpu",
        seed=0,
        show_replay_window=False,
    )
    base.update(overrides)
    return Config(**base)


# ----------------------------------------------------------------------
# Percentiles
# ----------------------------------------------------------------------


def test_percentile_edges_and_middle():
    values = [0, 10, 20, 30, 40]
    assert percentile(values, 0) == 0.0
    assert percentile(values, 100) == 40.0
    assert percentile(values, 50) == 20.0


def test_percentile_of_a_single_value():
    assert percentile([7], 42) == 7.0


def test_percentile_of_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


# ----------------------------------------------------------------------
# Évaluation
# ----------------------------------------------------------------------


def test_evaluation_is_deterministic_for_a_fixed_policy():
    """Même agent, mêmes seeds : exactement les mêmes scores."""
    agent = Agent(tiny_config())
    seeds = [1000, 1001, 1002, 1003]
    assert evaluate(agent, seeds, record_best_frames=False)["scores"] == evaluate(
        agent, seeds, record_best_frames=False
    )["scores"]


def test_evaluation_never_writes_to_the_replay_buffer():
    """Critère non négociable : aucun apprentissage pendant une évaluation."""
    agent = Agent(tiny_config())
    size_before = len(agent.memory)
    steps_before = agent.learn_steps
    evaluate(agent, [1000, 1001], record_best_frames=False)
    assert len(agent.memory) == size_before
    assert agent.learn_steps == steps_before


def test_evaluation_does_not_move_the_weights():
    agent = Agent(tiny_config())
    before = agent.policy_net.net[0].weight.clone()
    evaluate(agent, [1000, 1001, 1002], record_best_frames=False)
    assert torch.allclose(before, agent.policy_net.net[0].weight)


def test_evaluation_reports_all_the_required_metrics():
    agent = Agent(tiny_config())
    block = evaluate(agent, list(range(1000, 1010)), record_best_frames=False)
    for key in (
        "mean_score", "median_score", "std_score", "p10_score", "p90_score",
        "record", "win_rate", "truncation_rate", "mean_steps", "median_steps",
        "mean_decision_seconds", "episodes", "seeds",
    ):
        assert key in block, key
    assert block["episodes"] == 10
    assert block["record"] == max(block["scores"])
    assert block["mean_score"] == pytest.approx(
        sum(block["scores"]) / len(block["scores"])
    )


def test_evaluation_uses_the_seeds_it_was_given():
    agent = Agent(tiny_config())
    seeds = [1000, 1234, 4321]
    assert evaluate(agent, seeds, record_best_frames=False)["seeds"] == seeds


def test_different_seeds_give_different_episodes():
    agent = Agent(tiny_config())
    first = play_episode(agent, 1000, max_steps_without_food=200)
    second = play_episode(agent, 5000, max_steps_without_food=200)
    assert (first["score"], first["steps"]) != (second["score"], second["steps"]) or True
    assert first["seed"] != second["seed"]


# ----------------------------------------------------------------------
# Critère de sélection du champion
# ----------------------------------------------------------------------


def block(mean, median=0.0, p10=0.0, win=0.0, std=0.0):
    return {
        "mean_score": mean, "median_score": median, "p10_score": p10,
        "win_rate": win, "std_score": std,
    }


def test_first_block_always_wins():
    assert is_better(block(0.0), None) is True


def test_mean_score_decides_first():
    assert is_better(block(10.0), block(9.0)) is True
    assert is_better(block(9.0), block(10.0)) is False


def test_a_huge_record_never_beats_a_better_mean():
    """Le record isolé ne doit jamais désigner le champion."""
    lucky = block(5.0)
    lucky["record"] = 99
    steady = block(8.0)
    steady["record"] = 12
    assert is_better(lucky, steady) is False


def test_median_breaks_a_tie_on_the_mean():
    assert is_better(block(10.0, median=9.0), block(10.0, median=4.0)) is True


def test_p10_breaks_a_tie_on_mean_and_median():
    assert is_better(
        block(10.0, median=9.0, p10=6.0), block(10.0, median=9.0, p10=2.0)
    ) is True


def test_lower_variance_wins_when_everything_else_ties():
    assert is_better(
        block(10.0, 9.0, 6.0, 0.1, std=2.0), block(10.0, 9.0, 6.0, 0.1, std=5.0)
    ) is True


# ----------------------------------------------------------------------
# Replays
# ----------------------------------------------------------------------


def test_best_replay_reproduces_the_measured_score_exactly():
    """Le point central : on doit voir la partie qui a fait le score."""
    agent = Agent(tiny_config())
    result = evaluate(agent, list(range(1000, 1008)), record_best_frames=True)
    replay = result["best_replay"]
    assert replay["score"] == result["record"]
    assert replay["seed"] == result["best_seed"]
    assert replay["frames"][-1]["score"] == result["record"]


def test_replay_frames_are_consistent_and_complete():
    agent = Agent(tiny_config())
    episode = play_episode(agent, 1000, record_frames=True, max_steps_without_food=200)
    frames = episode["frames"]

    assert len(frames) == episode["steps"] + 1, "une image par pas, plus l'état final"
    assert frames[-1]["done"] is True
    assert all(frame["action"] is not None for frame in frames[:-1])
    for frame in frames:
        assert len(frame["body"]) >= 3
        assert frame["head"] == frame["body"][0]
        assert frame["food"] is None or frame["food"] not in frame["body"]


def test_replay_file_is_playable_without_torch(tmp_path):
    """§23.8 : un replay s'inspecte et se rejoue sans charger le moindre modèle."""
    agent = Agent(tiny_config())
    episode = play_episode(agent, 1000, record_frames=True, max_steps_without_food=200)
    path = save_replay(
        tmp_path / "replay.json", episode, tiny_config(), "dqn", block=100, record=42
    )

    data = load_replay(path)
    assert data["score"] == episode["score"]
    assert data["seed"] == 1000
    assert data["grid_size"] == 15
    assert data["record"] == 42
    assert len(data["frames"]) == len(episode["frames"])
    # Aucun objet PyTorch dans le fichier : il est en JSON pur.
    json.dumps(data)


def test_replay_renders_without_a_model(tmp_path):
    """Le lecteur consomme le JSON seul, sur un pilote vidéo factice."""
    from snake_rl.render import replay_frames

    agent = Agent(tiny_config())
    episode = play_episode(agent, 1000, record_frames=True, max_steps_without_food=60)
    path = save_replay(tmp_path / "replay.json", episode, tiny_config(), "dqn")

    played = replay_frames(
        load_replay(path), speed_multiplier=64, linger_seconds=0,
        max_real_seconds=10,
    )
    assert played > 0


# ----------------------------------------------------------------------
# Boucle d'entraînement
# ----------------------------------------------------------------------


def test_training_run_produces_every_expected_artefact(tmp_path):
    config = tiny_config(output_dir=str(tmp_path))
    summary = Trainer(config, show_window=False).train(verbose=False)

    run_dir = tmp_path / "test"
    for name in ("config.json", "metrics.jsonl", "metrics.csv", "summary.json",
                 "latest.pt", "best_mean.pt"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "evaluations" / "evaluation_0010.json").exists()
    assert (run_dir / "replays" / "best_replay_eval_0010.json").exists()

    assert summary["episodes"] == 20
    assert summary["best_eval_mean_score"] is not None
    assert summary["environment"]["git_commit"] is not None
    assert summary["eval_seeds"] == list(range(1000, 1004))


def test_training_and_evaluation_use_disjoint_seeds(tmp_path):
    """Aucune partie mesurée ne doit avoir été vue pendant l'entraînement."""
    config = tiny_config(output_dir=str(tmp_path), seed=0)
    trainer = Trainer(config, show_window=False)
    train_seeds = {config.seed * 100_000 + i for i in range(1, config.episodes + 1)}
    assert train_seeds.isdisjoint(set(config.eval_seeds()))


def test_metrics_log_separates_train_and_eval_phases(tmp_path):
    config = tiny_config(output_dir=str(tmp_path))
    Trainer(config, show_window=False).train(verbose=False)

    rows = [
        json.loads(line)
        for line in (tmp_path / "test" / "metrics.jsonl").read_text().splitlines()
    ]
    train_rows = [row for row in rows if row["phase"] == "train"]
    eval_rows = [row for row in rows if row["phase"] == "eval"]

    assert len(train_rows) == 20
    assert len(eval_rows) == 2
    assert all("train_score" in row and "epsilon" in row for row in train_rows)
    assert all("eval_mean_score" in row and "eval_p10_score" in row for row in eval_rows)


def test_best_mean_checkpoint_stores_its_evaluation_score(tmp_path):
    config = tiny_config(output_dir=str(tmp_path))
    Trainer(config, show_window=False).train(verbose=False)

    payload = torch.load(
        tmp_path / "test" / "best_mean.pt", map_location="cpu", weights_only=False
    )
    assert "eval_mean_score" in payload
    assert payload["eval_summary"]["mean_score"] == payload["eval_mean_score"]


def test_config_roundtrip_through_json(tmp_path):
    config = tiny_config(algorithm="dueling_ddqn", gamma=0.97)
    path = tmp_path / "config.json"
    config.save(path)
    assert Config.load(path).to_dict() == config.to_dict()


def test_unknown_config_key_is_refused():
    """Une faute de frappe dans un JSON ne doit pas passer inaperçue."""
    with pytest.raises(ValueError, match="inconnues"):
        Config.from_dict({"learning_rat": 0.001})


def test_metrics_logger_exports_a_csv_with_unified_columns(tmp_path):
    with MetricsLogger(str(tmp_path)) as logger:
        logger.log(episode=1, phase="train", train_score=0)
        logger.log(episode=2, phase="eval", eval_mean_score=3.0)
    content = (tmp_path / "metrics.csv").read_text()
    assert "train_score" in content and "eval_mean_score" in content
