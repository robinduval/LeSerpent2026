"""Periodic checkpoints supplement champion/latest and support short resumes."""

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from snake_rl.config import Config
from snake_rl.train import Trainer, build_parser, config_from_args


def tiny_config(tmp_path, **overrides):
    options = dict(
        run_id="milestone_test", output_dir=str(tmp_path), device="cpu", seed=7,
        hidden_size=8, batch_size=4, learning_starts=4, replay_capacity=100,
        episodes=12, eval_interval=5, eval_episodes=2, checkpoint_every=5,
        max_steps_without_food=4, long_without_food_threshold=2,
        show_replay_window=False, replay_best_after_eval=False,
    )
    options.update(overrides)
    return Config(**options)


@pytest.fixture
def trained_run(tmp_path, monkeypatch):
    import snake_rl.plotting as plotting

    monkeypatch.setattr(plotting, "plot_run", lambda *args, **kwargs: None)
    trainer = Trainer(tiny_config(tmp_path), show_window=False)
    trainer.train(verbose=False)
    return trainer


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_milestones_at_five_and_ten_preserve_latest_and_champion(trained_run):
    run_dir = Path(trained_run.run_dir)
    assert sorted(path.name for path in run_dir.glob("milestone_*.pt")) == [
        "milestone_0005.pt", "milestone_0010.pt",
    ]
    latest = read_checkpoint(run_dir / "latest.pt")
    champion = read_checkpoint(run_dir / "best_mean.pt")
    assert latest["episode"] == latest["episodes_done"] == 12
    assert champion["episode"] in {5, 10}
    assert champion["eval_mean_score"] == trained_run.best_eval["mean_score"]

    for episode in (5, 10):
        checkpoint = read_checkpoint(run_dir / f"milestone_{episode:04d}.pt")
        assert {
            "policy", "target", "optimizer", "episode", "episodes_done",
            "learn_steps", "epsilon", "config", "seed", "rng", "agent_rng",
            "trainer_state", "replay_buffer_saved",
        } <= checkpoint.keys()
        assert checkpoint["episode"] == checkpoint["episodes_done"] == episode
        assert checkpoint["seed"] == checkpoint["config"]["seed"] == 7
        assert checkpoint["config"] == trained_run.config.to_dict()
        assert checkpoint["policy"] and checkpoint["target"]
        assert checkpoint["optimizer"]["state"]
        assert checkpoint["learn_steps"] > 0
        assert checkpoint["epsilon"] == pytest.approx(
            1.0 - 0.98 * episode / trained_run.config.epsilon_decay_episodes
        )
        assert checkpoint["replay_buffer_saved"] is False
        state = checkpoint["trainer_state"]
        assert {
            "train_scores", "episode_stats", "record_train", "best_eval",
            "best_record_eval", "training_episode_seconds", "elapsed_seconds",
        } <= state.keys()
        assert len(state["train_scores"]) == len(state["episode_stats"]) == episode
        assert state["record_train"] == max(state["train_scores"])
        assert state["best_eval"]["mean_score"] is not None
        assert "best_replay" not in state["best_eval"]
        assert "stagnation_replay" not in state["best_eval"]


def test_zero_interval_disables_only_milestones(tmp_path, monkeypatch):
    import snake_rl.plotting as plotting

    monkeypatch.setattr(plotting, "plot_run", lambda *args, **kwargs: None)
    trainer = Trainer(tiny_config(tmp_path, episodes=2, eval_interval=1,
                                 checkpoint_every=0), show_window=False)
    trainer.train(verbose=False)
    run_dir = Path(trainer.run_dir)
    assert not list(run_dir.glob("milestone_*.pt"))
    assert (run_dir / "latest.pt").is_file()
    assert (run_dir / "best_mean.pt").is_file()


def test_checkpoint_interval_default_config_and_cli_precedence(tmp_path):
    assert Config().checkpoint_every == 500
    parser = build_parser()
    assert config_from_args(parser.parse_args(["--no-window"])).checkpoint_every == 500
    path = tmp_path / "config.json"
    Config(checkpoint_every=10).save(path)
    assert config_from_args(parser.parse_args(
        ["--config", str(path), "--no-window"]
    )).checkpoint_every == 10
    assert config_from_args(parser.parse_args(
        ["--config", str(path), "--checkpoint-interval", "5", "--no-window"]
    )).checkpoint_every == 5
    assert config_from_args(parser.parse_args(
        ["--config", str(path), "--checkpoint-interval", "0", "--no-window"]
    )).checkpoint_every == 0
    with pytest.raises(ValueError, match="checkpoint"):
        config_from_args(parser.parse_args(["--checkpoint-interval", "-1"]))


@pytest.mark.parametrize("interval", [-1, 1.5, True])
def test_checkpoint_interval_rejects_negative_or_noninteger_values(interval):
    with pytest.raises(ValueError, match="checkpoint"):
        Config(checkpoint_every=interval)


def test_resume_cli_uses_checkpoint_config_and_applies_overrides(trained_run):
    path = Path(trained_run.run_dir) / "milestone_0010.pt"
    args = build_parser().parse_args([
        "--resume", str(path), "--run-id", "resumed", "--episodes", "14",
        "--checkpoint-interval", "2", "--no-window",
    ])
    config = config_from_args(args)
    assert config.run_id == "resumed"
    assert config.episodes == 14
    assert config.checkpoint_every == 2
    assert config.hidden_size == trained_run.config.hidden_size
    assert config.seed == trained_run.config.seed
    assert config.show_replay_window is False
    assert "resume" not in config.to_dict()


def test_resume_restores_networks_optimizer_schedule_statistics_and_rng(trained_run):
    path = Path(trained_run.run_dir) / "milestone_0010.pt"
    payload = read_checkpoint(path)
    expected_agent_rng = np.random.default_rng()
    expected_agent_rng.bit_generator.state = copy.deepcopy(payload["agent_rng"])
    expected_private_draws = expected_agent_rng.random(5)
    expected_python_rng = random.Random()
    expected_python_rng.setstate(payload["rng"]["python"])
    expected_numpy_rng = np.random.RandomState()
    expected_numpy_rng.set_state(payload["rng"]["numpy"])
    expected_torch_rng = torch.Generator()
    expected_torch_rng.set_state(payload["rng"]["torch"])

    restored = Trainer(trained_run.config.replace(run_id="restored"),
                       show_window=False, resume=path)
    for name, weights in restored.agent.policy_net.state_dict().items():
        assert torch.equal(weights, payload["policy"][name])
    for name, weights in restored.agent.target_net.state_dict().items():
        assert torch.equal(weights, payload["target"][name])
    optimizer = restored.agent.optimizer.state_dict()
    assert optimizer["param_groups"] == payload["optimizer"]["param_groups"]
    for parameter, state in optimizer["state"].items():
        for key, value in state.items():
            assert torch.equal(value, payload["optimizer"]["state"][parameter][key])
    assert restored.agent.episodes_done == 10
    assert restored.agent.learn_steps == payload["learn_steps"]
    assert restored.agent.epsilon == pytest.approx(payload["epsilon"])
    assert restored.train_scores == payload["trainer_state"]["train_scores"]
    assert restored.episode_stats == payload["trainer_state"]["episode_stats"]
    assert restored.record_train == payload["trainer_state"]["record_train"]
    assert restored.best_eval == payload["trainer_state"]["best_eval"]
    assert restored.best_record_eval == payload["trainer_state"]["best_record_eval"]
    assert restored.training_episode_seconds == payload["trainer_state"]["training_episode_seconds"]
    assert len(restored.agent.memory) == 0
    assert np.array_equal(restored.agent._rng.random(5), expected_private_draws)
    assert random.random() == expected_python_rng.random()
    assert np.array_equal(np.random.random(5), expected_numpy_rng.random(5))
    assert torch.equal(torch.rand(5), torch.rand(5, generator=expected_torch_rng))


def test_resume_continues_absolute_episode_numbers_without_mutating_source(trained_run):
    path = Path(trained_run.run_dir) / "milestone_0010.pt"
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = read_checkpoint(path)
    resumed = Trainer(trained_run.config.replace(run_id="continued"),
                      show_window=False, resume=path)
    summary = resumed.train(verbose=False)
    rows = [json.loads(line) for line in
            (Path(resumed.run_dir) / "metrics.jsonl").read_text().splitlines()]
    train_rows = [row for row in rows if row["phase"] == "train"]
    assert [row["episode"] for row in train_rows] == [11, 12]
    assert resumed.agent.episodes_done == 12
    assert train_rows[0]["epsilon"] == pytest.approx(round(payload["epsilon"], 4))
    assert resumed.train_scores[:10] == payload["trainer_state"]["train_scores"]
    assert resumed.episode_stats[:10] == payload["trainer_state"]["episode_stats"]
    assert len(resumed.train_scores) == len(resumed.episode_stats) == 12
    assert summary["episodes"] == 12
    assert summary["record_train"] == max(resumed.train_scores)
    assert read_checkpoint(Path(resumed.run_dir) / "latest.pt")["episode"] == 12
    assert not list(Path(resumed.run_dir).glob("milestone_*.pt"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize('field', ['eval_episodes', 'eval_seed_start'])
def test_resume_refuses_a_different_champion_evaluation_suite(trained_run, field):
    path = Path(trained_run.run_dir) / 'milestone_0010.pt'
    config = trained_run.config.replace(run_id=f'incompatible_{field}',
                                        **{field: getattr(trained_run.config, field) + 1})
    with pytest.raises(ValueError, match='configuration incompatible'):
        Trainer(config, show_window=False, resume=path)


def test_resumed_evaluation_records_cumulative_elapsed_time(trained_run, monkeypatch):
    from types import SimpleNamespace
    import snake_rl.train as train_module

    path = Path(trained_run.run_dir) / 'milestone_0010.pt'
    restored = Trainer(trained_run.config.replace(run_id='resumed_time'),
                       show_window=False, resume=path)
    restored.started_at = 7.0
    restored.elapsed_before_resume = 100.0
    monkeypatch.setattr(train_module, 'time', SimpleNamespace(perf_counter=lambda: 10.0))
    rows = []
    logger = SimpleNamespace(log=lambda **row: rows.append(row))
    restored.run_evaluation(10, logger)
    assert rows[0]['wall_time_seconds'] == 103.0
