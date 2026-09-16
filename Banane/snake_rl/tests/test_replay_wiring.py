"""Regression tests for measured evaluation trajectories and visual replay wiring."""

import json
from pathlib import Path

import pytest

from snake_rl.config import Config
from snake_rl.evaluate import evaluate
from snake_rl.metrics import load_replay
from snake_rl.train import Trainer, build_parser, config_from_args


def short_config(tmp_path, **overrides):
    options = dict(
        run_id="replay_wiring", output_dir=str(tmp_path), device="cpu",
        hidden_size=8, batch_size=8, learning_starts=100,
        replay_capacity=100, episodes=4, eval_interval=2, eval_episodes=2,
        checkpoint_every=0, max_steps_without_food=4,
        long_without_food_threshold=2, show_replay_window=False,
    )
    options.update(overrides)
    return Config(**options)


@pytest.mark.parametrize("arguments, expected", [([], True), (["--no-window"], False)])
def test_cli_overrides_headless_json_unless_no_window(tmp_path, monkeypatch, arguments, expected):
    import snake_rl.train as train_module

    path = tmp_path / "config.json"
    Config(show_replay_window=False).save(path)
    args = build_parser().parse_args(["--config", str(path), *arguments])
    assert args.show_window is expected
    captured = {}

    class CapturingTrainer:
        def __init__(self, config, show_window):
            captured.update(config=config, show_window=show_window)

        def train(self, verbose):
            return {}

    monkeypatch.setattr(train_module, "Trainer", CapturingTrainer)
    train_module.main(["--config", str(path), "--quiet", *arguments])
    assert captured["config"].show_replay_window is expected
    assert captured["show_window"] is expected


def test_replay_fps_default_config_and_cli_override(tmp_path):
    assert Config().replay_fps == 60
    path = tmp_path / "config.json"
    Config(replay_fps=90).save(path)
    parser = build_parser()
    assert config_from_args(parser.parse_args(["--config", str(path)])).replay_fps == 90
    assert config_from_args(parser.parse_args(
        ["--config", str(path), "--replay-fps", "120"]
    )).replay_fps == 120


@pytest.mark.parametrize("fps", [0, -1, 1.5, True])
def test_replay_fps_rejects_nonpositive_or_noninteger_values(fps):
    with pytest.raises(ValueError, match="replay_fps"):
        Config(replay_fps=fps)


class CountingStraightAgent:
    """A deliberately stateful policy exposes any second evaluation pass."""

    def __init__(self):
        self.calls = 0

    def act(self, state, mask=None, greedy=False):
        assert greedy
        self.calls += 1
        # The initial direction is right; continuing straight survives the cap.
        return 3


def test_best_and_stagnation_replays_use_measured_frames_without_policy_rerun():
    agent = CountingStraightAgent()
    block = evaluate(agent, [1000, 1001], record_best_frames=True,
                     max_steps_without_food=4, long_without_food_threshold=2)
    assert agent.calls == 8
    assert block["episode_metrics"][0]["steps"] == 4
    assert block["episode_metrics"][1]["steps"] == 4
    best = block["best_replay"]
    assert best["seed"] == block["best_seed"]
    assert best["score"] == block["record"]
    assert len(best["frames"]) == best["steps"] + 1
    assert best["frames"][-1]["score"] == best["score"]
    assert block["stagnation_replay"]["frames"]


def test_visual_replay_runs_after_each_block_before_training_resumes(
    tmp_path, monkeypatch, capsys,
):
    import snake_rl.plotting as plotting
    import snake_rl.render as render

    config = short_config(tmp_path, replay_best_after_eval=False, replay_fps=120)
    trainer = Trainer(config, show_window=True)
    events = []
    original_training = trainer.run_training_episode

    def training(episode):
        events.append(("train", episode))
        return original_training(episode)

    def replay(path, **kwargs):
        replay_data = load_replay(path)
        block = int(Path(path).stem.rsplit("_", 1)[-1])
        summary = json.loads((Path(trainer.run_dir) / "evaluations"
                              / f"evaluation_{block:04d}.json").read_text())
        assert Path(path).name == f"best_replay_eval_{block:04d}.json"
        assert replay_data["score"] == summary["record"]
        assert replay_data["seed"] == summary["best_seed"]
        assert replay_data["frames"]
        assert kwargs == {"fps": 120, "close_when_done": True, "linger_seconds": 0}
        # The replay announcement must precede the synchronous renderer call.
        output = capsys.readouterr().out
        assert (f"REPLAY BEST EVAL | episode {block} | score {replay_data['score']} "
                f"| seed {replay_data['seed']}") in output
        events.append(("replay", block))

    monkeypatch.setattr(trainer, "run_training_episode", training)
    monkeypatch.setattr(render, "replay_file", replay)
    monkeypatch.setattr(plotting, "plot_run", lambda *args, **kwargs: None)
    trainer.train(verbose=False)
    assert events == [("train", 1), ("train", 2), ("replay", 2),
                      ("train", 3), ("train", 4), ("replay", 4)]


def test_no_window_training_never_creates_pygame_display(tmp_path, monkeypatch):
    import pygame
    import snake_rl.plotting as plotting
    import snake_rl.train as train_module

    path = tmp_path / "input_config.json"
    short_config(tmp_path, show_replay_window=True).save(path)
    calls = []

    def forbidden_display(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("--no-window must not create a Pygame display")

    monkeypatch.setattr(pygame.display, "set_mode", forbidden_display)
    monkeypatch.setattr(plotting, "plot_run", lambda *args, **kwargs: None)
    train_module.main(["--config", str(path), "--no-window", "--quiet"])
    assert calls == []
    saved = json.loads((tmp_path / "replay_wiring" / "config.json").read_text())
    assert saved["show_replay_window"] is False
