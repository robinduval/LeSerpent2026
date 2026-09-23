"""Integration contracts with virtual time; no performance claims use these runs."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CHOU = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHOU))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import torch

from snake_rl.agent import DQNAgent, DQNConfig
from snake_rl.env import Env, RIGHT, UP
from snake_rl import evaluation, ui
from snake_rl.metrics import individual_key, select_candidate, summarize
from snake_rl.training import train


class VirtualClock:
    def __init__(self):
        self.now = 0.0
        self.previous_tick = 0.0
        self.calls = []

    def tick(self, hz):
        self.calls.append(hz)
        self.now = max(self.now, self.previous_tick + 1.0 / hz)
        self.previous_tick = self.now

    def time(self):
        return self.now

    def perf_counter(self):
        return self.now


class StraightPolicy:
    def select_action(self, env):
        return RIGHT


def collision_fixture(*args, **kwargs):
    env = Env(*args, **kwargs)
    env.body = [(1, 1), (1, 2), (2, 2), (2, 1), (3, 1)]
    env.direction = UP
    env.apple = (10, 10)
    return env


def small_agent():
    agent = DQNAgent(DQNConfig(hidden=8, batch_size=2, capacity=8, warmup=2))
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        agent.online.layers[-1].bias[RIGHT] = 1.0
    return agent


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        (CHOU / "runs").mkdir(exist_ok=True)
        self.work = tempfile.TemporaryDirectory(prefix="test_integration_", dir=CHOU / "runs")
        self.addCleanup(self.work.cleanup)
        self.base = Path(self.work.name)
        (self.base / "runs").mkdir()
        self.addCleanup(pygame.quit)

    def virtual_episode(self, policy, *, max_steps=6, render=None, trace=None, env_factory=None):
        clock = VirtualClock()
        with patch.object(evaluation, "time", clock), \
             patch.object(evaluation.pygame.time, "Clock", return_value=clock), \
             patch.object(evaluation.pygame.display, "get_init", return_value=False), \
             patch.object(evaluation, "Env", env_factory or Env):
            row = evaluation.play_episode(policy, 47, "test_policy", max_steps,
                                          render(clock) if render else None, trace)
        return row, clock

    def test_one_transition_per_five_hz_interval_and_immediate_first_move(self):
        trace = self.base / "trace.jsonl"
        row, clock = self.virtual_episode(StraightPolicy(), trace=trace)
        self.assertEqual(row["steps"], 6)
        self.assertEqual(clock.calls, [5] * 5)
        self.assertAlmostEqual(row["official_time_seconds"], 1.0)
        self.assertAlmostEqual(row["observed_move_hz"], 5.0)
        timings = [json.loads(line)["elapsed_seconds"] for line in trace.read_text().splitlines()]
        for actual, expected in zip(timings, (0, .2, .4, .6, .8, 1.0)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(row["termination_reason"], "external_step_limit")
        self.assertFalse(row["completed"])
        self.assertIn("not established", row["classability"])

    def test_terminal_timestamp_excludes_final_render_work(self):
        def slow_render(clock):
            def render(env, start_time, ended):
                clock.now += 3.0
            return render
        row, clock = self.virtual_episode(StraightPolicy(), render=slow_render,
                                          env_factory=collision_fixture)
        self.assertEqual(row["termination_reason"], "self_collision")
        self.assertEqual(row["steps"], 1)
        self.assertEqual(clock.calls, [])
        # Terminal event happened at t=0; rendering is not part of that event.
        self.assertEqual(row["official_time_seconds"], 0.0)

    def test_evaluation_never_explores_trains_or_changes_weights(self):
        agent = small_agent()
        before = {k: value.clone() for k, value in agent.online.state_dict().items()}
        counters = (agent.env_steps, agent.updates, len(agent.replay))
        rng = json.dumps(agent.rng.bit_generator.state, sort_keys=True)
        with patch.object(agent, "train_step", side_effect=AssertionError("evaluation trained")), \
             patch.object(agent, "observe", side_effect=AssertionError("evaluation recorded replay")), \
             patch.object(agent, "select_action", wraps=agent.select_action) as select:
            row, _ = self.virtual_episode(agent)
        self.assertEqual(row["steps"], 6)
        self.assertEqual(select.call_count, 6)
        self.assertTrue(all(call.kwargs == {"explore": False} for call in select.call_args_list))
        self.assertEqual(counters, (agent.env_steps, agent.updates, len(agent.replay)))
        self.assertEqual(rng, json.dumps(agent.rng.bit_generator.state, sort_keys=True))
        for key, value in agent.online.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))

    def test_quit_before_first_action_remains_external(self):
        clock = VirtualClock()
        policy = StraightPolicy()
        with patch.object(evaluation, "time", clock), \
             patch.object(evaluation.pygame.time, "Clock", return_value=clock), \
             patch.object(evaluation.pygame.display, "get_init", return_value=True), \
             patch.object(evaluation.pygame.event, "get", return_value=[pygame.event.Event(pygame.QUIT)]), \
             patch.object(policy, "select_action", side_effect=AssertionError("action after close")):
            row = evaluation.play_episode(policy, 1, "test_close")
        self.assertEqual(row["steps"], 0)
        self.assertEqual(row["termination_reason"], "user_quit")
        self.assertFalse(row["completed"])
        self.assertEqual(row["decision_latency"], 0)
        self.assertEqual(clock.calls, [])

    def test_smoke_dummy_display_closes_without_changing_checkpoint(self):
        pygame.quit()
        checkpoint = self.base / "test_agent.pt"
        small_agent().save(checkpoint, include_replay=False)
        original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        clock = VirtualClock()
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}), \
             patch.object(ui.pygame.time, "Clock", return_value=clock), \
             patch.object(ui, "time", clock), \
             redirect_stdout(io.StringIO()):
            ui.play(checkpoint, self.base, seed=47, smoke_steps=2)
        self.assertEqual(clock.calls, [5, 5])
        self.assertFalse(pygame.get_init())
        self.assertFalse(pygame.display.get_init())
        self.assertEqual(original_hash, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        rows = [json.loads(s) for s in (self.base / "runs/interactive.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assert_interactive_fields(rows[0])
        self.assertEqual(rows[0]["termination_reason"], "user_quit")
        self.assertFalse(rows[0]["completed"])
        self.assertEqual(rows[0]["steps"], 2)
        self.assertAlmostEqual(rows[0]["official_time_seconds"], .4)

    def assert_interactive_fields(self, row):
        expected = {"model_id", "configuration_id", "seed", "official_score",
                    "official_time_seconds", "rl_return", "apples", "steps",
                    "completed", "termination_reason", "final_length", "decision_latency",
                    "training_transitions", "training_updates", "training_wall_time",
                    "demonstration_cost", "simulation_cost"}
        self.assertFalse(expected - set(row), f"Missing telemetry: {expected - set(row)}")
        self.assertEqual(row["apples"], row["official_score"])
        self.assertGreaterEqual(row["decision_latency"], 0)

    def test_terminal_interactive_telemetry_is_complete_and_logged_once(self):
        pygame.quit()
        checkpoint = self.base / "test_terminal_agent.pt"
        agent = small_agent()
        agent.save(checkpoint, metadata={"run_id": "integration_fixture", "training_wall_time": 12.0},
                   include_replay=False)
        clock = VirtualClock()
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}), \
             patch.object(ui.pygame.time, "Clock", return_value=clock), \
             patch.object(ui, "time", clock), \
             patch.object(ui, "Env", collision_fixture), \
             redirect_stdout(io.StringIO()):
            ui.play(checkpoint, self.base, seed=47, smoke_steps=1)
        rows = [json.loads(s) for s in (self.base / "runs/interactive.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)  # Closing after a terminal must not log a second game.
        row = rows[0]
        self.assert_interactive_fields(row)
        self.assertEqual(row["configuration_id"], "integration_fixture")
        self.assertEqual(row["seed"], 47)
        self.assertEqual(row["termination_reason"], "self_collision")
        self.assertEqual(row["steps"], 1)
        self.assertEqual(row["official_time_seconds"], 0.0)
        self.assertEqual(row["training_wall_time"], 12.0)
        self.assertEqual(row["training_transitions"], agent.env_steps)
        self.assertEqual(row["training_updates"], agent.updates)

    def test_missing_and_corrupt_checkpoint_fail_explicitly_without_fallback(self):
        invalid = self.base / "corrupt.pt"
        invalid.write_bytes(b"this is not a torch checkpoint")
        for checkpoint in (self.base / "missing.pt", invalid):
            completed = subprocess.run(
                [sys.executable, str(CHOU / "serpent-algo.py"), "--checkpoint", str(checkpoint)],
                cwd=self.base, capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn("Snake RL:", completed.stderr)
            self.assertNotIn("Loaded ", completed.stdout)

    def test_default_entry_resources_are_absolute_and_need_no_arguments(self):
        spec = importlib.util.spec_from_file_location("snake_entry_contract", CHOU / "serpent-algo.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        args = entry.parser().parse_args([])
        self.assertEqual(entry.BASE_DIR, CHOU)
        self.assertEqual(args.checkpoint, CHOU / "checkpoints" / "selected.pt")
        self.assertTrue(args.checkpoint.is_absolute())
        self.assertFalse(args.train)
        self.assertFalse(args.evaluate)

    def test_bounded_training_truncation_and_resumable_cost_accounting(self):
        spec = importlib.util.spec_from_file_location("snake_train_contract", CHOU / "serpent-algo.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        args = entry.parser().parse_args([
            "--train", "--run-id", "tiny", "--transitions", "4", "--seed", "47",
            "--batch-size", "2", "--capacity", "8", "--warmup", "2",
            "--n-step", "3", "--update-every", "1", "--save-every", "100",
            "--episode-limit", "2", "--stagnation-limit", "2"])
        with redirect_stdout(io.StringIO()):
            candidate = train(args, self.base)
        rows = [json.loads(s) for s in (self.base / "runs/tiny/episodes.jsonl").read_text().splitlines()]
        self.assertEqual([row["steps"] for row in rows], [2, 2])
        self.assertTrue(all(row["termination_reason"] == "training_interruption" for row in rows))
        self.assertTrue(all(row["official_time_seconds"] is None for row in rows))
        self.assertTrue(candidate.exists())
        self.assertFalse((self.base / "checkpoints/selected.pt").exists())
        resume = self.base / "runs/tiny/resume.pt"
        agent = DQNAgent.load(resume, require_resume=True)
        self.assertEqual(agent.env_steps, 4)
        self.assertGreater(agent.updates, 0)
        self.assertEqual(len(agent.replay), 4)
        self.assertEqual(len(agent.pending), 0)
        self.assertFalse(agent.replay.terminated[:4].any())
        for actual, expected in zip(agent.replay.discounts[:4], (.99**2, .99, .99**2, .99)):
            self.assertAlmostEqual(float(actual), expected, places=6)
        # Give the prior cost an unmistakable value so a reset to session-only
        # duration cannot accidentally pass a short real-time smoke run.
        agent.save(resume, metadata={"training_wall_time": 123.0}, include_replay=True)
        args.run_id, args.resume, args.transitions = "tiny_resumed", resume, 2
        with redirect_stdout(io.StringIO()):
            final = train(args, self.base)
        continued = DQNAgent.load(final)
        self.assertEqual(continued.env_steps, 6)
        self.assertGreaterEqual(continued.metadata["training_wall_time"], 123.0)
        self.assertGreater(continued.updates, agent.updates)

    def test_official_examples_and_no_times_for_interrupted_games(self):
        def result(score, seconds, *, reason="self_collision", seed=1):
            return dict(official_score=score, official_time_seconds=seconds,
                        termination_reason=reason, completed=False, seed=seed,
                        evaluation_step_budget=1000)
        for a, b, expected in [(result(300, 200), result(250, 50), 0),
                               (result(300, 200), result(300, 100), 1),
                               (result(301, 200), result(300, 100), 0)]:
            self.assertIs(sorted([a, b], key=individual_key)[0], (a, b)[expected])
        interrupted = result(9, 10, reason="external_step_limit")
        report = summarize([interrupted, result(9, 20)])
        self.assertEqual(report["external_cutoffs"], 1)
        self.assertEqual(report["terminal_time_count_by_exact_score"], {"9": 1})
        self.assertEqual(report["terminal_time_mean_by_exact_score"], {"9": 20})
        # At equal partial score the lower interrupted duration cannot select a model.
        slower_name_first = {"a_longer": [result(9, 90, reason="external_step_limit")],
                             "z_shorter": [interrupted]}
        self.assertEqual(select_candidate(slower_name_first), "a_longer")


if __name__ == "__main__":
    unittest.main()
