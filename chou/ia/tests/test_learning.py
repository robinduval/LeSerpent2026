"""Learning invariants; tests do not require a long training run or GUI."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_rl.agent import DQNAgent, DQNConfig, double_dqn_targets
from snake_rl.replay import ReplayBuffer
from snake_rl.state import STATE_DIMS, encode
from snake_rl.env import Env
from snake_rl.training import train


class FakeEnv:
    body = [(3, 7), (2, 7), (1, 7)]
    direction = 1
    apple = (12, 4)
    grow_pending = False

    def would_collide(self, action):
        return action == 0


class ConstantNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.expand(len(states), -1)


class LearningTests(unittest.TestCase):
    def agent(self, **kwargs):
        params = dict(hidden=16, capacity=32, warmup=4, batch_size=4,
                      epsilon_decay_steps=20, seed=123)
        params.update(kwargs)
        return DQNAgent(DQNConfig(**params))

    def test_encoder_layout_and_purity(self):
        env = FakeEnv()
        before = copy.deepcopy(env.__dict__)
        np.testing.assert_array_equal(encode(env),
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0])
        ordered = encode(env, "ordered")
        self.assertEqual(ordered.shape, (STATE_DIMS["ordered"],))
        self.assertEqual(ordered.dtype, np.float32)
        # Head rank 3, next segment rank 2, tail rank 1, in head coordinates.
        self.assertAlmostEqual(float(ordered[11]), 3 / 225)
        self.assertAlmostEqual(float(ordered[11 + 14]), 2 / 225)
        self.assertAlmostEqual(float(ordered[11 + 13]), 1 / 225)
        self.assertEqual(env.__dict__, before)
        state = encode(env)
        state[:] = 8
        self.assertNotEqual(encode(env)[0], 8)
        with self.assertRaises(ValueError):
            encode(env, "oracle")

    def test_real_engine_encoding_preserves_rng_and_game_state(self):
        env = Env(seed=71)
        before = env.copy()
        for kind, dimension in STATE_DIMS.items():
            state = encode(env, kind)
            self.assertEqual(state.shape, (dimension,))
            self.assertTrue(np.isfinite(state).all())
        self.assertEqual(env.body, before.body)
        self.assertEqual(env.apple, before.apple)
        self.assertEqual(env.direction, before.direction)
        self.assertEqual(env.grow_pending, before.grow_pending)
        self.assertEqual(env.score, before.score)
        self.assertEqual(env.steps, before.steps)
        self.assertEqual(env.rng.getstate(), before.rng.getstate())

    def test_double_dqn_selection_evaluation_and_terminal(self):
        online = ConstantNetwork([1, 9, 2, 3])
        target = ConstantNetwork([10, 4, 30, 20])
        targets = double_dqn_targets(online, target, torch.zeros(2, 11),
                                     torch.tensor([2., -10.]), torch.tensor([.9, .9]),
                                     torch.tensor([False, True]))
        torch.testing.assert_close(targets, torch.tensor([5.6, -10.]))
        self.assertFalse(targets.requires_grad)
        self.assertIsNone(online.values.grad)
        self.assertIsNone(target.values.grad)

    def test_networks_are_independent_and_target_is_frozen(self):
        agent = self.agent(target_update=2)
        for a, b in zip(agent.online.parameters(), agent.target.parameters()):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
            self.assertFalse(b.requires_grad)
        old_target = {k: v.clone() for k, v in agent.target.state_dict().items()}
        for i in range(8):
            agent.observe(np.full(11, i / 10), i % 4, 1., np.full(11, (i + 1) / 10), False)
        stats = agent.train_step()
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))
        for k, v in agent.target.state_dict().items():
            torch.testing.assert_close(v, old_target[k])
        agent.train_step()
        for k, v in agent.target.state_dict().items():
            torch.testing.assert_close(v, agent.online.state_dict()[k])

    def test_transition_copies_and_evaluation_does_not_explore(self):
        agent = self.agent()
        state, nxt = np.zeros(11), np.ones(11)
        agent.observe(state, 1, 2., nxt, False)
        state[:] = 99
        nxt[:] = 99
        np.testing.assert_array_equal(agent.replay.states[0], np.zeros(11))
        np.testing.assert_array_equal(agent.replay.next_states[0], np.ones(11))
        before = copy.deepcopy(agent.rng.bit_generator.state)
        action = agent.select_action(np.zeros(11), explore=False)
        self.assertTrue(all(agent.select_action(np.zeros(11)) == action for _ in range(10)))
        self.assertEqual(before, agent.rng.bit_generator.state)
        self.assertEqual(agent.env_steps, 1)

    def test_nstep_terminal_flush_and_no_episode_crossover(self):
        agent = self.agent(n_step=3, gamma=.5)
        a, b, c = np.zeros(11), np.ones(11), np.full(11, 2.)
        agent.observe(a, 0, 2., b, False)
        a[:] = 9  # The pending queue must own its observations too.
        agent.observe(b, 1, 4., c, True)
        self.assertEqual(len(agent.replay), 2)
        self.assertEqual(len(agent.pending), 0)
        np.testing.assert_array_equal(agent.replay.rewards[:2], [4., 4.])
        np.testing.assert_array_equal(agent.replay.discounts[:2], [.25, .5])
        np.testing.assert_array_equal(agent.replay.terminated[:2], [True, True])
        np.testing.assert_array_equal(agent.replay.states[0], np.zeros(11))
        agent.observe(np.zeros(11), 2, 100., b, False)
        self.assertEqual(len(agent.replay), 2)

    def test_full_nstep_horizon_uses_gamma_power_and_last_observation(self):
        agent = self.agent(n_step=3, gamma=.5)
        for index, reward in enumerate((2., 4., 8.)):
            agent.observe(np.full(11, index), 0, reward,
                          np.full(11, index + 1), False)
        self.assertEqual(len(agent.replay), 1)
        self.assertEqual(len(agent.pending), 2)
        self.assertEqual(agent.replay.rewards[0], 6.)
        self.assertEqual(agent.replay.discounts[0], .125)
        np.testing.assert_array_equal(agent.replay.next_states[0], np.full(11, 3.))

    def test_truncation_flushes_but_bootstraps(self):
        agent = self.agent(n_step=3, gamma=.5)
        state = np.zeros(11)
        agent.observe(state, 0, 1., state + 1, False)
        agent.observe(state + 1, 0, 2., state + 2, False, truncated=True)
        self.assertEqual(len(agent.pending), 0)
        self.assertFalse(agent.replay.terminated[:2].any())
        np.testing.assert_array_equal(agent.replay.rewards[:2], [2., 2.])
        np.testing.assert_array_equal(agent.replay.discounts[:2], [.25, .5])

    def test_external_interruption_flushes_pending_without_false_terminal(self):
        agent = self.agent(n_step=3, gamma=.5)
        state = np.zeros(11)
        agent.observe(state, 0, 2., state + 1, False)
        agent.observe(state + 1, 0, 4., state + 2, False)
        agent.end_episode()
        self.assertEqual(len(agent.pending), 0)
        self.assertEqual(agent.env_steps, 2)
        np.testing.assert_array_equal(agent.replay.rewards[:2], [4., 4.])
        self.assertFalse(agent.replay.terminated[:2].any())

    def test_priorities_probabilities_weights_and_duplicate_updates(self):
        buffer = ReplayBuffer(4, 11, prioritized=True, alpha=1., priority_epsilon=.01)
        for i in range(4):
            buffer.add(np.zeros(11), i, 0., np.zeros(11), False, .9)
        buffer.update_priorities([0, 1, 2, 3], [1., 2., 4., 8.])
        result = buffer.sample(4, np.random.default_rng(4), beta=1.)
        probabilities = buffer.priorities[:4] / buffer.priorities[:4].sum()
        np.testing.assert_allclose(result["probabilities"], probabilities[result["indices"]])
        np.testing.assert_allclose(result["weights"].numpy(), probabilities.min() /
                                   probabilities[result["indices"]], rtol=1e-6)
        buffer.update_priorities([1, 1], [.1, 9.])
        self.assertAlmostEqual(float(buffer.priorities[1]), 9.01, places=5)
        with self.assertRaises(FloatingPointError):
            buffer.update_priorities([1], [float("nan")])

    def test_checkpoint_exact_resume_including_pending_and_rng(self):
        agent = self.agent(n_step=3, prioritized=True)
        for i in range(11):
            state = np.full(11, i / 10, dtype=np.float32)
            agent.observe(state, agent.select_action(state, explore=True), .2,
                          state + .1, i == 7)
            agent.train_step()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "runs") as temp:
            path = Path(temp) / "resume.pt"
            agent.save(path, {"purpose": "test"})
            restored = DQNAgent.load(path, require_resume=True)
            self.assertEqual(restored.metadata, {"purpose": "test"})
            self.assertEqual(restored.env_steps, agent.env_steps)
            self.assertEqual(restored.updates, agent.updates)
            self.assertEqual(len(restored.pending), len(agent.pending))
            np.testing.assert_array_equal(restored.replay.states[:len(restored.replay)],
                                          agent.replay.states[:len(agent.replay)])
            for i in range(10):
                state = np.full(11, .12 * i, dtype=np.float32)
                self.assertEqual(agent.select_action(state), restored.select_action(state))
                self.assertEqual(agent.select_action(state, True), restored.select_action(state, True))
            stats_a, stats_b = agent.train_step(), restored.train_step()
            self.assertEqual(stats_a, stats_b)
            for k, v in agent.online.state_dict().items():
                torch.testing.assert_close(v, restored.online.state_dict()[k], rtol=0, atol=0)
            agent.save(path, include_replay=False)
            self.assertEqual(DQNAgent.load(path).select_action(np.zeros(11)), agent.select_action(np.zeros(11)))
            with self.assertRaises(ValueError):
                DQNAgent.load(path, require_resume=True)
            with self.assertRaises(ValueError):
                DQNAgent.load(path, expected_encoder="ordered")
            damaged = torch.load(path, weights_only=True)
            damaged["rule_signature"] = "different game"
            torch.save(damaged, path)
            with self.assertRaises(ValueError):
                DQNAgent.load(path)

    def training_args(self, **overrides):
        params = dict(run_id="integration", resume=None, encoder="classic11", seed=9,
                      gamma=.95, lr=.001, batch_size=4, capacity=32, warmup=4,
                      epsilon_decay=20, n_step=3, per=False, reward_step=0.,
                      transitions=12, shaping=0., episode_limit=100,
                      stagnation_limit=100, update_every=1, updates_per_step=1,
                      save_every=100)
        params.update(overrides)
        return SimpleNamespace(**params)

    def test_driver_interrupt_checkpoint_and_episode_boundary_resume(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "runs") as temp:
            base = Path(temp)
            args = self.training_args()
            select_action = DQNAgent.select_action

            def interrupt_after_seven(agent, state, explore=False):
                if agent.env_steps >= 7:
                    raise KeyboardInterrupt
                return select_action(agent, state, explore)

            with patch.object(DQNAgent, "select_action", interrupt_after_seven):
                with contextlib.redirect_stdout(io.StringIO()):
                    train(args, base)
            resume_path = base / "runs" / args.run_id / "resume.pt"
            saved = DQNAgent.load(resume_path, require_resume=True)
            self.assertEqual(saved.env_steps, 7)
            self.assertEqual(len(saved.pending), 0)
            self.assertEqual(len(saved.replay), 7)
            self.assertFalse(saved.replay.terminated[:len(saved.replay)].any())
            previous_updates = saved.updates
            args.resume = str(resume_path)
            args.transitions = 5
            with contextlib.redirect_stdout(io.StringIO()):
                candidate = train(args, base)
            resumed = DQNAgent.load(resume_path, require_resume=True)
            self.assertEqual(resumed.env_steps, 12)
            self.assertEqual(len(resumed.pending), 0)
            self.assertGreater(resumed.updates, previous_updates)
            self.assertEqual(len(resumed.replay), 12)
            self.assertTrue(candidate.is_file())
            log_path = base / "runs" / args.run_id / "episodes.jsonl"
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertTrue(rows)
            self.assertIsNone(rows[-1]["official_time_seconds"])

    def test_driver_refuses_training_resume_from_inference_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "runs") as temp:
            base = Path(temp)
            path = base / "inference.pt"
            self.agent().save(path, include_replay=False)
            args = self.training_args(resume=str(path))
            with self.assertRaises(ValueError):
                train(args, base)

    def test_reject_nonfinite_states_and_rewards(self):
        agent = self.agent()
        with self.assertRaises(ValueError):
            agent.select_action(np.full(11, np.nan))
        with self.assertRaises(ValueError):
            agent.observe(np.zeros(11), 0, np.nan, np.zeros(11), False)


if __name__ == "__main__":
    unittest.main()
