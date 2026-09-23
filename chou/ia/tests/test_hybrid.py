import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from snake_rl.env import Env
from snake_rl.hybrid_agent import HybridAgent, HybridConfig, action_features
from snake_rl.cycle_shield import allowed_actions

class HybridTests(unittest.TestCase):
    def test_features_pure_and_numerically_finite(self):
        env = Env(seed=27)
        before = (list(env.body), env.rng.getstate(), env.apple, env.steps)
        features, mask = action_features(env)
        self.assertEqual(features.shape, (4,20))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(np.flatnonzero(mask).tolist(), list(allowed_actions(env)))
        self.assertEqual(before, (list(env.body), env.rng.getstate(), env.apple, env.steps))

    def test_numpy_inference_equals_torch_and_action_is_masked(self):
        agent = HybridAgent()
        env = Env(seed=38)
        for _ in range(40):
            features, mask = agent.encode(env)
            with torch.no_grad():
                expected = agent.online(torch.from_numpy(features)).numpy()
            np.testing.assert_allclose(agent.values(features), expected, atol=1e-6)
            action = agent.select_action(env)
            self.assertTrue(mask[action])
            env.step(action)

    def test_td_update_changes_weights_and_roundtrip_preserves_choices(self):
        agent = HybridAgent(HybridConfig(batch_size=8, capacity=64))
        env = Env(seed=68)
        for _ in range(32):
            old, mask = agent.encode(env)
            action = agent.select_features(old, mask)
            agent.commit(env, action)
            _, done, _ = env.step(action)
            successor, next_mask = agent.encode(env)
            agent.observe(old, action, 1., successor, next_mask, done)
        before = [p.detach().clone() for p in agent.online.parameters()]
        loss = agent.train_step()
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.online.parameters())))
        self.assertTrue(all(p.grad is None for p in agent.target.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            agent.save(path)
            restored = HybridAgent.load(path)
            self.assertEqual(restored.env_steps, 32)
            self.assertEqual(restored.updates, 1)
            env.reset()
            self.assertEqual(agent.select_action(env), restored.select_action(env))
            features, _ = action_features(env)
            np.testing.assert_array_equal(agent.values(features), restored.values(features))

    def test_network_weights_control_choice_within_identical_safe_set(self):
        agent = HybridAgent()
        for seed in range(100):
            env = Env(seed=seed)
            features, mask = agent.encode(env)
            allowed = np.flatnonzero(mask)
            if len(allowed) >= 2:
                break
        self.assertGreaterEqual(len(allowed), 2)
        choices = []
        for preferred in allowed[:2]:
            with torch.no_grad():
                for parameter in agent.online.parameters():
                    parameter.zero_()
                agent.online.layers[0].weight[0, 15+preferred] = 1
                agent.online.layers[2].weight[0, 0] = 1
                agent.online.layers[4].weight[0, 0] = 1
            agent.refresh_weights()
            choices.append(agent.select_features(features, mask))
        self.assertEqual(choices, allowed[:2].tolist())
        self.assertNotEqual(choices[0], choices[1])

    def test_reset_clears_private_cycle_between_identical_episodes(self):
        agent = HybridAgent()
        env = Env(seed=16)
        traces = []
        for episode in range(2):
            env.reset(seed=16)
            trace = []
            for step in range(300):
                action = agent.select_action(env)
                self.assertTrue(agent.shield.is_aligned(env))
                env.step(action)
                trace.append((action, env.body[0], env.score))
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_missing_and_untrained_checkpoints_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.pt'
            with self.assertRaises(FileNotFoundError):
                HybridAgent.load(path)
            HybridAgent().save(path)
            with self.assertRaisesRegex(ValueError, 'not undergone reinforcement learning'):
                HybridAgent.load(path)

if __name__ == '__main__':
    unittest.main()
