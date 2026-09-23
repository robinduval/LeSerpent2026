"""The checkpoint declares whether inference uses a safety filter."""
from pathlib import Path
import tempfile
import unittest

from snake_rl.agent import DQNAgent, DQNConfig
from snake_rl.env import Env
from snake_rl.policy_io import load_policy, policy_kind
from snake_rl.shielded_policy import ShieldedPolicy

BASE = Path(__file__).resolve().parents[1]


class PolicyDispatchTests(unittest.TestCase):
    def test_checkpoint_metadata_activates_only_the_declared_filter(self):
        with tempfile.TemporaryDirectory(dir=BASE/'runs') as directory:
            path = Path(directory)/'agent.pt'
            agent = DQNAgent(DQNConfig(hidden=8, capacity=8, batch_size=2, warmup=2))
            agent.save(path, metadata={'safety_filter': 'tail2'}, include_replay=False)
            loaded = load_policy(path)
            self.assertIsInstance(loaded, ShieldedPolicy)
            self.assertEqual(policy_kind(loaded), 'rl_with_safety_filter')
            self.assertIn(loaded.select_action(Env(seed=5)), range(4))
            self.assertIn('proposed', loaded.last_decision)
            agent.save(path, metadata={'safety_filter': 'unknown'}, include_replay=False)
            with self.assertRaisesRegex(ValueError, 'Unknown safety filter'):
                load_policy(path)
            agent.save(path, metadata={}, include_replay=False)
            self.assertIsInstance(load_policy(path), DQNAgent)
            self.assertEqual(policy_kind(load_policy(path)), 'pure_rl')
