import importlib.util
from pathlib import Path
import unittest


class SelectionTests(unittest.TestCase):
    def test_selection_rejects_fast_collision(self):
        path = Path(__file__).with_name('compare_methods.py')
        self.assertTrue(path.exists())
        spec = importlib.util.spec_from_file_location('compare', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        safe = dict(summary=dict(collisions=0, truncated=0, score_total=22300, victories=100, total_steps=600000))
        fast_failure = dict(summary=dict(collisions=1, truncated=0, score_total=22299, victories=99, total_steps=1000))
        self.assertGreater(m.ranking(safe), m.ranking(fast_failure))
        faster = dict(summary=dict(collisions=0, truncated=0, score_total=22300, victories=100, total_steps=500000))
        self.assertGreater(m.ranking(faster), m.ranking(safe))
