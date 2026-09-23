import importlib.util
from pathlib import Path
import unittest


class BenchmarkTests(unittest.TestCase):
    def test_batch_reproducibility_and_units(self):
        path = Path(__file__).with_name('benchmark.py')
        self.assertTrue(path.exists(), 'Benchmark absent')
        spec = importlib.util.spec_from_file_location('bench', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        rows = m.run_batch([1000, 1001])
        repeat = m.run_batch([1000, 1001])
        self.assertEqual([r['steps'] for r in rows], [r['steps'] for r in repeat])
        for r in rows:
            self.assertEqual(r['score'], 223)
            self.assertEqual(r['status'], 'victory')
            self.assertEqual(r['theoretical_seconds_5hz'], r['steps']/5)
            self.assertGreater(r['steps_to_10'], 0)
        report = m.summarize(rows, 2.0)
        self.assertEqual(report['score_total'], 446)
        self.assertEqual(report['victories'], 2)
        self.assertEqual(report['batch_compute_wall_seconds'], 2.0)
        self.assertEqual(report['score_min'], 223)


if __name__ == '__main__':
    unittest.main()
