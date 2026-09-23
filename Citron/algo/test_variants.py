import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

spec = importlib.util.spec_from_file_location('variants_game', Path(__file__).with_name('snake-algo.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class VariantTests(unittest.TestCase):
    def assert_cycle(self, game):
        cells = game.cycle.cells
        self.assertEqual(len(set(cells)), 225)
        for a, b in zip(cells, cells[1:]+cells[:1]):
            self.assertIn(((b[0]-a[0])%15, (b[1]-a[1])%15),
                          ((1,0),(14,0),(0,1),(0,14)))

    def test_geometry_preserves_start_and_cycle(self):
        for offset in (1, 3, 5, 8, 11):
            game = m.Game(1, 'shortcut', turn_offset=offset, cutoff=90)
            self.assert_cycle(game)
            self.assertEqual(game.cycle.next((1,7)), (2,7))
            self.assertEqual(game.cycle.next((2,7)), (3,7))

    def test_lookahead_and_dynamic_complete(self):
        for algorithm in ('lookahead', 'dynamic'):
            game = m.Game(42, algorithm)
            for _ in range(51000):
                if game.done:
                    break
                game.step()
                if game.steps%200 == 0:
                    self.assert_cycle(game)
            self.assertEqual(game.reason, 'victory')

    def test_invalid_parameters(self):
        for kwargs in ({'cutoff':0}, {'turn_offset':14}):
            with self.assertRaises(ValueError):
                m.Game(1, 'shortcut', **kwargs)

    def test_export_keeps_geometry_and_cutoff(self):
        with tempfile.TemporaryDirectory() as folder:
            m.Recorder(Path(folder), 42, 'shortcut', cutoff=90, turn_offset=3)
            config = json.loads((Path(folder)/'config.json').read_text())
            self.assertEqual(config['cutoff'], 90)
            self.assertEqual(config['turn_offset'], 3)


if __name__ == '__main__':
    unittest.main()
