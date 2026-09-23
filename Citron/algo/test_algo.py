import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
PATH = Path(__file__).with_name('snake-algo.py')


class AlgoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PATH.exists():
            spec = importlib.util.spec_from_file_location('algo', PATH)
            cls.m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.m)

    def setUp(self):
        self.assertTrue(PATH.exists(), 'Le programme algorithmique doit exister')

    def test_cycle_covers_torus_and_initial_body(self):
        cycle = self.m.Cycle()
        self.assertEqual(len(set(cycle.cells)), 225)
        for a, b in zip(cycle.cells, cycle.cells[1:] + cycle.cells[:1]):
            dx, dy = (b[0]-a[0]) % 15, (b[1]-a[1]) % 15
            self.assertIn((dx, dy), [(1, 0), (14, 0), (0, 1), (0, 14)])
        self.assertEqual(cycle.next((1, 7)), (2, 7))
        self.assertEqual(cycle.next((2, 7)), (3, 7))

    def test_growth_is_delayed(self):
        game = self.m.Game(42)
        game.apple.position = (4, 7)
        game.step()
        self.assertEqual(game.snake.score, 1)
        self.assertEqual(len(game.snake.body), 3)
        self.assertTrue(game.snake.grow_pending)
        game.step()
        self.assertEqual(len(game.snake.body), 4)

    def test_bfs_shortcut_and_cycle_order(self):
        game = self.m.Game(42, algorithm='bfs-safe')
        game.apple.position = (3, 8)
        game.step()
        self.assertEqual(game.snake.score, 1)
        self.assertEqual(game.snake.head_pos, [3, 8])
        for _ in range(20000):
            if game.done:
                break
            game.step()
            ranks = [game.cycle.distance(game.snake.body[-1], cell)
                     for cell in reversed(game.snake.body)]
            self.assertEqual(ranks, sorted(set(ranks)))
        self.assertEqual(game.reason, 'victory')

    def test_optimized_preserves_order_and_completes(self):
        for seed in range(1000, 1100):
            game = self.m.Game(seed, algorithm='shortcut')
            for _ in range(51000):
                if game.done:
                    break
                game.step()
                if game.steps % 100 == 0:
                    ranks = [game.cycle.distance(game.snake.body[-1], cell)
                             for cell in reversed(game.snake.body)]
                    self.assertEqual(ranks, sorted(set(ranks)))
            self.assertEqual(game.reason, 'victory')

    def test_optimized_uses_left_and_right(self):
        game = self.m.Game(42, algorithm='shortcut')
        cycle = game.cycle.cells
        edges = {((b[0]-a[0]) % 15, (b[1]-a[1]) % 15)
                 for a, b in zip(cycle, cycle[1:]+cycle[:1])}
        self.assertEqual(edges, {(1,0), (14,0), (0,1), (0,14)})
        self.assertEqual(len(set(cycle)), 225)
        directions = set()
        for _ in range(51000):
            if game.done:
                break
            game.step()
            directions.add(game.snake.direction)
        self.assertIn((-1, 0), directions)
        self.assertIn((1, 0), directions)
        self.assertEqual(game.reason, 'victory')

    def test_complete_games_without_collision(self):
        for seed in range(5):
            game = self.m.Game(seed)
            for _ in range(51000):
                game.step()
                if game.done:
                    break
            self.assertEqual(game.reason, 'victory', (seed, game.steps))
            self.assertEqual(len(game.snake.body), 225)
            self.assertEqual(game.snake.score, 223)

    def test_metrics_real_time_and_export(self):
        game = self.m.Game(42)
        game.apple.position = (4, 7)
        game.step()
        data = game.metrics(2.0)
        self.assertEqual(data['score_per_real_second'], 0.5)
        self.assertEqual(data['real_seconds'], 2.0)
        self.assertEqual(data['steps'], 1)
        with tempfile.TemporaryDirectory() as folder:
            log = self.m.Recorder(Path(folder), 42)
            log.write(data)
            self.assertEqual(json.loads((Path(folder)/'metrics.json').read_text())['score'], 1)
            self.assertEqual(len((Path(folder)/'steps.csv').read_text().splitlines()), 2)

    def test_render_populates_surface(self):
        self.m.pygame.init()
        view = self.m.View()
        game = self.m.Game(42)
        view.draw(game, game.metrics(0), [(0, 0)])
        self.assertNotEqual(view.screen.get_at((10, 10)), view.screen.get_at((35, 135)))
        self.m.pygame.quit()

    def test_board_matches_original_render_pixel_for_pixel(self):
        m = self.m
        m.pygame.init()
        view = m.View()
        game = m.Game(42)
        view.draw(game, game.metrics(0), [(0, 0)])
        expected = m.pygame.Surface((m.base.SCREEN_WIDTH, m.base.SCREEN_HEIGHT))
        expected.fill(m.base.GRIS_FOND)
        m.pygame.draw.rect(expected, m.base.NOIR, (0, m.base.SCORE_PANEL_HEIGHT,
                                                 m.base.SCREEN_WIDTH, m.base.SCREEN_WIDTH))
        m.base.draw_grid(expected)
        game.apple.draw(expected)
        game.snake.draw(expected)
        rect = (0, m.base.SCORE_PANEL_HEIGHT, m.base.SCREEN_WIDTH, m.base.SCREEN_WIDTH)
        self.assertEqual(m.pygame.image.tostring(view.screen.subsurface(rect), 'RGB'),
                         m.pygame.image.tostring(expected.subsurface(rect), 'RGB'))
        m.pygame.quit()


if __name__ == '__main__':
    unittest.main()
