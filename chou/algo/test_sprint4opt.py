"""Independent certificates for the four-edge sprint neighborhood."""
import hashlib
import time
import unittest
from pathlib import Path

from test_serpent_algo import game
from test_optimization import assert_certificate


class FourOptTests(unittest.TestCase):
    def setUp(self):
        self.policy = game.make_policy('bridge_uphill', time_limit_ms=None)
        self.state = game.Game(201).state()
        base = self.policy._normalized(self.state)
        self.cycle = self.policy._apply(base, ('reverse', 0, 209))
        assert_certificate(self, self.cycle, self.state)

    def test_all_bridges_preserve_geometry_body_and_exact_rank_in_each_block(self):
        options = list(self.policy._bridges(self.cycle, 110, 222, neutral=None))
        self.assertGreater(len(options), 0)
        for _, descriptor in options:
            _, first, second, third, last = descriptor
            self.assertTrue(0 <= first < second < third < last <= 222)
            candidate = self.policy._apply(self.cycle, descriptor)
            for apple_rank in (first + 1, second, second + 1, third, third + 1, last):
                state = self.state._replace(apple=self.cycle[apple_rank])
                index = assert_certificate(self, candidate, state)
                proposals = dict((move, rank) for rank, move in
                                 self.policy._bridges(self.cycle, apple_rank, 222, neutral=None))
                self.assertEqual(index[state.apple], proposals[descriptor])
            self.assertEqual(candidate[223:], self.cycle[223:])

    def test_modes_partition_all_moves_and_allow_virtual_uphill_only(self):
        for rank in (5, 110, 197, 221, 222):
            all_moves = dict((move, new_rank) for new_rank, move in
                             self.policy._bridges(self.cycle, rank, 222, neutral=None))
            improving = dict((move, new_rank) for new_rank, move in
                              self.policy._bridges(self.cycle, rank, 222, neutral=False))
            neutral = dict((move, new_rank) for new_rank, move in
                           self.policy._bridges(self.cycle, rank, 222, neutral=True))
            self.assertEqual(improving, {move: value for move, value in all_moves.items() if value < rank})
            self.assertEqual(neutral, {move: value for move, value in all_moves.items() if value == rank})
        moves = list(self.policy._moves(self.cycle, 110, 222, neutral=None))
        self.assertTrue(any(rank > 110 for rank, _ in moves))
        self.assertTrue(any(rank < 110 for rank, _ in moves))
        self.assertTrue(any(rank == 110 for rank, _ in moves))

    def test_body_limits_and_pending_growth_survive_commit(self):
        for length in (3, 4, 10):
            body = (self.cycle[0],) + tuple(reversed(self.cycle[226-length:]))
            self.assertEqual(len(body), length)
            state = self.state._replace(body=body, apple=self.cycle[100], grow_pending=True, steps=10)
            options = list(self.policy._bridges(self.cycle, 100, 225-length, neutral=None))
            self.assertTrue(options)
            for _, descriptor in options:
                candidate = self.policy._apply(self.cycle, descriptor)
                assert_certificate(self, candidate, state)
            candidate = self.policy._apply(self.cycle, options[0][1])
            self.policy.shield.cycle = list(candidate)
            self.policy.shield.index = {cell: rank for rank, cell in enumerate(candidate)}
            self.policy.shield.validate(state)
            action = min(self.policy.shield._options(state), key=lambda option: option[2])[0]
            self.policy.shield.commit(state, action)
            future, ate = game.simulate_known_apple(state, action)
            self.assertEqual(len(future.body), length + 1)
            self.assertEqual(future.body[-1], state.body[-1])
            assert_certificate(self, tuple(self.policy.shield.cycle), future)

    def test_uphill_search_returns_safe_non_worsening_best_at_expiry(self):
        state = self.state._replace(apple=self.cycle[110], steps=10)
        self.policy.shield.cycle = list(self.cycle)
        self.policy.shield.index = {cell: rank for rank, cell in enumerate(self.cycle)}
        self.policy.shield.validate(state)
        candidate = self.policy._search(self.cycle, state, deadline=time.perf_counter()-1)
        self.assertEqual(candidate, self.cycle)
        self.policy.search_budget = 12
        candidate = self.policy._search(self.cycle, state)
        index = assert_certificate(self, candidate, state)
        self.assertLessEqual(index[state.apple], 110)
        self.assertLessEqual(self.policy.last_search_stats['candidates'], 12)

    def test_zero_deadline_safe_progress_and_reset(self):
        for name in ('bridge128', 'bridge_uphill'):
            policy = game.make_policy(name, time_limit_ms=0)
            engine = game.Game(202)
            for _ in range(20):
                state = engine.state()
                before = policy.shield.distance(state.body[0], state.apple)
                action = policy.select_action(state)
                event = engine.step(action)
                self.assertFalse(event['collision'])
                assert_certificate(self, tuple(policy.shield.cycle), engine.state())
                if not event['ate_apple']:
                    self.assertLess(policy.shield.distance(engine.state().body[0], state.apple), before)
            engine.reset(seed=203)
            fresh = game.make_policy(name, time_limit_ms=0)
            self.assertEqual(policy.select_action(engine.state()), fresh.select_action(engine.state()))
            self.assertEqual(policy.shield.cycle, fresh.shield.cycle)

    def test_consecutive_apples_deferred_growth_223_and_restart(self):
        for name in ('bridge128', 'bridge_uphill'):
            policy = game.make_policy(name, time_limit_ms=20)
            engine = game.Game(201)
            for score in range(1, 224):
                head = tuple(engine.snake.head_pos)
                engine.apple.position = policy.shield.cycle[(policy.shield.index[head] + 1) % 225]
                action = policy.select_action(engine.state())
                event = engine.step(action)
                self.assertTrue(event['ate_apple'])
                self.assertFalse(event['collision'])
                self.assertEqual(event['grew'], score > 1)
                self.assertEqual(len(engine.snake.body), score + 2)
                assert_certificate(self, tuple(policy.shield.cycle), engine.state())
            self.assertTrue(engine.completed)
            self.assertEqual(engine.snake.score, 223)
            self.assertEqual(engine.steps, 223)
            self.assertIsNone(engine.apple.position)
            engine.reset(seed=204)
            fresh = game.make_policy(name, time_limit_ms=0)
            policy.time_limit_ms = 0
            self.assertEqual(policy.select_action(engine.state()), fresh.select_action(engine.state()))


if __name__ == '__main__':
    source = Path(__file__).with_name('serpent-algo.py')
    print('source_sha256=' + hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
    unittest.main()
