"""Independent safety checks for the bounded Hamiltonian search extensions."""

import random
import time
import unittest
from unittest.mock import patch

from test_serpent_algo import game


def assert_certificate(test, cycle, state):
    """Check the actual graph and body, without calling policy validation."""
    test.assertEqual(len(cycle), 225)
    test.assertEqual(set(cycle), {(x, y) for x in range(15) for y in range(15)})
    index = {cell: rank for rank, cell in enumerate(cycle)}
    for first, second in zip(cycle, cycle[1:] + cycle[:1]):
        dx = (second[0] - first[0]) % 15
        dy = (second[1] - first[1]) % 15
        test.assertIn((dx, dy), ((1, 0), (14, 0), (0, 1), (0, 14)))
    test.assertEqual(len(set(state.body)), len(state.body))
    for offset, cell in enumerate(state.body):
        test.assertEqual(index[cell], (index[state.body[0]] - offset) % 225)
    return index


class SearchNeighborhoodTests(unittest.TestCase):
    def setUp(self):
        self.state = game.Game(seed=101).state()
        self.policy = game.AdvancedCyclePolicy(
            search_budget=64, beam_width=4, apple_neutral=True, or_opt=True
        )
        self.cycle = self.policy._normalized(self.state)

    def test_normalization_identifies_rotations_of_one_oriented_cycle(self):
        self.assertEqual(self.cycle[0], self.state.body[0])
        for rotation in (1, 37, 224):
            self.policy.shield.cycle = list(self.cycle[rotation:] + self.cycle[:rotation])
            self.policy.shield.index = {
                cell: rank for rank, cell in enumerate(self.policy.shield.cycle)
            }
            self.assertEqual(self.policy._normalized(self.state), self.cycle)
        assert_certificate(self, self.cycle, self.state)

    def test_neutral_inversion_can_contain_the_apple(self):
        # Both new edges cross a vertical wrap. The apple is the midpoint,
        # so this reversal changes the graph without changing its rank.
        apple_rank = 105
        state = self.state._replace(apple=self.cycle[apple_rank])
        witness = ("reverse", 0, 209)
        options = list(self.policy._two_opt(self.cycle, apple_rank, 222, neutral=True))
        self.assertIn((apple_rank, witness), options)
        candidate = self.policy._apply(self.cycle, witness)
        self.assertNotEqual(candidate, self.cycle)
        index = assert_certificate(self, candidate, state)
        self.assertEqual(index[state.apple], apple_rank)
        self.assertEqual(candidate[-2:], self.cycle[-2:])

    def test_disabling_apple_neutral_preserves_old_neutral_neighborhood(self):
        self.policy.apple_neutral = False
        for rank, descriptor in self.policy._two_opt(self.cycle, 105, 222, neutral=True):
            self.assertEqual(rank, 105)
            _, first, last = descriptor
            self.assertFalse(first < 105 <= last)

    def test_or_opt_relocation_preserves_all_body_edges_and_three_bridges(self):
        base = self.policy._apply(self.cycle, ("reverse", 0, 209))
        state = self.state._replace(apple=base[210])
        witness = ("relocate", 209, 210, 0, False)
        options = list(self.policy._or_opt(base, 210, 222))
        self.assertIn((2, witness), options)
        candidate = self.policy._apply(base, witness)
        index = assert_certificate(self, candidate, state)
        self.assertEqual(index[state.apple], 2)
        self.assertEqual(candidate[-2:], base[-2:])
        # Check each proposal independently, including reversed blocks.
        for rank, descriptor in options:
            with self.subTest(descriptor=descriptor):
                proposal = self.policy._apply(base, descriptor)
                proposal_index = assert_certificate(self, proposal, state)
                self.assertEqual(proposal_index[state.apple], rank)
                self.assertLess(rank, 210)

    def test_expired_search_preserves_a_safe_non_worsening_continuation(self):
        before_cycle = list(self.policy.shield.cycle)
        before_index = dict(self.policy.shield.index)
        candidate = self.policy._search(self.cycle, self.state, deadline=time.perf_counter() - 1)
        index = assert_certificate(self, candidate, self.state)
        self.assertLessEqual(index[self.state.apple], self.cycle.index(self.state.apple))
        self.assertEqual(self.policy.shield.cycle, before_cycle)
        self.assertEqual(self.policy.shield.index, before_index)
        self.assertTrue(self.policy.last_search_stats["timed_out"])

    def test_research_budget_is_bounded_and_best_initial_cycle_is_retained(self):
        for budget in (0, 5, 64, 96):
            with self.subTest(budget=budget):
                policy = game.AdvancedCyclePolicy(
                    search_budget=budget, beam_width=4, apple_neutral=True, or_opt=True
                )
                cycle = policy._normalized(self.state)
                candidate = policy._search(cycle, self.state)
                index = assert_certificate(self, candidate, self.state)
                self.assertLessEqual(index[self.state.apple], cycle.index(self.state.apple))
                self.assertLessEqual(policy.last_search_stats["candidates"], budget)

    def test_distinct_exploratory_cycles_are_not_revisited_and_budget_exceeds_64(self):
        policy = game.AdvancedCyclePolicy(
            search_budget=96, beam_width=4, apple_neutral=True, or_opt=True
        )
        state = self.state._replace(apple=self.cycle[110])
        with patch.object(policy, "_improve", wraps=policy._improve) as descent:
            candidate = policy._search(self.cycle, state)
        explored = [tuple(call.args[0]) for call in descent.call_args_list]
        self.assertEqual(len(explored), len(set(explored)))
        self.assertEqual(policy.last_search_stats["candidates"], 96)
        assert_certificate(self, candidate, state)

    def test_relocation_rank_accounting_on_both_sides_of_a_block(self):
        base = self.policy._apply(self.cycle, ("reverse", 0, 209))
        relocated = self.policy._apply(base, ("relocate", 209, 210, 0, False))
        self.policy.or_lengths = (2, 4, 6, 13)
        proposals_checked = 0
        for cycle in (base, relocated):
            for apple_rank in (1, 2, 15, 105, 209, 210, 222):
                state = self.state._replace(apple=cycle[apple_rank])
                for neutral in (False, True):
                    for rank, descriptor in self.policy._or_opt(cycle, apple_rank, 222, neutral):
                        candidate = self.policy._apply(cycle, descriptor)
                        index = assert_certificate(self, candidate, state)
                        self.assertEqual(index[state.apple], rank)
                        if neutral:
                            self.assertEqual(rank, apple_rank)
                        else:
                            self.assertLess(rank, apple_rank)
                        proposals_checked += 1
        self.assertGreater(proposals_checked, 5)

    def test_all_block_lengths_match_an_independent_exhaustive_or_opt_oracle(self):
        policy = game.AdvancedCyclePolicy(or_opt=True, or_lengths=None)
        cycle = policy._apply(self.cycle, ("reverse", 0, 209))
        apple_rank, free_end = 110, 222
        state = self.state._replace(apple=cycle[apple_rank])

        def adjacent(first, second):
            displacement = ((second[0] - first[0]) % 15,
                            (second[1] - first[1]) % 15)
            return displacement in ((1, 0), (14, 0), (0, 1), (0, 14))

        expected = set()
        for start in range(1, free_end + 1):
            for end in range(start, free_end + 1):
                if not adjacent(cycle[start - 1], cycle[end + 1]):
                    continue
                for after in range(free_end + 1):
                    if start - 1 <= after <= end:
                        continue
                    for reverse in (False, True) if end > start else (False,):
                        block = cycle[start:end + 1]
                        if reverse:
                            block = block[::-1]
                        if not adjacent(cycle[after], block[0]):
                            continue
                        if not adjacent(block[-1], cycle[after + 1]):
                            continue
                        rest = cycle[:start] + cycle[end + 1:]
                        insertion = rest.index(cycle[after]) + 1
                        candidate = rest[:insertion] + block + rest[insertion:]
                        rank = candidate.index(state.apple)
                        if rank <= apple_rank:
                            expected.add((rank, ("relocate", start, end, after, reverse)))
        actual = {
            entry for neutral in (False, True)
            for entry in policy._or_opt(cycle, apple_rank, free_end, neutral)
        }
        self.assertEqual(actual, expected)
        self.assertTrue(any(end - start + 1 > 3
                            for _, (_, start, end, _, _) in actual))
        for rank, descriptor in actual:
            candidate = policy._apply(cycle, descriptor)
            index = assert_certificate(self, candidate, state)
            self.assertEqual(index[state.apple], rank)

    def test_three_opt_certifies_bridges_body_and_apple_in_each_block(self):
        policy = game.AdvancedCyclePolicy(three_opt=True)
        witnesses = 0
        for inversion_start, free_end in ((7, 45), (67, 110), (142, 222)):
            cycle = policy._apply(self.cycle, ("reverse", inversion_start, inversion_start + 16))
            body = (cycle[0],) + tuple(reversed(cycle[free_end + 1:]))
            state = self.state._replace(body=body, apple=cycle[free_end])
            assert_certificate(self, cycle, state)
            descriptors = {
                descriptor for neutral in (False, True)
                for _, descriptor in policy._three_opt(cycle, free_end, free_end, neutral)
            }
            self.assertTrue(descriptors)
            for descriptor in descriptors:
                _, first, middle, last = descriptor
                candidate = policy._apply(cycle, descriptor)
                self.assertEqual(candidate[:first + 1], cycle[:first + 1])
                self.assertEqual(candidate[last + 1:], cycle[last + 1:])
                self.assertEqual(candidate[first + 1:middle + 1], cycle[first + 1:middle + 1][::-1])
                self.assertEqual(candidate[middle + 1:last + 1], cycle[middle + 1:last + 1][::-1])
                index = assert_certificate(self, candidate, state)
                # Verify the three physical replacement edges independently.
                for source, destination in ((first, middle), (first + 1, last),
                                            (middle + 1, last + 1)):
                    dx = (cycle[destination][0] - cycle[source][0]) % 15
                    dy = (cycle[destination][1] - cycle[source][1]) % 15
                    self.assertIn((dx, dy), ((1, 0), (14, 0), (0, 1), (0, 14)))
                for apple_rank in {1, first + 1, middle - 1, middle,
                                   middle + 1, last - 1, last, free_end}:
                    rank = index[cycle[apple_rank]]
                    for neutral in (False, True):
                        actual = dict((move, result) for result, move in policy._three_opt(
                            cycle, apple_rank, free_end, neutral))
                        admissible = rank == apple_rank if neutral else rank < apple_rank
                        self.assertEqual(descriptor in actual, admissible)
                        if admissible:
                            self.assertEqual(actual[descriptor], rank)
                witnesses += 1
        self.assertGreaterEqual(witnesses, 3)


class LookaheadTransitionTests(unittest.TestCase):
    def test_multiple_roots_are_deduplicated_by_normalized_cycle(self):
        state = game.Game(seed=109).state()
        shield = game.RewiredCycleShield()
        cycle = tuple(shield.cycle)
        kwargs = dict(depth=4, width=4, max_nodes=64, optimizer=None)
        ordinary = game.choose_lookahead(shield, state, **kwargs)
        duplicates = game.choose_lookahead(
            shield, state, **kwargs,
            initial_cycles=(cycle, cycle[19:] + cycle[:19], cycle[124:] + cycle[:124]),
        )
        self.assertEqual(ordinary, duplicates)
        self.assertEqual(tuple(shield.cycle), cycle)

    def test_distinct_root_is_explored_and_returned_first_move_is_certified(self):
        state = game.Game(seed=109).state()
        policy = game.AdvancedCyclePolicy()
        cycle = policy._normalized(state)
        state = state._replace(apple=cycle[105])
        alternative = policy._apply(cycle, ("reverse", 0, 209))
        before = list(policy.shield.cycle), dict(policy.shield.index)
        kwargs = dict(depth=4, width=4, max_nodes=64, optimizer=None)
        _, _, ordinary_stats = game.choose_lookahead(policy.shield, state, **kwargs)
        chosen_cycle, action, stats = game.choose_lookahead(
            policy.shield, state, **kwargs, initial_cycles=(alternative,)
        )
        self.assertGreater(stats["distinct_states"], ordinary_stats["distinct_states"])
        self.assertLessEqual(stats["nodes"], 64)
        self.assertEqual((policy.shield.cycle, policy.shield.index), before)
        assert_certificate(self, chosen_cycle, state)
        predicted, _ = game.simulate_known_apple(state, action)
        assert_certificate(self, chosen_cycle, predicted)

    def test_invalid_lookahead_root_is_rejected_without_mutating_fallback(self):
        state = game.Game(seed=110).state()
        shield = game.RewiredCycleShield()
        before_cycle, before_index = list(shield.cycle), dict(shield.index)
        malformed = list(shield.cycle)
        malformed[1] = malformed[0]
        with self.assertRaises(ValueError):
            game.choose_lookahead(shield, state, initial_cycles=(malformed,))
        self.assertEqual(shield.cycle, before_cycle)
        self.assertEqual(shield.index, before_index)
        assert_certificate(self, tuple(shield.cycle), state)

    def test_tail_cell_is_only_available_without_pending_growth(self):
        initial = game.Game(seed=102).state()
        state = initial._replace(
            body=((1, 1), (1, 2), (2, 2), (2, 1)),
            direction=game.DIRECTIONS.index(game.RIGHT), apple=(0, 0),
        )
        predicted, consumed = game.simulate_known_apple(state, state.direction)
        self.assertFalse(consumed)
        self.assertEqual(predicted.body, ((2, 1), (1, 1), (1, 2), (2, 2)))
        with self.assertRaises(ValueError):
            game.simulate_known_apple(state._replace(grow_pending=True), state.direction)

    def test_simulation_matches_engine_and_stops_at_the_known_apple(self):
        for pending in (False, True):
            for consumes in (False, True):
                with self.subTest(pending=pending, consumes=consumes):
                    engine = game.Game(seed=102)
                    engine.snake.grow_pending = pending
                    engine.apple.position = (4, 7) if consumes else (5, 7)
                    before = engine.state()
                    action = game.DIRECTIONS.index(game.RIGHT)
                    with patch.object(random.Random, "choice", side_effect=AssertionError("RNG accessed")):
                        predicted, ate = game.simulate_known_apple(before, action)
                    self.assertEqual(engine.state(), before)
                    event = engine.step(action)
                    actual = engine.state()
                    for name in ("body", "direction", "grow_pending", "score", "steps"):
                        self.assertEqual(getattr(predicted, name), getattr(actual, name))
                    self.assertEqual(ate, event["ate_apple"])
                    self.assertEqual(ate, consumes)
                    self.assertEqual(predicted.terminated, consumes)
                    self.assertEqual(predicted.apple, None if consumes else before.apple)

    def test_lookahead_does_not_mutate_shield_state_or_global_randomness(self):
        engine = game.Game(seed=103)
        shield = game.RewiredCycleShield()
        state = engine.state()
        before_cycle, before_index = list(shield.cycle), dict(shield.index)
        before_random = random.getstate()
        cycle, action, stats = game.choose_lookahead(
            shield, state, depth=6, width=3, max_nodes=72
        )
        self.assertEqual(shield.cycle, before_cycle)
        self.assertEqual(shield.index, before_index)
        self.assertEqual(engine.state(), state)
        self.assertEqual(random.getstate(), before_random)
        index = assert_certificate(self, cycle, state)
        self.assertLessEqual(stats["nodes"], 72)
        self.assertLessEqual(stats["best_estimate"], stats["initial_estimate"])
        dx, dy = game.DIRECTIONS[action]
        next_head = ((state.body[0][0] + dx) % 15, (state.body[0][1] + dy) % 15)
        self.assertEqual(index[next_head], (index[state.body[0]] + 1) % 225)
        next_state, ate = game.simulate_known_apple(state, action)
        assert_certificate(self, cycle, next_state)
        if not ate:
            remaining = (index[state.apple] - index[next_head]) % 225
            self.assertLess(remaining, shield.distance(state.body[0], state.apple))

    def test_expired_lookahead_returns_the_validated_fallback(self):
        state = game.Game(seed=104).state()
        shield = game.RewiredCycleShield()
        cycle, action, stats = game.choose_lookahead(
            shield, state, depth=8, width=4, max_nodes=72,
            deadline_ns=time.perf_counter_ns() - 1,
        )
        assert_certificate(self, cycle, state)
        predicted, _ = game.simulate_known_apple(state, action)
        assert_certificate(self, cycle, predicted)
        self.assertTrue(stats["expired"])


class AdvancedPolicyIntegrationTests(unittest.TestCase):
    def test_zero_time_budget_still_executes_a_safe_progressing_move(self):
        engine = game.Game(seed=108)
        policy = game.AdvancedCyclePolicy(
            search_budget=128, beam_width=8, apple_neutral=True,
            or_opt=True, or_lengths=None, adaptive=True,
            lookahead_depth=8, time_limit_ms=0,
        )
        for _ in range(10):
            state = engine.state()
            before = policy.shield.distance(state.body[0], state.apple)
            action = policy.select_action(state)
            assert_certificate(self, tuple(policy.shield.cycle), state)
            event = engine.step(action)
            self.assertFalse(event["collision"])
            assert_certificate(self, tuple(policy.shield.cycle), engine.state())
            if not event["ate_apple"]:
                self.assertLess(policy.shield.distance(engine.state().body[0], state.apple), before)

    def test_reproducible_actions_progress_and_restart_without_rng_access(self):
        first = game.Game(seed=105)
        second = game.Game(seed=105)
        config = dict(search_budget=64, beam_width=4, apple_neutral=True,
                      or_opt=True, adaptive=True)
        first_policy = game.AdvancedCyclePolicy(**config)
        second_policy = game.AdvancedCyclePolicy(**config)
        for _ in range(150):
            state = first.state()
            before = first_policy.shield.distance(state.body[0], state.apple)
            original_rng = first._rng.getstate()
            action = first_policy.select_action(state)
            self.assertEqual(first._rng.getstate(), original_rng)
            self.assertEqual(action, second_policy.select_action(second.state()))
            assert_certificate(self, tuple(first_policy.shield.cycle), state)
            event = first.step(action)
            self.assertEqual(event, second.step(action))
            self.assertEqual(first.state(), second.state())
            self.assertFalse(event["collision"])
            assert_certificate(self, tuple(first_policy.shield.cycle), first.state())
            if not event["ate_apple"]:
                self.assertLess(first_policy.shield.distance(first.state().body[0], state.apple), before)
        first.reset(seed=106)
        state = first.state()
        fresh = game.AdvancedCyclePolicy(**config)
        self.assertEqual(first_policy.select_action(state), fresh.select_action(state))
        self.assertEqual(first_policy.shield.cycle, fresh.shield.cycle)

    def test_all_consecutive_apples_keep_pending_growth_safe_until_victory(self):
        engine = game.Game(seed=107)
        policy = game.AdvancedCyclePolicy(
            search_budget=64, beam_width=4, apple_neutral=True, or_opt=True, adaptive=True
        )
        for score in range(1, 224):
            head = tuple(engine.snake.head_pos)
            successor = policy.shield.cycle[(policy.shield.index[head] + 1) % 225]
            engine.apple.position = successor
            state = engine.state()
            action = policy.select_action(state)
            assert_certificate(self, tuple(policy.shield.cycle), state)
            event = engine.step(action)
            self.assertTrue(event["ate_apple"])
            self.assertFalse(event["collision"])
            self.assertEqual(event["grew"], score > 1)
            self.assertEqual(engine.snake.score, score)
            self.assertEqual(len(engine.snake.body), score + 2)
            assert_certificate(self, tuple(policy.shield.cycle), engine.state())
        self.assertTrue(engine.completed)
        self.assertEqual(engine.steps, 223)
        self.assertIsNone(engine.apple.position)


if __name__ == "__main__":
    unittest.main()
