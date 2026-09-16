import unittest
from snake_rl.metrics import individual_key, select_candidate


class RankingTests(unittest.TestCase):
    def test_official_lexicographic_examples(self):
        for a, b, first in [((300,200),(250,50),'A'), ((300,200),(300,100),'B'), ((301,200),(300,100),'A')]:
            rows = [dict(name=name,official_score=s,official_time_seconds=t) for name,(s,t) in [('A',a),('B',b)]]
            self.assertEqual(sorted(rows,key=individual_key)[0]['name'],first)

    def test_accelerated_time_not_official(self):
        with self.assertRaises(ValueError):
            individual_key(dict(official_score=10,official_time_seconds=None))

    def test_one_lucky_game_cannot_win_by_maximum(self):
        def rows(scores):
            return [dict(seed=i,evaluation_step_budget=100,official_score=s,official_time_seconds=20,
                         completed=False,termination_reason='self_collision') for i,s in enumerate(scores)]
        self.assertEqual(select_candidate({'steady':rows([40,40,40]),'lucky':rows([0,0,100])}),'steady')

    def test_mismatched_seed_sets_rejected(self):
        with self.assertRaises(ValueError):
            select_candidate({'a':[dict(seed=1,evaluation_step_budget=1)],'b':[dict(seed=2,evaluation_step_budget=1)]})
