"""Official individual ranking and explicitly non-official model validation."""
from __future__ import annotations

from collections import defaultdict
import math
import statistics


def individual_key(result):
    """No rounding, score/time ratio, weighted sum, or score tolerance."""
    elapsed = result['official_time_seconds']
    if elapsed is None:
        raise ValueError('An accelerated training episode has no official time.')
    return -result['official_score'], elapsed


def summarize(results):
    if not results:
        raise ValueError('Empty evaluation sample')
    scores = [r['official_score'] for r in results]
    time_by_score = defaultdict(list)
    for row in results:
        if row['official_time_seconds'] is not None and row['termination_reason'] not in ('external_step_limit', 'user_quit'):
            time_by_score[str(row['official_score'])].append(row['official_time_seconds'])
    return {
        'episodes': len(results), 'score_mean': statistics.mean(scores),
        'score_median': statistics.median(scores), 'score_max': max(scores),
        'score_min': min(scores), 'score_std': statistics.pstdev(scores),
        'completed': sum(r['completed'] for r in results),
        'completion_rate': sum(r['completed'] for r in results) / len(results),
        'collision_rate': sum(r['termination_reason'] == 'self_collision' for r in results) / len(results),
        'external_cutoffs': sum(r['termination_reason'] == 'external_step_limit' for r in results),
        'target_223_rate': sum(r['official_score'] == 223 for r in results) / len(results),
        'terminal_time_mean_by_exact_score': {s: statistics.mean(ts) for s, ts in time_by_score.items()},
        'terminal_time_count_by_exact_score': {s: len(ts) for s, ts in time_by_score.items()},
    }


def model_selection_key(results):
    """Local validation convention, NOT an invented official multi-game rule.

    Compare entire score distributions (mean, median, lower quartile, completion).
    Only if those are identical, and the multiset of exact scores is identical,
    may the selector below compare times of naturally ended episodes.
    """
    scores = sorted(r['official_score'] for r in results)
    return (statistics.mean(scores), statistics.median(scores),
            scores[(len(scores)-1)//4], sum(r['completed'] for r in results))


def select_candidate(candidates):
    """Same seeds/budgets required; never choose by one lucky maximum."""
    if not candidates:
        raise ValueError('No candidates')
    signatures = {tuple((r['seed'], r['evaluation_step_budget']) for r in rows)
                  for rows in candidates.values()}
    if len(signatures) != 1:
        raise ValueError('Candidates need identical seeds and evaluation budgets')
    best_key = max(model_selection_key(rows) for rows in candidates.values())
    ties = [name for name, rows in candidates.items() if model_selection_key(rows) == best_key]
    if len(ties) == 1:
        return ties[0]
    score_sets = {tuple(sorted(r['official_score'] for r in candidates[name])) for name in ties}
    if len(score_sets) == 1 and all(r['termination_reason'] not in ('external_step_limit', 'user_quit') for name in ties for r in candidates[name]):
        return min(ties, key=lambda name: statistics.mean(r['official_time_seconds'] for r in candidates[name]))
    return sorted(ties)[0]  # Stable label tie-break; explicitly no speed claim.
