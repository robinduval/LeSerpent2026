"""Benchmark aléatoire headless, sans limite artificielle ni apprentissage."""

import argparse
from collections import Counter
import hashlib
import json
import random
import statistics
import time

from . import rules
from .game import SnakeGame


def run_benchmark(episodes=2000, seed=123):
    if episodes < 1:
        raise ValueError("episodes doit être positif")
    rng = random.Random(seed)
    digest = hashlib.sha256()
    results = []
    crossings = 0
    longest_without_food = 0
    started = time.perf_counter()
    for index in range(episodes):
        game = SnakeGame(seed=seed + index, max_steps_without_food=None)
        digest.update(json.dumps(game.snapshot(), sort_keys=True).encode())
        while not game.done:
            action = rng.choice([i for i, legal in enumerate(game.legal_action_mask()) if legal])
            dx, dy = rules.ACTIONS[action]
            x, y = game.head
            crossings += int(not (0 <= x + dx < rules.GRID_SIZE
                                  and 0 <= y + dy < rules.GRID_SIZE))
            longest_without_food = max(longest_without_food, game.steps_since_food + 1)
            result = game.step(action)
            digest.update(json.dumps(game.snapshot(action=action, reward=result.reward),
                                     sort_keys=True).encode())
        results.append({"seed": seed + index, "score": game.score, "steps": game.steps,
                        "cause": result.info["cause"], "truncated": result.truncated})
    elapsed = time.perf_counter() - started
    steps = sorted(row["steps"] for row in results)
    scores = [row["score"] for row in results]
    causes = dict(Counter(row["cause"] for row in results))

    def percentile(q):
        position = q / 100 * (len(steps) - 1)
        low = int(position)
        high = min(low + 1, len(steps) - 1)
        return steps[low] + (steps[high] - steps[low]) * (position - low)

    return {
        "ruleset": rules.RULESET, "policy": "uniform_over_non_reverse_actions",
        "seed": seed, "episodes": episodes, "max_steps_without_food": None,
        "termination_counts": causes, "wall_deaths": causes.get("wall", 0),
        "truncations": sum(row["truncated"] for row in results),
        "mean_score": statistics.fmean(scores), "record": max(scores),
        "total_steps": sum(steps), "mean_steps": statistics.fmean(steps),
        "median_steps": statistics.median(steps), "p90_steps": percentile(90),
        "p99_steps": percentile(99), "max_steps": max(steps),
        "longest_without_food": longest_without_food,
        "episodes_over_1000_steps": sum(step > 1000 for step in steps),
        "episodes_over_10000_steps": sum(step > 10000 for step in steps),
        "boundary_crossings": crossings, "trajectory_sha256": digest.hexdigest(),
        "elapsed_seconds": elapsed, "steps_per_second": sum(steps) / elapsed,
        "episode_results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", help="rapport JSON optionnel")
    args = parser.parse_args(argv)
    result = run_benchmark(args.episodes, args.seed)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "episode_results"}, indent=2))
    return result


if __name__ == "__main__":
    main()
