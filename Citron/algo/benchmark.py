"""Simulations sans rendu : temps de calcul != durée officielle de jeu."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('snake_algo', ROOT/'snake-algo.py')
algo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(algo)


def run_batch(seeds, algorithm='hamiltonian', cutoff=112, turn_offset=5):
    # Environnements indépendants, entrelacés ; plusieurs lots en processus.
    games = [(seed, algo.Game(seed, algorithm, cutoff, turn_offset), {}) for seed in seeds]
    rows = []
    while games:
        active = []
        for seed, game, milestones in games:
            game.step()
            if game.snake.score > 0:
                milestones.setdefault(str(game.snake.score), game.steps)
            if game.done or game.steps >= 51000:
                seconds = game.steps / algo.base.GAME_SPEED
                rows.append(dict(seed=seed, score=game.snake.score, steps=game.steps,
                    length=len(game.snake.body), status=game.reason if game.done else 'truncated',
                    completed=game.done, theoretical_seconds_5hz=seconds,
                    theoretical_score_per_second_5hz=game.snake.score/seconds,
                    steps_per_apple=game.steps/max(1, game.snake.score),
                    steps_to_10=milestones.get('10'), milestone_steps=milestones,
                    wraps=game.wraps, direction_counts=game.direction_counts,
                    max_decision_ms=game.max_decision_ms))
            else:
                active.append((seed, game, milestones))
        games = active
    return sorted(rows, key=lambda row: row['seed'])


def summarize(rows, wall_seconds):
    scores = [r['score'] for r in rows]
    durations = [r['theoretical_seconds_5hz'] for r in rows]
    steps = sum(r['steps'] for r in rows)
    return dict(parties=len(rows), score_total=sum(scores), score_mean=statistics.mean(scores),
        score_median=statistics.median(scores), score_min=min(scores), score_max=max(scores),
        score_population_std=statistics.pstdev(scores),
        victories=sum(r['status']=='victory' for r in rows),
        collisions=sum(r['status']=='collision' for r in rows),
        truncated=sum(not r['completed'] for r in rows),
        at_least_10=sum(r['score']>=10 for r in rows),
        total_steps=steps, mean_steps=steps/len(rows),
        theoretical_seconds_5hz_mean=statistics.mean(durations),
        theoretical_seconds_5hz_min=min(durations), theoretical_seconds_5hz_max=max(durations),
        theoretical_score_per_second_5hz_aggregate=sum(scores)/sum(durations),
        mean_steps_per_apple=statistics.mean(r['steps_per_apple'] for r in rows),
        theoretical_seconds_to_10_mean=statistics.mean(r['steps_to_10']/5 for r in rows if r['steps_to_10'] is not None)
        if any(r['steps_to_10'] is not None for r in rows) else None,
        batch_compute_wall_seconds=wall_seconds,
        simulation_steps_per_compute_second=steps/wall_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--algorithm', choices=algo.ALGORITHMS, default='hamiltonian')
    p.add_argument('--cutoff', type=int, default=112)
    p.add_argument('--turn-offset', type=int, default=5)
    args = p.parse_args()
    if args.games < 1 or args.workers < 1:
        p.error('games et workers doivent être positifs')
    workers = min(args.workers, args.games)
    seeds = list(range(args.seed, args.seed+args.games))
    folder = ROOT/'runs'/('benchmark-'+args.algorithm+'-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True)
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_batch, seeds[i::workers], args.algorithm, args.cutoff, args.turn_offset) for i in range(workers)]
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f'{len(rows)}/{args.games} parties terminées', flush=True)
    elapsed = time.perf_counter()-started
    report = dict(algorithm=args.algorithm, cutoff=args.cutoff, turn_offset=args.turn_offset,
        seed_start=args.seed, workers=workers,
        measurement='Simulation sans attente ; durées 5 Hz théoriques, non observées.',
        source_sha256=hashlib.sha256((ROOT/'snake-algo.py').read_bytes()).hexdigest(),
        summary=summarize(rows, elapsed), episodes=sorted(rows, key=lambda r:r['seed']))
    (folder/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary'], indent=2))
    print(f'RESULTS={folder / "results.json"}')


if __name__ == '__main__':
    main()
