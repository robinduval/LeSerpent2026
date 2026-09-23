"""Campagne comparative, archives par partie et validation séparée."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('snake_benchmark', ROOT/'benchmark.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def configurations():
    configs = [dict(algorithm=a, cutoff=112, turn_offset=5)
               for a in ('hamiltonian', 'bfs-safe')]
    configs += [dict(algorithm='shortcut', cutoff=cutoff, turn_offset=offset)
                for cutoff in (60, 90, 112, 140, 170)
                for offset in (1, 3, 5, 8, 11)]
    configs += [dict(algorithm='lookahead', cutoff=cutoff, turn_offset=5)
                for cutoff in (90, 112, 140)]
    configs += [dict(algorithm='dynamic', cutoff=112, turn_offset=offset)
                for offset in (1, 3, 5, 8, 11)]
    return configs


def identifier(config):
    return f"{config['algorithm']}-c{config['cutoff']}-o{config['turn_offset']}"


def ranking(report):
    s = report['summary']
    return (s['collisions'] == 0 and s['truncated'] == 0,
            s['score_total'], s['victories'], -s['total_steps'])


def evaluate(config, seeds, folder, phase):
    started = time.perf_counter()
    rows = bench.run_batch(seeds, **config)
    summary = bench.summarize(rows, time.perf_counter()-started)
    summary['max_decision_ms'] = max(r['max_decision_ms'] for r in rows)
    # All scores retained; milestones at every apple stay in compressed archives.
    report = dict(config=config, summary=summary, seed_first=seeds[0], seed_last=seeds[-1])
    archive = f'{phase}-{identifier(config)}.json.gz'
    with gzip.open(Path(folder)/archive, 'wt', encoding='utf-8') as handle:
        json.dump(dict(**report, episodes=rows), handle)
    report['archive'] = archive
    return report


def run_phase(pool, configs, seeds, folder, phase):
    futures = [pool.submit(evaluate, config, seeds, str(folder), phase) for config in configs]
    reports = []
    for future in as_completed(futures):
        report = future.result()
        reports.append(report)
        s = report['summary']
        print(f"{phase} {len(reports)}/{len(configs)} {identifier(report['config'])}: "
              f"score={s['score_total']} wins={s['victories']}/{len(seeds)} "
              f"pas/pomme={s['mean_steps_per_apple']:.2f}", flush=True)
        (folder/f'{phase}.json').write_text(json.dumps(sorted(reports,key=ranking,reverse=True),indent=2))
    return sorted(reports, key=ranking, reverse=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.workers < 1:
        p.error('workers doit être positif')
    folder = ROOT/'results'/datetime.now().strftime('selection-%Y%m%d-%H%M%S')
    folder.mkdir(parents=True)
    manifest = dict(protocol='100 réglage + 1000 validation finalistes ; score prioritaire ; simulations sans attente',
        workers=args.workers, sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                     for name in ('snake-algo.py','benchmark.py','compare_methods.py')},
        not_tested={'cell_tree':'Le pavage 2×2 de la méthode publiée ne couvre pas le tore impair 15×15 ; pas de modification des règles.'})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        screening = run_phase(pool, configurations(), list(range(1000,1100)), folder, 'screening')
        eligible = [r for r in screening if ranking(r)[0]]
        # Freeze finalists BEFORE opening validation seeds; compare baseline too.
        finalists = [r['config'] for r in eligible[:3]]
        reference = dict(algorithm='shortcut',cutoff=112,turn_offset=5)
        if reference not in finalists:
            finalists.append(reference)
        (folder/'finalists.json').write_text(json.dumps(finalists,indent=2))
        validation = run_phase(pool, finalists, list(range(10000,11000)), folder, 'validation')
    winner = next((r for r in validation if ranking(r)[0]), None)
    result = dict(winner=winner, total_wall_seconds=time.perf_counter()-started,
                  folder=str(folder), note='Durées de jeu théoriques à 5 Hz ; pas de preuve de sûreté universelle.')
    (folder/'selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
