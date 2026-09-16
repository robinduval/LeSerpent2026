"""Crée un snapshot ZIP d'analyse, y compris pendant un batch d'entraînement."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    destination = ROOT / 'exports' / datetime.now().strftime('analysis-%Y%m%d-%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('snake-ia.py', 'serpent-algo.py', 'test_snake_ia.py', 'requirements.txt',
                 'RUNNING.md', 'README.md', 'ANALYSIS_HANDOFF.md', 'export_analysis.py'):
        shutil.copy2(ROOT / name, destination / name)
    all_rows = []
    summaries = []
    for path in sorted((ROOT / 'runs').glob('*/metrics.json')):
        # metrics.json est remplacé atomiquement : un seul snapshot cohérent par run.
        data = json.loads(path.read_text())
        folder = destination / 'runs' / path.parent.name
        folder.mkdir(parents=True)
        (folder / 'metrics.json').write_text(json.dumps(data, indent=2))
        shutil.copy2(path.parent / 'config.json', folder / 'config.json')
        # latest.pt est lui aussi remplacé atomiquement. Ce checkpoint peut précéder
        # le snapshot des métriques si le batch est encore en cours.
        checkpoint = path.parent / 'latest.pt'
        if checkpoint.exists():
            shutil.copy2(checkpoint, folder / 'latest.pt')
        rows = []
        for original in data['episodes']:
            row = dict(original)
            row['run_id'] = path.parent.name
            row['wall_duration_s'] = row['duration_s']
            row['wall_score_per_second'] = row['score_per_second']
            duration = row['duration_s'] if row['mode'] == 'eval' else row['steps'] / 5
            row['game_duration_s'] = duration
            row['game_score_per_second'] = row['score'] / duration if duration else 0
            row['game_timing_kind'] = 'measured_5hz' if row['mode'] == 'eval' else 'estimated_5hz'
            rows.append(row)
        all_rows.extend(rows)
        complete = [r for r in rows if r['completed']]
        best = max(complete, key=lambda r: (r['score'], r['game_score_per_second']), default=None)
        summaries.append(dict(run_id=path.parent.name, mode=data['live']['mode'],
                              completed=len(complete), interrupted=len(rows)-len(complete),
                              best_score=best['score'] if best else None,
                              best_game_ratio=best['game_score_per_second'] if best else None,
                              mean_score=sum(r['score'] for r in complete)/len(complete) if complete else None,
                              success_rate=sum(r['score']>=10 for r in complete)/len(complete) if complete else None,
                              snapshot_live=data['live']))
    fields = sorted({key for row in all_rows for key in row})
    with (destination / 'episodes_all.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (destination / 'summary.json').write_text(json.dumps(summaries, indent=2))
    manifest = dict(created_at=datetime.now().astimezone().isoformat(),
                    python=platform.python_version(), platform=platform.platform(),
                    packages={name: importlib.metadata.version(name) for name in ('torch','pygame','numpy')},
                    snapshot_note='Lecture atomique par run, pas de transaction entre runs et checkpoints. Les runs actifs sont partiels.',
                    files={str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in destination.rglob('*') if p.is_file()})
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = shutil.make_archive(str(destination), 'zip', destination)
    print(archive)


if __name__ == '__main__':
    main()
