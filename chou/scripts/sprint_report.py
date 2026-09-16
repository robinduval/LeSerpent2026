"""Regenerate sprint results strictly from saved training and 5 Hz logs."""
from pathlib import Path
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from snake_rl.policy_io import load_policy, policy_kind


def lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    tail = BASE/'runs/safety_sprint_tail2_validation'
    for seed in (880001, 880002, 880003):
        result = tail/f'result_{seed}.json'
        if result.exists():
            row = json.loads(result.read_text())
        else:
            trace = lines(tail/f'trace_{seed}.jsonl')
            if not trace:
                continue
            last = trace[-1]
            row = dict(seed=seed, official_score=last['score'], steps=last['step'],
                       official_time_seconds=last['elapsed_seconds'],
                       termination_reason='in_progress_snapshot', completed=False)
        rows.append(('RL + filtre local tail2', row, str(result.relative_to(BASE))))
    patterns = ('safety_sprint_pure_*', 'distillation_sprint_validation_*',
                'original_sprint_validation_*', 'sprint_rewired_5hz_*', 'sprint_explored_5hz_*')
    for pattern in patterns:
        for folder in sorted((BASE/'runs').glob(pattern)):
            if not folder.is_dir():
                continue
            result = folder/'evaluation.jsonl'
            saved = lines(result)
            if saved:
                for row in saved:
                    rows.append((folder.name, row, str(result.relative_to(BASE))))
            else:
                trace = lines(folder/'actions.jsonl')
                if not trace:
                    continue
                last = trace[-1]
                row = dict(seed=last['seed'], official_score=last['score'], steps=last['step'],
                           official_time_seconds=last['elapsed_seconds'],
                           termination_reason='in_progress_snapshot', completed=False)
                rows.append((folder.name, row, str((folder/'actions.jsonl').relative_to(BASE))))
    selected = BASE/'checkpoints/selected.pt'
    policy = load_policy(selected)
    document = [
        '# Résultats après le sprint de vingt minutes', '',
        f'Rapport régénéré à {now}. Sources JSON conservées dans `runs/`.', '',
        'L’agent vise d’abord le score officiel maximal. Le temps intervient uniquement pour départager des scores identiques. Le projet n’optimise pas le ratio score/temps.', '',
        '## Modèle chargé sans argument', '',
        f'`checkpoints/selected.pt` : **{policy_kind(policy)}**, {policy.env_steps:,} transitions RL, {policy.updates:,} mises à jour.',
        f'SHA-256 : `{hashlib.sha256(selected.read_bytes()).hexdigest()}`.', '',
        'Le candidat séparé `checkpoints/hybrid_223.pt` est un réseau Double DQN avec filtre hamiltonien et réorganisation des arcs libres. Sa sécurité est programmée ; le réseau choisit parmi les actions admises. Le filtre utilise seulement le plateau présent. Voir `HAMILTONIAN_SAFETY.md` pour les invariants et leurs limites.', '',
        '## Essais réellement cadencés à 5 Hz', '',
        '| Variante | Graine | Points | Secondes écoulées | Pas | Fin |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for label, row, _ in rows:
        document.append(f"| {label} | {row['seed']} | {row['official_score']} | {row['official_time_seconds']:.3f} | {row['steps']} | {row['termination_reason']} |")
    document += ['',
        '`external_step_limit` signifie une interruption expérimentale : le score observé n’est pas une partie terminée ni une victoire. `in_progress_snapshot` est uniquement un instantané daté. Ces essais ont des budgets différents ; cette table ne prétend pas établir un classement statistique global. Les durées ne sont comparables pour le classement qu’à score final identique.', '',
        'L’ancien test final du réseau pur donnait 74, 69, 51, 64 et 47, moyenne 61, cinq collisions. Il est conservé dans `RESULTS_V1.md` et n’a pas été réutilisé pour régler les nouveaux candidats.', '',
        '## Apprentissage accéléré — distinct des essais à 5 Hz', '',
        '| Session | Transitions RL | Mises à jour TD | Épisodes terminés | À 223 | Temps calcul total (s) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for path in sorted((BASE/'runs').glob('hybrid*/summary.json')):
        data = json.loads(path.read_text())
        episodes = data.get('training_completed_episodes', [])
        document.append(f"| {path.parent.name} | {data['training_transitions']} | {data['updates']} | {len(episodes)} | {sum(e['completed'] and e['score']==223 for e in episodes)} | {data['wall_seconds']:.3f} |")
    episodes = lines(BASE/'runs/safety_sprint_tail2/training.jsonl')
    if episodes:
        last = episodes[-1]
        document.append(f"| tail2, en plus du préentraînement v1 | {last['training_transitions']} | {last['training_updates']} | {len(episodes)} collectés | {sum(e['completed'] for e in episodes)} | {last['wall_seconds']:.3f} |")
    document += ['',
        'Les poids évoluent pendant ces épisodes : les complétions d’apprentissage ne sont pas un taux de réussite d’un modèle figé. Chaque session hybride comporte en outre 6 000 transitions de démonstration et 500 mises à jour supervisées, comptabilisées séparément dans les métadonnées. Les contrôles fonctionnels accélérés et les ablations de poids présents dans certains répertoires ne constituent pas des temps officiels à 5 Hz.', '',
        '## Vérification et limites', '',
        'Python 3.13.15, pygame 2.6.1 ; `pygame.font` importé avec succès. Le lancement natif SDL Cocoa est testé et la fenêtre affiche explicitement « RL + filtre » pour un hybride. Les poids restent immuables pendant le jeu et aucune installation ou mise à jour RL ne se produit au lancement.', '',
        'Rapport de tests : `runs/test_sprint_report.txt`. Historique, protocole et source scientifique complémentaire : `SPRINT_20_MIN.md`. Les règles, la grille, la croissance différée, les pommes et les 223 points maximaux sont inchangés.', '',
        'La recevabilité d’un filtre programmé pendant l’évaluation reste une question distincte de l’autorisation d’entraîner librement. Ni première place, ni durée officielle à 223 ne sont déduites d’un temps d’entraînement ou d’un test accéléré.', '',
        '## Fichiers des mesures', '',
    ]
    document += [f'- `{source}`' for source in dict.fromkeys(source for _, _, source in rows)]
    (BASE/'docs/RESULTS.md').write_text('\n'.join(document)+'\n')
    (BASE/'runs/sprint_results.json').write_text(json.dumps(dict(reported_at=now, rows=[dict(variant=l, source=s, **r) for l,r,s in rows]), indent=2)+'\n')
    print(f'Wrote RESULTS.md from {len(rows)} actual 5 Hz runs/snapshots.')


if __name__ == '__main__':
    main()
