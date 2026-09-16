"""Materialize the delivery report from completed, auditable experiment logs."""
from pathlib import Path
import json
import statistics
import hashlib
import sys
from datetime import datetime
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from snake_rl.metrics import summarize


def read_rows(path):
 return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
 manifest=json.loads((BASE/'checkpoints/selection.json').read_text())
 final=read_rows(BASE/'runs/final_test/evaluation.jsonl')
 if len(final)!=5 or [r['seed'] for r in final]!=manifest['final_test_seeds']:
  raise SystemExit('Final test is not complete; do not publish a partial report as final.')
 expected_digest=hashlib.sha256((BASE/'checkpoints/selected.pt').read_bytes()).hexdigest()
 if expected_digest!=manifest['selected_sha256']:
  raise SystemExit('Delivered checkpoint does not match its selection manifest.')
 expected_id='selected:'+expected_digest[:12]
 if not (BASE/'runs/final_test/summary.json').exists() or any(
     r['model_id']!=expected_id or r['evaluation_step_budget']!=2000 or
     r['clock_hz']!=5 or r['termination_reason']=='user_quit' for r in final):
  raise SystemExit('Final test provenance/protocol mismatch.')
 if not (BASE/'runs/exact_launch.json').exists():
  raise SystemExit('Exact entry-point launch has not been recorded.')
 lines=['# Résultats mesurés — Snake RL', '',
  'Rapport produit le '+datetime.now().astimezone().isoformat(timespec='seconds')+'.', '',
  '**Le programme et l’apprentissage fonctionnent ; la première place et la complétion ne sont pas démontrées.** '
  'Le maximum du moteur est 223 points. Les valeurs ci-dessous proviennent des fichiers de cette session, '
  'jamais des scores publiés dans les articles.', '',
  '## Protocole', '',
  '- Python 3.13.15, pygame 2.6.1, Torch 2.14.0, NumPy 2.5.3 ; CPU Apple M5 Pro, 18 cœurs logiques, 48 Gio ; Torch limité à un thread par entraînement.',
  '- Entraînement accéléré autorisé par l’utilisateur. Aucun démonstrateur, filtre de sécurité ni planificateur dans l’agent RL.',
  '- Validation : trois graines communes 10001–10003, 600 déplacements maximum par épisode, un mouvement par tick de la clock originale à 5 Hz. Processus de validation parfois concurrents.',
  '- La borne de 600 est une interruption expérimentale ; elle ne constitue ni une défaite ni une victoire du jeu. Ces scores partiels ne sont pas présumés classables officiellement.',
  '- Sélection locale : moyenne de score, médiane, quartile inférieur, complétions ; temps seulement en cas de distributions de scores identiques et de parties naturellement terminées. Ce n’est pas une règle d’agrégation officielle inventée.',
  '- Test final : cinq graines réservées 20001–20005, cinq processus indépendants à 5 Hz chacun, sans entraînement concurrent, borne externe de 2000 déplacements ; aucun réglage ni nouveau choix de modèle à partir de ces résultats.',
  '- Le temps est mesuré par time.time() à la collision, à la victoire ou à l’interruption, après un premier déplacement immédiat. Le chronomètre original ne se fige pas ; la convention exacte du professeur reste inconnue.',
  '- Score, récompense RL et durée restent séparés. Les temps accélérés d’entraînement ne sont pas des temps officiels de partie.', '',
  '## Comparaisons à budget identique', '',
  '| Variante | n | Scores individuels | Moyenne | Médiane | Écart-type | Max | Collisions | Interruptions | Victoires |',
  '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
 for directory in sorted((BASE/'runs').glob('validation_*')):
  path=directory/'evaluation.jsonl'
  if not (directory/'summary.json').exists(): continue
  rows=read_rows(path); s=summarize(rows)
  lines.append(f"| {directory.name} | {len(rows)} | {', '.join(str(r['official_score']) for r in rows)} | {s['score_mean']:.2f} | {s['score_median']:g} | {s['score_std']:.2f} | {s['score_max']} | {sum(r['termination_reason']=='self_collision' for r in rows)} | {s['external_cutoffs']} | {s['completed']} |")
 lines+=['',
  'DDQN11, PER, n-step=3 et PER+n-step ont chacun un million de transitions, la même graine d’entraînement 17 et les mêmes hyperparamètres communs. '
  'Les variantes ordered utilisent 248 entrées et un MLP de 48 900 paramètres, contre 18 564 pour les onze informations. '
  'La seconde graine d’entraînement est 23. Les checkpoints à 2 et 5 millions prolongent celui à 1 million de la graine 17.', '',
  'PER et n-step ne sont pas retenus par défaut : ils n’améliorent pas cette comparaison. '
  'L’amélioration de la représentation enrichie est celle du paquet entier (rang du corps, deltas toriques, dangers, rayons, croissance). '
  'L’ablation du rang seul n’a pas été faite : ces résultats n’isolent pas son effet causal. '
  'Le réseau sur la grille est un petit MLP sur grille aplatie ; aucun CNN n’a été comparé.', '',
  'L’échantillon de validation est petit et plusieurs checkpoints sont comparés : le gagnant peut être optimiste. '
  'Les scores à 600 déplacements privilégient les progrès observables dans ce budget et ne prouvent pas le meilleur score éventuel sans limite. '
  'Il faut regarder les collisions et interruptions, pas seulement la moyenne.', '',
  '![Scores de validation](../runs/report/validation_scores.png)', '',
  '## Modèle chargé par défaut', '',
  f"- Checkpoint : `checkpoints/selected.pt`, sélection `{manifest['selected_validation']}`.",
  f"- Source : `{manifest['source_checkpoint']}`.",
  f"- Entraînement : {manifest['training_transitions']:,} transitions et {manifest['training_updates']:,} mises à jour ; encodeur `{manifest['encoder']}`.",
  f"- SHA-256 livré : `{manifest['selected_sha256']}`.",
  '- Provenance et critères complets : `checkpoints/selection.json`. Les poids sont chargés sur CPU ; pas d’exploration ni de modification du checkpoint lors du jeu.',
  '- `checkpoints/reference_classic11.pt` conserve la référence évaluée à onze informations, distincte du modèle choisi.', '',
  '## Test final réservé', '',
  '| Graine | Score | Temps mesuré (s) | Pas | Longueur finale | Fin |',
  '|---:|---:|---:|---:|---:|---|']
 for r in final:
  lines.append(f"| {r['seed']} | {r['official_score']} | {r['official_time_seconds']:.3f} | {r['steps']} | {r['final_length']} | {r['termination_reason']} |")
 s=summarize(final)
 lines+=['',f"**n={len(final)} ; moyenne {s['score_mean']:.2f} ; médiane {s['score_median']:g} ; écart-type {s['score_std']:.2f} ; minimum {s['score_min']} ; maximum {s['score_max']}.**",
  f"Complétion : **{s['completed']}/{len(final)}** ; collisions : **{sum(r['termination_reason']=='self_collision' for r in final)}/{len(final)}** ; interruptions externes : **{s['external_cutoffs']}/{len(final)}**.", '',
  'Aucun temps n’est moyenné entre des scores différents pour désigner un vainqueur. Les fichiers JSON/CSV conservent chaque association score–temps. '
  'Les comparaisons de durée à score strictement identique sont dans les summaries et dans le graphe dédié ; avec ces faibles effectifs, aucune supériorité temporelle générale n’est établie.', '',
  f"Cadence effective du test : {min(r['observed_move_hz'] for r in final):.3f} à {max(r['observed_move_hz'] for r in final):.3f} mouvements/s, avec le même clock.tick(5) que la source.",
  f"Latence moyenne d’une décision, moyennée sur les cinq parties : {statistics.mean(r['decision_latency'] for r in final)*1000:.3f} ms.",
  'Cette latence inférieure au tick ne prouve pas une réduction de la durée de jeu. Le score maximal n’étant pas atteint régulièrement, aucune optimisation temporelle au détriment de points n’a été retenue.', '',
  '## Coût et courbes d’entraînement', '',
  '| Session | Transitions supplémentaires | Compteur final | Updates cumulées | Durée de session (s) |',
  '|---|---:|---:|---:|---:|']
 total=0; wall=0
 for directory in sorted((BASE/'runs').iterdir()):
  cfgp=directory/'config.json'; lp=directory/'learning.jsonl'
  if not cfgp.exists() or not lp.exists(): continue
  cfg=json.loads(cfgp.read_text())
  if not cfg.get('train'): continue
  rows=read_rows(lp); first,last=rows[0],rows[-1]
  added=last['transitions']-first['transitions']+1000
  total+=added; wall+=last['wall_seconds']
  lines.append(f"| {directory.name} | {added:,} | {last['transitions']:,} | {last['updates']:,} | {last['wall_seconds']:.2f} |")
 lines+=['',f"Total effectivement collecté, sans recompter les préfixes repris : **{total:,} transitions**. Somme des durées de session enregistrées : **{wall:.2f} s** ; certaines sessions étaient concurrentes, cette somme n’est pas la durée murale de toute la mission.", '',
  'Les durées couvrent la collecte et les mises à jour à partir du début instrumenté, sans le coût initial d’import de Python. '
  'Les checkpoints de reprise stockent replay/optimizer/RNG. Les premiers journaux de reprise DDQN11 indiquaient le temps de la session seule ; '
  'le tableau utilise délibérément les sessions séparées. Le pilote final cumule maintenant ce temps dans les checkpoints et les nouvelles lignes d’épisodes. '
  'La latence d’entraînement n’était pas instrumentée dans les premiers runs (champ null) ; les latences d’évaluation sont mesurées. '
  'Coût des démonstrations et de recherche dans un simulateur : zéro.', '',
  '![Courbes des scores d’entraînement](../runs/report/training_scores.png)', '',
  'Les courbes sont des parties exploratoires accélérées, pas des évaluations officielles. Le dernier checkpoint n’est pas automatiquement choisi. '
  'Les losses, valeurs Q, gradients et transitions sont conservés dans learning.jsonl ; les états et cibles non finis provoquent une erreur.', '',
  '## Baseline et portée du maximum', '',
  'Le jeu initial n’avait aucun agent : sans intervention, il continue tout droit. Cette référence est conservée et mesurée. '
  'Random et greedy sont explicitement algorithmiques. Un cycle hamiltonien du tore sert également à la preuve constructive du maximum et aux fixtures de correction. '
  'Il n’a pas été chronométré sur une partie complète à 5 Hz, n’est pas le réseau livré et n’entre pas dans la sélection des modèles RL.', '',
  '## Vérifications effectuées', '',
  '- 38 tests unitaires et d’intégration réussis sous Python 3.13 ; rapport `runs/test_report.txt`.',
  '- Parité avec la source sur 1000 graines ; collisions, queue, croissance différée, score 223, reset et pureté des observations.',
  '- DDQN avec cible indépendante sans gradient, PER, poids d’importance, n-step, vrais terminaux et interruptions, copies, finitude, sauvegarde et reprise.',
  '- Classement exact des trois exemples demandés ; aucun ratio score/temps dans la sélection.',
  '- CPU, checkpoints absents/corrompus, fermeture de fenêtre, logs complets, absence d’apprentissage en évaluation et checkpoint non modifié.',
  '- Python 3.13.15 / pygame.font vérifiés ; smoke test natif avec capture `runs/smoke.png`.',
  '- Lancement sans argument et fermeture propre : preuve dans `runs/exact_launch.json` ; chemins testés depuis un autre répertoire.', '',
  '## Points à confirmer et limites', '',
  'Le professeur doit préciser l’agrégation de plusieurs essais, l’admissibilité des défaites/interruption, le relevé exact du temps, '
  'l’autorisation des poids préentraînés et des observations enrichies. L’utilisateur a bien autorisé entraînement accéléré et récompenses libres. '
  'Le programme n’atteint pas encore régulièrement le maximum ; il ne démontre donc ni complétion fiable ni première place. '
  'Les résultats ne suffisent pas à conclure qu’un autre budget de formation, une autre graine ou une autre architecture ne ferait pas mieux.', '',
  'Les sources effectivement consultées et leurs niveaux de lecture figurent dans [RESEARCH.md](RESEARCH.md) ; '
  'les faits du moteur et les ambiguïtés du cours dans [RULES_AUDIT.md](RULES_AUDIT.md).']
 (BASE/'docs/RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('Wrote',BASE/'docs/RESULTS.md')

if __name__=='__main__': main()
