# Protocole de sélection

Demande : comparer les techniques proposées, conserver les scores, retenir et
publier la meilleure version. README reste réservé à la timeline.

## Plan et contraintes

- Préserver le moteur, la croissance différée, le tore 15×15, le rendu et 5 Hz.
- Ajouter des paramètres de géométrie et de seuil sans changer les anciens modes.
- Tester une anticipation de profondeur 3 et une réparation dynamique 2-opt du
  cycle, en conservant un chemin compatible avec le corps.
- Vérifier Cell Tree : le pavage 2×2 publié ne couvre pas 225 cases ; ne pas
  changer la grille pour obtenir artificiellement une comparaison. Une adaptation
  complète n'est pas assimilée à un simple réglage et sera signalée non testée.
- Comparer toutes les variantes retenues sur seeds 1000–1099. Sauvegarder chaque
  partie et les paramètres, pas seulement les meilleurs résultats.
- Sélection lexicographique : score total, victoires, moins de mouvements.
  Les candidats ayant des collisions/troncatures sont exclus du remplacement.
- Valider les finalistes sur seeds 10000–10999, distinctes du réglage. Ces seeds
  constituent une validation, pas un jeu pour ajuster les paramètres.
- Mettre le gagnant par défaut, tester le lancement sans argument, publier les
  résultats et le code sur main (demande explicite de l'utilisateur).

## Mesures

Scores, victoires, collisions, troncatures, mouvements/pomme, durées théoriques
à 5 Hz, temps réel de calcul du lot, temps maximal de décision. Même seed ne
signifie pas même position de pomme après divergence des corps. Ni un benchmark
ni l'absence de collision observée ne constituent une preuve de sûreté universelle.

## Sources

- https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/
- https://github.com/twanvl/snake#algorithms

## Progression

- Protocole fixé avant l'évaluation ; publication des essais rejetés incluse.

## Résultat de la campagne

Archives et graphiques : `results/selection-20260923-210057/REPORT.md`.
35 configurations × 100 parties, puis 3 finalistes × 1 000 nouvelles parties :
6 500 parties sauvegardées. Toutes ont atteint 223 points.

Le gagnant mesuré est `shortcut`, seuil 112, géométrie 3 : 1 000 victoires,
223 000 points cumulés, 6 651,437 mouvements moyens, soit 22 min 10,287 s
et 5,965 s/pomme **théoriques à 5 Hz**. L'ancienne géométrie 5 donne
22 min 15,855 s sur la même validation. Le gain de 0,417 % est modeste :
ce classement empirique ne démontre pas une supériorité statistique générale.

L'anticipation profondeur 3 et la réparation locale 2-opt n'ont pas gagné.
Cell Tree n'a pas été implémenté/testé : son pavage 2×2 demande une adaptation
spécifique au tore impair, hors de cette campagne. La réparation locale testée
ne prétend pas reproduire une méthode complète de réparation dynamique publiée.

`python snake-algo.py` lance désormais le gagnant, avec le rendu et la clock
originaux. Les variantes restent disponibles pour reproduire les expériences.
Les empreintes du manifeste identifient le code au lancement des benchmarks ;
les changements suivants concernent le défaut CLI et l'export des paramètres,
pas la logique simulée. README reste strictement la timeline demandée.
