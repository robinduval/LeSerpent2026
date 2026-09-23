# Réglages du sprint de dix minutes

Ces essais utilisent exclusivement les graines **201, 202 et 203**. Ils servent à sélectionner des finalistes, pas à mesurer leur généralisation. Les résultats du lot réservé sont volontairement absents de ce document.

Le moteur, le tirage des pommes et la cadence de 5 mouvements/s sont inchangés. Chaque partie vérifie le cycle complet, le corps contigu et la progression vers la pomme. Les durées x1 sont des équivalents `déplacements / 5`, calculés depuis des simulations accélérées.

| Variante | Échéance | Seed 201 | Seed 202 | Seed 203 | Moyenne déplacements | Moyenne x1 |
|---|---:|---:|---:|---:|---:|---:|
| Ancien agent `all128` | 20 ms | 4 642 | 4 853 | 5 012 | 4 835,67 | 967,13 s |
| Ajout 4-opt, `bridge128` | 20 ms | 3 423 | 3 505 | 3 959 | 3 629,00 | 725,80 s |
| 4-opt + détours virtuels, `bridge_uphill` | 20 ms | 3 627 | 3 657 | 3 202 | **3 495,33** | **699,07 s** |
| Même exploration avec échéance supérieure | 50 ms | 3 728 | 3 526 | 3 564 | 3 606,00 | 721,20 s |
| 4-opt + anticipation de 4 mouvements | 20 ms | 4 001 | 3 390 | 3 959 | 3 783,33 | 756,67 s |
| 4-opt + permutation de blocs inversés | 20 ms | 3 623 | 3 444 | 3 814 | 3 627,00 | 725,40 s |
| Recherche dirigée + 4-opt | 20 ms | 3 778 | 3 631 | 3 804 | 3 737,67 | 747,53 s |

Toutes les parties de ce tableau terminent à **223 points**. Aucun résultat ne prouve une moyenne de 600 secondes. La meilleure partie de réglage du finaliste avec détours virtuels prend **3 202 déplacements, soit 640,4 secondes x1**.

## Modifications retenues pour la comparaison finale

**4-opt sur l'arc libre.** La transformation `A+B+C+D+E → A+D+C+B+E` échange les blocs libres B et D. Elle vérifie les quatre nouvelles connexions toriques. Les trois blocs sont non vides, l'ensemble des 225 cases est conservé et le suffixe contenant le corps ne bouge pas. Elle ouvre des configurations que les seuls 2-opt et Or-opt atteignent difficilement. C'est le principal gain observé ici.

**Détours virtuels bornés.** La recherche peut explorer un cycle dont la distance à la pomme dépasse temporairement de huit cases au maximum celle du meilleur cycle rencontré. Ces détours concernent uniquement des copies certifiées : aucun mouvement réel ne s'éloigne de la pomme. Après chaque proposition, la descente locale reprend. Si elle rejoint un optimum déjà vu, la branche conserve sa proposition distincte pour diversifier les essais. Le meilleur cycle, départ inclus, reste toujours disponible et lui seul revient au jeu.

Le finaliste `bridge128` conserve quatre branches et 128 propositions. `bridge_uphill` utilise quatre branches et un plafond de 256 propositions. Leur échéance commune est de 20 ms. Le gain additionnel du second combine donc un voisinage exploratoire élargi et un plafond de travail supérieur : ce petit essai ne sépare pas complètement ces deux effets.

## Variantes non retenues

L'échéance de 50 ms n'a pas amélioré la moyenne du prototype avec détours virtuels. L'anticipation, la recherche dirigée et le voisinage avec inversions supplémentaires n'ont pas apporté de gain suffisamment clair face aux deux finalistes. Trois graines restent un échantillon trop petit pour conclure qu'une variante est universellement inférieure.

Une échéance est coopérative : une énumération ou une certification en cours peut la dépasser légèrement. Elle rend aussi les trajectoires sensibles à la charge de la machine. Des essais lancés simultanément peuvent donc choisir des cycles différents malgré la même graine de pommes. Ces chiffres sont des observations, pas une garantie de répétition exacte sous échéance réelle. Le plafond de travail peut être utilisé seul pour des répétitions déterministes.

## Traçabilité

- Référence : [tuning-baseline.json](tuning/tuning-baseline.json).
- Détours virtuels, 20 ms : [résultats](tuning/sprint_uphill_bridge_20_8.json).
- Détours virtuels, 50 ms : [résultats](tuning/sprint_uphill_bridge_50_8.json).
- Anticipation : [résultats](tuning/sprint-bridge-lookahead.json).
- Blocs inversés : [résultats](tuning/sprint-signed-bridge.json).
- Recherche dirigée : [résultats](tuning/sprint-directed-bridge.json).
- Le premier essai `bridge128` est issu de la sortie terminal du prototype temporaire, communiquée pendant le sprint : 3 423 / 3 505 / 3 959 déplacements. Il n'a pas de fichier JSON dédié. Les scripts de prototype ont été retirés au nettoyage ; les résultats JSON et le code figé des finalistes sont conservés.
- Paramètres figés et séparation des lots : [protocol.json](protocol.json).
- Code figé avant le lot réservé : [frozen-agent.py](frozen-agent.py).

Les critères enregistrés avant validation prévoyaient de comparer les deux finalistes à `all128` sur les graines réservées, puis de choisir d'abord la fiabilité et ensuite la moyenne des déplacements. Cette validation est terminée et documentée dans [BENCHMARK.md](../../BENCHMARK.md). Aucun paramètre supplémentaire n'a été réglé à partir de ces résultats réservés.
