# Optimisation mesurée de l’agent Snake

> Rapport archivé de la version `all128`. Les résultats et la politique actuellement livrée sont décrits dans [le benchmark actuel](../../BENCHMARK.md).

**Agent livré par défaut : `all128`. Cible de 4 500 déplacements moyens non atteinte.**
Sur les 100 graines réservées, il atteint 223 points dans **100/100 parties**, sans collision ni interruption, en **4 892,48 déplacements moyens**. Cela réduit les déplacements de **19,23 %** par rapport à l’ancien agent, mais laisse **392,48 pas** au-dessus de la cible.
La durée moyenne équivalente x1 vaut **978,50 s**, soit environ **16 min 18,50 s**. La cible de 3 000 déplacements n’est pas atteinte non plus. Aucun optimum global n’est revendiqué.

## Agent retenu et changements expérimentés

`serpent-algo.py` reste autonome avec la bibliothèque standard et Pygame. Le dossier `ia` n’a pas été consulté.

L’agent retenu combine :

- quatre départs partageant un plafond total de **128 propositions exploratoires**, après comparaison initiale à budget 64 ;
- cycles normalisés sur la tête et mémorisation des propositions déjà visitées ; plusieurs cycles de même distance peuvent être conservés ;
- inversions 2-opt améliorantes et neutres, y compris celles contenant la pomme si `i+j+1 = 2*a` ;
- **Or-opt sur toutes les longueurs de blocs libres**, avec déplacement et inversion éventuelle du bloc ;
- descentes locales entre les recherches, meilleure continuation certifiée conservée et échéance coopérative de **20 ms**.

La suppression des revisites porte sur les propositions exploratoires et leurs résultats ; des descentes différentes peuvent encore rejoindre le même optimum. La version expérimentale qui conserve alors le plateau précédent (`diverse64`) n’a pas amélioré les réglages.

Le 3-opt supplémentaire a été implémenté et validé, puis comparé comme finaliste `three64` : il n’est **pas activé par défaut**. Sa moyenne réservée est supérieure de seulement 13,10 pas à celle de `all128` ; cet écart faible ne démontre pas une supériorité générale d’un voisinage sur l’autre.
Les déclenchements adaptatifs en cours de trajet et l’anticipation de six mouvements, avec simulation exacte de la croissance différée et arrêt à la pomme présente, ont aussi été implémentés et mesurés. Leurs compromis de coût et de déplacements ne justifient pas leur activation dans l’agent livré. Les budgets plus élevés à 40 ms n’ont pas été retenus. Ces variantes restent accessibles pour reproduire les expériences.

Les mesures de chaque ajout, d’abord à plafond 64 comparable, puis avec voisinages/budgets élargis, figurent dans [ABLATIONS.md](../../benchmarks/ABLATIONS.md). La référence historique `explored64` conserve sa logique et son ancien plafond de 64. `fixed` reste disponible ; sa comparaison historique sur 20 graines est [archivée](../../benchmarks/original-BENCHMARK.md).

## Protocole

- **Régression : graines 1–20**, conservées depuis la version précédente.
- **Réglages : graines 101–110**, dont les cinq premières pour les ablations initiales. Les paramètres et finalistes ont été [figés](../../benchmarks/finalists-freeze.json) avant toute simulation réservée.
- **Validation réservée : graines 1001–1100**, jamais utilisées pour régler les paramètres. Comparaison des deux finalistes figés et de la référence, soit **300 parties**, sur quatre processus. La règle de sélection préannoncée privilégie les complétions sans échec, puis la moyenne de déplacements. Aucun réglage supplémentaire n’a suivi ces résultats.
- **Latences : graines 1001–1005 rejouées séparément sur un seul processus**, pour la référence et le gagnant. Cette seconde mesure n’a pas servi à régler l’agent.

Toutes les politiques utilisent exactement `Game.step`, le placement uniforme des pommes libres dans l’ordre x puis y et les mêmes graines. Les pommes rencontrées peuvent différer entre politiques puisque les cases occupées diffèrent. L’agent reçoit seulement un état public immuable, sans référence au générateur des pommes. Son exploration utilise un générateur indépendant dérivé de cet état.

Chaque simulation vérifie le cycle, le corps contigu et la progression **avant et après chaque mouvement**. Le futur corps est certifié avant l’exécution. Le plafond de 49 952 déplacements est une interruption de l’évaluation, pas une modification du jeu. Les paramètres du protocole sont conservés dans [optimization-protocol.json](../../optimization-protocol.json).

## Résultats sur les 100 graines réservées

| Agent | Complètes à 223 | Collisions | Interruptions | Pas moyens | Médiane | p95 | Maximum | Moyenne x1 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `explored64` | 100/100 | 0 | 0 | 6 057,48 | 6 046,0 | 6686 | 7162 | 1 211,50 |
| `all128` | 100/100 | 0 | 0 | 4 892,48 | 4 916,5 | 5306 | 5853 | 978,50 |
| `three64` | 100/100 | 0 | 0 | 4 905,58 | 4 916,0 | 5434 | 5847 | 981,12 |

Tous les scores valent 223. Pour `all128`, **10/100** parties utilisent moins de 4 500 déplacements ; aucune n’est sous 3 000. Le seuil demandé concerne la **moyenne**, il n’est donc pas atteint.

**Meilleure partie réservée de l’agent livré : seed 1063, score 223, 4247 déplacements, temps x1 : 849,4 secondes (14 min 9,4 s).**

La meilleure partie de cet agent parmi les lots de régression et réservé est **seed 15, score 223, 4105 déplacements, 821,0 secondes x1 (13 min 41 s)**. Le finaliste non retenu `three64` a obtenu son meilleur résultat réservé avec la seed 1009 : 4 012 pas, 802,4 s x1 ; ce résultat isolé ne remplace pas la comparaison des moyennes.

Les cinq parties les plus lentes de l’agent livré sur le lot réservé :

| Graine | Score | Déplacements | Équivalent x1 (s) |
|---:|---:|---:|---:|
| 1035 | 223 | 5853 | 1 170,6 |
| 1046 | 223 | 5439 | 1 087,8 |
| 1019 | 223 | 5402 | 1 080,4 |
| 1096 | 223 | 5334 | 1 066,8 |
| 1092 | 223 | 5309 | 1 061,8 |

Déplacements moyens par phase sur ce même lot :

| Pommes consommées | Ancien agent | Agent retenu |
|---|---:|---:|
| 1-50 | 1 239,01 | 879,00 |
| 51-100 | 1 677,57 | 1 309,41 |
| 101-150 | 1 797,28 | 1 501,87 |
| 151-223 | 1 343,62 | 1 202,20 |

Les pommes 51–150 représentent encore **2 811,28 pas**, soit **57,46 %** du trajet. L’attente maximale d’une pomme observée pour `all128` est de **135 mouvements**, sous la borne conservatrice de 224.

Données complètes par graine : [benchmark-results.json](../../benchmark-results.json). Son champ historique `movement_reduction_percent` compare la première politique à la dernière (`three64`) et vaut donc 19,02 %. Le gain du gagnant `all128`, **19,23 %**, est calculé explicitement depuis les moyennes et conservé dans [selection.json](../../benchmarks/selection.json).

## Régression sur les 20 graines historiques

| Agent | Complètes à 223 | Collisions | Interruptions | Pas moyens | Médiane | p95 | Maximum | Moyenne x1 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `explored64` | 20/20 | 0 | 0 | 6 016,85 | 5 924,0 | 6457 | 6519 | 1 203,37 |
| `all128` | 20/20 | 0 | 0 | 4 869,90 | 4 914,5 | 5240 | 5278 | 973,98 |
| `three64` | 20/20 | 0 | 0 | 4 916,10 | 4 920,0 | 5362 | 5517 | 983,22 |

La référence reproduit **exactement les déplacements de chacune des 20 anciennes parties**, moyenne 6 016,85. Les trois politiques atteignent 223 points, sans collision ni interruption. Données : [regression-finalists.json](../../benchmarks/regression-finalists.json).

## Latences mesurées sans concurrence

Python **3.13.15**, Pygame **2.6.1**, graines 1001–1005, un processus et échéance de recherche de 20 ms. Les dix parties sont complètes sans échec.

| Agent / type de décision | Nombre | Moyenne (ms) | p95 (ms) | Maximum (ms) |
|---|---:|---:|---:|---:|
| `explored64`, ordinaire | 29020 | 0,122 | 0,144 | 0,307 |
| `explored64`, avec recherche | 1115 | 2,380 | 6,946 | 7,863 |
| `all128`, ordinaire | 23107 | 0,262 | 0,336 | 2,850 |
| `all128`, avec recherche | 1115 | 13,692 | 20,356 | 20,471 |

Le coût moyen global du gagnant vaut **0,880 ms** ; son p95 global vaut **0,390 ms**. Les recherches ne représentent qu’environ 4,6 % des décisions et doivent être lues séparément : un p95 global masque une grande partie de leur coût.

Ces durées incluent `select_action`, les recherches et validations internes ; elles excluent le snapshot, le moteur, le rendu et les contrôles externes du benchmark. Les p95 sont des percentiles empiriques au rang supérieur, calculés sur les échantillons bruts regroupés. Le maximum observé est inférieur aux 200 ms disponibles à 5 mouvements/s, sans constituer une garantie de latence sur une autre machine.

L’échéance est **coopérative** : une énumération ou validation en cours, puis la préparation finale, peuvent finir après 20 ms. Elle a été atteinte dans 694 décisions du gagnant sur ce lot sériel. Une expiration conserve une continuation sûre ; elle n’est ni une collision ni une interruption de partie.

Source : [latency-serial.json](../../benchmarks/latency-serial.json). Les latences du fichier réservé correspondent à quatre processus concurrents ; elles sont conservées comme observations sous cette charge, et non présentées comme les coûts isolés ci-dessus.

## Temps x1 et reproductibilité

La cadence et le chronomètre du jeu sont inchangés : **`GAME_SPEED = 5`, `clock.tick(5)` et chronomètre mural**. Toutes les parties de ce rapport ont été simulées sans attente ni affichage. Les temps x1 sont **calculés** par `déplacements / 5`, jamais annoncés comme des chronométrages réels à la vitesse officielle.

Temps mural des campagnes, vérifications comprises : **289,238 s** pour les 300 parties réservées en parallèle ; **59,799 s** pour les 60 parties de régression ; **31,440 s** pour les dix parties de mesure sérielle. Le premier mouvement de l’interface précède son premier `clock.tick`, comme auparavant ; l’équivalent x1 ne cherche pas à reproduire cet écart initial.

Le plafond de travail, l’ordre de recherche et le hasard exploratoire sont reproductibles. **Une échéance murale rend le point d’arrêt sensible à la charge du système** : le même numéro de graine ne garantit donc pas exactement les déplacements indiqués. Le lot sériel donne par exemple 4 844,40 pas moyens pour le gagnant sur ses cinq graines ; il ne remplace pas la moyenne des 100 graines. `--time-limit-ms 0` désactive la garde temporelle pour les ablations déterministes, tout en conservant le plafond de travail ; il ne change pas la vitesse du jeu.

## Vérifications et traçabilité

**47 tests passent** en 11,038 s sur la version livrée : moteur différé, pommes consécutives, quatre bords, victoire et redémarrage, intégrité du cycle et du corps, nouvelles adjacences, comptabilité des rangs, neutres contenant la pomme, Or-opt comparé à une énumération exhaustive indépendante, reconnexion 3-opt, anti-revisites, budgets supérieurs à 64, anticipation sans pommes futures, arrêt sûr sur délai expiré et absence d’accès au hasard des pommes.

Le rendu a également été contrôlé sur **20 images avec le vrai Pygame**, sans attente et en pilote SDL sans fenêtre : surface 450×530, grille, pomme, serpent, score, chronomètre, demande de cadence 5 Hz et fermeture correcte. La capture temporaire n’est plus conservée ; le [compte rendu du contrôle](../../benchmarks/ui-validation.json) reste disponible. Ce contrôle accéléré ne constitue pas une partie chronométrée réelle.

Le moteur, le rendu, l’ordre des opérations et les algorithmes sont identiques entre la [source figée mesurée](../../benchmarks/frozen-agent.py) et la version livrée. Seuls la sélection par défaut et les textes d’aide ont changé après la validation réservée.

- SHA-256 mesuré au gel : `9a196bb39ae2c1cbfe959b4d94f207ed40ab91e0027bb60d919ba1efce8804ff`.
- SHA-256 livré : `b8337b3b3b9ef847c16fc7c8259361b2828dbc2a7f4947799e36df13cdd6f177`.
- Résultats antérieurs conservés : [référence originale](../../benchmarks/original-baseline-results.json).
- Les sauvegardes intermédiaires `.partial.json` et le résultat du test de démarrage du calcul parallèle ont été retirés lors du nettoyage de livraison. Aucun de ces fichiers temporaires n'entre dans les tableaux de performances ; les résultats complets sont conservés.

## Commandes

```sh
.venv/bin/python serpent-algo.py --policy all128
.venv/bin/python serpent-algo.py --policy explored64 --seed 15
.venv/bin/python -m unittest discover -v
.venv/bin/python serpent-algo.py --benchmark --policies explored64,all128,three64 --seeds 1:21 --workers 4 --time-limit-ms 20 --output benchmarks/recheck-regression.json
.venv/bin/python serpent-algo.py --benchmark --policies explored64,all128,three64 --seeds 1001:1101 --workers 4 --time-limit-ms 20 --output benchmarks/recheck-heldout.json
.venv/bin/python serpent-algo.py --benchmark --policies explored64,all128 --seeds 1001:1006 --workers 1 --time-limit-ms 20 --output benchmarks/recheck-latency.json
```
