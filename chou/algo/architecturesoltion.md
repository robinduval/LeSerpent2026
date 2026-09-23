# Architecture de la solution Snake

La solution livrée est un agent **entièrement algorithmique** : un cycle hamiltonien reconfigurable, des améliorations locales et une exploration bornée. La priorité est d’atteindre **223 points sans collision**, puis de réduire les déplacements. Aucun apprentissage ni modèle neuronal n’intervient.

## Organisation du programme

Tout le fonctionnement reste dans [serpent-algo.py](serpent-algo.py), sans dépendance à un autre dossier du projet.

| Composant | Responsabilité |
|---|---|
| `Snake`, `Apple` | Corps, déplacement, croissance différée et tirage des pommes libres. |
| `Game.state()` | Produit un état public immuable : corps, direction, croissance en attente, score, déplacements et pomme présente. |
| `Game.step()` | Applique une transition du jeu, commune à l’interface et aux simulations. |
| `RewiredCycleShield` | Conserve le cycle, vérifie les invariants et valide le futur corps avant de jouer. |
| `AdvancedCyclePolicy` | Recherche des reconfigurations et choisit le mouvement admissible. |
| `main()` | Affichage Pygame, événements, chronomètre, cadence et redémarrage. |
| `simulate()`, `benchmark()` | Évaluation accélérée, vérifications et export des mesures. |

Cette séparation permet de tester le même moteur que celui de la partie affichée. La décision reçoit seulement `GameState` : elle ne dispose ni des pommes futures, ni du générateur aléatoire du moteur.

## Règles conservées

La grille est un **tore de 15 × 15 cases** : les coordonnées du prochain déplacement sont calculées modulo 15. Le corps initial contient trois cases ; la tête est en `(3, 7)` et avance vers la droite.

L’ordre de `Game.step()` est essentiel :

1. Appliquer la direction autorisée et déplacer la tête.
2. Retirer la queue, sauf si une croissance était déjà en attente ; dans ce dernier cas, conserver la queue et consommer ce drapeau.
3. Détecter une collision avec le corps ainsi obtenu.
4. Si la tête atteint la pomme, incrémenter le score et programmer la croissance **du prochain mouvement**.
5. Tirer la prochaine pomme parmi les cases libres ; s’il n’en reste aucune, terminer en victoire.

Avec cette croissance différée, le score maximal est **223**, et non 222. Les pommes sont tirées uniformément parmi les cases libres, énumérées dans l’ordre x puis y. Le placement n’a pas été adapté à l’agent.

`GAME_SPEED = 5` et `clock.tick(GAME_SPEED)` sont conservés. Le chronomètre affiché mesure le temps mural réel ; le rendu reste celui du jeu d’origine.

## Un parcours sûr disponible à tout instant

Le cycle initial classe les cases par :

```text
rank(x, y) = 15*y + ((x+y) % 15)
```

Le successeur est à droite tant que `(x+y) % 15 < 14`, puis en bas. Ce parcours visite les 225 cases une seule fois et revient au départ grâce au tore. Les rangs du corps initial, de la tête vers la queue, sont **115, 114, 113** : son orientation est compatible dès le premier mouvement.

L’invariant maintenu est plus fort qu’un simple ordre entre les segments : **le corps forme un segment contigu du cycle, orienté de la queue vers la tête**. Le cycle entier est vérifié : couverture exacte des cases, rangs cohérents, adjacence torique de chaque arête et alignement du corps.

Le successeur du cycle constitue la continuation de secours. Une reconfiguration est préparée dans une copie, certifiée, puis adoptée. Avant de renvoyer une action, `commit()` vérifie aussi le corps obtenu après ce mouvement, en tenant compte de `grow_pending`. Cette précaution couvre notamment les pommes consécutives et la conservation différée de la queue.

## Raccourcir le cycle sans compromettre le corps

Le cycle est temporairement normalisé avec la tête au rang 0. Les cases libres occupent alors l’arc compris entre la tête et la queue. Les transformations ne touchent pas aux arêtes internes du corps.

**2-opt** inverse une portion libre `[i+1 … j]`. Les deux nouveaux raccordements doivent être voisins sur le tore. Si la pomme est dans cette portion, son nouveau rang est `i+j+1-a`, où `a` désigne son ancien rang. Une inversion est améliorante si ce rang diminue ; elle peut aussi être neutre lorsque `i+j+1 = 2*a`.

**Or-opt** extrait un bloc libre, éventuellement inversé, puis le réinsère ailleurs dans l’arc libre. Les trois nouvelles connexions sont vérifiées. La configuration retenue considère toutes les longueurs admissibles ; l’énumération exploite les quatre voisins toriques des extrémités pour éviter de tester tous les raccordements possibles.

Ces opérations réordonnent les mêmes cases dans une unique liste cyclique. La validation complète après transformation confirme qu’aucun sous-cycle, doublon ou raccordement invalide n’est introduit.

## Recherche retenue : `all128`

À chaque décision, une descente locale applique les transformations qui rapprochent directement la pomme. À l’initialisation et après chaque consommation, une exploration plus large complète cette descente :

1. Quatre branches partent du meilleur cycle local disponible et partagent un budget total de **128 propositions exploratoires**.
2. Des transformations neutres permettent d’explorer d’autres configurations de même distance.
3. Chaque proposition est certifiée, puis améliorée par une nouvelle descente locale.
4. Les cycles sont représentés par des tuples normalisés sur la tête. Un ensemble d’empreintes évite de revisiter les propositions déjà rencontrées ; plusieurs configurations distinctes de même distance peuvent subsister.
5. Le meilleur cycle rencontré reste disponible, y compris la continuation initiale si aucune amélioration n’est obtenue.

Le budget de 128 ne compte pas toutes les opérations élémentaires : les descentes locales, l’énumération des voisins et les validations s’y ajoutent. Une échéance coopérative de **20 ms** borne également la recherche par défaut. Une opération déjà commencée et la préparation finale peuvent dépasser légèrement cette échéance.

L’action choisie minimise la distance cyclique restante parmi les mouvements admissibles. Le cycle adopté ne doit jamais augmenter la distance à la pomme ; le mouvement exécuté doit la réduire **strictement**. Cette quantité entière décroissante empêche une boucle sans consommation. La borne conservatrice est de 224 déplacements par pomme.

Le hasard exploratoire est indépendant de celui des pommes et dérivé de l’état public. Sans limite murale, le plafond de travail et l’ordre de recherche rendent les essais reproductibles. Avec l’échéance de 20 ms, la charge de la machine peut changer le point d’arrêt et donc le parcours : une graine seule ne garantit pas les mêmes déplacements.

## Choix expérimentaux et limites

L’ancien agent `explored64` et le cycle fixe `fixed` sont conservés comme comparateurs. Les autres variantes restent accessibles dans `POLICY_CONFIGS` pour reproduire les ablations, mais ne sont pas activées par défaut.

Les principaux gains viennent de l’élargissement du voisinage par Or-opt. Les branches seules n’ont pas amélioré le petit lot initial. Les essais de **3-opt supplémentaire**, de relance adaptative en cours de trajet et d’anticipation sur six mouvements n’ont pas présenté un compromis suffisamment favorable pour remplacer `all128`. L’anticipation simulait la croissance exacte et s’arrêtait à la pomme présente, sans consulter la suivante.

Les paramètres ont été réglés sur les graines 101–110, puis figés avant la comparaison finale sur 100 graines réservées, 1001–1100. Les graines historiques 1–20 servent à la régression. Les résultats finaux sont :

| Mesure sur les 100 graines réservées | Ancien `explored64` | Agent livré `all128` |
|---|---:|---:|
| Parties terminées à 223 points | 100/100 | 100/100 |
| Collisions / interruptions | 0 / 0 | 0 / 0 |
| Déplacements moyens | 6 057,48 | **4 892,48** |
| Déplacements médians | 6 046 | 4 916,5 |
| Durée moyenne équivalente à 5 mouvements/s | 1 211,50 s | **978,50 s** |

Le gain mesuré est de **19,23 %**, mais la cible de **4 500 déplacements moyens n’est pas atteinte**, ni celle de 3 000. La sécurité des parcours ne constitue pas une preuve d’optimalité des déplacements.

La meilleure partie de l’agent livré parmi les lots de régression et réservé est la **graine 15 : score 223, 4 105 déplacements, soit 821 secondes x1**. Sur le lot réservé seul, sa meilleure partie est la graine 1063 : 4 247 déplacements, soit 849,4 secondes. Ces temps sont calculés par `déplacements / 5`, à partir de simulations accélérées ; ce ne sont pas des chronométrages réels de l’interface.

Sur une mesure séparée sans concurrence, une décision ordinaire coûte en moyenne **0,262 ms** et une décision avec recherche **13,692 ms** ; le maximum de recherche observé est **20,471 ms**. Les 47 tests couvrent notamment le moteur, les bords, les invariants, les transformations, les délais expirés, la fin de partie et le redémarrage. Le rendu Pygame a également été contrôlé.

Les mesures détaillées, les latences et leurs limites sont dans [BENCHMARK.md](BENCHMARK.md), les essais intermédiaires dans [benchmarks/ABLATIONS.md](benchmarks/ABLATIONS.md) et les résultats par graine dans [benchmark-results.json](benchmark-results.json).

## Installer, lancer et vérifier

Depuis `chou/algo`, avec Python 3.13 :

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
.venv/bin/python serpent-algo.py
```

Le code utilise la bibliothèque standard et Pygame. NumPy reste présent dans l’environnement déclaré, mais l’agent ne l’utilise pas. Espace redémarre après la fin de partie ; `--manual` active le contrôle au clavier.

```sh
.venv/bin/python serpent-algo.py --seed 15
.venv/bin/python serpent-algo.py --policy explored64 --seed 15
.venv/bin/python -m unittest discover -v
.venv/bin/python serpent-algo.py --benchmark --policies explored64,all128 --seeds 1:21 --output benchmarks/recheck-regression.json
.venv/bin/python serpent-algo.py --benchmark --policies explored64,all128 --seeds 1001:1101 --workers 4 --output benchmarks/recheck-heldout.json
```

La borne finale des plages de graines est exclue : `1:21` correspond à 1–20. Utiliser `--workers 1` pour mesurer les latences sans concurrence. `--time-limit-ms 0` désactive seulement l’échéance de recherche, pas le plafond de propositions ni la cadence du jeu ; cette option permet les comparaisons déterministes au prix d’un calcul potentiellement plus long.
