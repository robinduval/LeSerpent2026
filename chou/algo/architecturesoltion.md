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

## Transformation supplémentaire : 4-opt

Le voisinage libre inclut `A+B+C+D+E → A+D+C+B+E`. Les blocs B, C et D sont non vides et entièrement libres. Les quatre raccordements A→D, D→C, C→B et B→E doivent être voisins sur le tore. La permutation conserve un unique cycle de 225 cases, et le suffixe contenant le corps reste intact. `_bridges()` calcule le nouveau rang de la pomme, `_apply()` prépare la permutation et `_certify()` confirme l’invariant complet.

## Recherche retenue : `bridge_uphill`

Chaque mouvement commence par les améliorations locales 2-opt, Or-opt et 4-opt. Au départ et après chaque pomme, quatre branches explorent plusieurs cycles avec des empreintes normalisées sur la tête. Les propositions déjà visitées sont écartées et le meilleur cycle rencontré reste disponible, départ inclus.

La recherche autorise des détours **virtuels** jusqu’à huit rangs au-delà du meilleur cycle, puis reprend les descentes locales. Seul le meilleur cycle est finalement adopté ; les déplacements réels restent strictement progressifs. Quatre branches partagent 256 propositions.

L’échéance coopérative de 20 ms borne les recherches en plus du plafond de travail. Une opération déjà commencée peut dépasser légèrement cette échéance. La recherche retourne une continuation certifiée ; elle ne modifie jamais le jeu pendant ses explorations. Le cycle finalement adopté ne doit pas augmenter la distance à la pomme, et le mouvement exécuté doit la réduire strictement. Cette quantité entière décroissante exclut une boucle sans consommation.

Le hasard exploratoire dépend uniquement de l’état public et reste indépendant de celui des pommes. Le plafond de travail seul est reproductible ; l’échéance murale peut modifier le point d’arrêt selon la charge de la machine.

## Compromis et résultats mesurés

L’ancien `all128`, `explored64` et `fixed` restent disponibles comme comparateurs. Le 4-opt est le principal changement utile du sprint : il ouvre des permutations que 2-opt et Or-opt seuls atteignent difficilement. Les variantes d’anticipation et les autres voisinages testés n’ont pas donné un gain assez net sur le petit lot de réglage. Le détail des choix se trouve dans [les ablations](benchmarks/sprint10/ABLATIONS.md).

Après réglage sur 201–203 et gel des paramètres, le lot réservé 2001–2100 donne **100/100 parties à 223**, **0 collision(s)** et **0 interruption(s)**. Le gagnant réalise **3 626,32 déplacements moyens**, soit **725,26 secondes x1**, avec **25,55 %** de déplacements en moins que `all128` rejoué sur ce même lot.

La cible de 4 500 mouvements est atteinte ; la cible de 3 000 mouvements / 600 secondes est **non atteinte**. La meilleure partie du gagnant parmi régression et lot réservé est la **seed 2010 : 223 points, 3152 mouvements, 630,4 secondes x1**. Ces durées sont calculées depuis les simulations accélérées, pas mesurées dans l’interface.

Le **record toutes variantes** est de **612,2 secondes x1** : `bridge128`, graine **2081**, **223 points en 3 061 déplacements**, sans collision ni interruption. Cette variante reste disponible avec `--policy bridge128`, mais `bridge_uphill` est retenu pour sa meilleure moyenne. Les deux records et leur source sont mis en évidence dans [BENCHMARK.md](BENCHMARK.md).

La sécurité impose un corps contigu dans un cycle complet et écarte donc certains trajets potentiellement plus courts. L’optimisation porte sur la pomme actuelle ; elle ne prouve pas l’optimalité d’une partie entière. Le calcul supplémentaire reste borné, mais les latences sous six processus concurrents doivent être distinguées du fonctionnement interactif.

La mesure complémentaire sans concurrence, sur deux graines, donne **0,437 ms** par décision ordinaire et **18,306 ms** par recherche en moyenne ; le maximum de recherche observé est **21,455 ms**. L’échéance est coopérative, et ces observations locales ne sont pas une garantie de latence sur toute machine.

Les **53 tests** vérifient moteur, croissance, bords, fin et redémarrage ainsi que les transformations, leurs rangs et les arrêts sûrs. Les chiffres complets et les limites de reproductibilité figurent dans [BENCHMARK.md](BENCHMARK.md) et [heldout.json](benchmarks/sprint10/heldout.json).

## Installer, lancer et vérifier

Depuis `chou/algo`, avec Python 3.13 :

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
.venv/bin/python serpent-algo.py
```

Le code utilise la bibliothèque standard et Pygame. NumPy reste présent dans l’environnement déclaré, mais l’agent ne l’utilise pas. Espace redémarre après la fin de partie ; `--manual` active le contrôle au clavier.

```sh
.venv/bin/python serpent-algo.py --seed 2010
.venv/bin/python serpent-algo.py --policy bridge128 --seed 2081
.venv/bin/python serpent-algo.py --policy all128 --seed 15
.venv/bin/python -m unittest discover -v
.venv/bin/python serpent-algo.py --benchmark --policies all128,bridge128,bridge_uphill --seeds 1:21 --output benchmarks/recheck-regression.json
.venv/bin/python serpent-algo.py --benchmark --policies all128,bridge128,bridge_uphill --seeds 2001:2101 --workers 6 --output benchmarks/recheck-heldout.json
```

La borne finale des plages de graines est exclue : `1:21` correspond à 1–20. Utiliser `--workers 1` pour mesurer les latences sans concurrence. `--time-limit-ms 0` désactive seulement l’échéance de recherche, pas le plafond de propositions ni la cadence du jeu ; cette option permet les comparaisons déterministes au prix d’un calcul potentiellement plus long.
