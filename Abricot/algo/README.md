# Groupe ABRICOT — Snake algorithmique

Livrable : `snake-algo.py`. Aucun apprentissage : chaque coup est calculé.

```bash
python3 snake-algo.py play                          # notre algo, à la vitesse du jeu
python3 snake-algo.py play --algo cycle --trace     # voir le cycle suivi
python3 snake-algo.py play --speed 60               # même partie, affichée plus vite
python3 snake-algo.py bench --games 30              # compare les quatre algos
```

Le script se relance tout seul avec `Abricot/.venv/` si pygame manque au Python du système.

## Le point de départ : le jeu est un tore

`move()` fait `% GRID_SIZE` : le serpent traverse les bords (voir `../ia/AGENTS.md` §6bis).
Pour l'algorithmique, c'est décisif. Sur une grille à murs 15×15, **aucun cycle hamiltonien
n'existe** : 225 cases, nombre impair, graphe biparti. Sur le tore, il en existe un très simple :
14 pas à droite, 1 pas en bas, répété 15 fois. Le suivre remplit la grille à coup sûr. La position
de départ du serpent est déjà sur ce cycle.

## Résultats — 30 parties par algorithme, graines 0 à 29

| Algo | Victoires | Score moyen | Pas moyens | Pas par pomme | Durée à 5 pas/s |
|---|---|---|---|---|---|
| `dijkstra` | 0/30 | 59,8 | 538 | 9,0 | 1,8 min |
| `glouton` | 0/30 | 64,9 | 604 | 9,3 | 2,0 min |
| `cycle` | **30/30** | 223 | 12 514 | 56,1 | 41,7 min |
| **`tapsell`** | **30/30** | **223** | **6 803** | 30,5 | **22,7 min** |

223 pommes = grille pleine = victoire. Temps de calcul : moins de 3 ms par coup dans tous les cas.

Dijkstra et le glouton sont rapides par pomme mais meurent vers 60 pommes : ils foncent sur la
pomme sans se soucier de ce qui reste derrière. Le cycle ne peut pas perdre mais fait le tour
complet de la grille pour chaque pomme. Notre algorithme combine les deux.

## Notre algorithme : cycle hamiltonien + raccourcis de Tapsell

On numérote les cases dans l'ordre du cycle. Suivre le cycle, c'est avancer de +1. Un raccourci,
c'est aller sur une case voisine qui porte un numéro plus loin devant. Il est permis si :

- la case est libre ;
- il **ne dépasse pas la pomme** (sinon il faut refaire le tour) ;
- il **ne rattrape pas la queue**, avec 2 cases de marge parce que le serpent grandit un pas après
  avoir mangé.

Avec ces règles, le corps reste toujours rangé dans l'ordre du cycle : devant la tête, jusqu'à la
queue, tout est libre. Le serpent peut donc toujours reprendre le cycle à +1 sans se mordre.
**La victoire reste garantie**, et la partie est deux fois plus courte.

Les raccourcis sont coupés quand le serpent occupe plus de la moitié de la grille. Mesuré : 6 803
pas avec cette coupure, 10 235 sans. Hypothèse, non vérifiée : les sauts tardifs tassent le corps
juste derrière la tête, et plus aucun saut n'est possible ensuite pendant des centaines de pas.

Méthode décrite par John Tapsell : <https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/>

## Timeline

23/09 : Constat : le tore rend un cycle hamiltonien possible sur 15×15, ce qui est impossible sur
la grille à murs. Vérification par programme du cycle hélicoïdal (225 cases, chaque pas entre voisins).

23/09 : Étude de faisabilité en simulation, 30 parties par stratégie : cycle strict 12 514 pas,
cycle + raccourcis 6 803 (coupés à 50 %) ou 10 235 (toujours actifs), glouton avec contrôle de la
queue 0 victoire sur 30.

23/09 : Essai d'un solveur « certifié » : aller au plus court vers la pomme seulement si l'on sait
construire un nouveau cycle hamiltonien contenant le corps d'arrivée. Échec : la construction
(Warnsdorff + retour arrière borné) réussit rarement, le gain est nul, et une partie sur trois s'est
terminée par une mort à 221 pommes, donc la méthode telle qu'écrite a un bug. Abandonnée pour l'instant.

23/09 20:13 : `snake-algo.py` : les classes `Snake` et `Apple` du jeu de base sont importées telles
quelles, la boucle de `main()` est rejouée coup pour coup. Les mesures sur le vrai jeu retombent au
pas près sur celles de la simulation (12 514 et 6 803), ce qui valide la simulation.

## Suite

Déformer le cycle en cours de partie au lieu de le garder fixe, pour réduire le nombre de pas sans
perdre la garantie de victoire.
