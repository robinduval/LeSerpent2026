# Groupe ABRICOT — Snake par algorithme, sans apprentissage
Livrable du 22/09 : `snake-algo.py` (cycle hamiltonien recomposé à chaque pas). Jeu de base : `serpent-algo.py`, copie de celui du prof dont seule l'injection de prompt a été retirée (voir `../ia/AGENTS.md` §5).
## Timeline
20:01 : Constat de départ : `move()` fait `% GRID_SIZE`, la grille est un tore (`../ia/AGENTS.md` §6bis). Sur une grille à murs 15x15, aucun cycle hamiltonien n'existe (225 cases, nombre impair, graphe biparti). Sur le tore, il en existe un : 14 pas à droite, 1 en bas, 15 fois. Vérifié par programme, et la position de départ du serpent est déjà dessus. Le suivre remplit la grille à coup sûr.
20:01 : Étude de faisabilité en simulation, 30 parties par stratégie. Cycle strict : 30/30 victoires en 12 514 pas. Cycle + raccourcis de Tapsell : 6 803 pas s'ils sont coupés à 50 % de remplissage, 10 235 s'ils restent actifs. Plus court chemin + queue joignable : 0/30, mort vers 162 pommes.
20:04 : Premier essai de solveur « certifié » : n'aller au plus court vers la pomme que si l'on sait construire un nouveau cycle contenant le corps d'arrivée (Warnsdorff + retour arrière borné). Échec : construction réussie une fois sur 500, aucun gain, et une mort à 221 pommes qui trahit un bug. Abandonné.
20:04 : Avis critique demandé à Gemini. Recommandation retenue : déformer le cycle localement au lieu de le reconstruire. Son estimation de 1 800 à 2 800 pas n'est pas sourcée.
20:11 : Page de démonstration interactive (cycle strict contre raccourcis, chaque décision expliquée) pour rendre la règle de Tapsell lisible.
20:16 : Première livraison, algo `tapsell`. Les classes `Snake` et `Apple` du jeu de base sont importées telles quelles et la boucle de `main()` est rejouée coup pour coup. Mesures sur le vrai jeu identiques au pas près à la simulation (12 514 et 6 803) : la simulation est validée.
20:17 : Déformation par échanges 2x2 : deux arêtes parallèles de même sens dans un carré sont remplacées par les deux autres côtés, le morceau entre elles est parcouru à l'envers. Avec les raccourcis : 5 926 pas contre 6 861 sur 10 parties, soit -14 %. Gain réel mais faible : un échange isolé rapproche rarement la pomme.
20:19 : Mesure de ce qui reste à gagner : en prenant toujours le plus court chemin vers chaque pomme, il faudrait ~1 800 pas. La perte est maximale entre 20 et 60 % de remplissage (25 à 50 pas par pomme contre ~10).
20:21 : Recomposition par rotations de Pósa : à chaque pas, plus court chemin vers la pomme, puis parcours de toutes les autres cases libres. 5 869 pas au mieux, car la recherche échoue souvent.
20:24 : Bug de garantie trouvé en laissant les raccourcis actifs jusqu'au bout : mort à 224 cases. Cause : un raccourci laisse des cases libres derrière la tête ; si deux pommes tombent ensuite juste devant elle, la queue n'avance pas (elle reste en place un pas après chaque pomme) et la tête la rattrape.
20:26 : Conséquence pour `tapsell` : même coupés à 50 %, les raccourcis laissent une marge de 3 cases seulement en début de partie. Aucune mort sur 30 parties, mais le risque n'est pas nul. La première livraison annonçait une victoire garantie : c'était faux au sens strict, corrigé ici.
20:27 : Diagnostic des échecs de la recomposition : une fois sur deux, le plus court chemin vers la pomme coupe la zone libre en deux, et plus aucun parcours ne peut passer par toutes les cases.
20:28 : Correction : parmi les chemins vers la pomme (jusqu'à 6 pas de détour), on prend le premier qui longe les obstacles et laisse la zone libre d'un seul tenant. 3 940 pas sur 5 parties, sans aucun raccourci, donc sans case sautée.
21:03 : Intégration dans `snake-algo.py` sous le nom `recompose`, algo par défaut. Sur le vrai jeu, 30 parties : 30/30 victoires en 3 983 pas en moyenne (13,3 min au rythme du jeu), contre 6 803 pour `tapsell` et 12 514 pour le cycle strict. Temps de calcul au pire : 418 ms pour un pas, soit plus que les 200 ms d'un pas à 5 pas/s ; l'affichage prend alors ponctuellement du retard, la partie n'est pas affectée.
## Résultats
Métrique : **toutes les pommes d'abord, puis le moins de pas possible**. Le temps est le temps *de jeu* (GAME_SPEED = 5 pas/s), jamais le temps de calcul. Jeu torique 15x15, scoring inchangé. 223 pommes = grille pleine = victoire.
### Les cinq algorithmes — 30 parties chacun, graines 0 à 29
| Algo | Victoires | Score moyen | Pas moyens | Pas par pomme | Durée à 5 pas/s | Calcul max par pas |
|---|---|---|---|---|---|---|
| **`recompose`** | **30/30** | **223** | **3 983** | **17,9** | **13,3 min** | 418 ms |
| `tapsell` | 30/30 | 223 | 6 803 | 30,5 | 22,7 min | 0,5 ms |
| `cycle` | 30/30 | 223 | 12 514 | 56,1 | 41,7 min | 0,5 ms |
| `dijkstra` | 0/30 | 59,8 | 538 | 9,0 | 1,8 min | 1,6 ms |
| `glouton` | 0/30 | 64,9 | 604 | 9,3 | 2,0 min | 1,6 ms |

Dijkstra et le glouton sont rapides par pomme mais meurent vers 60 pommes : ils foncent sans se soucier de ce qu'ils laissent derrière eux. Le cycle strict ne peut pas perdre mais refait presque le tour de la grille pour chaque pomme. `recompose` garde la garantie du cycle et divise son nombre de pas par 3,1.
### Algorithme retenu : `recompose`
Le cycle se découpe en deux arcs : le corps, de la queue à la tête, auquel on ne touche jamais ; et les cases libres devant la tête. N'importe quel ordre de ces cases libres qui part d'une voisine de la tête et finit sur une voisine de la queue forme encore un cycle hamiltonien contenant le corps. Le suivre gagne à coup sûr. À chaque pas, on choisit donc l'ordre où la pomme arrive le plus tôt :
1. **Un chemin vers la pomme qui ne coupe pas la zone libre.** Le plus court d'abord, puis jusqu'à 6 pas de détour, en longeant les obstacles.
2. **Un parcours de toutes les autres cases libres qui revient à la queue**, par extension gloutonne (règle de Warnsdorff) et rotations de Pósa quand l'extension bloque.
3. **Une vérification complète du nouveau cycle avant de l'adopter.** Si la recherche échoue, on garde l'ancien : la garantie ne dépend jamais de la réussite de la recherche.

Aucune case n'est jamais sautée, contrairement aux raccourcis de Tapsell : la victoire est certaine, pas seulement très probable.
### Pourquoi pas les raccourcis de Tapsell
`tapsell` (<https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/>) saute en avant dans l'ordre du cycle, sans dépasser la pomme ni rattraper la queue. Deux fois plus rapide que le cycle strict, mais les cases sautées restent derrière la tête : tant qu'il en reste, une série de pommes juste devant elle peut lui faire rattraper sa queue. Observé une fois avec les raccourcis actifs jusqu'au bout ; jamais observé avec la coupure à 50 %, mais non exclu. `recompose` fait mieux sur les deux tableaux : moins de pas, et une garantie stricte.
### Ce qui n'a pas été atteint
La borne basse grossière (plus court chemin vers chaque pomme, sans se soucier de la suite) est de ~1 800 pas. `recompose` en fait un peu plus du double (3 983) : la zone libre ne se laisse pas toujours parcourir entièrement après le plus court chemin, et le détour de 6 pas n'y suffit pas toujours.
### Reproduire
```bash
python3 snake-algo.py                               # démonstration : recompose, à la vitesse du jeu (~14 min)
python3 snake-algo.py play --speed 60 --trace       # même partie, affichée plus vite, cycle dessiné
python3 snake-algo.py play --algo tapsell
python3 snake-algo.py bench --games 30              # les cinq algorithmes, ~20 min
```
Le script se relance tout seul avec `Abricot/.venv/` si pygame manque au Python du système.
