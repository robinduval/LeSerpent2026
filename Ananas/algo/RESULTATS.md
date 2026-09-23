# Résultats des parties

Grille 15×15 torique, jeu à 5 déplacements par seconde (`clock.tick(5)`).
Maximum possible : **223 pommes** (grille pleine).

- **Graine** : nombre qui fixe le hasard du jeu, donc l'endroit où apparaissent les pommes. Même graine = même partie rejouable à l'identique ; graine différente = autre partie.
- **Dernière pomme** : temps de jeu (horloge du jeu) au moment de la dernière pomme mangée.
- **Fin** : `collision` (le serpent s'est mordu), `victoire` (grille pleine) ou `boucle` (le serpent tourne sans fin sans pouvoir manger ; le vrai jeu ne s'arrêterait jamais).

Chaque partie lancée avec `python3 snake-algo.py` ajoute automatiquement une ligne en bas du tableau « Parties » (`interrompue` = fenêtre fermée avant la fin).

## Moyennes (run 1, graines 0 à 9)

| Algo | Pommes moyennes | Min | Max | Collisions | Boucles | Victoires |
|---|---|---|---|---|---|---|
| BFS v1 | 135,4 | 100 | 162 | 2 | 8 | 0 |
| SafePath | 211,5 | 177 | 223 | 4 | 5 | 1 |

## Moyennes (run 6, graines 0 à 49, 50 parties par algo)

| Algo | Pommes moyennes | Min | Max | Victoires | Collisions | Boucles | Record (223 pommes) |
|---|---|---|---|---|---|---|---|
| SafePath | 215,0 | 176 | 223 | 4 (8 %) | 26 | 20 | **21:47** (graine 20) |
| SafePath v2 | 212,2 | 134 | 223 | 2 (4 %) | 31 | 17 | 22:52 (graine 22) |

Conclusion du run 6 : SafePath v2 (détours + anti-boucle profond) est moins bon que SafePath et ses décisions montent à 841 ms (> 1 tick). On garde SafePath.

## Moyennes (run 7, graines 0 à 49) : cycle hamiltonien + raccourcis

| Réglage (raccourcis tant que le serpent occupe moins de…) | Victoires | Temps moyen pour 223 | Record |
|---|---|---|---|
| 50 % de la grille | **50/50** | **22:19** | **20:53** (graine 47) |
| 75 % | 50/50 | 26:31 | 22:40 (graine 20) |
| 100 % | 49/50 (1 collision) | 35:24 | 32:15 (graine 20) |

Conclusion du run 7 : victoire à chaque partie avec le réglage 50 %, plus rapide en moyenne que les rares victoires de SafePath. Les lignes « Parties » du run 7 correspondent au réglage 50 %.

## Moyennes (run 8, graines 0 à 49) : réglage du cycle + raccourcis

Toutes les variantes ci-dessous gagnent 50/50 sauf mention. Temps = moyenne du temps de jeu pour atteindre 223.

| Variante | Seuil raccourcis | Marge | Temps moyen | Record |
|---|---|---|---|---|
| zigzag | 30 % | 3 | 25:46 | 23:01 |
| zigzag | 40 % | 1 | 23:13 | **19:58** (graine 25) |
| **zigzag** | **50 %** | **1** | **22:10** | 20:19 (graine 25) |
| zigzag | 60 % | 1 | 22:46 | 20:26 |
| hélice torique | 50 % | 1 | 22:16 | 20:19 |
| hélice torique | 80 % | 1 | 28:07 | 24:19 |
| zigzag + petits raccourcis tardifs (≤ 2 cases) | 50 % | 1 | 33:08, **47/50** | 28:20 |

Conclusions du run 8 :
- réglage retenu (par défaut dans `snake-algo.py`) : zigzag, raccourcis jusqu'à 50 % de remplissage, marge 1 → 50/50 victoires, 22:10 en moyenne ;
- l'hélice torique n'apporte rien ; autoriser des raccourcis plus tard ralentit (trous laissés derrière la tête) et fait perdre des parties.

## Parties

| Run | Heure | Algo | Graine | Pommes | Dernière pomme | Fin |
|---|---|---|---|---|---|---|
| 1 | — | BFS v1 | 0 | 128 | — | boucle |
| 1 | — | BFS v1 | 1 | 129 | — | boucle |
| 1 | — | BFS v1 | 2 | 121 | — | collision |
| 1 | — | BFS v1 | 3 | 158 | — | boucle |
| 1 | — | BFS v1 | 4 | 149 | — | boucle |
| 1 | — | BFS v1 | 5 | 159 | — | boucle |
| 1 | — | BFS v1 | 6 | 134 | — | collision |
| 1 | — | BFS v1 | 7 | 162 | — | boucle |
| 1 | — | BFS v1 | 8 | 100 | — | boucle |
| 1 | — | BFS v1 | 9 | 114 | — | boucle |
| 1 | — | SafePath | 0 | 219 | — | boucle |
| 1 | — | SafePath | 1 | 219 | — | collision |
| 1 | — | SafePath | 2 | 192 | — | collision |
| 1 | — | SafePath | 3 | 204 | — | boucle |
| 1 | — | SafePath | 4 | 220 | — | boucle |
| 1 | — | SafePath | 5 | 221 | — | collision |
| 1 | — | SafePath | 6 | 219 | — | boucle |
| 1 | — | SafePath | 7 | 223 | — | victoire |
| 1 | — | SafePath | 8 | 221 | — | boucle |
| 1 | — | SafePath | 9 | 177 | — | collision |
| 2 | — | BFS v1 | 0 | 128 | 5:54 | boucle |
| 2 | — | BFS v1 | 1 | 129 | 6:34 | boucle |
| 3 | — | SafePath | 0 | 219 | 21:37 | boucle |
| 3 | — | SafePath | 1 | 219 | 18:45 | collision |
| 4 | — | SafePath | 216833 | 6 | 0:10 | interrompue |
| 5 | 20:56 | SafePath | 983666 | 70 | 2:10 | interrompue |
| 6 | 21:00 | SafePath | 0 | 219 | 21:37 | boucle |
| 6 | 21:00 | SafePath | 1 | 219 | 18:45 | collision |
| 6 | 21:00 | SafePath | 2 | 192 | 13:35 | collision |
| 6 | 21:00 | SafePath | 3 | 204 | 14:25 | boucle |
| 6 | 21:00 | SafePath | 4 | 220 | 21:52 | boucle |
| 6 | 21:00 | SafePath | 5 | 221 | 18:13 | collision |
| 6 | 21:00 | SafePath | 6 | 219 | 20:12 | boucle |
| 6 | 21:00 | SafePath | 7 | 223 | 28:06 | victoire |
| 6 | 21:00 | SafePath | 8 | 221 | 21:17 | boucle |
| 6 | 21:00 | SafePath | 9 | 177 | 10:20 | collision |
| 6 | 21:00 | SafePath | 10 | 221 | 18:42 | collision |
| 6 | 21:00 | SafePath | 11 | 214 | 17:42 | collision |
| 6 | 21:00 | SafePath | 12 | 223 | 35:33 | victoire |
| 6 | 21:00 | SafePath | 13 | 182 | 11:34 | collision |
| 6 | 21:00 | SafePath | 14 | 218 | 17:41 | boucle |
| 6 | 21:00 | SafePath | 15 | 217 | 20:51 | boucle |
| 6 | 21:00 | SafePath | 16 | 217 | 20:57 | boucle |
| 6 | 21:00 | SafePath | 17 | 220 | 21:15 | collision |
| 6 | 21:00 | SafePath | 18 | 219 | 19:21 | collision |
| 6 | 21:00 | SafePath | 19 | 220 | 17:25 | boucle |
| 6 | 21:00 | SafePath | 20 | 223 | 21:47 | victoire |
| 6 | 21:00 | SafePath | 21 | 217 | 19:27 | boucle |
| 6 | 21:00 | SafePath | 22 | 221 | 19:39 | boucle |
| 6 | 21:00 | SafePath | 23 | 212 | 16:25 | collision |
| 6 | 21:00 | SafePath | 24 | 221 | 25:18 | boucle |
| 6 | 21:00 | SafePath | 25 | 220 | 18:40 | boucle |
| 6 | 21:00 | SafePath | 26 | 220 | 25:25 | boucle |
| 6 | 21:00 | SafePath | 27 | 215 | 18:30 | collision |
| 6 | 21:00 | SafePath | 28 | 220 | 20:22 | collision |
| 6 | 21:00 | SafePath | 29 | 220 | 21:21 | boucle |
| 6 | 21:00 | SafePath | 30 | 221 | 20:15 | collision |
| 6 | 21:00 | SafePath | 31 | 217 | 15:55 | collision |
| 6 | 21:00 | SafePath | 32 | 195 | 14:02 | collision |
| 6 | 21:00 | SafePath | 33 | 217 | 20:34 | boucle |
| 6 | 21:00 | SafePath | 34 | 176 | 11:09 | collision |
| 6 | 21:00 | SafePath | 35 | 219 | 42:15 | collision |
| 6 | 21:00 | SafePath | 36 | 213 | 18:40 | collision |
| 6 | 21:00 | SafePath | 37 | 217 | 21:56 | boucle |
| 6 | 21:00 | SafePath | 38 | 221 | 20:59 | collision |
| 6 | 21:00 | SafePath | 39 | 222 | 22:09 | collision |
| 6 | 21:00 | SafePath | 40 | 223 | 22:42 | victoire |
| 6 | 21:00 | SafePath | 41 | 218 | 18:29 | collision |
| 6 | 21:00 | SafePath | 42 | 222 | 21:40 | collision |
| 6 | 21:00 | SafePath | 43 | 206 | 17:14 | collision |
| 6 | 21:00 | SafePath | 44 | 218 | 17:55 | boucle |
| 6 | 21:00 | SafePath | 45 | 221 | 19:21 | boucle |
| 6 | 21:00 | SafePath | 46 | 213 | 17:31 | collision |
| 6 | 21:00 | SafePath | 47 | 221 | 24:21 | collision |
| 6 | 21:00 | SafePath | 48 | 221 | 19:46 | boucle |
| 6 | 21:00 | SafePath | 49 | 216 | 18:36 | collision |
| 6 | 21:00 | SafePath v2 | 0 | 218 | 18:39 | collision |
| 6 | 21:00 | SafePath v2 | 1 | 219 | 18:39 | boucle |
| 6 | 21:00 | SafePath v2 | 2 | 204 | 15:14 | collision |
| 6 | 21:00 | SafePath v2 | 3 | 206 | 15:02 | collision |
| 6 | 21:00 | SafePath v2 | 4 | 222 | 19:59 | collision |
| 6 | 21:00 | SafePath v2 | 5 | 220 | 18:13 | collision |
| 6 | 21:00 | SafePath v2 | 6 | 221 | 20:32 | boucle |
| 6 | 21:00 | SafePath v2 | 7 | 222 | 20:15 | collision |
| 6 | 21:00 | SafePath v2 | 8 | 207 | 17:01 | collision |
| 6 | 21:00 | SafePath v2 | 9 | 219 | 20:42 | boucle |
| 6 | 21:00 | SafePath v2 | 10 | 219 | 19:47 | collision |
| 6 | 21:00 | SafePath v2 | 11 | 219 | 20:04 | collision |
| 6 | 21:00 | SafePath v2 | 12 | 213 | 20:40 | boucle |
| 6 | 21:00 | SafePath v2 | 13 | 219 | 18:33 | boucle |
| 6 | 21:00 | SafePath v2 | 14 | 220 | 19:45 | boucle |
| 6 | 21:00 | SafePath v2 | 15 | 220 | 18:51 | boucle |
| 6 | 21:00 | SafePath v2 | 16 | 219 | 34:40 | collision |
| 6 | 21:00 | SafePath v2 | 17 | 219 | 19:22 | boucle |
| 6 | 21:00 | SafePath v2 | 18 | 213 | 16:11 | collision |
| 6 | 21:00 | SafePath v2 | 19 | 218 | 22:12 | boucle |
| 6 | 21:00 | SafePath v2 | 20 | 215 | 19:49 | collision |
| 6 | 21:00 | SafePath v2 | 21 | 219 | 20:00 | collision |
| 6 | 21:00 | SafePath v2 | 22 | 223 | 22:52 | victoire |
| 6 | 21:00 | SafePath v2 | 23 | 221 | 26:46 | collision |
| 6 | 21:00 | SafePath v2 | 24 | 220 | 29:40 | boucle |
| 6 | 21:00 | SafePath v2 | 25 | 221 | 27:28 | collision |
| 6 | 21:00 | SafePath v2 | 26 | 201 | 15:34 | collision |
| 6 | 21:00 | SafePath v2 | 27 | 213 | 19:12 | boucle |
| 6 | 21:00 | SafePath v2 | 28 | 214 | 19:23 | collision |
| 6 | 21:00 | SafePath v2 | 29 | 219 | 19:31 | boucle |
| 6 | 21:00 | SafePath v2 | 30 | 220 | 20:02 | collision |
| 6 | 21:00 | SafePath v2 | 31 | 222 | 18:58 | collision |
| 6 | 21:00 | SafePath v2 | 32 | 218 | 18:24 | boucle |
| 6 | 21:00 | SafePath v2 | 33 | 220 | 19:52 | boucle |
| 6 | 21:00 | SafePath v2 | 34 | 170 | 10:35 | collision |
| 6 | 21:00 | SafePath v2 | 35 | 219 | 21:25 | collision |
| 6 | 21:00 | SafePath v2 | 36 | 223 | 27:09 | victoire |
| 6 | 21:00 | SafePath v2 | 37 | 212 | 17:10 | collision |
| 6 | 21:00 | SafePath v2 | 38 | 212 | 20:05 | collision |
| 6 | 21:00 | SafePath v2 | 39 | 215 | 16:46 | boucle |
| 6 | 21:00 | SafePath v2 | 40 | 218 | 18:58 | collision |
| 6 | 21:00 | SafePath v2 | 41 | 220 | 20:26 | collision |
| 6 | 21:00 | SafePath v2 | 42 | 134 | 6:24 | collision |
| 6 | 21:00 | SafePath v2 | 43 | 159 | 8:51 | collision |
| 6 | 21:00 | SafePath v2 | 44 | 210 | 16:38 | collision |
| 6 | 21:00 | SafePath v2 | 45 | 210 | 19:34 | collision |
| 6 | 21:00 | SafePath v2 | 46 | 219 | 18:02 | boucle |
| 6 | 21:00 | SafePath v2 | 47 | 220 | 17:45 | collision |
| 6 | 21:00 | SafePath v2 | 48 | 218 | 20:56 | boucle |
| 6 | 21:00 | SafePath v2 | 49 | 169 | 11:05 | collision |
| 7 | 21:03 | Cycle+raccourcis | 0 | 223 | 21:30 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 1 | 223 | 21:43 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 2 | 223 | 22:51 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 3 | 223 | 23:12 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 4 | 223 | 21:53 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 5 | 223 | 23:05 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 6 | 223 | 21:02 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 7 | 223 | 23:06 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 8 | 223 | 22:16 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 9 | 223 | 22:53 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 10 | 223 | 22:20 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 11 | 223 | 22:27 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 12 | 223 | 23:44 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 13 | 223 | 22:12 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 14 | 223 | 22:14 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 15 | 223 | 22:30 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 16 | 223 | 23:15 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 17 | 223 | 22:49 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 18 | 223 | 21:26 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 19 | 223 | 22:38 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 20 | 223 | 22:18 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 21 | 223 | 22:16 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 22 | 223 | 22:10 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 23 | 223 | 20:54 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 24 | 223 | 21:58 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 25 | 223 | 22:04 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 26 | 223 | 22:50 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 27 | 223 | 21:39 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 28 | 223 | 22:41 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 29 | 223 | 23:00 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 30 | 223 | 23:33 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 31 | 223 | 23:05 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 32 | 223 | 22:38 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 33 | 223 | 23:24 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 34 | 223 | 21:30 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 35 | 223 | 22:25 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 36 | 223 | 21:23 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 37 | 223 | 22:58 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 38 | 223 | 22:43 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 39 | 223 | 21:10 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 40 | 223 | 24:23 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 41 | 223 | 22:20 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 42 | 223 | 21:15 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 43 | 223 | 21:49 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 44 | 223 | 22:29 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 45 | 223 | 21:52 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 46 | 223 | 22:04 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 47 | 223 | 20:53 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 48 | 223 | 22:02 | victoire |
| 7 | 21:03 | Cycle+raccourcis | 49 | 223 | 20:56 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 0 | 223 | 22:12 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 1 | 223 | 23:20 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 2 | 223 | 21:23 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 3 | 223 | 22:53 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 4 | 223 | 21:11 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 5 | 223 | 22:13 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 6 | 223 | 21:50 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 7 | 223 | 20:48 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 8 | 223 | 23:01 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 9 | 223 | 22:17 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 10 | 223 | 22:07 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 11 | 223 | 20:54 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 12 | 223 | 22:19 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 13 | 223 | 22:44 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 14 | 223 | 22:06 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 15 | 223 | 23:11 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 16 | 223 | 22:23 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 17 | 223 | 22:11 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 18 | 223 | 21:37 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 19 | 223 | 22:48 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 20 | 223 | 22:53 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 21 | 223 | 21:46 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 22 | 223 | 22:16 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 23 | 223 | 21:08 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 24 | 223 | 22:37 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 25 | 223 | 20:19 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 26 | 223 | 24:07 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 27 | 223 | 21:50 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 28 | 223 | 22:46 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 29 | 223 | 21:28 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 30 | 223 | 24:09 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 31 | 223 | 21:44 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 32 | 223 | 22:13 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 33 | 223 | 21:22 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 34 | 223 | 22:42 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 35 | 223 | 22:17 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 36 | 223 | 22:57 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 37 | 223 | 23:20 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 38 | 223 | 23:21 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 39 | 223 | 22:43 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 40 | 223 | 22:32 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 41 | 223 | 20:46 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 42 | 223 | 22:38 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 43 | 223 | 20:56 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 44 | 223 | 21:56 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 45 | 223 | 21:50 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 46 | 223 | 21:42 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 47 | 223 | 22:00 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 48 | 223 | 21:12 | victoire |
| 8 | 21:06 | Cycle+raccourcis | 49 | 223 | 21:34 | victoire |
| 9 | 21:15 | Circuit dynamique | 1 | 223 | 13:38 | victoire |
| 10 | 21:18 | Cycle+raccourcis | 762596 | 223 | 23:12 | victoire |
| 10 | 21:19 | Circuit dynamique | 1 | 90 | 5:26 | interrompue |
| 11 | 21:20 | Circuit dynamique | 674917 | 11 | 0:29 | interrompue |
| 12 | 21:22 | Circuit dynamique | 240141 | 223 | 14:12 | victoire |
| 13 | 21:24 | Circuit dynamique | 50475 | 5 | 0:15 | interrompue |
| 14 | 21:29 | Circuit dynamique | 579411 | 223 | 14:43 | victoire |
| 15 | 21:41 | Circuit dynamique | 293321 | 223 | 12:01 | victoire |
