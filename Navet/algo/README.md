# Navet — Étape 2 : agent algorithmique (Dijkstra + flood-fill, puis Hamilton)

Journal des décisions : horodaté, une entrée par décision/processus.
Convention (validée) : une découverte ou un plan n'est écrit ici QUE lorsqu'il est
validé, pas avant.

Fichiers :
- `player.py` — le cerveau algorithmique (réutilise `ia/game.py` tel quel)
- `main.py` — visualisation pygame (touches 1/2/3/4 = x1/x5/x20/max, espace = pause,
  c = afficher le cycle)
- `sweep.py` — benchmark parallèle du seuil de switch, CSV dans `results/`

## Journal

### 2026-09-23 — 19:52 — début du prompt engineering (Étape 2)
Requête du client (traduite de l'anglais) :

> Reprendre la logique du dossier `ia`, où nous avons implémenté l'apprentissage par
> renforcement pour jouer au snake, interagir de la même façon, et avoir aussi un
> `main.py` pour lancer le test visuellement — mais cette fois c'est un algorithme qui
> joue. Voici ce qu'il nous faut :
> 1. Un **algorithme de Dijkstra** pour trouver le chemin le plus rapide — il doit
>    prendre en compte les **distances toriques**, pas seulement les distances en ligne
>    droite.
> 2. Un **flood fill** pour éviter de choisir les options qui mettraient le serpent en
>    mauvaise position.
> 3. Une idée innovante : à un moment donné, il faut un **switch** de l'algorithme
>    Dijkstra + flood-fill vers un **simple Hamilton** — bien sûr, Hamilton en cours de
>    partie est difficile car le serpent a déjà pris des chemins aléatoires. Si vous avez
>    un plan pour que ça marche, pour bien finir la partie comme un **speedrunner**, ce
>    serait parfait.

(L'oubli de préciser ce journal est signalé et corrigé dans cette même entrée ;
les horodatages des décisions à partir de maintenant seront consignés à la volée.)

### 2026-09-23 — 20:03 — plan technique validé
- Fichiers dans `algo/`, `ia/game.py` réutilisé sans modification (aucune logique de jeu
  dupliquée) ; même interface que le RL : action relative one-hot
  [tout droit, droite, gauche] vers `play_step()`.
- Phase 1 : Dijkstra torique vers la pomme ; flood-fill de sécurité sur le serpent
  virtuel après le chemin — rejet si la queue n'est plus atteignable ou si l'espace
  atteignable < longueur. Filet : suivre sa queue.
- Phase 2 : switch vers un cycle hamiltonien fixe avec raccourcis.
- Réponses du client : le critère de switch sera choisi par benchmark (plusieurs valeurs,
  plusieurs parties, en parallèle, rapport des meilleurs paramètres) ; `main.py` =
  pygame + stats finales ; accélération x5 et x20 demandée pour observer l'algorithme.

### 2026-09-23 — vers 20:05 — recentrage demandé par le client
Un seul paramètre benchmarké : le seuil de remplissage qui déclenche le switch Hamilton.
Retiré : le critère « N rejets consécutifs du flood-fill ». La marge des raccourcis
Hamilton est figée à 4 (non benchmarkée).

### 2026-09-23 — vers 20:05 — choix techniques pendant l'implémentation
- **Cycle en hélice** : un boustrophédon colonne par colonne ne se referme pas sur une
  grille impaire (15×15). Sur le tore, on descend chaque colonne en partant une ligne
  plus haut que la précédente ; après 15 colonnes le décalage revient à 0 et le cycle
  se ferme.
- **Obstacles temporels** : le segment i du corps se libère au pas L − i (+1 si
  croissance en cours). Dijkstra et le flood-fill acceptent une case du corps si on y
  arrive après sa libération — généralisation de « la queue exclue si elle va se
  libérer ».
- **Pas de faim** : le jeu de base (`serpent-algo.py`) n'en a pas ; elle est neutralisée
  dans `game.py` via ses paramètres (`hunger_k` énorme). Nos scripts arrêtent une partie
  après 2 250 pas (10 × 225) sans pomme : cause « boucle » (abaissé à 675 à 21:15).

### 2026-09-23 — vers 20:08 — correction du plan : entrée dans le cycle
- **Erreur du plan corrigée** : les raccourcis ne sont garantis que si le corps est rangé
  dans l'ordre du cycle, pas « même avec un corps en désordre ». Ajout d'un test exact :
  suivre le cycle depuis la tête est sûr si chaque segment situé devant sur le cycle est
  libéré avant que la tête y arrive.
- **Constat** : laisser Dijkstra reprendre la main pendant la transition remélange le
  corps (aucune convergence à 50 %) ; minimiser le « retard » du corps sur le cycle fait
  éviter les pommes et tourner en boucle.
- **Solution retenue** : toutes les cases sont sur le cycle, la question est *où* y
  entrer. Dijkstra vers toutes les cases, on retient la plus proche où le serpent virtuel
  peut ensuite suivre le cycle sans se mordre, et on s'engage sur ce chemin. Sans entrée
  sûre, Dijkstra + flood-fill continue et on réessaie au pas suivant.

### 2026-09-23 — 20:14 — benchmark du seuil de switch Hamilton
30 parties par seuil (seeds 1000..1029, identiques pour chaque seuil), 226 s sur 10 cœurs.
CSV : `results/sweep_20260923_2014.csv`.

| seuil | victoires | pas moyens (victoires) | défaites |
|---|---|---|---|
| 30 % | 100 % | 10 127 | – |
| 25 % | 100 % | 10 439 | – |
| 20 % | 100 % | 10 598 | – |
| 0 % (Hamilton dès le départ) | 100 % | 11 068 | – |
| 35 % | 90 % | 9 773 | 3 |
| 40 % | 60 % | 9 465 | 12 |
| 45 % | 33 % | 9 166 | 20 |
| 50 % | 7 % | 8 567 | 28 |
| ≥ 60 % et « jamais » | 0 % | – | 30 (meurt vers 214/223) |

- 30 % est le seuil le plus rapide parmi ceux qui gagnent toutes les parties.
- Au-delà, un switch plus tardif donne des victoires plus courtes, mais le corps en
  désordre rejoint le cycle de moins en moins souvent ; à partir de 60 %, jamais.
- Dijkstra + flood-fill seul ne gagne aucune partie.

### 2026-09-23 — 20:19 — retour du client sur le cycle
Le cycle en hélice laisse des bandes vides qui font perdre beaucoup de temps en fin de
partie, et un switch à 30 % est jugé trop tôt. Question ouverte : proposer de meilleures
alternatives.

### 2026-09-23 — 20:21 — plan validé : cycle construit à partir du corps
- Au switch, on cherche un chemin tête → toutes les cases libres → queue. Corps + chemin
  = cycle hamiltonien où le corps est déjà rangé : plus de transition, switch possible
  tard (60–80 %). Sans solution (recherche bornée), Dijkstra + flood-fill continue et on
  réessaie.
- Contre les bandes vides : le cycle est reconstruit après chaque pomme, en passant le
  plus tôt possible par la nouvelle pomme.
- Mesure : même `sweep.py`, seuils de 40 à 90 %, comparé à la référence (hélice fixe,
  30 %, 10 127 pas). L'hélice fixe reste disponible pour la comparaison.

### 2026-09-23 — 20:24 — le plan de 20:21 est infaisable tel quel
Un chemin tête → toutes les cases libres → queue n'existe presque jamais après une phase
Dijkstra : l'espace libre est coupé en plusieurs morceaux (13/20 parties à 30 %, 20/20 à
50 %) ; cycle construit dans 0/20 parties de 30 à 80 %.
Bug de recherche corrigé : sur une grille vide, viser la pomme en premier pouvait bloquer
la recherche ; un deuxième essai en Warnsdorff pur a été ajouté. Hélice rétablie comme
mode par défaut.

### 2026-09-23 — 20:25 — décision : option A (hélice pour entrer, puis cycle reconstruit)
- Le switch reste celui de l'hélice (transition par point d'entrée).
- Une fois dans le cycle, après chaque nouvelle pomme, on reconstruit le cycle à partir
  du corps (tête → cases libres → queue) en visant la pomme au plus tôt ; on garde le
  nouveau cycle seulement s'il amène à la pomme plus vite que l'actuel.
- Raccourcis désactivés dans ce mode : le corps reste collé le long du cycle, donc le
  cycle en cours est toujours une solution et la reconstruction reste possible.
- Objectif : supprimer les bandes vides. Le switch tôt n'est pas traité (option B,
  construction tenant compte du temps, gardée pour plus tard).
- Mesure : `sweep.py` sur les mêmes seeds, comparé à la référence hélice (30 %, 10 127 pas).

### 2026-09-23 — 20:35 — benchmark de l'option A
Mêmes seeds 1000..1029. Au seuil de 30 %, victoire en 4 894 pas en moyenne contre 10 127
avec l'hélice fixe (−52 %), 100 % de victoires. Les gains vont de −32 % à −57 % selon le
seuil. Victoires et défaites inchangées pour chaque seuil : les défaites viennent de
l'entrée dans l'hélice au-delà de 35 %, un problème qu'A ne traite pas.
CSV : `results/sweep_reconstruit_20260923_2035.csv`.

### 2026-09-23 — 20:40 — décision : essayer B + déclencher le switch juste avant d'être coincé
- **B, construction qui tient compte du temps** : cycle hamiltonien de toute la grille
  qui peut passer par des cases du corps, à condition d'y arriver après leur libération
  (segment i à une position ≥ L − i + croissance). Plus besoin que l'espace libre soit
  d'un seul tenant : les poches s'ouvrent quand la queue avance.
- **Déclenchement « juste avant d'être coincé »** (idée du client) : tant qu'on joue
  Dijkstra, on vérifie qu'un cycle B existe encore depuis l'état suivant ; le jour où il
  n'existe plus, on switche sur le cycle trouvé au pas précédent.
- Première étape : mesurer la faisabilité de B selon le remplissage (même mesure que 20:24).

### 2026-09-23 — 20:50 — nouveau critère du client
La victoire garantie n'est pas exigée : il suffit qu'environ **50 % des parties au moins**
atteignent 223. Parmi les réglages qui y parviennent, on retient le plus rapide.

### 2026-09-23 — 20:45 — benchmark de l'option B
Seuils 30 à 90 %, seeds 1000..1029. B entre dans le cycle à tous les seuils ; 80 à 83 %
de victoires de 40 à 80 %, 47 % à 90 %. Plus lent qu'A : 5 145 pas au mieux (seuil 70 %),
contre 4 112 pour A à 40 %. Défaites : morsures quand une pomme imprévue rend le cycle
dangereux. Fausses alertes du « piège » corrigées (40 relances avant de conclure).
CSV : `results/sweep_temporel_20260923_2045.csv`.

### 2026-09-23 — 20:55 — décision : switch piloté par les îlots
Le vrai obstacle au switch n'est pas le remplissage mais la **fragmentation** de l'espace
libre (îlots, impasses). Plan validé :
1. **Switch juste avant la fragmentation** : à chaque pas, un flood-fill vérifie si le coup
   prévu couperait l'espace libre en plusieurs îlots ou créerait une impasse ; si oui, on
   switche maintenant avec le cycle d'A (corps encore d'un tenant). Plus de % à deviner.
2. **Dijkstra qui évite les îlots** : parmi les coups sûrs, on écarte ceux qui fragmentent
   l'espace libre, pour faire reculer le moment du switch.
3. **Mesure** : remplissage au moment du switch, avec et sans le filtre 2, sur les mêmes
   seeds ; critère inchangé (≥ 50 % de victoires, puis la vitesse).

### 2026-09-23 — 21:08 — décision : Dijkstra prudent + seuil tardif
Retour à un seuil fixe, mais tardif (40 à 80 %). Avant le seuil, Dijkstra « prudent » :
parmi les coups sûrs, on préfère ceux qui gardent l'espace libre d'un seul tenant (sinon
coup Dijkstra normal, sans switch). Au seuil : cycle d'A s'il existe, sinon cycle B, sinon
nouvel essai quelques pas plus tard. Après le switch : reconstructions d'A (B tant que le
corps n'est pas collé au cycle). Un seul sweep, 30 seeds.

### 2026-09-23 — 21:15 — parties qui tournent en rond arrêtées plus tôt
Une seule partie (seuil 80 %, seed 1007) prenait 81 s sur les 301 s du sweep de 21:10 :
serpent bloqué en mode secours, une recherche de cycle B à chaque pas.
- Limite « boucle » abaissée de 2 250 à 675 pas sans pomme (3 × 225) ; écart maximal
  mesuré entre deux pommes dans des parties gagnées : 214 pas.
- En secours, la recherche de cycle B n'est relancée qu'au plus tous les 10 pas.
Seed 1007 : 81 s → 10,6 s. Parties gagnées testées (seeds 1000, 1001) inchangées.

### 2026-09-23 — 21:25 — où part le temps
Prudent 60 %, seeds 1000..1009, 10/10 victoires, 3 525 pas. Phase 1 : 14,6 pas par pomme
pour 11,3 au plus court, surcoût 432 pas par partie. Phase 2 (cycles) : 17,5 pas par pomme
pour 7,5 au plus court, surcoût 903 pas par partie, dont 436 juste après le switch
(60–70 %, cycles B). Priorité : construire les cycles à partir du plus court chemin vers la
pomme. Mesure rapide (~2 s) : `regret.py`.

### 2026-09-23 — 21:25 — décision : piste 1, cycles construits depuis le plus court chemin
Le cycle (A comme B) commence par un plus court chemin Dijkstra vers la pomme (plusieurs
variantes tirées au hasard parmi les plus courts), puis est complété de la pomme jusqu'à la
fermeture. En cas d'échec, on revient à la recherche actuelle. Évaluation avec `regret.py`,
sweep complet seulement pour confirmer.

### 2026-09-23 — 21:33 — piste 1 : pas de gain, cause trouvée
Construire les cycles depuis le plus court chemin (avec petits détours et relances) ne
change rien : 3 571 pas contre 3 525 sur 10 seeds. Le serpent joue exactement le cycle
annoncé. 264 pommes sur 270 de la phase 2 utilisent un cycle d'A, qui doit parcourir toutes
les cases libres avant la queue ; l'espace libre devenant un couloir, le cycle annonce
16,5 pas pour un plus court de 7,0. Reconstruire toujours en B : 4/10 victoires, 5 362 pas.

### 2026-09-23 — 21:33 — décision : on ne garde que le meilleur
- Piste 1 retirée du code (retour à la version à 3 525 pas).
- `main.py` suit toujours le meilleur réglage mesuré.
- À explorer : raccourcis dans le cycle d'A (sauter un bout du couloir quand le test de
  sécurité exact l'autorise), puis choix du cycle qui laisse l'espace libre le plus large.

### 2026-09-23 — 21:35 — raccourcis dans le cycle d'A : négatif, retirés
Mesure `regret.py` (prudent 60 %, 10 seeds ; référence 10/10, 3 525 pas). Raccourcis de
marge 1, 2, 4, 8 avec reconstruction en A d'abord : 6 à 9/10 victoires, 5 267 à 6 171 pas.
Sans A d'abord (marge 4) : 8/10, 5 864 pas. Après un raccourci, le corps laisse des trous ;
le cycle d'A se reconstruit mal et la phase 2 passe de 17 à 40–47 pas par pomme.
Reconstruction en A d'abord seule : 3 520 pas (dans le bruit). Les deux options sont retirées.

### 2026-09-23 — 21:37 — décision : régler la phase 1 et le seuil
Tolérance T du filtre anti-îlots (0, 5, 10, 20 : les îlots bordés par les T derniers
segments de la queue ne comptent pas) × seuils 55, 60, 65 %. Tri rapide avec `regret.py`
(10 seeds), confirmation des meilleurs sur 30 seeds.

### 2026-09-23 — 21:39 — tolérance 5 retenue (gain modeste)
Tri sur 10 seeds (seuils 55/60/65 % × tol 0/5/10/20) : écarts dans le bruit (± 100 pas).
Confirmation, moyenne sur les victoires :
- 30 seeds : tol 0 → 30/30, 3 606 pas ; tol 5 → 29/30, 3 566 ; tol 10 → 29/30, 3 626 ;
  seuil 65 % tol 5 → 29/30, 3 583.
- 100 seeds, seuil 60 % : tol 0 → 98/100, 3 603 pas ; **tol 5 → 99/100, 3 571 pas**.
Retenu : prudent, seuil 60 %, tol 5 (défaut de `main.py`). Gain d'environ 1 % : la
phase 1 n'est pas le levier principal. `regret.py` donne désormais la moyenne sur les
victoires (comme `sweep.py`).

### 2026-09-23 — 21:40 — décision : choisir le cycle qui laisse l'espace libre le plus large
À chaque reconstruction en A, plusieurs cycles candidats (égalités départagées au hasard).
Largeur = nombre moyen de voisins libres par case libre, après avoir suivi le cycle jusqu'à
la pomme (≈ 2 pour un couloir, → 4 pour un espace large). Parmi les candidats qui
atteignent la pomme au plus tard « meilleur + s » pas, on prend le plus large ; s = 0, 2, 4.
Tri avec `regret.py`, confirmation sur 100 seeds.

### 2026-09-23 — 21:42 — cycle le plus large : −2,5 %, retenu
Moyenne sur les victoires, prudent 60 %, tol 5. Référence (100 seeds) : 99/100, 3 571 pas.
- 10 seeds : 4 candidats avec s = 0, 2, 4 et 8 candidats avec s = 2 → 3 563 à 3 581 pas,
  contre 3 660 sans choix.
- 100 seeds : **k = 4, s = 2 → 99/100, 3 480 pas (−91, −2,5 %)** ; k = 4, s = 0 → 3 500 ;
  k = 4, s = 4 → 3 483 ; k = 8, s = 2 → 3 485.
Retenu : 4 candidats, 2 pas de plus tolérés (défauts de `main.py`). Environ 11:36 de jeu.
