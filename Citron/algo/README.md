# Citron — version algorithmique

## Lancement

Depuis `Citron/algo/` :

```sh
python -m pip install -r requirements.txt
python snake-algo.py
```

Le jeu joue seul à 5 Hz. Échap termine.
Le moteur est importé depuis `../ia/serpent-algo.py` : conserver le dossier
Citron complet. Aucun import de Torch ni modèle nécessaire.

## Méthode v1

Cycle hamiltonien déterministe sur la grille torique 15×15 : avancer à droite
14 fois, descendre, recommencer avec une origine décalée d'une colonne.
Après 15 lignes, retour au départ. Les 225 cases sont visitées une fois.
Le cycle contient le corps initial dans son ordre de déplacement.
Suivre ses successeurs maintient cet ordre et évite de croiser le corps.
Les passages entre bords sont de véritables arêtes du graphe.
Pas de raccourcis dans cette baseline : la sûreté prime sur la rapidité.

Le déplacement, l'interdiction de demi-tour, la collision et la croissance
différée utilisent le moteur original. La pomme est tirée uniformément parmi
les cases libres avec un générateur local reproductible (`--seed 42`).
La fin et le score suivent le moteur : avec sa croissance différée, le score
peut atteindre 223 au remplissage, et ne vaut pas toujours longueur moins 3.

## Instruments

Même rendu que la version RL : appels directs à `draw_grid`, `Apple.draw` et
`Snake.draw` du moteur original, mêmes dimensions et couleurs, sans surimpression.
Le plateau reste inchangé ; à sa droite, panneau
score, longueur/occupation, durée réelle, ratio score/durée réelle, mouvements,
traversées des bords, pas sans pomme, distance sur le cycle jusqu'à la pomme,
pas par pomme, temps de calcul de décision et courbe score/temps réel.

Chaque lancement écrit `runs/<session>/config.json`, `steps.csv` et
`metrics.json` (dernier état, actualisé à chaque mouvement). Horloge de mesure :
`time.perf_counter()`. Aucun temps simulé présenté comme temps réel.
Les parties interrompues ont `completed=false`, les victoires/collisions
`completed=true`. À la fin naturelle, le chronomètre reste figé et la fenêtre
reste ouverte. La prochaine amélioration se comparera à cette baseline sur
les mêmes seeds : score, complétion, pas/pomme et durée réelle à 5 Hz.

## Vérification

```sh
python -m unittest discover -s . -v
python snake-algo.py --max-steps 10
```

Les tests appellent directement les transitions, sans attente de clock : ce
sont des vérifications logiques, pas des résultats chronométrés de jeu.
`--max-steps` est uniquement une limite de diagnostic, jamais une victoire.

## Batterie reproductible

`python benchmark.py --games 100 --workers 8 --seed 1000`

100 environnements indépendants répartis sur 8 processus, entrelacés dans chaque
processus. Même Game et mêmes transitions que la fenêtre, sans attente ni rendu.
La clock de la démonstration reste inchangée à 5 Hz. `runs/benchmark-*/results.json`
contient chaque partie, les seeds, les étapes aux seuils 10/50/100/200, le résumé
et l'empreinte du code. Temps de calcul mesuré pour le lot ; durée théorique à
5 Hz = mouvements / 5, explicitement distincte d'une durée réellement observée.
La limite de 51 000 mouvements marque une troncature, jamais une victoire.

## Variante BFS torique sécurisée

`python snake-algo.py --algorithm bfs-safe`

Recherche en largeur (BFS) sur les voisins modulo 15, sous contrainte d'indices
strictement croissants sur le cycle, sans dépasser la pomme ou la queue.
Deux indices et la croissance en attente sont réservés avant la queue.
Si la pomme n'est pas accessible dans ce graphe contraint, avancer vers le point
atteignable le plus avancé sans dépasser la pomme ; sinon suivre le cycle.
Seul le premier mouvement est exécuté, puis on recalcule. Cette variante reste
hybride : elle utilise le cycle comme invariant de sécurité et repli.
Ce n'est ni un BFS libre, ni une preuve d'optimalité globale.

Sources primaires :
- https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/
- https://github.com/chynl/snake#algorithms

Comparaison reproductible :

```sh
python benchmark.py --algorithm hamiltonian --seed 1000
python benchmark.py --algorithm bfs-safe --seed 1000
python -m pip install -r requirements-analysis.txt
python plot_comparison.py runs/<baseline>/results.json runs/<bfs>/results.json
```

100 seeds identiques ne donnent pas les mêmes positions de pommes après divergence
des corps : le tirage dépend des cases libres. Les résultats concernent cette
batterie et ne démontrent pas une supériorité universelle. La baseline reste
accessible via `--algorithm hamiltonian`. Le rendu et les règles n'ont pas changé.

## Raccourcis gloutons (optimisés)

`python snake-algo.py --algorithm shortcut`

C'est désormais le choix par défaut de `python snake-algo.py`.

Correction du biais directionnel : la variante `shortcut` utilise maintenant un
cycle bidirectionnel obtenu par échanges 2-opt sur des paires de lignes disjointes.
Le corps initial et les 225 cases sont conservés. Le cycle historique est gardé
pour les modes `hamiltonian` et `bfs-safe`. Les compteurs des quatre directions
sont affichés et exportés. Les raccourcis sont désactivés dès 112 segments pour
résorber les trous avant la fin de partie et éviter les pièges de croissance
consécutive. Validation : 100/100 victoires sur seeds 1000–1099 et 2000–2099 ;
6,00 s/pomme théoriques à 5 Hz sur la première batterie. Ce résultat appartient
à la combinaison nouveau cycle + seuil, pas au seul changement de direction.

Évalue les actions légales et choisit celle qui avance le plus sur le cycle,
sans dépasser la pomme ni la queue et en réservant de la place pour la croissance.
Contrairement au BFS, ne cherche pas le chemin le plus court vers la pomme :
réduit directement les détours du cycle. Le moteur et le rendu restent inchangés.
La première variante anticipant le déplacement de queue a été rejetée (3 collisions
sur 100). La version retenue garde une réserve conservatrice avant la queue actuelle.

## Timeline

- 23/09 : baseline cycle torique, affichage PyGame, métriques CSV/JSON et tests.
- 23/09 : restauration du rendu original et vérification pixel par pixel.
- 23/09 : comparaison du cycle et du BFS sécurisé sur 100 seeds.
- 23/09 : ajout des raccourcis gloutons puis correction du biais directionnel
  avec un cycle bidirectionnel et un retour au cycle seul dès 112 segments.
- 23/09 : validation sur deux batteries de 100 parties ; 200 victoires.

## Auteurs

`AUTHORS` contient les auteurs au format `NOM Prénom login`, avec les logins
`gaspard.jolly` et `valentin.oison`.
