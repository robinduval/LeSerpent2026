# Groupe Melon — Snake ALGO (cycle hamiltonien dynamique)

**Auteurs** : LAMECHE Nazim, DE-SOUSA Basile

## Installation (Python 3)

```bash
python -m pip install --upgrade pip
python -m pip install pygame
```

## Utilisation

```bash
python snake-algo.py                                  # partie visible, pilotée par l'algo, horloge du socle (5 coups/s)
python snake-algo.py bench --games 100 --seed 0       # parties sans affichage avec les vraies classes Snake/Apple
python snake-algo.py --planner hamilton               # ancienne version (cycle fixe + raccourcis), pour comparer
```

`play` affiche le score, le temps réel et le remplissage, et imprime en console, à la fin de la partie :
score, nombre de déplacements, temps réel et ratio score/temps.
`bench` donne les victoires, les morts, le score moyen et les déplacements (moyenne, min, max).

## Architecture

- **Socle** : les lignes 1 à 186 sont une copie à l'identique de `serpent-algo.py` (grille 15×15, horloge, score et règles non modifiés).
  La boucle `main()` est celle du socle ; seule la lecture du clavier est remplacée par `planner.next_direction(...)`.
- **Planner** : `VirtualPlanPlanner` (par défaut). Il est déterministe : aucun aléa, et les budgets sont des nombres d'étapes, jamais un chronomètre.

### Le tore
La grille du socle est **torique** (`% GRID_SIZE` dans `move()`) : le serpent réapparaît de l'autre côté, et seule la morsure tue.
Sur une grille 15×15 avec murs (225 cases, nombre impair), aucun cycle hamiltonien n'existe. Sur le tore, si : c'est tout le point de départ.

### L'algorithme en 4 étapes
1. **Un circuit qui passe par toutes les cases** (cycle hamiltonien). Au départ, c'est l'« hélice » : 14 pas à droite, puis 1 pas en bas.
2. **Règle de sécurité (certificat)** : le corps occupe toujours un morceau *contigu* du circuit, et toutes les cases vides sont
   devant la tête. Suivre le circuit ne peut donc jamais tuer le serpent, **quelle que soit la place des pommes**.
3. **À chaque coup, on redessine le circuit** pour que la pomme arrive le plus tôt possible :
   - on calcule la distance BFS *d* entre la tête et la pomme. Si le circuit y mène déjà en *d* coups, on avance ;
   - sinon, on cherche un chemin tête → pomme de longueur *d*, *d*+1, *d*+2, *d*+4, *d*+8, puis *d*+16, sans jamais dépasser le trajet actuel.
     On vérifie ensuite qu'après la pomme, on peut encore passer par **toutes** les cases vides et revenir à la queue
     (DFS avec heuristique de Warnsdorff, élagage des culs-de-sac et test de connexité) ;
   - **plan virtuel** : la pomme ne bouge pas tant qu'on ne l'a pas mangée. On peut donc viser un chemin qui passe par des cases
     que la queue aura libérées à temps (BFS temporel), en exigeant le certificat à l'arrivée ;
   - si rien n'est trouvé dans le budget (1 500 étapes par coup), on garde le circuit actuel, qui est toujours sûr, et on réessaie au coup suivant.
4. **On joue la case suivante du circuit.**

**Pas de seuil de remplissage.** L'ancienne version coupait les raccourcis sous 50 % de cases vides, parce qu'ils laissaient des trous
derrière la tête. Ici, il n'y a jamais de trou. Quand la grille se remplit, les raccourcis deviennent impossibles d'eux-mêmes, et le serpent suit le circuit.

**Garantie de victoire (preuve par récurrence)** : le corps de départ est contigu dans l'hélice. Jouer la case suivante garde le corps
contigu. Cette case est toujours vide, et jamais le cou, donc jamais un demi-tour. Chaque nouveau circuit est vérifié par assertion
(mêmes cases, voisins consécutifs). La pomme est donc toujours atteinte : **victoire à chaque partie, score 223.**

### Score maximal = 223
`grow()` compte le point immédiatement, et la victoire est déclarée quand `relocate()` ne trouve plus de case libre.
La dernière pomme compte donc un point sans faire grandir le serpent : 225 − 3 + 1 = **223**.

## Résultats

Mesures sur les **mêmes seeds** (comparaison appariée), avec les vraies classes du socle :

| | Ancienne version (cycle fixe) | **Cycle dynamique (retenu)** |
|---|---|---|
| Victoires / morts (500 parties) | 500 / 0 | **500 / 0** |
| Déplacements moyens (500 parties) | 6 672 | **3 174 (−52 %)** |
| Déplacements par pomme | 30 | **14** |
| Temps réel moyen (100 parties chronométrées) | ≈ 22 min 34 s | **10 min 43 s** |
| Meilleur temps réel (100 parties) | ≈ 19 min 50 s | **9 min 30 s** |
| Ratio score/temps moyen | ≈ 0,165 pomme/s | **≈ 0,348 pomme/s** |

Le temps réel a été chronométré sur la vraie boucle `main()` du socle, avec l'affichage, en accéléré.
L'horloge `clock.tick(5)` mesurée fait 202 ms par image (et non 200). Le calcul + dessin le plus lent a pris 58 ms :
le serpent avance toujours au rythme de l'horloge. Les temps de l'ancienne version sont extrapolés
(déplacements × 202 ms, mêmes 100 seeds). Le détail est dans `../experiments/RAPPORT.md`.

**Choix de la métrique** : le classement se fait au ratio score/temps. En théorie, ce ratio favorise un serpent qui meurt tôt avec un bon début de partie.
Nous avons choisi de viser la **grille pleine** (score maximal de 223) et de réduire le temps pour y arriver.
Notre ratio (≈ 0,35 pomme/s) est obtenu sans aucune mort.

---

## Timeline

```
19:50 benchmark des choix algos
19:53 Décision de garantir la victoire à chaque partir
19:55 choix d'un algo: Circuit Hamiltonien avec raccourcis et code
20:00 optimisation du code avec l'absence des walls
20:05 Test du code
20:20 Réponse au formulaire score 223 en 22min
20:21 Recherche d'optimisations
20:23 Tests de toutes les options d'optimisations
21:30 Fin des test et choix de "cycle hamiltonien dynamique à planification virtuelle"
21:31 Lancement de tests de Seeds pour trouver la moyenne et le best
21:45 réponse finale au formulaire avec best 9min07 et moyenne à 10min

```

## Pistes testées et écartées (détail dans `../experiments/RAPPORT.md`)
- **Seuil adaptatif selon les trous** : de −32 % à −84 %. Il réactive les raccourcis en fin de partie, là où ils coûtent le plus.
- **Raccourcis qui limitent les trous** et **autre cycle torique (peigne)** : ±0 %. Le goulot n'est pas la forme du cycle, mais le fait qu'il soit fixe.
- **Vérification par simulation** au lieu des marges : +0,6 % seulement. Avec une marge trop faible, 1 mort sur 200 parties : sur un cycle fixe
  avec des trous, une suite de pommes mal placées peut tuer le serpent. Le cycle dynamique n'a pas ce défaut.
- **BFS/A\* glouton seul** : meurt dans 500 parties sur 500 (à 28 % de remplissage en moyenne).

## Pistes d'amélioration
- Entre 40 et 80 % de remplissage, environ 19 déplacements par pomme, contre une distance BFS d'environ 10. Le plus court chemin y est souvent
  *prouvé* impossible à refermer en circuit. On pourrait choisir le chemin après la pomme de façon à préparer la pomme suivante
  (garder une zone vide compacte).
