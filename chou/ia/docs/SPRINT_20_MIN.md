# Sprint demandé le 16 septembre 2026, 21:14–21:34 (Paris)

Objectif : rapprocher le score de 223 dans vingt minutes de développement. L'entraînement accéléré et les récompenses libres ont été autorisés ; les essais de qualification conservent `pygame.time.Clock().tick(5)`, une transition par tick et les règles initiales. Le budget de développement ne devient pas un timeout du jeu.

## Trois pistes effectivement implémentées

1. **Double DQN + filtre local `tail2`** : le réseau existant est entraîné sur les actions réellement exécutées. Le filtre élimine les collisions immédiates, certains pièges au mouvement suivant et préfère conserver un accès à la queue. Le réseau départage les actions restantes. Un flood fill ne prouve pas la complétion ; cette variante peut encore perdre.
2. **Double DQN + cycle hamiltonien réorganisable** : un réseau à valeurs d'action choisit entre les déplacements admis par `RewiredCycleShield`. Le filtre conserve un cycle couvrant les 225 cases et contenant le corps comme un segment contigu. Il autorise le réseau à inverser des arcs libres via une transformation 2-opt. Ce système est explicitement hybride : sa sécurité vient du filtre programmé, ses choix entre possibilités viennent des poids appris.
3. **Réseau seul** : les mêmes poids que la première piste sont évalués sans filtre pour mesurer sa contribution. Une distillation pendant l'entraînement est également explorée ; les résultats de cette piste ne sont pas confondus avec ceux du système filtré.

Les anciens poids et leur sélection sont conservés dans `checkpoints/selected_v1_pure.pt` et `checkpoints/selection_v1.json`. Le rapport antérieur reste dans `RESULTS_V1.md`. Le modèle par défaut et les mesures finales sont indiqués dans `RESULTS.md`.

## Pourquoi le cycle réorganisable traite la croissance différée

Au reset, le corps suit les trois cases consécutives d'un cycle torique de 225 cases. Le déplacement normal suit son successeur. Une réorganisation ne touche qu'un arc de cases libres et reconnecte ses deux extrémités par des arêtes voisines ; le cycle reste unique, couvre toutes les cases et conserve toutes les arêtes du corps. Après le mouvement, le corps reste donc contigu, que la queue soit retirée ou conservée pour la croissance en attente.

Le successeur reste accessible tant que le moteur n'a pas terminé. Les autres déplacements admis doivent aussi diminuer strictement la distance dirigée vers la pomme dans le cycle résultant. Une pomme est donc atteinte en au plus 224 déplacements. Cela exclut une survie indéfinie sans progression. Cet argument dépend des invariants implémentés, de l'état initial et des règles présentes ; il ne s'applique pas à un corps arbitraire ou à un autre plateau.

Des tests vérifient l'adjacence du cycle, la contiguïté du corps, la pureté des observations, des suites de choix admissibles aléatoires et 223 pommes consécutives imposées comme **fixture de test**. Ces fixtures ne sont pas des parties classables. L'ancien masque à rang fixe a révélé des ensembles d'actions vides : il reste expérimental, sans garantie de complétion.

## Mesures et séparation des données

- `runs/safety_sprint_tail2/training.jsonl` contient les épisodes d'apprentissage, avec mises à jour actives : aucun temps n'est un temps officiel.
- `runs/safety_sprint_tail2_validation/` contient trois essais indépendants du checkpoint figé, graines 880001–880003, chacun cadencé à 5 Hz. Le budget annoncé de 4 500 déplacements est une interruption expérimentale externe, pas une défaite et pas une victoire.
- `runs/safety_sprint_pure_88010*/` contient l'ablation réseau seul à 5 Hz sur d'autres graines.
- `runs/hybrid_rewired/` contient l'entraînement du modèle avec cycle réorganisable. Les épisodes d'apprentissage complets à 223 sont distingués des essais à poids figés.
- `runs/sprint_rewired_5hz_890001/` qualifie à 5 Hz un checkpoint figé de 100 000 transitions, avec une borne externe de 2 500 pas.

Les processus d'évaluation peuvent tourner simultanément, mais chaque partie a sa propre horloge à 5 Hz. Les poids et le générateur de pommes sont propres à chaque processus. Aucun réglage ne consulte les prochains tirages. Des vérifications fonctionnelles accélérées éventuellement présentes dans les répertoires ne remplacent pas ces essais de qualification et ne fournissent aucun temps officiel.

## Recherche complémentaire réellement consultée

[John Tapsell, *Nokia 6110 Part 3 – Algorithms*](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/), consulté le 16 septembre 2026. L'auteur explique comment un cycle hamiltonien et l'ordre tête/corps/queue permettent des raccourcis, à condition de réserver la place nécessaire à la croissance. Il limite les raccourcis dans son implémentation. Notre croissance différée exige une analyse distincte ; le cycle contigu et les inversions d'arcs libres sont une implémentation locale, et non une garantie importée de cet article. Aucun score externe n'est comparé à nos points.

## Recevabilité

Le cours demande du RL. Les deux variantes hybrides comportent de vraies mises à jour Double DQN et un rôle mesurable des poids dans le choix des actions, mais la logique de sécurité est programmée. L'autorisation donnée pour l'entraînement n'établit pas à elle seule l'acceptation d'un filtre pendant l'évaluation ; une question explicite a été posée. Les variantes et leurs checkpoints restent identifiables séparément.
