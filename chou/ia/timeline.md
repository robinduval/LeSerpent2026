# Notes d’expérience — Snake avec Reinforcement Learning

**Équipe : Hamza Hannat et Leadro Tolaini**  
**Date : 16 septembre 2026 — horaires de Paris**

Notre objectif est d’obtenir le score officiel maximal, puis de réduire le temps uniquement à score égal. Nous n’optimisons ni le ratio score/temps ni une somme pondérée de ces deux critères.

Les créneaux ci-dessous reprennent les horaires indicatifs fournis par l’équipe. Le sprint de 21h14 à 21h34 et les résultats chiffrés sont documentés dans les journaux du projet. Les observations d’apprentissage sont distinguées des évaluations à poids figés et à 5 Hz.

## Déroulement

### 20h00 — Analyse du sujet et recherche

Lecture des consignes et étude des méthodes existantes : Double DQN, Prioritized Experience Replay, retours multi-step, apprentissage par démonstrations et filtres de sécurité. AlphaSnake est étudié comme piste de recherche, sans reproduire son infrastructure dans notre solution.

L’audit du moteur identifie une grille torique de **15 × 15**, sans murs mortels, une croissance différée d’un déplacement et un maximum exact de **223 points**. L’ordre de croissance et de retrait de la queue doit être respecté pour éviter de déclarer sûrs des mouvements qui ne le sont pas.

### 20h15 — Première piste PyTorch et cycle hamiltonien

Les notes de l’équipe proposent d’associer un modèle PyTorch à un cycle hamiltonien pour privilégier la complétion. Un tel cycle visite toutes les cases du plateau ; le suivre fournit une base de sécurité, mais peut imposer de longs détours avant chaque pomme.

L’implémentation distingue ensuite deux approches : un **Double DQN pur**, qui choisit ses actions à partir d’une représentation du corps, et un **hybride**, dans lequel un filtre programmé restreint les actions disponibles avant le choix du réseau. La sécurité du cycle ne doit pas être attribuée à l’apprentissage seul.

### 20h30 — Priorité à la complétion, premiers constats sur la lenteur

Le constat rapporté par l’équipe est qu’un parcours structuré autour d’un cycle favorise le score maximal, au prix d’un temps de partie élevé. Cette observation motive la recherche de raccourcis.

Les expériences documentées par la suite confirment des complétions à **223 en apprentissage** pour l’hybride hamiltonien. Elles ne démontrent pas une réussite systématique d’un modèle figé en évaluation complète à 5 Hz.

### 20h45 — Raccourcis, représentation de l’état et variantes d’apprentissage

Comparaison d’un état pédagogique à onze informations avec une représentation enrichie décrivant l’ordre du corps, la pomme, les dangers et la croissance en attente. Expérimentation de Double DQN, de replay prioritaire et de retours multi-step, séparément puis en combinaison.

La recherche sur les chemins et les cycles conduit ensuite à deux extensions : un filtre local fondé sur l’espace accessible et l’accès à la queue, puis un cycle réorganisable par transformations **2-opt**. Dans ce dernier cas, les réorganisations conservent les arêtes du corps et ne modifient que des arcs libres. AlphaSnake reste une référence étudiée, pas un algorithme implémenté sous ce nom.

### 21h05–21h10 — Première évaluation réservée du RL pur

Le modèle sélectionné est un Double DQN entraîné sur **1 000 000 de transitions**. Sur cinq graines réservées, à 5 Hz, il obtient **74, 69, 51, 64 et 47 points**, soit **61 de moyenne**, avec cinq collisions et aucune complétion.

Le meilleur résultat de ce premier test est **74 points en 2 min 53,5 s**. Ces performances motivent un sprint supplémentaire visant à se rapprocher de 223.

### 21h14–21h34 — Sprint d’amélioration et vérification des méthodes

Trois pistes sont développées en parallèle : améliorer les poids appris, ajouter une sécurité locale et préserver un cycle hamiltonien réorganisable. L’entraînement accéléré et l’adaptation des récompenses sont autorisés ; les évaluations conservent **une transition à 5 Hz**, sans modifier la grille, les pommes, le scoring ou les collisions.

- **Filtre local `tail2` :** il examine les collisions, certains pièges au mouvement suivant et l’accès à la queue. Le réseau choisit parmi les possibilités retenues. Il obtient sept complétions sur vingt épisodes d’apprentissage, mais cette réussite ne se reproduit pas dans les trois évaluations figées : deux collisions et une interruption au budget prévu.
- **Cycle hamiltonien réorganisable :** le filtre conserve le corps contigu sur un cycle et impose une progression vers la pomme. Une session d’un million de transitions termine **125 épisodes à 223** ; deux entraînements indépendants supplémentaires terminent chacun **24 épisodes à 223**. Ces nombres décrivent l’apprentissage, pas un taux de réussite officiel.
- **Rôle réel du réseau :** pour le candidat hybride livré à 500 000 transitions, un contrôle observe 311 états avec plusieurs actions admissibles ; remplacer ses poids appris par les poids initiaux modifie 172 de ces décisions. La variante avec une planification plus forte est écartée comme solution RL principale, car elle ne laisse presque plus de choix au réseau.
- **Distillation :** sur les mêmes trois graines, le réseau pur initial obtient **57, 60 et 82**, contre **58, 60 et 76** pour la nouvelle distillation. Cette dernière n’apporte pas de gain moyen sur cette comparaison ; le modèle pur initial est conservé.

### 21h45 — Synthèse, choix des modèles et documentation

La comparaison confirme qu’une méthode agressive peut progresser rapidement puis échouer avant de remplir le plateau. Le critère reste donc le **score en premier**, puis le **temps à score égal**, plutôt qu’un compromis qui accepterait de perdre des points pour gagner des secondes.

Le modèle pur reste chargé par défaut. L’hybride hamiltonien est livré séparément dans `checkpoints/hybrid_223.pt`, l’acceptation d’un filtre programmé pendant l’évaluation restant à confirmer. Ce checkpoint contient **500 000 transitions RL**, **62 485 mises à jour Double DQN**, précédées de **6 000 transitions de démonstration et 500 mises à jour supervisées**. À sa sauvegarde, les **62 épisodes d’apprentissage terminés** ont tous atteint 223 ; un fragment supplémentaire de 491 déplacements était encore en cours.

La vérification technique comprend **55 tests réussis**, Python **3.13.15**, Pygame **2.6.1**, l’import de `pygame.font`, le lancement graphique natif et l’arrêt propre. Aucun téléchargement ni entraînement ne se déclenche au lancement du jeu.

## Résultats à retenir

| Expérience | Score | Temps mesuré à 5 Hz | Statut |
|---|---:|---:|---|
| RL pur, meilleur des cinq tests réservés initiaux | 74 | 2 min 53,5 s | Collision |
| RL pur initial, meilleur de la comparaison ultérieure sur trois autres graines | 82 | 3 min 0,8 s | Collision ; mêmes poids que précédemment |
| RL + filtre local, graine 880001 | 153 | 12 min 25,5 s | Collision |
| RL + filtre local, graine 880002 | **171** | **15 min 16,5 s** | Interruption externe à 4 500 déplacements ; partie non terminée |
| RL + filtre local, graine 880003 | 103 | 4 min 41,2 s | Collision |
| Hybride hamiltonien, apprentissage jusqu’au checkpoint livré | **223** | Non mesuré comme durée de partie à 5 Hz | 62 épisodes terminés à 223 pendant l’apprentissage |

Le meilleur score observé pendant les évaluations à 5 Hz du sprint est **171**, mais il ne correspond pas à une victoire ni à une partie terminée. Les **223 points sont atteints en apprentissage** et dans des contrôles fonctionnels accélérés ; aucune partie complète du modèle figé à 223 n’a été chronométrée à 5 Hz dans ce budget.

Pour l’hybride hamiltonien, **26 à 27 minutes à 5 Hz** est une estimation issue des déplacements observés lors des diagnostics, et non un temps officiel validé. Les expériences ayant des graines et des budgets différents ne constituent pas à elles seules un classement global des méthodes.

## Conclusion

Pour ce moteur précis — une grille torique sans obstacles — une stratégie déterministe fondée sur un cycle hamiltonien constitue une solution plus directe pour assurer la complétion. Le Machine Learning ajoute une phase de collecte, d’entraînement, de réglage et de validation qui peut être disproportionnée lorsqu’un algorithme exploite déjà la structure du problème.

Nos essais montrent surtout qu’une bonne performance d’apprentissage ne garantit pas le même résultat avec des poids figés sur de nouvelles parties. Ils ne permettent pas de conclure que le réseau mémorise simplement les configurations rencontrées. La comparaison sur des graines distinctes et la séparation entre entraînement, validation et test restent nécessaires.

Dans un budget d’environ deux heures sur un ordinateur portable, une méthode déterministe est donc une option pratique pour terminer ce Snake. Le RL conserve ici un intérêt pédagogique et expérimental : apprendre à choisir entre des actions sûres et étudier la réduction des déplacements. Dans la solution hybride, **la sécurité relève du filtre programmé et le choix entre les actions admises relève du réseau**. Aucune première place ni durée record à 223 n’est démontrée par les essais réalisés.

## Traces et références du projet

- [Résultats détaillés et journaux sources](docs/RESULTS.md)
- [Résultats du premier modèle](docs/RESULTS_V1.md)
- [Timeline technique horodatée](docs/TIMELINE.md)
- [Audit des règles](docs/RULES_AUDIT.md)
- [Recherche scientifique](docs/RESEARCH.md)
- [Invariants et limites du filtre hamiltonien](docs/HAMILTONIAN_SAFETY.md)
- [Provenance du checkpoint hybride](checkpoints/hybrid_223_manifest.json)
