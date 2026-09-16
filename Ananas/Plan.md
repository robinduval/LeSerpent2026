# Plan d'exécution - Le Serpent 2026

## Décision de stratégie
L’objectif n’est pas d’entraîner un agent spécialisé dans la recherche du chemin le plus court vers la pomme. Le système doit devenir un agent Snake généraliste de très haut niveau, capable de maximiser le score final, de préserver sa mobilité et d’anticiper les risques d’enfermement. La stratégie principale sera un Rainbow DQN enrichi par des démonstrations issues d’un planificateur sûr. Dijkstra sera utilisé comme baseline et comme outil auxiliaire, mais pas comme politique finale.

Développer en parallèle :

- un agent déterministe `SafePath`, optimisé pour prendre le chemin court seulement lorsqu'il reste sûr ;
- un DQN compact, conforme à l'exigence Deep Learning et rapide à entraîner.

Le succès se mesure d'abord par la survie et les pommes obtenues, puis par le nombre de pas par pomme et le temps de décision. Le moteur, la clock, la grille et le scoring restent inchangés.

## Phase 0 - Clarifier et figer les règles

- [ ] Identifier le dossier du groupe, les auteurs et les pseudos GitHub.
- [ ] Confirmer les deux dates de rendu.
- [ ] Demander la formule officielle du ratio score/temps.
- [ ] Confirmer si le reward interne du DQN est modifiable sans modifier le scoring.
- [ ] Confirmer si l'entraînement headless est autorisé.
- [ ] Confirmer CPU/GPU et budget d'entraînement disponible.
- [ ] Écrire dans la timeline les décisions obtenues.

Sortie : une configuration d'évaluation incontestable avant toute comparaison.

## Phase 1 - Comprendre le jeu sans le modifier

- [ ] Lire `serpent.py` et relever les interfaces : état, déplacement, collision, pomme, score, reset et clock.
- [ ] Exécuter le jeu de base et noter la commande, les dépendances et le comportement attendu.
- [ ] Créer un adaptateur minimal entre le jeu et une interface d'agent.
- [ ] Ajouter une seed si le moteur possède déjà un point d'injection qui ne change pas ses règles.
- [ ] Créer un enregistreur CSV/JSON commun aux deux agents.

Sortie : une partie peut être pilotée par un agent vide et produire un résultat mesuré.

## Phase 2A - Baseline algorithmique

- [ ] Implémenter les mouvements légaux et la représentation de la grille.
- [ ] Implémenter Dijkstra demandé par le cours, ou BFS si l'enseignant accepte l'équivalence sur coûts uniformes.
- [ ] Recalculer le chemin vers la pomme après chaque mouvement.
- [ ] Mesurer score, pas par pomme et collisions sur un lot fixe de seeds.

Sortie : baseline plus-court-chemin fonctionnelle, même si elle peut encore s'enfermer.

## Phase 2B - Baseline DQN

- [ ] Encoder l'état compact présenté dans le cours.
- [ ] Convertir les trois actions relatives en directions du jeu.
- [ ] Implémenter le MLP PyTorch, la prédiction et la sauvegarde.
- [ ] Ajouter replay buffer, mini-batches, epsilon-greedy et réseau cible.
- [ ] Reproduire d'abord les rewards du cours.
- [ ] Enregistrer loss, reward, score, epsilon et meilleur checkpoint.

Sortie : un vrai DQN apprend, sauvegarde puis rejoue sans exploration.

## Phase 3A - Rendre l'algorithme sûr

- [ ] Simuler le chemin vers la pomme, croissance comprise.
- [ ] Vérifier qu'une route tête-vers-queue subsiste après la pomme.
- [ ] Ajouter un calcul de zone accessible par flood-fill.
- [ ] Ajouter le mode de repli vers la queue.
- [ ] Ajouter le mouvement de dernier recours maximisant l'espace libre.
- [ ] Tester les corridors, spirales, poches fermées et grilles presque pleines.

Sortie : `SafePath` bat Dijkstra/GBFS naïfs sur la moyenne des mêmes seeds.

## Phase 3B - Stabiliser le DQN

- [ ] Vérifier que l'action choisie influence réellement le jeu et que les transitions sont correctes.
- [ ] Vérifier la gestion des états terminaux et l'absence de fuite entre épisodes.
- [ ] Ajuster seulement quelques hyperparamètres à fort impact : learning rate, gamma, batch size et décroissance epsilon.
- [ ] Si autorisé, comparer reward du cours et faible coût par pas afin d'éviter les boucles.
- [ ] Conserver séparément le meilleur score et la meilleure moyenne glissante.
- [ ] Arrêter les essais qui n'améliorent plus les résultats pour préserver le temps.

Sortie : courbe d'apprentissage exploitable et checkpoint reproductible.

## Phase 4 - Comparaison honnête

- [ ] Geler les versions des deux agents.
- [ ] Évaluer sur les mêmes seeds, sans exploration pour le DQN.
- [ ] Utiliser au moins 30 parties si le temps le permet.
- [ ] Produire moyenne, médiane, écart-type, maximum, taux de collision et pas par pomme.
- [ ] Mesurer séparément le temps de décision, le temps de partie et le temps d'entraînement.
- [ ] Identifier les scénarios où chaque agent échoue.

Critères de sélection :

1. taux de parties sans collision prématurée ;
2. nombre moyen de pommes ;
3. nombre médian de pas par pomme ;
4. temps moyen de décision.

## Phase 5 - Livrables et présentation

- [ ] Placer `snake-algo.py` et `snake-ia.py` dans le dossier du groupe.
- [ ] Ajouter le checkpoint DQN selon les règles de taille du dépôt, ou documenter sa génération.
- [ ] Compléter `AUTHORS`.
- [ ] Compléter la timeline/changelog avec qui, quand, quoi et résultat.
- [ ] Ajouter les commandes d'installation, entraînement, démonstration et évaluation.
- [ ] Préparer une courbe d'apprentissage et un tableau de comparaison.
- [ ] Préparer le tour de table : difficultés, débogage, validation, résultats et transfert en entreprise.

## Ordre de priorité en cas de manque de temps

1. DQN minimal réellement entraînable et démontrable.
2. Dijkstra/BFS baseline exécutable.
3. Validation de sécurité tête-vers-queue.
4. Protocole d'évaluation reproductible.
5. Flood-fill et repli avancé.
6. Tuning limité du DQN.
7. Améliorations visuelles ou refactors non indispensables.

## Pièges à surveiller

- Plus court chemin ne signifie pas meilleure survie : la pomme peut fermer la seule sortie.
- Dijkstra n'apporte aucun avantage de distance sur BFS lorsque chaque déplacement coûte 1 ; son intérêt est surtout pédagogique ou futur si les coûts deviennent pondérés.
- Le reward `+0,1` par déplacement peut encourager le serpent à survivre longtemps sans manger vite.
- L'état local du cours ne décrit pas la forme complète du corps ; le DQN peut plafonner quand le serpent grandit.
- Mélanger actions absolues dans le jeu et relatives dans le réseau produit des rotations incohérentes si l'adaptateur n'est pas testé.
- Entraîner avec PyGame rendu et clock active peut rendre les essais inutilement longs.
- Comparer les meilleurs scores sur des pommes aléatoires différentes ne prouve rien.
- Modifier clock, grille ou scoring rend le résultat non comparable et viole les consignes.
- Un cycle hamiltonien peut maximiser la survie mais être jugé hors intention pédagogique et lent pour atteindre les pommes.
- Une architecture Deep Learning trop grande augmente le temps sans garantir un meilleur score sur cet état compact.

## Première session de travail recommandée

1. Répondre aux questions de la phase 0.
2. Inspecter uniquement les interfaces nécessaires de `serpent.py`.
3. Construire l'adaptateur et l'enregistreur communs.
4. Lancer en parallèle la baseline Dijkstra/BFS et le squelette DQN.
5. Obtenir une première mesure reproductible avant toute optimisation.

