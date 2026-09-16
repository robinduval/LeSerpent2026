# Timeline du groupe

20:10 : On envoie le premier prompt pour comprendre exactement comment on va itérer pour faire du RL, ainsi qu'un prompt pour expliquer à notre IA le sujet et le but.
20:15 : Recherche sur l'autre agent : comment aider le premier pour gagner du temps et des performances.
20:20 : Combinaison des réflexions des deux agents.
20:27 : Premier prompt pour l'écriture de code, qui s'appuie sur toutes les réflexions que nous avons eues avec l'agent auparavant.
20:33 : Premier résultat → pas terrible.
20:36 : Lancement d'un deuxième prompt où l'on change plusieurs règles : utilisation de plusieurs serpents et accélération possible du temps pendant l'entraînement, même si la clock ne peut pas être modifiée lors de l'évaluation qui compte.
20:39 : Deuxième résultat → ça s'améliore grandement, mais l'agent n'a pas compris ce qu'on lui a dit vis-à-vis du temps.
20:46 : Sortie d'un premier score en situation réelle → 19 points pour un objectif de 10 → nul en gros, au regard de nos ambitions.
21:00 : Fin du premier gros batch d'entraînement → meilleur score : 61. Nous trouvons cela insuffisant.
21:14 : On a changé un peu les règles et ça fonctionne bien mieux, notamment le fait qu'il sait mieux où il se trouve maintenant. Techniquement, nous avons enrichi les observations de l'agent, sans modifier les règles du jeu.
21:20 : Meilleur score à 90, on avance. L'entraînement s'améliore bien plus rapidement.
21:25 : Entraînement centré sur le meilleur score.
21:35 : Score maximal : 117 pour environ 315 secondes (315,8 secondes équivalentes à 5 Hz, calculées à partir des 1 579 déplacements de la partie d'entraînement).

## Conclusion du groupe

On n'a pas assez réfléchi avant de lancer l'entraînement ; il n'était donc pas optimal et nous avons perdu près d'un tiers du temps. C'était un peu abrutissant de faire cela pendant une heure et demie : nous réfléchissions moins bien à la fin, alors que c'était le moment où nous aurions dû être les plus frais.

## Journal technique détaillé

20:27 : Début de l'implémentation RL ; règles du socle conservées, budget provisoirement fixé de 20:00 à 22:00.
20:30 : Lancement de la première partie autonome avec Double DQN, replay buffer et clock inchangée à 5 Hz.
20:32 : Première partie terminée par collision avec le corps : 2 points, 108,52 secondes, 540 déplacements, 477 mises à jour ; checkpoint et métriques sauvegardés dans runs/20260916-203043-456478-train.
20:32 : Quatre tests validés : bords traversables, croissance différée, danger selon la queue, apprentissage et reprise de checkpoint ; seuil de 10 points non atteint.
20:35 : Nouvelle règle utilisateur : accélération et plusieurs serpents autorisés à l'entraînement ; toutes les durées restent mesurées en temps réel.
20:37 : Essai avec 8 serpents depuis le premier checkpoint : 330 parties terminées, record d'entraînement 50 points, 67 408 transitions en 30,09 secondes réelles ; évaluation du modèle figé encore à réaliser.
20:43 : Reprise du dernier checkpoint pour 60 secondes avec 8 serpents et nouvelles seeds à partir de 2000.
20:44 : Batch terminé : 594 parties complètes, meilleur score 48, moyenne 22,38, taux à 10 points de 87,04 %, 130 224 nouvelles transitions en 60 secondes ; lancement d'une partie visible du modèle figé à 5 Hz, seed 1000.
20:44 : Ajout d'un export ZIP réutilisable pendant les batchs : données consolidées, distinction temps réel/équivalent 5 Hz, configurations, checkpoints, code, méthode et pistes d'analyse ; intégrité de l'archive vérifiée.
20:45 : Évaluation visible à 5 Hz du modèle figé, seed 1000 : 19 points en 36,99 secondes, 184 déplacements, ratio 0,514 point/s ; seuil 10 atteint à 19,30 secondes, fin naturelle par collision ; epsilon nul et compteurs d'apprentissage inchangés.
20:46 : Nouveau batch intensif de 60 secondes depuis le checkpoint précédent, 8 serpents, seeds à partir de 4000 : 623 parties terminées, meilleur score 55, moyenne 21,75, taux à 10 points 89,89 %, 130 032 nouvelles transitions.
20:48 : Partie visible du nouveau modèle figé à 5 Hz sur la même seed 1000 : 13 points en 22,91 secondes, ratio 0,567 point/s ; seuil 10 atteint à 17,89 secondes. Régression sur cette seed face aux 19 points précédents ; les deux checkpoints sont conservés, aucune supériorité générale conclue sans batterie.
20:50 : Lancement d'un entraînement massif de 600 secondes réelles, 8 serpents, depuis le checkpoint évalué à 19 points ; nouvelles seeds à partir de 10000, sauvegarde toutes les 30 secondes, historique séparé dans runs/20260916-205009-109580-train. Résultats finaux en attente.
21:07 : Ajout d'une variante spatiale à 22 entrées avec espace accessible et distances normalisées ; transfert des poids du modèle massif, nouvelles entrées initialement neutres, replay et optimiseur réinitialisés, règles et clock d'évaluation conservées. Neuf tests passent.
21:08 : Batch spatial de 60 secondes : 349 parties terminées, record d'entraînement 74, moyenne 29,84 ; lancement d'une comparaison de trois seeds 1000 à 1002 entre baseline massif et modèle spatial figés à 5 Hz.
21:09 : Baseline massif évalué sur seeds 1000, 1001, 1002 : 25, 13, 16 points, moyenne 18, meilleur score 25 ; trois parties terminées naturellement.
21:12 : Modèle spatial figé à 5 Hz : 82 points atteints sur seed 1000 en 233,55 secondes, puis interruption via événement de fermeture Pygame. Ce score n'est pas un score de partie terminée ; comparaison de trois parties encore incomplète. Modèle spatial et baseline conservés séparément.
21:13 : À la demande du groupe, reprise de l'entraînement spatial accéléré pour 600 secondes avec 8 serpents et seeds à partir de 30000 ; checkpoints distincts toutes les 30 secondes, sans évaluation visible automatique à la fin. Session runs/20260916-211339-898975-train.
21:25 : Nouvelle priorité utilisateur : score maximal indépendamment du temps. Préparation d'un fine-tuning spatial de 300 secondes, epsilon minimum 0,01 au lieu de 0,05 et learning rate 1e-4 au lieu de 3e-4 ; récompenses et règles inchangées, ancienne version conservée pour comparaison.
