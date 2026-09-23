19:48 : Partir sur les algorithmes Dijkstra et A\*.
19:53 : Implémenter A\* en Python pour cibler la pomme.
20:00 : Installer pygame et tester le jeu. Constater que A* seul s'enferme après environ 60 pommes.
20:05 : Combiner A* avec un algorithme de survie.
20:08 : Prioriser l'évitement des culs-de-sac.
20:10 : Créer un algorithme hybride pour minimiser les pas. Constater un score multiplié par 3,3, mais l'apparition de boucles infinies.
20:18 : Ajouter une limite de pas sans manger pour casser les boucles.
20:28 : Intégrer une option --rapide pour accélérer la simulation.
20:35 : Afficher le score sur l'interface et corriger le timer.
20:45 : Constater une mort à 172 de score en partie réelle.
20:55 : Affiner l'algorithme avec un BFS temporel (filet « queue joignable ») après une mort à 212. Constater une moyenne de 210,2 et 4 victoires sur 20.
21:05 : Tester trois pistes d'optimisation (limites de pas, chemin sûr, cycle hamiltonien). Constater qu'une limite de 300 pas est optimale, que le chemin sûr donne 210,8 de moyenne, et que le cycle hamiltonien garantit 223 points mais rallonge considérablement le temps de jeu.
21:20 : Combiner l'hybride A\* et le cycle hamiltonien via une longueur seuil. Constater que ce seuil permet de contrôler le compromis entre le score et le temps de jeu.
21:30 : Accélérer les tests : parties en parallèle et commande --bench (60 parties en quelques secondes au lieu de plusieurs minutes).
21:40 : Viser 223 à chaque partie en réduisant le temps. Constater qu'arrêter les raccourcis du cycle à 120 cases passe le temps de 33:44 à 22:32.
21:55 : Rendre le cycle hamiltonien dynamique (échanges et déplacements de morceaux devant la tête) et en faire l'autopilote par défaut, sans option. Constater 223 à chaque partie en 14:16, contre 18:32 pour l'hybride A\* (--astar) qui meurt dans 57 parties sur 60.
