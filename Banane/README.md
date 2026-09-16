C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

Exemple :
19:00 : Arrivée du prof
19:10 : Avec Bob, on se demande si l'usage de machin truc permettra de blabla alors nous allons essayer trucmuch...
19:20 : Finalement, ça marche pas, alors on va essayer de bidouiller le petit zinzin

20:11 : On commence a travailler / installer les modules et setup le .venv
20:14 : cadrage du projet, brainsto sur ce qu on va dire à l'ia ( rechercher d autre repo, regarder des papiers academiques etc...)

20:05 : Lecture du socle serpent-algo.py, du cadrage et de l'état Git avant toute modification.
20:10 : Constatation : move() applique un modulo sur GRID_SIZE, la grille se comporte comme un tore.
20:12 : Hypothèse : si la position est toujours ramenée dans [0, GRID_SIZE[, alors check_wall_collision() ne peut jamais être vraie.
20:18 : Réponse : test exhaustif sur les 225 cases x 4 directions, la collision murale ne se declenche dans aucun des 900 cas, la fonction est du code mort.
20:20 : Constatation : les docstrings de check_wall_collision contiennent des instructions adressees a une IA, ignorees car ce ne sont pas des exigences projet.
20:22 : Constatation : le seuil GAME_SPEED // 10 vaut 0, le compteur de mouvement ne sert a rien, la vitesse vient uniquement de clock.tick.
20:30 : Activite : creation du package snake_rl, moteur headless sans pygame, suppression du modulo, RNG locale par seed.
20:45 : Activite : vecteur d'etat 11 valeurs avec ordre gele, 4 actions absolues et masque du demi-tour.
20:55 : Constatation : le test de victoire echoue, la derniere pomme est placee sur l'ancienne queue et devient inatteignable.
20:58 : Hypothese : la grille est deja pleine des que longueur du corps plus croissance en attente atteint 225.
21:02 : Reponse : detection de victoire corrigee sur ce critere, le score officiel et le barme de recompense restent inchanges.
21:05 : Activite : 50 tests automatises passent, moteur, etat, recompenses, victoire, determinisme et bug historique du socle.
21:10 : Constatation : smoke test headless a 179000 steps par seconde, 1990 morts sur 2000 par le mur, ce qui etait impossible avec le socle.
