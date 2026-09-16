C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

Exemple :
19:00 : Arrivée du prof
19:10 : Avec Bob, on se demande si l'usage de machin truc permettra de blabla alors nous allons essayer trucmuch...
19:20 : Finalement, ça marche pas, alors on va essayer de bidouiller le petit zinzin

20:11 : On commence a travailler / installer les modules et setup le .venv
20:14 : cadrage du projet, brainsto sur ce qu on va dire à l'ia ( rechercher d autre repo, regarder des papiers academiques etc...)
20:15 le prof vient nous parler avec tom et gabriel, nous dérange légèrement mais ça reste acceptable. Gare à lui si ça se reproduit...
20:19 on ouvre le forms du prof mais on voit que ça sert a rien pour l'instant

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
21:25 : Activite : DQN PyTorch, replay buffer NumPy prealloue, epsilon greedy, target network et checkpoints.
21:35 : Activite : protocole d'evaluation separe, epsilon a zero, seeds 1000 et suivantes, aucune ecriture en memoire.
21:45 : Constatation : le premier smoke test ne se termine jamais, l'entrainement fait ses 60 parties en 1.1 seconde mais l'evaluation bloque.
21:47 : Hypothese : avec epsilon a zero la politique est deterministe et la prime de survie de +0.1 la pousse a tourner en rond indefiniment.
21:52 : Reponse : garde-fou technique de 500 pas sans pomme, tres au dela des 225 cases de la grille, declare dans la configuration et reporte via truncation_rate, une troncature n'est ni une defaite ni une victoire.
21:55 : Constatation : une troncature n'est pas un etat terminal, on conserve donc le bootstrap sinon le reseau apprend que ces situations ne valent rien.
22:05 : Reponse : smoke test de 200 parties en 8 secondes, score moyen d'evaluation de 3.1 puis 9.3, record 23, pipeline complet valide.
22:10 : Constatation : le score d'evaluation depasse tres largement le score d'entrainement, ce qui confirme qu'il faut mesurer sans exploration.
22:15 : Activite : replay graphique d'une trajectoire enregistree, sans rejouer le modele, plus courbes d'apprentissage automatiques.
20:30 : Activite : reprise de session, la campagne de comparaison dqn / ddqn / dueling_ddqn tourne toujours en tache de fond.
20:35 : Activite : implementation du Prioritized Experience Replay, arbre de sommes pour un tirage et une mise a jour en O(log n).
20:38 : Constatation : 19 tests PER passent, dont la propriete centrale mesuree et non supposee, la transition d'erreur TD 100 est rejouee plus de 4 fois la mediane.
20:40 : Reponse : smoke test dueling_ddqn_per sur 200 episodes, evaluation moyenne 11.10, mediane 11.0, record 23, aucune troncature.
20:42 : Hypothese : le 1.0 episode/s mesure n'est pas un cout du PER mais la concurrence des 4 trials de la campagne sur les memes coeurs, a reverifier machine libre.
20:46 : Activité : inspection Git et lecture complète du cadrage, SESSION_HANDOFF.md absent ; quatre essais de grid search orphelins encore actifs sont identifiés et arrêtés.
20:47 : Constatation : le modulo du professeur définit un plateau torique ; l'interprétation des murs mortels et le benchmark de 1990 morts murales sont invalides.
20:48 : Réponse : déplacement et trois dangers corrigés avec modulo, suppression de la mort murale, masque du seul demi-tour et récompenses conservés.
20:49 : Activité : ajout de tests de wraparound et de provenance torique ; anciens poids et résultats exclus du classement, rapports E00 et E01 marqués legacy invalides.
20:50 : Constatation : benchmark aléatoire de 2000 parties sans limite, zéro mort murale et 2000 auto-collisions, 568,9745 pas moyens et maximum 2994 ; même seed reproductible et seed différent donnant une autre trajectoire.
20:52 : Réponse : 184 tests passent, déplacements RL conformes au socle sur 900 cas, git diff vérifié et rapport de validation enregistré ; aucune longue grid search relancée.
20:56 : Constatation : la limite expérimentale compte 500 pas consécutifs sans pomme et conserve déjà le bootstrap ; les flags de fin et les replays doivent distinguer terminated et truncated explicitement.
20:58 : Réponse : flags séparés, troncature sans mort ni reward de mort, limite zéro réellement désactivée ; ajout de métriques de pommes, Q, loss et stagnation analytique sans nouvelle règle terminale.
20:58 : Activité : lancement DQN seul sur trois seeds, 120 épisodes par seed, évaluations greedy tous les 40 épisodes, configuration courte fixe et reward du cours inchangé.
21:00 : Constatation : les 60 évaluations finales donnent un score moyen de 28,90, médiane 25,5, record 56, zéro troncature et maximum 75 pas sans pomme ; les politiques initiales étaient toutes tronquées.
21:01 : Activité : trois replays intermédiaires de cycles de 15 pas sont reproduits action par action, sans collision finale ; campagne totale de 105,38 secondes, dont 77,64 secondes dans les épisodes d'entraînement.
21:03 : Réponse : 223 tests passent, rapports et options expérimentales enregistrés ; la baseline apprend la recherche des pommes sur cet échantillon, aucun reward changé et aucune longue grid search relancée.
21:07 : Constatation : le JSON désactive le replay graphique et le CLI conservait ce flag sans --window ; l'évaluation tournait correctement mais aucun renderer n'était appelé.
21:08 : Réponse : replay automatique activé sauf --no-window, annonce terminal et cadence visuelle de 60 FPS configurable ; capture de la meilleure trajectoire dès la mesure, sans refaire jouer le modèle.
21:09 : Activité : smoke réel de quatre épisodes, fenêtres WSLg/X11 ouvertes après les évaluations 2 et 4, chacune 501 frames lues, fermeture automatique et reprise du training ; fenêtre visible confirmée par xwininfo.
21:10 : Activité : contrôle --no-window réussi avec un pilote graphique volontairement inexistant, aucun display créé ; paramètres DQN, règles et récompenses inchangés.
21:12 : Réponse : 236 tests passent, dont les régressions de branchement et de cadence graphique ; rapport de validation et git diff vérifiés, sorties des smoke tests conservées sans écrasement.
