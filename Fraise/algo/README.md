C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse). Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

19:50 : Début, nous allons lancer notre ami Claudius Opus V.V et nous préparons son super contexte
19:57 : Décision de structure : on reprend le socle validé la semaine dernière (jeu de base importé sans être modifié, lancement sans argument pour le prof, logs score et temps, mode rapide xN avec temps compté x N, mode eval sans affichage) et on met les trois algorithmes dans un seul fichier pour pouvoir les comparer.
19:58 : Dijkstra et GBFS codés sur le tore (les bords se traversent, donc la distance passe parfois par le bord) ; mesure sur 50 parties identiques : Dijkstra moyenne 65,9 record 100, GBFS moyenne 68,1 record 105.
19:59 : Constat conforme au cours : GBFS explore 13,1 cases par coup contre 54,4 pour Dijkstra, mais aucun des deux ne finit jamais la partie (0 victoire sur 50) car ils foncent vers la pomme et s'enferment.
20:00 : Hypothèse : comme le meilleur score gagne et que le temps ne départage qu'à égalité, il faut viser la victoire (remplir la grille) plutôt que le meilleur score moyen.
20:00 : Idée clé : un circuit passant une seule fois par chaque case est impossible sur une grille fermée 15x15 (225 cases, nombre impair), mais il existe ici parce que les bords se rejoignent ; circuit construit (chaque ligne parcourue vers la droite en traversant le bord, puis on descend) et vérifié : 225 cases, aucune répétition, tous les pas valides.
20:01 : Version intermédiaire « glouton sûr » (plus court chemin vers la pomme, accepté seulement si la tête peut encore rejoindre sa queue après avoir mangé) : bug, score 0 partout ; cause trouvée : le test de sécurité partait de la case de la tête, toujours occupée par le serpent lui-même.
20:01 : Bug corrigé ; le glouton sûr monte à 104,4 de moyenne et 177 de record, mais 0 victoire sur 50 parties (42 morts, 8 boucles) : le test « queue atteignable » ne garantit rien.
20:02 : Notre algorithme « hybride » : on suit le circuit hamiltonien, et on prend un raccourci dès qu'il est sûr ; règle de sûreté : un raccourci ne doit jamais dépasser la queue dans l'ordre du circuit, donc la tête reste toujours devant le corps et la collision devient impossible.
20:02 : Premier essai de l'hybride : 20 victoires sur 20, score 223 (grille pleine), 1377 s de jeu en moyenne.
20:03 : Recherche des réglages : le critère « aller vers la case la plus proche de la pomme au sens de Manhattan » et le critère « prendre le plus grand raccourci » donnent 0 victoire (le serpent dépasse la pomme et tourne en boucle) ; on garde « la case la plus proche de la pomme dans l'ordre du circuit ».
20:04 : Constat contre-intuitif : arrêter les raccourcis plus tôt (seuil 0,45 de cases libres au lieu de 0,25) fait gagner du temps, car les raccourcis tardifs collent la tête à la queue et obligent ensuite à de longs détours.
20:04 : La marge 1 case avant la queue a fini par tuer le serpent (score 103) sur 25 parties ; marges 2 et 3 gagnent 60 parties sur 60 ; décision : marge 3 et seuil 0,45, car le score prime sur le temps et la pire partie est plus courte (1437 s contre 1552 s).
20:05 : Mesure officielle sur 50 parties : hybride 50 victoires sur 50, record 223 en 1237,6 s, calcul 0,02 ms par coup (budget de 200 ms par image à 5 images/seconde), donc la clock n'est pas perturbée.
20:06 : La commande « python Fraise/algo/snake-algo.py » lance l'hybride en partie réelle, sans argument ; graphique de suivi (evolution.svg) et mesures (mesures.csv) générés par « python Fraise/algo/graphique.py ».
20:11 : Méthode hamiltonienne gagné (223 points) en 21 minutes
20:12 : Tom se met à abboyer sur Claudius Opus V.V
20:16 : Maintenant on va chercher une autre solution que le hamiltonien et on va essayer de trouver une solution plus rapide quite à être moins constant 
20:21 : Glouton plus sûre 1 victoire (223 points) sur 30 partie, en 1070 s
20:55 : Une version optimisé de l'algorithme glouton (Foncer vers la pomme, mais seulement si on peut encore rejoindre sa queue et Au-delà de 55 % de la grille, ne plus viser la pomme du tout) nous assure la victoire dans 10% des parties en 950sec
20:55 : Tom me pique le clavier et essaye cette idée "Le but lorsque l'on se poursuit plus les pommes 55% passer on cherche egalement a reduire le nombre de trou pour que toutes les prochaines pommes soit devant lui et pas au milieu piege dans son corp", spoiler elle ne fonctionnera pas et aura de moins bon résultats que la précédente
21:17 : Claude devient autonome et essaye un autre truc "longer les murs et son propre corps" : victoire en 992sec (2 / 300)
21:19 : victoire en 835 seconde on a repris le glouton boosté et on lance 20000 partie à la fois sur le macbook m3pro tah les fous (Tom a la rage qu'un mac fonctionne si bien)
21:21 : 763 secondes pour la victoire

20:56 : On va tenter de faire tourner le plus de partie possible avec un algorithme qui a peu de % de victoire mais qui va le plus vite possible

