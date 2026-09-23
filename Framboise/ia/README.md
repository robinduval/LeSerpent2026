C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

Exemple :
19:00 : Arrivée du prof 
19:10 : Avec Bob, on se demande si l'usage de machin truc permettra de blabla alors nous allons essayer trucmuch...
19:20 : Finalement, ça marche pas, alors on va essayer de bidouiller le petit zinzin

20:05 : Intervention de Robin Duval
20:10 : Début du travail, lecture du README, recherches et plan de de
20:30 : Squelette de script
20:47 : Début de l'entrainement
20:49 : L'IA est etonnement assez performante, mais dans ses meilleurs runs, elles finie par mourir en se bloquant elle meme. Analyse des runs, assisté par IA (Meilleur score : 65)
21:05 : Lancement de la nouvel IA optimisée
21:20 : L'IA a fait enormement d'entrainement, elle atteint un score moyen de 144 en 300 secondes, et un max a 208
21:25 : Nous essayons de mettre en place une solution dans laquelle, a partir d'un certain pourcentage de remplissage de la grille, le serpent suit un pattern precis (Methode hamiltonienne) pour remplir terme 100%.
21:40 : Robin nous dit que c'est de la triche et que c'est de l'algo. Abort mission.
21:41 : Fin de tournage, on a bien géré

---

19:49 : On commence
19:50 : On cherche des solutions, sur internet et avec nos amis magiques, pour des algos ou compositions d'algos
20:00 : On se dit que faire un algo combiné avec parcours le plus cours (au debut de la partie) puis une methode hamiltonienne (pour la fin de partie) pourrait etre une bonne chose
20:08 : L'ami magique est sur le coup, avec des directives bien claires ! Il va meme comparer l'approche a laquelle nous avons reflechie, et une approche un peu retravaillée, qu'il nous a proposé qui consite a faire uniquement du hamiltonien, mais un peu ameliorer au debut, qui va faire en sorte de trouver des raccourcis si possible et que cela ne le met pas en difficulté pour la suite
20:19 : L'approche de l'ami magique est bien plus performante (10/10 parties gagnees contre 6/10, en 22 min en moyenne, contre 37)
L'approche en une phrase : le serpent suit un cycle qui passe par toutes les cases, prend des raccourcis vers la pomme (trouvés par BFS) sans jamais dépasser sa queue sur le cycle, jusqu'à 50 % de remplissage, puis suit le cycle sans raccourci jusqu'à 100 %.
Resultats : 222 (100%) en 19 min 28 s.