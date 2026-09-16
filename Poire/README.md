C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

Exemple :
19:00 : Arrivée du prof 
19:10 : Avec Bob, on se demande si l'usage de machin truc permettra de blabla alors nous allons essayer trucmuch...
19:20 : Finalement, ça marche pas, alors on va essayer de bidouiller le petit zinzin
20:12 : Lancement du jeu pour voir comment cela fonctionne, voir comment on meurt dans le jeu et on a trouvé le code pour le prompt caché
20:13 : Variable à définir que l'on regarde au niveau du code, le corps il fait 5 de longueur
20:17 : Compréhension de la fin du code
20:23 : Bonne compréhension du jeu & du code , sympa à lire

20:37 : 
- Formule de Bellman pour la mise a jour de la fonction d'apprentissage: `Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]`
- Nouvelle estimation = Ancienne estimation + (Vitesse d'apprentissage) × (Erreur de prédiction) (c'est une formule mathématique)
20:45 : Compréhension derrière du code de l'application des forumes maths 
20:48 : Ecriture du code à présent avec OpenAPI Gym Algo
20:56 : 
Représentation	Taille	Avantage	Problème
Grille brute (15×15, 1 canal ou plus)	225+ valeurs	Aucune perte d'info, généralise en théorie très bien	Nécessite un CNN, beaucoup plus long à entraîner et à faire converger avec le temps dont tu disposes
Table Q sur état brut	3^225 environ	—	Irréaliste, ne rentre dans aucune mémoire
Vecteur de features compact	~11 booléens	Q-learning tabulaire ou petit réseau dense, convergence en quelques milliers d'épisodes	Perd de l'information (ex : forme précise du corps) → plafond de score plus bas mais atteignable vite

Donc on modelise le pb, en un vecteur a seulement 11 parametres (au lieu de 225 ou 3^225), donc petit et realiste dans le tmeps du cours pour faire lun entrainement qui ~~~ fonctionne ~~~

| Index | Feature           |
| ----: | ----------------- |
|     0 | danger tout droit |
|     1 | danger droite     |
|     2 | danger gauche     |
|     3 | direction gauche  |
|     4 | direction droite  |
|     5 | direction haut    |
|     6 | direction bas     |
|     7 | pomme à gauche    |
|     8 | pomme à droite    |
|     9 | pomme en haut     |
|    10 | pomme en bas      |


21h06: Choix d'un algo simple (Q-learning tabulaire) pour avoir un resulat correct dans le temps impartie avant d'ameliorer
21h17 : Lancement des premiers tests

21:26
--- Entraînement Q-learning ---
épisodes=8000 alpha=0.1 gamma=0.9 epsilon=1.0->0.01 decay=0.998
  épisode    500/8000 | score moyen (100 derniers) :  3.56 | epsilon : 0.368
  épisode   1000/8000 | score moyen (100 derniers) :  9.48 | epsilon : 0.135
  épisode   1500/8000 | score moyen (100 derniers) : 15.62 | epsilon : 0.050
  épisode   2000/8000 | score moyen (100 derniers) : 22.17 | epsilon : 0.018
  épisode   2500/8000 | score moyen (100 derniers) : 23.99 | epsilon : 0.010
  épisode   3000/8000 | score moyen (100 derniers) : 23.36 | epsilon : 0.010
  épisode   3500/8000 | score moyen (100 derniers) : 24.28 | epsilon : 0.010
  épisode   4000/8000 | score moyen (100 derniers) : 25.90 | epsilon : 0.010
  épisode   4500/8000 | score moyen (100 derniers) : 23.30 | epsilon : 0.010
  épisode   5000/8000 | score moyen (100 derniers) : 25.00 | epsilon : 0.010
  épisode   5500/8000 | score moyen (100 derniers) : 25.21 | epsilon : 0.010
  épisode   6000/8000 | score moyen (100 derniers) : 24.11 | epsilon : 0.010
  épisode   6500/8000 | score moyen (100 derniers) : 27.93 | epsilon : 0.010
  épisode   7000/8000 | score moyen (100 derniers) : 29.92 | epsilon : 0.010
  épisode   7500/8000 | score moyen (100 derniers) : 27.58 | epsilon : 0.010
  épisode   8000/8000 | score moyen (100 derniers) : 25.41 | epsilon : 0.010
--- Résultat ---
durée              : 24.0 s
meilleur score     : 60
score moyen (100)  : 25.41
epsilon final      : 0.0100
états découverts   : 256
sauvegardé dans    : /home/brian/Documents/aepita/other-projects/LeSerpent2026/Poire/qtable_poire.pkl


Sur le premier try, avec les points d'un algo extrement simple (model a 11 parametres)
On arrive a un score de 34 avec un temps de 1 min 26 soit un ratio score / temps de ~0.4