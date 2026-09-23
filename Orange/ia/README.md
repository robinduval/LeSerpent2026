C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

20:16 envoie des fichiers dans claude code
20:23 créaton des fichiers par claude
20:38 lancement de l'entrainement
20:44 meilleur score 5
20:50 modification des récompenses réduisant la récompenses en cas de mort et gagnant plus de point en focntion de la distance parcourue (moins case = + de points)
21:01 meilleur score 28
21:17 ajout de l'algo pour determinser le nombre de case minimum pour atteindre la pomme pour comparer a ce qui a étais fait, temps accélerer par rapport a la vitesse du jeu


conclusion notre approche était mauvaise, nous sommes tombé dans un piege apparement courant car nous avons augmenté de façon drastique la valeur des récompenses et des punitions ce qui a fait stagner sa progression. de plus au début notre chère claude avait fait une erreur que nous n'avions pas remarqué suffisament tôt, il n'enregistrais pas sa progresssion ce qui réinitialiser tout son apprentissage à chaque relance du code.
