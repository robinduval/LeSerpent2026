C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.


20:00 : Recherche de l'algo à utiliser avec ChatGPT
20:04 : Etape 1 : BFS + simulation du chemin Etape 2: Faire le SafePath  Etape 3: Faire un benchmark entre les 2 méthods
20:07 : Envoie du prompt pour Claude
20:25 : On demande de commencer à faire des benchmark pour faire le choix de l'algo
20:30 : Rendu du premier score, nous choisisons SafePath
20:40 : Demande de points d'amélioration et créer un nouveau benchmark SafePathV1 vs SafePathV2
20:58 : Nous proposons de faire un parcours hamiltonien sur les 10 dernières pommes uniquement pour voir les résultats 
21:10 : On est fatiguées on le laisse se auto performer et faire la technique de faire le choix de l'algo en fonction de la situation
21:23 : La technique retenue est BFS avec cycle hamiltonien pour sécuriser les 223 pommes
21:27 : On continue les tests en accéléré mais on fait en sorte que la décision ne retarde jamais la clock
21:32 : Arrêt du benchmark entre SafePath, Hamilton + raccourci et circuit dynamique (BFS + recherche en profondeur) et continuation avec circuit dynamique
21:44 : On arrête avec une moyenne de 12 minutes