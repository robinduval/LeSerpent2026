C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

Exemple :
19:50 : On commence nos lectures des resources fournies + setup du repo
20:11 : On commence les recherches sur Deep Q
20:22 : On décide qu'on va essayer de faire DQN + algo -> On fait des recherches sur les heuristiques /stratégies qu'on peut utiliser pour opti l'entrainement
20:46 : On explore une stratégie "Double DQN + garde fou A*".
20:55 : On valide la stratégie en question.
20:55 : On Lance l'implémentation "Double DQN + garde fou A*".
21:00 : pendant que l'implémentation est en cours, on fait + de recherches pour comprendre & vulgariser stratégie choisie.
21:07 : On teste que le code produit compile (oui) & on commence a fix le garde fou...
21:15 : L'implémentation fonctionne mais l'entrainement est lent , on essaye d'aller + loin que 600 coup / s 
21:20 : On est a ~900 coup /s , c'est mieux. On lance l'entrainement.
21:35 : On a un plateau sur l'entrainement. On arrive a gagner 100pourcent des parties en ~32min a 5cps.

Conclusion : 

Exercice intéressant, bon prétexte pour apprendre le RN + très intéressant d'explorer des stratégies un peu + pointues et d'allier rl & algos. Notre stratégie choisie (Double dqn + guardrails) est un peu trop safe au final.
