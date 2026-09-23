20:00 : Les premières pistes sont : approche naïve par cycle hamiltonien, et approche plus optimisée avec un « cell tree », comme variante hamiltonienne.

20:00 : On élimine le cycle hamiltonien naïf comme stratégie finale : il garantit un parcours sûr, mais impose trop de déplacements. On le conserve comme comparateur.

20:02 : En parallèle, on audite le code pour ne pas faire la même erreur que la dernière fois : vérification des commentaires, des bugs et des incohérences, notamment sur les collisions et la croissance.

20:10 : Correction des problèmes trouvés et vérification des règles : traversée des bords sur une grille torique, croissance différée, score et fin de partie.

20:18 : Prompt généré, lancement de l’agent pour implémenter une solution purement algorithmique, sans apprentissage et sans accès aux pommes futures.

20:29 : La technique du cycle hamiltonien reconfigurable, avec optimisation locale 2-opt et exploration de 64 transformations, nous permet d’atteindre le score maximal de 223 en 19 minutes. Le corps reste un segment contigu du cycle pour garantir la sécurité.

20:33 : On fait des recherches pour fixer un objectif clair. Le résultat précédent de 19 minutes est encourageant, mais reste objectivement très long : il faut réduire les déplacements tout en conservant le score maximal.

20:37 : Conclusion, avec la clock du repo à 5 mouvements/seconde : un bon algo vise 15 minutes, un très bon algo 10 minutes, soit respectivement 4 500 et 3 000 déplacements. On vise donc les 15 minutes pour commencer.

20:39 : Amélioration du premier résultat avec une nouvelle stratégie : davantage de diversité des cycles et un meilleur suivi de la queue. On explore plusieurs branches, évite les configurations déjà visitées et élargit les transformations avec Or-opt, qui déplace des blocs libres.

20:45 : Calcul de la distance de Manhattan adaptée au tore pour estimer la distance idéale sans obstacle. Elle donne une borne inférieure vers la pomme actuelle et aide à réfléchir à un nouvel objectif de temps, sans prédire les pommes suivantes.

20:54 : Exploration d’autres paradigmes et du compromis entre temps et risque de ne pas finir la partie. On teste notamment les recherches pendant le trajet, le 3-opt et l’anticipation du déplacement de la queue.

20:55 : Prendre plus de risques pourrait permettre de mieux performer sur une partie donnée. Pour la solution retenue, on privilégie néanmoins la fiabilité : chaque mouvement conserve une continuation certifiée et progresse vers la pomme.

21:14 : Super run : 223 points en 878 secondes, soit environ 14 minutes 38 secondes à la cadence x1.

21:20 : Plusieurs algorithmes sont évalués en parallèle ; on examine les logs des meilleures graines, mais aussi les moyennes, les complétions et les collisions pour ne pas choisir sur une seule bonne partie.

21:26 : Meilleur résultat observé de l’agent retenu : graine 15, score 223 en 821 secondes, soit 4 105 déplacements et 13 minutes 41 secondes. Les durées x1 sont calculées à 5 mouvements/seconde à partir des simulations accélérées. Cette meilleure partie ne signifie pas que la cible de 15 minutes est atteinte en moyenne.
