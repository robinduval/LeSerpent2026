C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse). Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

Exemple : 19:00 : Arrivée du prof 19:10 : Avec Bob, on se demande si l'usage de machin truc permettra de blabla alors nous allons essayer trucmuch... 19:20 : Finalement, ça marche pas, alors on va essayer de bidouiller le petit zinzin

20:14 : lancement de claude sur le repo et avec les slides.
20:16 : réponses aux questions de contexte de claude.
20:18 : Bloc Game (play_step) branché sur serpent-algo.py sans le modifier ; les murs sont traversants (modulo) donc seul le corps tue ; agent aléatoire de référence : record 3, moyenne 1,04 sur 200 parties.
20:20 : Bloc Model (Linear_QNet 11 → 256 → 3 + équation de Bellman) validé : la cible converge (Q = 10) ; CPU 118 µs/pas contre MPS 422 µs/pas → on reste sur CPU.
20:21 : Annonce du système de classement (score > 10 et meilleur score gagne ou pour même score moins de temps gagne). CPU plus performant que le GPU pour ce cas selon test de claude.
20:25 : Bloc Agent (état à 11 valeurs, epsilon-greedy, mémoire de rejeu) : 300 parties en 13 s ; évaluation sur 200 parties : record 66 en 111 s, moyenne 29,5, 0,58 pomme/s ; les 200 parties finissent enfermées dans le corps.
20:33 : Premier modèle installé (model/v1-11-entrees) + mode démo avec écran GAME OVER comme le jeu de base.
20:34 : Clarification des optimisation de temps possible.
20:40 : Hypothèse : le serpent s'enferme car il ne voit que 3 cases ; ajout de l'espace libre (flood fill) pour chaque action → état à 14 valeurs ; moyenne 29,5 → 84,8, record 66 → 123 ; coût 0,1 ms par coup (budget 200 ms à 5 FPS).
20:42 : Seed 3 à 300 parties : 10 boucles infinies sur 200 parties → écarté (en partie réelle une boucle ne se termine jamais) ; 600 parties n'apporte rien de net en V2.
20:43 : Constat : les pommes/s baissent (0,58 → 0,37) mais à score égal V2 est aussi rapide que V1 (30 pommes en 50 s contre 49 s) → la baisse vient seulement de la survie plus longue.
20:44 : Validation du chrono avec la vraie clock : 97,4 s réels contre 95,8 s prédits par eval (coups / 5), écart 1,7 %.
20:48 : Gabriel commence à avoir du mal avec le français. Une V2 commence a donner des résultats plus intéressant avec un meilleur score simuler à 120.
20:50 : La commande du prof « python Fraise/snake-ia.py » lance la partie sans argument, avec logs (score, temps) dans le terminal.
20:52 : Bug trouvé en test : un Ctrl+C pendant l'entraînement de secours lançait la partie avec un modèle incomplet → corrigé (arrêt + suppression du modèle partiel).
20:58 : Partie réelle (python Fraise/snake-ia.py) : 108 pommes en 04:22 avant interruption manuelle ; chrono conforme à 0,5 % près ; gros détours au-delà de 45 % de remplissage.
21:01 : Hypothèse : le flood fill voit tout le corps comme un mur fixe alors que la queue libère des cases ; nouvelle version propagée coup par coup qui tient compte de la queue (V3).
21:06 : Seed 3 : passer de 300 à 600 parties fait chuter la moyenne de 91,9 à 33,5 (le réseau « oublie ») → on choisit le modèle par évaluation et non par le record d'entraînement.
21:12 : Mode rapide x10 (temps compté = temps réel x10) : avec clock.tick() le temps est gonflé de 15,2 % (≈ 3 ms d'imprécision par image multipliés par 10).
21:13 : Passage à clock.tick_busy_loop() en mode rapide uniquement : écart 0,0 % en x10 et -0,1 % en x20.
21:15 : Espace libre tenant compte de la queue : moyenne 84,8 → 102,2, record 123 → 156 (seed 1, 600 parties)
21:16 : Test en x1000 : 107 pommes mais 28:36 affichés pour 5:43 de jeu réel (+400 %) car une image dure 0,2 ms pour ~1 ms de calcul → x20 retenu.
21:17 : Logs simplifiés au score et au temps uniquement.
21:19 : Score soumis : 110 pommes en 04:34.7 (modèle V3, mode rapide x20).
21:20 : 141 en 8min 31s
21:22 : Session de 6 parties en x20 : 141, 94, 97, 108, 136, 111 → moyenne 114,5, cohérente avec l'évaluation (médiane 105).
