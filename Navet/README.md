C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

Exemple :
19:00 : Arrivée du prof 
20:05 : Lecture du sujet socle serpent-algo.py (grille 15x15, GAME_SPEED=5, score+1, tore % GRID_SIZE) / Constat mur fantôme volontaire + prompt injection ignorée / Hypothèse tore à exploiter.
20:10 : Lancement Muse Spark 1.3 dans opencode / Demande plan RL pur sans hardcodé, journal horodaté à validation Navet.
20:25 : Plan d'implémentation validé + lancement premier jet baseline DQN Loeber / Réf prompt init_prompt.md, training libre headless, éval à clock intacte.
20:34 : Baseline 200 parties seed0 mean22.3 max57 last50-30.0 pas/pomme11-12 / Constat eps=0 après 80 mais apprend quand même grâce au tore / Hypothèse target seul ne suffira pas sans exploration.
20:50 : Fix1 target seul 200p mean20.3 last50-27.2 pas de gain / Fix2 eps-decay+target 1000p mean6.0 last100-13.6 en climb final 30pts / Contrôle baseline1000 mean23.1 last100-22.7 plateau eps=0 / Hypothèse exploration paie au-delà de 1000 + gamma 0.9 myope à fixer.
20:39 : Fix3 gamma0.97 1500p mean9.1 last100-17.8 / Eval eps=0 50p baseline23.9 eps25.6 gamma29.4 max51 / Constat train bruité ment, éval dit vrai, chaque fix gagne.
20:46 : Fix4 groupé (truncation faim bootstrap + step-0.01 anti-S) 2000p 98s / Eval eps=0 best mean33.4 max61 record battu / Constat pas/pomme stable 11.5 sans tourner en rond.
20:51 : Plus de tokens gratuits sur opencode / Bascule sur l'abonnement Claude de Tom depuis l'ordi d'Antoine.
20:58 : Volonté de visualiser nos modèles pour y ajouter notre compréhension humaine / Constat les CSV disent le score mais pas le comportement.
20:58 : play_visual.py rendu pygame du socle (couleurs, panneau, GAME_SPEED=5) réutilisant SnakeGame + get_state / Contrôle seed999 scores identiques à evaluate.py 35/404 16/196 48/638 donc la visu ne ment pas.
20:59 : Observation humaine "le serpent est timide à traverser les murs" / Mesure 30 parties 12280 pas : wraps réels 1.0% des pas, chemin court par un mur 26.8% des pas, état 11 bits pointe à l'opposé dans 100% de ces cas / Hypothèse ce n'est pas la durée d'entraînement, l'entrée ment.
21:02 : Fix5 bits pomme en distance torique (dx=(ax-hx)%15, droite si dx<15-dx) flag --torus-food, défaut off pour rejouer les modèles <=fix4 / Constat la visu a trouvé un bug que 2000 parties de training n'avaient pas révélé.
21:04 : train_parallel.py N seeds en processus séparés OMP_NUM_THREADS=1 / Choix multi-run plutôt que multi-worker car réseau 11-256-3 ne sature pas un coeur, et donne l'écart-type inter-seed qui manquait aux runs mono-seed.
21:06 : Lancement fix5 5 seeds x 20000 épisodes + contrôle ctrl20k 3 seeds x 20000 sans fix5 / But séparer l'effet du fix de l'effet de la durée, les deux ayant changé en même temps.
21:12 : watch_training.py suivi live lecture seule des CSV (barres, last200, max, eps, ETA, courbe ASCII) / Correction vitesse réelle 790 ép/min pas 290, ETA 25 min pas 70, la 1re mesure incluait l'import torch.
21:14 : Constat à 5000 ép fix5 last200 17.5 vs ctrl20k 13.7 écart-type 0.6 sur les deux / Hypothèse le fix torique mord déjà, écart 3.8 très supérieur au bruit inter-seed.
21:20 : Arrêt runs à 8600/20000 sur demande / Fix bug mesure diag_wrap le "100% opposé" ignorait --torus-food donc mentait sur les modèles fix5.
21:22 : Mesure comportement fix4 vs fix5 wrap 1.2%->7.1% des pas, pas/pomme 12.1->8.8 / Constat le fix change vraiment la politique.
21:25 : Eval eps=0 50p fix5 s1 38.9 max67 s3 max72 MAIS mean 5 seeds 26.8 ecart-type 7.7 vs contrôle 28.1 ecart-type 4.7 / Constat écart -1.4 plus petit que le bruit, fix5 et contrôle indistinguables en score, le 38.9 est un tirage pas une preuve.
21:26 : Constat fix5 s5 pas/pomme 138 tourne en rond survit par famine / Hypothèse -0.01/pas insuffisant, et le vrai plafond est l'état 11 bits qui ne voit ni la queue ni les impasses.
21:21 : Constat décisif éval 50 morts sur 50 = corps zéro famine / Hypothèse le plafond n'est ni la durée ni le tore mais les 3 bits de danger qui ne voient qu'une case devant, à 70 pommes le serpent fait 73 cases.
21:22 : Fix6 état 11->16 bits flood-fill espace libre par direction + distance queue + longueur, flag --rich-state, coût 12us vs 2us / Contrôle démarrage 1500ép fix6 3.26 vs fix5 3.46 courbes superposées pas de régression.
21:23 : Lancement fix6 5 seeds x 20000 --torus-food --rich-state.
21:24 : Ajout temps par score pour le rendu, métrique = steps/GAME_SPEED donc durée à la clock du socle indépendante de notre machine, temps CPU loggé à part / evaluate.py colonnes temps_jeu_s sec_par_pomme temps_cpu_ms + bilan.py tableau multi-modèles.
21:25 : Bilan fix5_s1 30p score moy 39.8 max 67 temps moy 102s temps du record 206s 2.56s/pomme 15ms CPU/partie.
21:26 : Kill seeds 4 et 5 du run fix6 pour alléger le Mac, 3 runs restants seeds 1-3 / Choix des seeds les plus basses et non des mieux classées car les 5 étaient à égalité 1480ép, sélectionner après coup aurait biaisé le résultat / Conséquence n=3 au lieu de 5 donc écart-type inter-seed moins fiable, à dire au prof.
21:54 : Génération courbes.py 6 figures PNG depuis results/*.csv, install matplotlib manquant / Choix moyenne glissante 200 sur les runs car score train bruité par eps>0, et points par seed + barre d'erreur sur les évals.
21:55 : Correction 2 défauts de lisibilité repérés à l'inspection des PNG, note qui chevauchait la barre d'erreur fig3 et annotation sur le titre fig1 / Ajout annotation piège de lecture baseline monte vite car eps=0 dès 80 mais plafonne.
21:56 : Constat fig2 fix6 (bleu) ~42 vs fix5 et contrôle ~22-24, 3 seeds resserrées, max 100/99/101 contre 72 avant / Hypothèse contrairement à fix5 l'écart dépasse largement le bruit inter-seed, le flood-fill a mordu.
22:10 : Arrêt fix6 à 18333/20000 sur demande / Eval eps=0 50p s1 mean88.1 max119 s2 86.0 max116 s3 85.5 max108.
22:12 : Constat fix6 mean 86.6 écart-type inter-seed 1.1 vs contrôle 28.1 soit +58.4 pour un bruit de 4.7 / Réponse écart 12x le bruit, premier fix concluant statistiquement contrairement à fix5 qui était dans le bruit.
22:13 : Régénération des 6 figures avec fix6 + main.py lance désormais model_fix6_s1_best avec --torus-food --rich-state, flags déclarés en un seul endroit pour éviter le piège modèle-joué-avec-le-mauvais-état.
22:14 : Fix chevauchement panneau pygame remplissage/temps repéré sur capture à 70 pommes / Constat les scores à 3 chiffres ne rentraient plus dans la mise en page prévue pour 2.
22:15 : Constat fix6 meurt encore à 100% par morsure comme fix4 / Hypothèse le flood-fill repousse le plafond de 33 à 88 pommes mais n'élimine pas la cause, prochaine piste serait un horizon plus long ou un état qui voit la queue accessible.
