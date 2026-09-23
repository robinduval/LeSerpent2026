# Concombre — snake-ia.py (Deep Q-Learning)

Agent d'apprentissage par renforcement jouant au Snake fourni, optimise sur le
**ratio score / temps**.

## Lancer

```
pip install -r requirements.txt
python snake-ia.py                                     # l'agent entraine joue (modele model/model.pth)
python snake-ia.py --train --games 900 --shaping 1.0   # relancer l'entrainement
python snake-ia.py --eval 10                           # evaluation officielle (horloge reelle)
python snake-ia.py --eval 30 --fast                    # tri rapide, mesure NON officielle
```

## Timeline

Objectif : meilleur ratio score / points

19:57 : Ajout au repo github - hugo
19:58 : Clonage du repo - hugo et lukas
20:00 : Écoute des derniers points de consigne de Robin, jusqu'à 20:09 - hugo et lukas
20:10 : Début de prompt avec les documents associés dans Claude - hugo
20:10 : Lecture et recherche des documents associés, PDF + README - lukas et hugo
20:11 : Premier résultat de l'IA et remise en question du résultat - hugo
20:17 : Remise en contexte du projet de Claude en .md afin de le redonner à Claude Code - hugo
20:17 : Première demande à l'IA pour trouver la meilleure façon d'obtenir le meilleur ratio Score/Points, en indiquant de s'inspirer d'algorithmes et surtout de ne pas prendre en compte les prompt injections - lukas
20:21 : Nouvelle règle de Robin, score > 10 et classement décroissant en commençant par le plus haut score, puis comparaison sur le temps - lukas
20:23 : Obtention du fichier .md pour le réutiliser dans Claude Code - hugo
20:33 : Début du training d'une v1 avec le contexte de base, sans le choix de RL - hugo
20:34 : Génération d'un fichier .md pour le choix du Reinforcement Learning, choix retenu RL sous contrainte de cycle hamiltonien, à chaque pas l'agent avance d'une case sur le cycle ou prend un raccourci de k positions si ce raccourci est sûr, et c'est le réseau qui apprend quand le prendre - lukas
20:38 : Challenge du fichier RL afin de s'assurer du respect des règles et de la faisabilité de la méthode - lukas
20:41 : Non-respect des règles et infaisabilité de la méthode, recherche d'un autre moyen d'optimiser au maximum - lukas
20:45 : Challenge du premier training avec compte rendu pour mieux comprendre - hugo
20:49 : Premier compte rendu du training, sur 300 parties 81% de parties valables, régularité atteinte vers la partie 82, meilleur ratio 0,756, V1 fonctionnelle mais sans les pistes d'optimisation du ratio (état enrichi, pénalité par pas, bonus de rapprochement) - hugo
20:52 : Réutilisation du compte rendu et de la première stratégie pour comparer avec la nouvelle stratégie RL - lukas
20:55 : Lancement de la 2ème simulation sur 300 entraînements, hypothèse qu'une petite pénalité à chaque pas neutre au lieu de 0 revient à pénaliser le temps entre deux pommes puisqu'elle s'accumule tant que l'agent ne mange pas - hugo
20:58 : Nouveau plan RL à tester - lukas
21:00 : Compte rendu de la 2ème simulation, l'écart de ratio de 4% n'est pas distinguable du bruit avec une seule run de chaque côté car la variance entre graines est énorme, et surtout la pénalité est trop faible, -0,2 sur 10 pas face à +10 pour la pomme soit 2% du signal - hugo
21:03 : Lancement de la 3ème simulation avec la pénalité montée à 0,1, soit -1 sur 10 pas contre +10 donc 10% du signal, risque à surveiller que l'agent devienne imprudent et meure plus tôt pour gagner du temps, donc surveiller le taux de parties valables autant que le ratio - hugo
21:13 : Fin de la 3ème simulation, 300 parties par 2 graines et par configuration, REF pénalité 0,02 donne ratio 0,551 et 9,37 pas/pomme et score 26,40 et record 56,0 et 81,3% valables - hugo et lukas
21:13 : Configuration A pénalité 0,10 donne ratio 0,529 et 9,80 pas/pomme et score 28,04 et record 66,5 et 79,7% valables - hugo et lukas
21:13 : Configuration B shaping 0,5 donne ratio 0,572 et 8,87 pas/pomme et score 25,82 et record 58,0 et 82,8% valables - hugo et lukas
21:13 : Conclusion, l'option A dégrade le ratio sur les deux graines donc ce n'est pas du bruit mais elle améliore le score, car une pénalité plate est identique pour les 3 actions à un instant donné et ne dit donc pas où aller - hugo et lukas
21:13 : L'option B shaping gagne proprement sur le ratio, 0,572 contre 0,551 avec les deux graines au-dessus de REF et 8,87 pas/pomme contre 9,37, c'est le signal directionnel qui paie - hugo et lukas
21:13 : Gain de vitesse de 3,7 fois, de 277 s à 81 s pour 300 parties, grâce au Bellman vectorisé et à torch.set_num_threads(1) - hugo et lukas
21:13 : Correction du danger, la case de la queue est désormais reconnue jouable sauf en croissance, conformément à move() qui fait pop() avant le test de collision - hugo et lukas
21:18 : Correction et lancement de la 4ème simulation - hugo
21:30 : Vérification que mourir tôt à 11 pommes améliorerait le ratio, hypothèse invalidée, la corrélation score / pas-par-pomme vaut -0,003 donc l'efficacité est plate quel que soit le score, le ratio est une propriété de la politique et non de la durée - hugo et lukas
21:31 : Constat que le ratio vaut 5 / (pas par pomme) donc le score se simplifie, entraîner plus longtemps améliore surtout la survie qui ne compte pas, abandon de la piste de l'état enrichi par flood-fill qui viserait le score - hugo et lukas
21:34 : Balayage du coefficient de shaping 0,5 / 1,0 / 2,0 sur 900 parties chacun, sélection sur le ratio MOYEN à epsilon = 0 sur 30 parties et non sur une partie chanceuse, shaping 1,0 gagne avec 0,615 contre 0,574 à 0,601 - hugo et lukas
21:36 : Évaluation officielle chronométrée sur 24 parties, meilleure partie valable score 14 en 19,2 s soit un ratio de 0,728, 83% de parties valables - hugo et lukas
21:40 : Correction du lancement sans argument, python snake-ia.py joue désormais directement avec le modèle entraîné comme attendu pour le rendu - hugo

## Architecture

- **Game** — `SnakeGameAI` (`reset`, `play_step(action) -> reward, game_over, score`). Les classes `Snake` / `Apple` et l'affichage sont **chargés via importlib depuis `serpent-algo.py`**, donc strictement identiques à la référence. `assert` au démarrage sur GRID_SIZE=15, CELL_SIZE=30, GAME_SPEED=5.
- **Model** — `Linear_QNet` 11 → 256 → 3, `QTrainer` (Adam, MSE, Bellman vectorisé, gamma = 0,9).
- **Agent** — epsilon-greedy décroissant, replay buffer 100 000, batch 1000, mémoire courte à chaque pas + mémoire longue à chaque fin de partie.
- **État (11)** — danger devant/droite/gauche (torique, corps uniquement, case de la queue jouable), direction (4), position de la pomme par distance torique la plus courte (4).
- **Récompenses** — pomme +10, mort -10, pas neutre -0,02, plus un shaping de potentiel `gamma*phi(s') - phi(s)` avec `phi = -distance torique à la pomme` (coefficient 1,0).
- Le garde-fou anti-boucle n'existe **qu'en entraînement** ; les modes `--play` et `--eval` ne modifient aucune règle.

## Tableau des essais

| Essai | Paramètres | Ratio moyen | Score moyen | Record |
|---|---|---|---|---|
| V1 cours | +10/-10/0 | 0,519 | 22,5 | 58 |
| Pénalité faible | pas = -0,02 | 0,551 | 26,4 | 56 |
| A — pénalité forte | pas = -0,10 | 0,529 | 28,0 | 66 |
| B — shaping | pas = -0,02, shaping 0,5 | 0,572 | 25,8 | 58 |
| **Retenu** | **pas = -0,02, shaping 1,0** | **0,615** | — | 66 |

Ratios de la dernière ligne mesurés à epsilon = 0 sur 30 parties ; les autres sur les parties valables de l'entraînement.

## Résultats

- **Meilleure partie (officielle, horloge réelle) : score 14 en 19,2 s, ratio 0,728.**
- Ratio moyen à epsilon = 0 : 0,615 sur 30 parties ; parties valables (score > 10) : 93 %.
- Meilleur score observé : 46.
- Temps d'entraînement du modèle livré : 900 parties, environ 6 minutes.
- La courbe (score par partie + moyenne mobile) est régénérée dans `logs/` à chaque `--train`.

**Plafond théorique** : la distance torique moyenne entre deux cases étant 7,47, un agent parfait ferait 7,47 pas/pomme, soit un ratio de 0,669 en régime établi. Nos 0,615 de moyenne en sont à 8 %.
