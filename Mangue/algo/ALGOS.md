# État de l'art : top 3 des algorithmes pour notre Snake

> Contexte (voir `PIEGES.md`) : tore 15×15 (225 cases, impair), seule mort = auto-morsure, croissance différée d'un coup, 5 coups/s. Objectif : **victoire garantie** (225 cases) en **un minimum de coups**. Le code du jeu fait foi et n'est pas modifié.

## 1. Ce que dit la littérature

| Constat | Source |
|---|---|
| Jouer de façon optimale est **NP-difficile** (collecte de toute la nourriture sur grilles pleines) et PSPACE-complet sur grilles générales | De Biasi & Ophelders, *The Complexity of Snake*, FUN 2016 |
| Savoir si un graphe est « gagnable » est NP-difficile, même sur grilles ; les graphes gagnables non hamiltoniens ont une maille ≤ 6 | Graafsma, Manthey & Skopalik, *Playing Snake on a Graph*, arXiv 2506.21281, 2025 |
| Le cycle fixe (« Almighty Move ») est le seul à gagner à 100 % sur 10×10 ; A* + forward checking est plus rapide mais pas fiable | Kong & Aguilar Mayans, *Automated Snake Game Solvers via AI Search Algorithms*, UC Irvine CS271 |
| Hybride BFS, puis A* avec forward checking, puis cycle en fin de partie (10×10) | Sagar et al., *Solving the Classic Snake Game Using AI...*, IJARCCE 2020 |
| Recherche de graphe (plus court chemin + retour vers la queue) : 94 % de victoires sur 6×6 ; RL : 50 % | chuyangliu/snake (GitHub) |
| RL (DQN) : score moyen ≈ 9,5 sur 12×12, très loin du remplissage | Tushar & Siddique, arXiv 2301.11977, 2023 |
| **Benchmark de référence** sur 30×30, 100 parties, nombre de coups pour remplir la grille (tableau ci-dessous) | twanvl/snake (Twan van Laarhoven, GitHub) |

Benchmark twanvl/snake (30×30, coups pour finir la partie) :

| Agent | Moyenne | Défaites | Gain vs cycle fixe |
|---|---|---|---|
| Cycle fixe (zig-zag) | 202 530 | 0 % | 1× |
| Cycle perturbé (PHC) | 103 497 | 0 % | ≈ 2,0× |
| DHCR (réparation dynamique du cycle) | 60 690 | **3 %** | ≈ 3,3× |
| **Cell tree (variant)** | **47 918** | 0 % | **≈ 4,2×** |

**Conclusion** : pas d'optimum calculable, donc heuristiques. Toutes les méthodes qui gagnent à 100 % reposent sur un **cycle hamiltonien** (fixe ou modifié). RL, glouton, A* et MCTS seuls sont exclus : aucune garantie de remplissage.

## 2. Top 3 pour notre contexte

Estimations 15×15 = extrapolation des ratios 30×30 appliqués à notre cycle pur (≈ 12 500 coups calculés). À valider par simulation.

| # | Algorithme | Coups estimés (15×15) | Temps à 5 fps | Garantie | Difficulté |
|---|---|---|---|---|---|
| 1 | **Cycle hamiltonien perturbé (PHC)** sur tore | ≈ 6 000 | ≈ 20 min | 100 % | Faible |
| 2 | **Cell tree** (arbre couvrant de blocs 2×2) | ≈ 3 000 | ≈ 10 min | 100 % | **Élevée** (grille impaire) |
| 3 | **DHCR** + repli sur cycle sûr | ≈ 3 800 | ≈ 13 min | < 100 % sans repli | Moyenne à élevée |

### #1 Cycle hamiltonien perturbé (Tapsell, 2015)

- **Principe** : fixer un cycle hamiltonien et numéroter les cases. Le serpent suit l'ordre du cycle mais peut **sauter en avant** (raccourci vers la pomme) si la case cible reste **entre la tête et la queue dans l'ordre du cycle**, avec une marge.
- **Règles d'origine** : distance disponible = distance(queue → tête) - longueur - 3 ; réduire encore si la pomme est sur le trajet ; **raccourcis désactivés au-delà de 50 %** de remplissage.
- **Adaptation à notre jeu** :
  - Cycle torique déjà vérifié (`RIGHT`/`DOWN` uniquement), compatible avec le corps initial.
  - Voisins toriques pour les raccourcis (4 voisins partout, pas de murs).
  - Marge **+1** pour la croissance différée (la queue ne bouge pas au coup qui suit un repas).
  - Raccourci choisi = voisin qui minimise la distance **dans l'ordre du cycle** jusqu'à la pomme, sous contrainte de sécurité.
- **Pourquoi en premier** : garantie mathématique, code court, sert de **filet de sécurité** aux deux autres.

### #2 Cell tree (twanvl/snake)

- **Principe** : grille découpée en blocs 2×2. Le serpent « roule à droite » dans chaque bloc, donc 2 coups possibles par case. Les blocs visités forment un **arbre couvrant** dont le contour est toujours un cycle hamiltonien : le cycle est **reconstruit en continu** vers la pomme. Plus court chemin sous contraintes (aller seulement vers le parent ou un bloc non visité) et heuristique d'accessibilité (une zone peut être temporairement coupée si le passage de la queue la libère).
- **Performance** : meilleure mesurée, ≈ 4,2× plus rapide que le cycle fixe, 0 % de défaite.
- **Obstacle majeur** : 15×15 est **impair**, donc impossible à paver en blocs 2×2. Piste : 7×7 blocs sur une zone 14×14 + une bande en L de 29 cases (ligne 14 et colonne 14) insérée comme couloir fixe dans le cycle, en exploitant l'enroulement. **Conception à prouver avant de coder.**

### #3 DHCR : Dynamic Hamiltonian Cycle Repair (Haidet / AlphaPhoenix, 2020)

- **Principe** : maintenir un cycle hamiltonien, suivre le plus court chemin vers la pomme en pondérant les arêtes pour rester proche du cycle existant, puis **réparer** le cycle autour de la déviation.
- **Performance** : ≈ 3,3× plus rapide que le cycle fixe, mais **3 % de défaites** sur 30×30.
- **Adaptation** : n'accepter un coup que si un cycle hamiltonien valide passant par tout le corps existe encore après ce coup (vérifiable en moins de 200 ms sur 225 cases), sinon repli sur le coup PHC.

## 3. Écartés

| Approche | Raison |
|---|---|
| RL (DQN, PPO, torch) | Aucun résultat publié de remplissage complet ; non déterministe ; hors du périmètre « purement algorithmique » |
| Glouton / BFS / A* seuls | Rapides en début de partie mais se piègent (≈ 90 à 94 % de victoires sur petites grilles) |
| A* + queue atteignable, puis cycle en fin de partie | Le passage au cycle n'est pas garanti depuis une configuration quelconque |
| MCTS / Minimax | Aucune preuve de gain en solo déterministe ; coûteux |
| « Multi-Algorithm Approach to Snake Game » (OpenReview, NeurIPS 2025, anonyme) | Soumission non validée ; métriques incohérentes (cycle hamiltonien qui ne remplit pas 20×20) |

## 4. Plan recommandé

1. Simulateur sans affichage fidèle au jeu (croissance différée, tore, une direction par coup), graine fixe.
2. **PHC** = référence + repli (livrable sûr).
3. **Cell tree** adapté 15×15, ou **DHCR** sécurisé si l'adaptation échoue.
4. Banc : 1 000 graines, 0 défaite exigée, comparaison du nombre moyen de coups et du pire cas.

## Sources

- De Biasi, M., Ophelders, T. *The Complexity of Snake*. FUN 2016. https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FUN.2016.11
- Graafsma, D., Manthey, B., Skopalik, A. *Playing Snake on a Graph*. arXiv 2506.21281, 2025. https://arxiv.org/abs/2506.21281
- Kong, S., Aguilar Mayans, J. *Automated Snake Game Solvers via AI Search Algorithms*. UC Irvine CS271. https://cpb-us-e2.wpmucdn.com/sites.uci.edu/dist/5/1894/files/2016/12/AutomatedSnakeGameSolvers.pdf
- Sagar, P. et al. *Solving the Classic Snake Game Using AI for Training Electronic Sport Players*. IJARCCE 2020. https://ijarcce.com/wp-content/uploads/2020/07/IJARCCE.2020.9615.pdf
- Merrill, P. *Snake: Artificial Intelligence Controller*. UNH CS730, 2011. https://www.cs.unh.edu/~ruml/cs730/paper-examples/merrill-2011.pdf
- Tushar, M.R., Siddique, S. *A Memory Efficient Deep Reinforcement Learning Approach for Snake Game*. arXiv 2301.11977, 2023. https://arxiv.org/abs/2301.11977
- *Perturbed hamiltonian-cycle-based algorithms to solve snake game*. NTU Singapore (contenu non consulté). https://dr.ntu.edu.sg/entities/publication/5892b710-89c3-43f6-a9e4-d8c834f07bae
- Tapsell, J. *Nokia 6110 Part 3: Algorithms*. 2015. https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/
- van Laarhoven, T. *twanvl/snake: Snake playing agents*. https://github.com/twanvl/snake
- chuyangliu. *snake*. https://github.com/chuyangliu/snake
