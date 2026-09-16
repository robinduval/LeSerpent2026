# Résultats après le sprint de vingt minutes

Rapport régénéré à 2026-09-16T19:33:59.367733+00:00. Sources JSON conservées dans `runs/`.

L’agent vise d’abord le score officiel maximal. Le temps intervient uniquement pour départager des scores identiques. Le projet n’optimise pas le ratio score/temps.

## Modèle chargé sans argument

`checkpoints/selected.pt` : **pure_rl**, 1,000,000 transitions RL, 249,751 mises à jour.
SHA-256 : `3b3a0bf2cfb1e07e0c796a4439dec853d4e9519c1571f4876f991f31aaa93374`.

Le candidat séparé `checkpoints/hybrid_223.pt` est un réseau Double DQN avec filtre hamiltonien et réorganisation des arcs libres. Sa sécurité est programmée ; le réseau choisit parmi les actions admises. Le filtre utilise seulement le plateau présent. Voir `HAMILTONIAN_SAFETY.md` pour les invariants et leurs limites.

## Essais réellement cadencés à 5 Hz

| Variante | Graine | Points | Secondes écoulées | Pas | Fin |
|---|---:|---:|---:|---:|---|
| RL + filtre local tail2 | 880001 | 153 | 745.465 | 3659 | self_collision |
| RL + filtre local tail2 | 880002 | 171 | 916.519 | 4500 | external_step_limit |
| RL + filtre local tail2 | 880003 | 103 | 281.188 | 1380 | self_collision |
| safety_sprint_pure_880101 | 880101 | 43 | 83.436 | 410 | self_collision |
| safety_sprint_pure_880102 | 880102 | 43 | 96.302 | 473 | self_collision |
| safety_sprint_pure_880103 | 880103 | 44 | 72.632 | 357 | self_collision |
| distillation_sprint_validation_880101 | 880101 | 58 | 124.394 | 611 | self_collision |
| distillation_sprint_validation_880102 | 880102 | 60 | 141.274 | 694 | self_collision |
| distillation_sprint_validation_880103 | 880103 | 76 | 165.472 | 813 | self_collision |
| original_sprint_validation_880101 | 880101 | 57 | 100.246 | 493 | self_collision |
| original_sprint_validation_880102 | 880102 | 60 | 138.923 | 683 | self_collision |
| original_sprint_validation_880103 | 880103 | 82 | 180.849 | 889 | self_collision |
| sprint_rewired_5hz_890001 | 890001 | 46 | 509.292 | 2500 | external_step_limit |
| sprint_explored_5hz_890002 | 890002 | 73 | 407.078 | 2000 | external_step_limit |

`external_step_limit` signifie une interruption expérimentale : le score observé n’est pas une partie terminée ni une victoire. `in_progress_snapshot` est uniquement un instantané daté. Ces essais ont des budgets différents ; cette table ne prétend pas établir un classement statistique global. Les durées ne sont comparables pour le classement qu’à score final identique.

L’ancien test final du réseau pur donnait 74, 69, 51, 64 et 47, moyenne 61, cinq collisions. Il est conservé dans `RESULTS_V1.md` et n’a pas été réutilisé pour régler les nouveaux candidats.

## Apprentissage accéléré — distinct des essais à 5 Hz

| Session | Transitions RL | Mises à jour TD | Épisodes terminés | À 223 | Temps calcul total (s) |
|---|---:|---:|---:|---:|---:|
| hybrid_explored | 200000 | 24985 | 33 | 33 | 30.226 |
| hybrid_explored_time | 200000 | 24985 | 33 | 33 | 31.238 |
| hybrid_optimized | 200000 | 24985 | 31 | 31 | 14.641 |
| hybrid_rewired | 1000000 | 124985 | 125 | 125 | 54.368 |
| hybrid_rewired_seed732 | 200000 | 24985 | 24 | 24 | 11.312 |
| hybrid_rewired_seed733 | 200000 | 24985 | 24 | 24 | 11.339 |
| hybrid_sprint | 50000 | 6235 | 5 | 5 | 3.262 |
| tail2, en plus du préentraînement v1 | 167138 | 10384 | 20 collectés | 7 | 29.060 |

Les poids évoluent pendant ces épisodes : les complétions d’apprentissage ne sont pas un taux de réussite d’un modèle figé. Chaque session hybride comporte en outre 6 000 transitions de démonstration et 500 mises à jour supervisées, comptabilisées séparément dans les métadonnées. Les contrôles fonctionnels accélérés et les ablations de poids présents dans certains répertoires ne constituent pas des temps officiels à 5 Hz.

## Vérification et limites

Python 3.13.15, pygame 2.6.1 ; `pygame.font` importé avec succès. Le lancement natif SDL Cocoa est testé et la fenêtre affiche explicitement « RL + filtre » pour un hybride. Les poids restent immuables pendant le jeu et aucune installation ou mise à jour RL ne se produit au lancement.

Rapport de tests : `runs/test_sprint_report.txt`. Historique, protocole et source scientifique complémentaire : `SPRINT_20_MIN.md`. Les règles, la grille, la croissance différée, les pommes et les 223 points maximaux sont inchangés.

La recevabilité d’un filtre programmé pendant l’évaluation reste une question distincte de l’autorisation d’entraîner librement. Ni première place, ni durée officielle à 223 ne sont déduites d’un temps d’entraînement ou d’un test accéléré.

## Fichiers des mesures

- `runs/safety_sprint_tail2_validation/result_880001.json`
- `runs/safety_sprint_tail2_validation/result_880002.json`
- `runs/safety_sprint_tail2_validation/result_880003.json`
- `runs/safety_sprint_pure_880101/evaluation.jsonl`
- `runs/safety_sprint_pure_880102/evaluation.jsonl`
- `runs/safety_sprint_pure_880103/evaluation.jsonl`
- `runs/distillation_sprint_validation_880101/evaluation.jsonl`
- `runs/distillation_sprint_validation_880102/evaluation.jsonl`
- `runs/distillation_sprint_validation_880103/evaluation.jsonl`
- `runs/original_sprint_validation_880101/evaluation.jsonl`
- `runs/original_sprint_validation_880102/evaluation.jsonl`
- `runs/original_sprint_validation_880103/evaluation.jsonl`
- `runs/sprint_rewired_5hz_890001/evaluation.jsonl`
- `runs/sprint_explored_5hz_890002/evaluation.jsonl`
