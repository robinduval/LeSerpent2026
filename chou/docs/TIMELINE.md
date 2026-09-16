# Timeline réelle — 16 septembre 2026

Cette timeline décrit cette seule session. Elle ne prétend pas dater le travail précédent de l'équipe. Fuseau : Europe/Paris.

| Étape | Trace |
|---|---|
| Inspection initiale | Lecture du moteur local, du README, du PDF de cours et de la demande utilisateur ; constat du dossier Chou et des suppressions préexistantes dans les autres groupes. |
| Audit et recherche | Copie intacte du moteur dans baseline, règles détaillées dans RULES_AUDIT, huit articles et dépôt pédagogique consultés dans RESEARCH. |
| Clarification utilisateur | Entraînement accéléré et récompenses libres autorisés ; évaluation maintenue à 5 Hz. |
| Première implémentation | Moteur, DDQN et tests développés en parallèle ; ordre des fichiers et expériences conservé dans les journaux, sans commits fictifs. |
| Clarification environnement | Exigence Python 3.13 via uv ; ancien chou/.venv supprimé et recréé, import pygame.font et rendu de texte vérifiés. |
| 20:43:36 | Benchmark CPU daté dans runs/hardware.json ; les coûts seuls ne constituent pas une mesure de score. |
| Expérimentations | Chaque runs/*/config.json porte son started_at ; les journaux learning.jsonl indiquent transitions, updates et temps écoulé. |
| Validation et choix | Chaque partie figure dans evaluation.jsonl ; les graines et le nombre de pas sont explicites. selection.json conserve le critère et la provenance du modèle retenu. |
| Livraison | Tests finaux, essai graphique, rapports et limites consignés dans RESULTS.md. |

Les entraînements accélérés ne sont pas convertis en durées de jeu. Les résultats des tests avec horloges simulées ne sont pas des mesures expérimentales.

Événements horodatés de fin de session :

- 2026-09-16T21:05:44+0200 : lancement du test réservé, cinq processus indépendants à 5 Hz, poids sélectionnés figés.
- 2026-09-16T21:06:18+0200 : lancement exact sans argument depuis la racine du dépôt, Python 3.13.15.
- 2026-09-16T21:07:14+0200 : arrêt propre du lancement exact, code retour 0, SHA du checkpoint inchangé.
- 21:09:42 : rapport final produit après cinq parties terminées naturellement : moyenne 61, maximum 74, aucune complétion ; 38 tests techniques réussis.

## Sprint demandé de 21:14 à 21:34, le 16 septembre 2026 (Paris)

- 21:14 : trois recherches parallèles : sécurité locale, cycle hamiltonien, apprentissage RL dans l'ensemble d'actions admissibles.
- 21:18 : trois essais du candidat local figé lancés à 5 Hz sur les graines 880001–880003, budget externe fixé à 4 500 pas.
- 21:20–21:24 : cycle réorganisable, entraînements indépendants, tests de croissance différée et preuve de progression ; ablations du réseau seul et essais de distillation.
- 21:21 puis 21:25 : qualifications à 5 Hz de deux candidats à cycle, respectivement bornées à 2 500 et 2 000 pas pour le budget restant.
- 21:26–21:29 : preuve causale du rôle des poids ; rejet comme modèle principal de la variante fortement planifiée dont le réseau ne fait presque aucun choix. Gel du candidat hybride plain-rewired, poids à 500 000 transitions.
- 21:29–21:34 : vérification native, tests, comparaison appariée, reconstruction des décisions des anciens journaux, documentation et synthèse des mesures effectivement terminées.

Les temps d'entraînement accéléré, les contrôles fonctionnels et les durées officielles à 5 Hz restent séparés. Aucun essai interrompu n'est compté comme une victoire.
