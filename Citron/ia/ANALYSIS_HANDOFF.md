# Contexte pour l'agent d'analyse

Nous disposons de deux heures, début supposé le 16/09/2026 à 20:00 Europe/Paris.
Objectif : meilleur score sur une batterie de parties, atteindre au moins 10
points ; à score égal, ratio score/temps de la partie correspondante.
La taille de la batterie officielle n'a pas encore été précisée.

## Règles actuelles

Mise à jour utilisateur à 21:25 : priorité au score maximal indépendamment du
ratio score/temps. Nouvelle expérience de fine-tuning : epsilon minimum 0,01 et
learning rate 1e-4, les autres réglages conservés. Les valeurs effectives sont dans
config.json et les checkpoints ; les valeurs 0,05 / 3e-4 ci-dessous décrivent la
baseline. Deux réglages changent ensemble : ce n'est pas une ablation individuelle.

Grille 15 × 15 torique, serpent initial de longueur 3. Score +1 par pomme.
Croissance au déplacement suivant. Les règles du fichier de base sont conservées.
Nouvelle autorisation utilisateur : entraînement accéléré et plusieurs serpents.
Évaluations à 5 Hz, modèle figé, epsilon 0, sans mise à jour ni masquage d'actions.
Budget et temps d'entraînement toujours mesurés réellement ; les ratios de jeu
d'entraînement sont des équivalents calculés à 5 Hz, non des résultats de jury.

## Méthode effectivement exécutée

Double DQN CPU, état 13 valeurs : dangers tout droit/droite/gauche prenant en
compte la queue et la croissance, direction one-hot, signe du déplacement torique
le plus court vers la pomme sur deux axes, longueur / 225, croissance en attente.
Trois actions relatives : tout droit, droite, gauche. Réseau 13 → 128 → 3, ReLU.
Adam 3e-4, gamma 0,99, Huber loss, clipping gradients 10 ; cible copiée toutes
les 500 transitions. Replay uniforme 20 000 ; batch 64 ; apprentissage dès 64
transitions puis une mise à jour par transition. Epsilon 1 → 0,05 en 8 000
transitions, commun aux environnements et conservé à la reprise.
Récompenses : déplacement +0,1 ; pomme +10 ; mort -10 ; victoire +100.
Pas de reward shaping, curriculum, imitation, algorithme de secours ni action masking.
Huit environnements entrelacés, inférence par lot, un modèle et replay partagés.
Ce n'est pas huit processus parallèles. Les durées d'épisodes se chevauchent.

## Lecture des fichiers

Variante ajoutée : `features=spatial` dans config.json et le checkpoint signifie
22 entrées. Les 13 premières restent identiques ; puis viennent 3 espaces
accessibles / 225, 3 distances au corps / 15, dx/7, dy/7 et Manhattan/14.
Composantes libres calculées sur le tore avec le corps figé après mouvement,
en respectant grow_pending. Ce n'est ni un planificateur ni un masque d'actions.
`warm_start` identifie le modèle basic transféré. Nouvelles colonnes de poids
initialisées à zéro, replay et optimiseur réinitialisés, compteurs/epsilon conservés.
Cette remise à zéro est un facteur de comparaison à expliciter. Les checkpoints
distincts intermédiaires restent dans le run local ; l'export inclut latest.pt.
Les runs sans champ features utilisent l'état basic historique de 13 valeurs.

`episodes_all.csv` rassemble les épisodes enregistrés dans les snapshots JSON.
`completed=false` signale une interruption : exclure ces lignes des scores de
parties terminées. Les runs et les modes doivent être analysés séparément.
`wall_duration_s` / `wall_score_per_second` : temps et ratio bruts chronométrés.
`game_duration_s` / `game_score_per_second` : steps / 5 et ratio correspondant pour
train ; durée chronométrée et ratio réel pour eval. `game_timing_kind` les distingue.
`session_elapsed_s` : temps monotone de session, sans addition des temps d'épisodes.
`training_steps` est cumulatif dans la chaîne des checkpoints ; `session_steps`
ne compte que les nouvelles transitions (disponible sur les runs multi-serpents).
`time_to_10_s` est un temps brut réel ; aucun historique du pas exact de la dixième
pomme n'est enregistré. Ne pas le présenter comme temps équivalent à 5 Hz.
Les identifiants d'épisodes sont attribués au lancement, l'ordre des lignes est
l'ordre des fins, qui diffère en multi-environnement.

Les checkpoints contiennent poids online/target, optimiseur, replay, RNG et compteurs.
Un checkpoint final d'entraînement n'est pas nécessairement celui ayant réalisé
le record historique : les poids changent pendant chaque partie d'entraînement.
`config.json` consigne chaque lancement et son parent. `manifest.json` contient
versions et empreintes SHA-256. Le code inclus fait autorité sur les détails.
Un snapshot pris pendant un batch peut contenir un checkpoint plus ancien que les
métriques. Le script ne bloque ni ne modifie le processus d'entraînement.

## Analyse demandée

1. Séparer progrès d'entraînement et qualité du modèle figé. Comparer les
   batteries à taille et seeds identiques ; le record dépend du nombre d'essais.
2. Tracer scores, moyenne mobile, taux ≥10 selon les transitions et le temps réel.
3. Proposer trois expériences prioritaires avec hypothèse, coût, critère de succès
   et protocole à budget égal. Garder un baseline et changer un élément à la fois.
4. Examiner exploration, stabilité des valeurs Q, pièges spatiaux et effets du
   bonus de survie. Les logs n'incluent pas les valeurs Q : demander cette mesure
   avant de conclure à une divergence. Ne pas modifier le scoring imposé.
5. Examiner l'enrichissement spatial de l'état, les n-step returns ou le replay
   prioritaire en fonction des échecs observés ; ne pas empiler sans ablation.
6. Signaler les limites du socle : victoire vérifiée au placement de pomme alors
   que croissance différée ; absence de limite sans pomme ; pas de garantie de
   terminaison. Les épisodes tronqués par la durée de session restent incomplets.

Livrable attendu : diagnostic étayé, expériences classées par gain espéré/coût,
modifications précises proposées et métriques de validation. Ne pas écraser les
checkpoints ni modifier les fichiers utilisés par un batch actif.

Pour refaire un export pendant les batchs : `python3 export_analysis.py`.
