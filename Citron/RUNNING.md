# Snake RL

Architecture nommée comme les slides dans `snake-ia.py` : `Game` (PyGame),
`Linear_QNet` (Torch) et `Agent`. `Game.play_step(action)` renvoie récompense,
fin de partie et score. `Agent.model.predict(state)` donne l'action sans
exploration ; `Agent.get_move` ajoute l'exploration lors de l'entraînement.
Les checkpoints existants restent compatibles, avec les mêmes clés et calculs.

Après récupération du dépôt, entrer dans `Citron`. Installer les dépendances une
fois : `python -m pip install -r requirements.txt` (Python 3.10 ou plus récent).

Partie autonome avec le modèle entraîné inclus :

```sh
python snake-ia.py
```

Sans option : mode évaluation, chargement automatique de `model/model.pt`, fenêtre
PyGame, jeu autonome à 5 Hz, epsilon zéro et aucune mise à jour. La démo n'est pas
limitée à l'horaire du projet : elle fonctionne aussi après 22 h. Fermer la fenêtre
pour arrêter. Une partie est jouée par défaut (`--episodes` pour en enchaîner).
Fichiers indispensables à publier : `snake-ia.py`, `serpent-algo.py`,
`requirements.txt`, `model/model.pt` et `model/metadata.json`.
Pas besoin de `runs/`, `exports/`, ni du PDF de cours sur la machine de démonstration.

La fenêtre montre le serpent et les métriques. Chaque session produit un dossier
`runs/<horodatage>-eval/` ou `-train/` contenant `dashboard.html`, `episodes.csv`,
`metrics.json` et `config.json`. Seul l'entraînement enregistre `latest.pt`.
La limite de deux heures à partir de 20 h reste appliquée à l'entraînement ;
`--budget-start` permet de la régler explicitement.

Continuer l'apprentissage en remplaçant le chemin par celui de votre session :

```sh
python3 snake-ia.py --mode train --resume runs/<session>/latest.pt --episodes 100
```

Nouvelle règle : clock accélérable et plusieurs serpents autorisés à l'entraînement.
Exemple : huit environnements indépendants, inférence groupée, replay et modèle
partagés, pendant au maximum 60 secondes réelles :

```sh
python3 snake-ia.py --mode train --resume runs/<session>/latest.pt --num-envs 8 --headless --episodes 1000 --max-seconds 60 --train-fps 0
```

Les environnements sont entrelacés sur un seul CPU, sans huit processus système.
`--train-fps 0` enlève la limite ; `--train-fps 5` reproduit l'ancienne cadence
pour chaque environnement. Le multi-environnement affiche ses métriques dans le
dashboard et n'ouvre pas de fenêtre Pygame. Une interruption de session est
enregistrée séparément des parties terminées, sans pénalité ni faux score final.
Le checkpoint est sauvegardé toutes les 30 secondes en multi-environnement et à
la fin de la session. Aucun paramètre d'apprentissage n'a été changé pour cet essai.

Évaluation sur une batterie de 10 parties, sans exploration ni apprentissage :

```sh
python3 snake-ia.py --mode eval --resume runs/<session>/latest.pt --episodes 10 --seed 1000
```

Comparer des batteries de même taille, mêmes seeds et mêmes conditions. Les
graphiques de chaque session restent séparés pour ne pas mélanger entraînement et
évaluation. Garder les dossiers précédents pour comparer les versions.

Le checkpoint restaure poids, optimiseur, réseau cible, replay buffer, nombre de
déplacements d'entraînement et état aléatoire de l'agent. La prochaine partie
redémarre ; le jeu interrompu n'est pas repris. Changer `--seed` pour changer les
placements de pommes des nouvelles parties.

## Règles conservées

Le programme importe `Snake` et l'affichage de `serpent-algo.py`, qui reste intact.
Grille 15 × 15, bords traversables, croissance différée et score +1 par pomme sont
conservés. En évaluation, `clock.tick(5)` s'applique aussi avec `--headless`,
et un seul environnement est autorisé. En entraînement la cadence est configurable.
Aucune limite de
déplacements sans pomme, modification de grille ou protection des actions n'est
ajoutée. Récompenses d'apprentissage : déplacement +0,1 ; pomme +10 ; mort -10 ;
victoire +100. Elles sont séparées du score en pommes.

Double DQN : état de 13 valeurs (11 indicateurs + longueur normalisée + croissance
en attente), couche cachée 128, replay 20 000, batch 64, Adam 3e-4, gamma 0,99,
clipping 10, synchronisation cible tous les 500 déplacements. Apprentissage dès
64 transitions ; epsilon de 1 à 0,05 sur 8 000 déplacements. Aucun pré-entraînement.

## Mesures

Objectif : meilleur score d'une batterie, seuil 10 ; à égalité, temps de la partie
correspondante. Le CSV conserve score, durée monotone, score/seconde, seed, nombre
de pas, epsilon, loss, mises à jour, récompense, temps jusqu'à 10 et cause de fin.
Le tableau de bord montre aussi le taux de parties à 10 points et le budget réel
restant. `duration_s`, `time_to_10_s` et `session_elapsed_s` utilisent une horloge
monotone réelle, jamais `steps / fps`. `throughput_steps_s` utilise toutes les
transitions de la session divisées par son temps réel, sans additionner les durées
des environnements simultanés. Les scores/seconde d'entraînement accéléré ne sont
pas comparables aux ratios d'évaluation à 5 Hz. Un seul épisode ne permet pas une
prévision fiable du score à deux heures.

Le dashboard affiche désormais en priorité la durée équivalente à 5 Hz
(`steps / 5`) et le ratio correspondant pour les parties d'entraînement. C'est une
estimation de la trajectoire enregistrée, pas une nouvelle évaluation. En mode
évaluation, il utilise le temps mesuré. Les champs bruts CSV/JSON `duration_s` et
`score_per_second` conservent leurs valeurs en temps réellement écoulé ; le débit
de calcul est séparé des performances de jeu dans le dashboard.

Limite héritée du socle : la victoire est vérifiée au replacement de la pomme,
alors que la croissance est différée. Ce comportement n'a pas été modifié.

Tests : `python3 -m unittest test_snake_ia.py -v`.

## Variante spatiale (22 entrées)

Poursuite orientée score maximal, sans changer les récompenses du jeu :

```sh
python3 snake-ia.py --mode train --resume runs/<spatial>/latest.pt --num-envs 8 --headless --episodes 1000000 --max-seconds 300 --epsilon-floor 0.01 --learning-rate 0.0001
```

Ces deux réglages sont persistés dans le checkpoint et la configuration. Ils
réduisent les actions aléatoires et l'amplitude des mises à jour en fin
d'entraînement ; leur supériorité doit être vérifiée en évaluation figée.
Une réduction de l'exploration peut améliorer le score d'entraînement sans
améliorer la politique : ne pas confondre les deux. La cadence d'évaluation et les
récompenses ne changent pas. Le ratio score/temps n'est plus l'objectif de sélection.

```sh
python3 snake-ia.py --mode train --features spatial --warm-start runs/<baseline>/latest.pt --num-envs 8 --headless --episodes 100000 --max-seconds 60 --seed 20000
python3 snake-ia.py --mode eval --resume runs/<spatial>/latest.pt --episodes 3 --seed 1000
```

Les neuf nouvelles entrées sont trois proportions d'espace accessible, trois
distances au corps dans les directions d'action, les deux distances signées à la
pomme et sa distance de Manhattan torique, normalisées. Le flood fill traite le
corps comme figé après le mouvement ; il estime l'espace, sans garantir la survie.
Ni filtrage des actions, ni changement des règles ou récompenses. L'évaluation
reste à 5 Hz. Les observations sont mises en cache pour éviter leur recalcul
quand le plateau n'a pas changé.

`--warm-start` reprend les poids basic avec poids zéro sur les nouvelles entrées,
ce qui conserve initialement les prédictions. Le replay et l'optimiseur sont
réinitialisés : les transitions précédentes ne contiennent pas la géométrie
nécessaire. Compteurs et epsilon sont conservés. Ce changement de replay est à
prendre en compte dans les comparaisons. `--resume` détecte automatiquement le
format du checkpoint ; les modèles basic restent compatibles et conservés.
En entraînement multi-environnement, un checkpoint distinct est désormais
conservé toutes les 30 secondes en plus de `latest.pt`.

## Export pour un autre agent

Exécuter `python3 export_analysis.py`, y compris pendant un entraînement actif.
Le dossier et ZIP `exports/analysis-<date>-<heure>` contiennent les snapshots des
sessions, les checkpoints disponibles, les configurations, le code, la méthode
`ANALYSIS_HANDOFF.md`, `summary.json`, `episodes_all.csv` et les empreintes des fichiers.
Les snapshots en cours sont partiels ; les parties interrompues sont identifiées.
Fournir le ZIP à l'autre agent en lui demandant de suivre `ANALYSIS_HANDOFF.md`.
