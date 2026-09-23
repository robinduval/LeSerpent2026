C'est ici que vous devez décrire votre timeline de la manière suivante (HH:MM : Activité / Constatation / Hypothèse / Réponse).
Règle : une heure : une minute : une instruction par ligne, pas de saut de ligne.

Exemple :
19:00 : Arrivée du prof 
19:10 : Avec Bob, on se demande si l'usage de machin truc permettra de blabla alors nous allons essayer trucmuch...
19:20 : Finalement, ça marche pas, alors on va essayer de bidouiller le petit zinzin

---

# snake-ia.py — Agent DQN guidé par A* (entraînement uniquement)

Un seul fichier, `snake-ia.py`. Le jeu (grille, horloge, scoring) n'est pas
modifié : seul un environnement RL et un agent DQN sont ajoutés autour.

- L'agent qui joue est un **DQN** (réseau de neurones tout petit : 11 entrées
  → 256 → 3 sorties).
- Pendant l'entraînement seulement, un **expert A\*** propose parfois l'action
  à la place du DQN (probabilité qui part de 1.0 et descend à 0.0). Le DQN
  apprend quand même de ces coups-là, comme des siens.
- Pendant l'évaluation et la partie « normale » (`python snake-ia.py` sans
  argument), l'A\* est **totalement désactivé** : seul le DQN entraîné joue.

Rappel : le score officiel du jeu est **1 point par pomme mangée** (non
modifié). Meilleur résultat obtenu jusqu'ici (entraînement A*-guidé, 400
épisodes, seed 0, évalué sur 100 parties DQN seul, sans A\*) :

- **Score : 72 pommes**, atteint en **148 secondes** de jeu (temps réel du
  jeu, cf. `best_score`/`best_score_time` dans `results/eval_*.csv`).
- Score moyen sur les 100 parties : 35.45 — 100% des parties atteignent au
  moins 10 points.

Ces chiffres évolueront à chaque nouvel entraînement (le seed, le nombre
d'épisodes et le hasard font varier le résultat) ; relancez `eval` pour les
mettre à jour.

## Installation

```bash
pip install --user numpy pygame torch
```

(`torch` installe automatiquement la version CUDA si un GPU NVIDIA est
détecté sur la machine ; sinon la version CPU suffit très bien vu la petite
taille du réseau.)

## Commandes

Toutes les commandes se lancent depuis ce dossier (`Courgette/`).

### Jouer directement (le plus simple)

```bash
python snake-ia.py
```

- S'il n'existe pas encore de modèle entraîné (`results/astar_seed0.pth`),
  un entraînement A*-guidé est lancé automatiquement (sans fenêtre, le plus
  rapide possible), puis 5 parties s'affichent avec la fenêtre du jeu, à la
  vitesse officielle (`GAME_SPEED`), **DQN seul, sans A\***.
- Si le modèle existe déjà, cette commande ne fait que le regarder jouer.

### Entraîner un agent

```bash
# Entraînement guidé par l'expert A* (le mode "proposé" du projet)
python snake-ia.py train --mode astar --episodes 400 --seed 0

# Entraînement DQN pur, sans aucune aide (le mode "baseline" de comparaison)
python snake-ia.py train --mode baseline --episodes 400 --seed 0
```

Toujours **sans fenêtre** (le plus rapide possible sur CPU/GPU) : pendant
l'entraînement, aucune interface visuelle n'est ouverte, uniquement des
prints réguliers dans le terminal. Résultats écrits dans :
- `results/<mode>_seed<seed>.pth` : les poids du réseau entraîné
- `results/<mode>_seed<seed>.csv` : une ligne par épisode (score, temps,
  score/temps, reward, epsilon, probabilité d'expert, loss...)

Pendant l'entraînement, un print apparaît environ toutes les 20 secondes
avec le **score/temps moyen des 1000 dernières parties**, pour vérifier
d'un coup d'œil que l'agent progresse (et pas seulement sur la dernière
partie, qui peut être un coup de chance isolé) :

```
[astar] Episode 3120/4000 avg_score/time (last 1000 games)=0.6123 epsilon=0.012 expert_p=0.000 loss=0.842
```

### Évaluer un modèle déjà entraîné

```bash
# Rapide (sans fenêtre) : pour tourner beaucoup de parties d'affilée
python snake-ia.py eval --model results/astar_seed0.pth --games 100 --fast

# Temps réel (fenêtre + horloge officielle du jeu) : pour regarder l'agent jouer
python snake-ia.py eval --model results/astar_seed0.pth --games 5
```

`eval` désactive systématiquement l'A\* et l'exploration aléatoire
(epsilon=0) : le DQN seul prend toutes les décisions. Le résultat est
sauvegardé dans `results/eval_<nom_du_modèle>.csv` et affiche notamment :
score moyen/max, temps moyen, **score/temps** (métrique principale du
projet), % de parties atteignant 10 points, longueur moyenne de partie.

### Comparer baseline vs A*-guidé (plusieurs simulations en parallèle)

```bash
python snake-ia.py compare --seeds 0 1 2 3 4 --episodes 400 --workers 10
```

Lance {baseline, A*-guidé} × chaque seed **en parallèle** (multiprocessing,
un seed = un entraînement complet + une évaluation sur 100 parties), pour
comparer les deux approches sur plusieurs runs plutôt que sur une seule
partie qui pourrait être un coup de chance. Résumé écrit dans
`results/summary.csv` (une ligne par run, plus une moyenne affichée dans le
terminal), directement exploitable pour tracer une courbe.

## Hyperparamètres (tous configurables, rien n'est codé en dur ailleurs)

Voir la classe `DQNConfig` en haut de `snake-ia.py` : `learning_rate`,
`gamma`, `batch_size`, `replay_memory_size`, `epsilon_start/end/decay`,
`target_update_frequency`, `number_of_episodes`, `expert_probability_start/
end/decay_episodes`, `hidden_size`, `stall_limit_factor`, `seed`.

## Récompenses (officielles, non modifiées)

+10 pomme mangée, -10 mort, +100 fin du jeu (plateau rempli), 0 déplacement
normal (choisi plutôt que +0.1, qui inciterait le serpent à tourner en rond
pour engranger du reward au lieu de manger). Le score officiel (nombre de
pommes) n'est jamais changé.

## État (ce que "voit" le DQN)

11 booléens : danger immédiat (tout droit / droite / gauche, en tenant
compte du fait que les bords sont enroulés), direction actuelle (4
booléens), et direction de la pomme par rapport à la tête (4 booléens,
distance la plus courte sur le plateau enroulé). Voir `get_state()` pour le
détail et la justification de ce choix.
