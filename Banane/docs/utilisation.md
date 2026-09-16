# Snake RL — documentation technique

`README.md` est la timeline demandée par le professeur. Toute la documentation
technique est ici.

## Installation

```bash
python3 -m pip install torch numpy matplotlib pygame pytest
```

Pygame n'est nécessaire que pour le replay visuel. L'entraînement et la
recherche d'hyperparamètres n'importent jamais pygame ni matplotlib tant qu'on
n'affiche ou ne trace rien.

## Organisation

| Module | Rôle |
| --- | --- |
| `rules.py` | Constantes officielles : grille 15×15, vitesse 5, 4 actions. Source unique de vérité. |
| `game.py` | Moteur headless. Aucune dépendance graphique, aucune horloge. |
| `state.py` | Vecteur d'état à 11 valeurs, ordre gelé. |
| `model.py` | `DQN` et `DuelingDQN`, choisis par la configuration. |
| `replay_buffer.py` | Mémoire uniforme sur tableaux NumPy préalloués. |
| `prioritized_replay.py` | Mémoire prioritaire (PER) et sélection selon l'algorithme. |
| `agent.py` | Politique epsilon-greedy, apprentissage, target network, checkpoints. |
| `evaluate.py` | Évaluation à epsilon nul, sans apprentissage. |
| `train.py` | Boucle d'entraînement, évaluations périodiques, résumé du run. |
| `grid_search.py` | Campagnes d'hyperparamètres reproductibles et parallèles. |
| `render.py` | Seul module qui importe pygame. Rejoue une trajectoire enregistrée. |
| `plotting.py` | Courbes d'apprentissage (backend `Agg`). |
| `metrics.py` | Journal JSONL, export CSV, sauvegarde des replays. |
| `seed.py` | Plages de seeds séparées et gestion des RNG. |
| `config.py` | Toute expérience est décrite par un `Config` sérialisable. |

Séparation Game / Agent / Model : le jeu ne connaît ni PyTorch ni l'agent,
l'agent ne connaît pas pygame, le modèle ne connaît que des tenseurs.

## Entraîner

```bash
python3 -m snake_rl.train --episodes 2000 --algorithm ddqn --run-id essai
python3 -m snake_rl.train --config runs/essai/config.json   # relance à l'identique
```

Algorithmes : `dqn`, `ddqn`, `dueling_ddqn`, `dueling_ddqn_per`.

Un run écrit dans `runs/<run_id>/` : `config.json`, `metrics.jsonl`,
`metrics.csv`, `summary.json`, `training_curve.png`, `best_mean.pt`,
`latest.pt`, les checkpoints jalons, un JSON par bloc d'évaluation et les
replays du meilleur épisode de chaque bloc.

## Voir jouer le modèle

L'entraînement tourne headless. À intervalle régulier le modèle est figé, un
bloc d'évaluation est joué à epsilon nul, et la **trajectoire** du meilleur
épisode du bloc est enregistrée.

```bash
python3 -m snake_rl.train --episodes 2000 --window       # ouvre Pygame après chaque bloc
python3 -m snake_rl.render runs/essai/replays/best_replay_eval_0500.json
```

Le replay rejoue la trajectoire enregistrée, il ne rejoue pas le modèle : ce
qu'on regarde est exactement la partie qui a produit le score annoncé. La
vitesse accélérée (`--speed`, touches `+` / `-`) est purement visuelle et ne
touche à aucune règle. `--no-window` coupe toute ouverture automatique, ce qui
est obligatoire pendant une campagne ; les replays restent écrits sur disque.

## Rechercher des hyperparamètres

```bash
python3 -m snake_rl.grid_search search_spaces/algorithms.json --workers 4
```

Un espace de recherche est un fichier JSON :

```json
{
  "name": "round1_algorithmes",
  "base": {"episodes": 1200, "eval_interval": 200, "eval_episodes": 20},
  "grid": {"algorithm": ["dqn", "ddqn", "dueling_ddqn"]},
  "seeds": [0, 1, 2],
  "workers": 4
}
```

Chaque combinaison est croisée avec chaque seed. Tous les trials reçoivent le
même budget d'épisodes et les mêmes seeds d'évaluation : c'est ce qui rend la
comparaison honnête, et c'est testé. La campagne écrit `search_summary.csv`,
`search_report.md` et `search_space.json`, qui contient l'espace et
l'environnement d'exécution.

Le pool utilise `spawn` : un contexte CUDA initialisé ne survit pas à un
`fork`. Un trial qui lève une exception est reporté en échec sans interrompre
la campagne.

## Choisir un champion

Le critère principal est le **score moyen d'évaluation à epsilon nul sur
plusieurs seeds**, puis le percentile 10, puis la médiane, puis la variance.
Un record isolé est conservé pour la démonstration et le replay mais ne décide
de rien : une configuration se juge sur sa moyenne à travers les seeds, pas sur
sa seed chanceuse.

## Reproductibilité

Trois plages de seeds strictement séparées (`seed.py`) :

| Plage | Usage |
| --- | --- |
| 0 – 9 | entraînement (les parties utilisent `seed * 100 000 + épisode`) |
| 1000 – 1049 | validation, évaluations périodiques et réglage |
| 10000 – 10099 | benchmark final, jamais vues pendant le réglage |

Aucun hyperparamètre n'est dispersé dans le code : relancer une expérience,
c'est relancer son `config.json`. Une clé inconnue dans un JSON de
configuration lève une erreur au lieu d'être ignorée. Chaque run enregistre sa
configuration, ses seeds, ses métriques, sa durée, ses checkpoints, le commit
Git et la version des bibliothèques.

Limite assumée : le replay buffer n'est pas sauvegardé dans les checkpoints,
une reprise d'entraînement repart donc d'une mémoire vide et n'est pas bit à
bit identique à un run continu.

## Écarts au socle, et pourquoi

1. **Suppression du modulo sur la position de la tête.** Le socle ramenait la
   tête dans la grille, ce qui rendait `check_wall_collision()` inatteignable :
   test exhaustif sur 900 cas dans `tests/test_legacy_wall_bug.py`. Correction
   d'un moteur incohérent avec ses propres règles, pas d'un assouplissement.
2. **Détection de victoire.** La grille pleine était indétectable parce que la
   croissance en attente n'était pas comptée. Le score officiel est inchangé.
3. **`max_steps_without_food` (défaut 500).** Garde-fou technique hors règles
   officielles : à epsilon nul une politique déterministe imparfaite tourne en
   rond et l'évaluation ne se termine jamais. 500 pas pour 225 cases, une
   politique compétente n'est jamais tronquée. Une troncature n'est ni une
   défaite ni une victoire et est comptée à part (`truncation_rate`). Mettre 0
   le désactive. **Le benchmark final s'exécute sous les règles officielles.**
4. **Mode headless.** Ni fenêtre, ni `clock.tick`, ni `sleep`. C'est une
   vitesse de simulation, pas une modification des règles.

Les récompenses (+10 pomme, -10 mort, +0.1 survie, +100 victoire) sont un
signal d'apprentissage. Le **score officiel** reste le nombre de pommes et
n'est jamais calculé à partir d'elles.

## Tests

```bash
SDL_VIDEODRIVER=dummy python3 -m pytest snake_rl/tests -q
```
