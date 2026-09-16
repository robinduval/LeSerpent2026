# Chou — Snake RL, objectif 223

> L’agent vise d’abord le score officiel maximal. Le temps intervient uniquement pour départager des scores identiques. Le projet n’optimise pas le ratio score/temps.

Le moteur fourni est conservé : tore 15 × 15, croissance différée, pommes aléatoires, maximum exact **223**, une transition à **5 Hz**. Python **3.13** est requis.

Le sprint a produit un **Double DQN avec filtre hamiltonien réorganisable**. Le filtre conserve un cycle sûr et une progression finie vers chaque pomme. Le réseau choisit parmi les actions admissibles : dans un contrôle de 7 790 états, 311 avaient plusieurs choix et changer les poids modifiait 172 décisions. La sécurité vient explicitement du code du filtre. L’entraînement de cette architecture a terminé 125 épisodes sur 125 à 223 ; ce n’est pas un taux de réussite en évaluation à poids figés.

Résultats réellement cadencés à 5 Hz, collisions et interruptions : [RESULTS.md](docs/RESULTS.md). Voir la [preuve des invariants](docs/HAMILTONIAN_SAFETY.md), le [protocole](docs/SPRINT_20_MIN.md) et les [anciens résultats](docs/RESULTS_V1.md). Aucun temps officiel à 223 ni première place n’est annoncé sans mesure correspondante.

## Lancement

Depuis la racine **LeSerpent2026** :

```sh
source chou/.venv/bin/activate
python --version
python -c "import pygame.font"
python chou/serpent-algo.py
```

Sans argument, `checkpoints/selected.pt` conserve le **RL pur** qualifié. L’autorisation donnée concernait explicitement l’entraînement ; l’acceptation d’un filtre pendant l’évaluation a été demandée et reste à confirmer. La comparaison appariée du sprint conserve ce modèle : **57, 60, 82** contre **58, 60, 76** pour la nouvelle distillation.

La variante visant la complétion est livrée séparément, prête à jouer :

```sh
python chou/serpent-algo.py --checkpoint chou/checkpoints/hybrid_223.pt
```

Elle charge **500 000 transitions RL et 62 485 mises à jour Double DQN**, après 6 000 transitions de démonstration et 500 mises à jour supervisées. À ce checkpoint, les 62 épisodes d'entraînement terminés ont atteint 223 ; le fragment suivant de 491 pas était encore en cours. Elle fonctionne sur CPU, sans téléchargement ni apprentissage au lancement. La fenêtre indique « RL + filtre ». Échap ou fermeture quittent ; Espace recommence après la fin. Aucun timeout n’est ajouté. Le chronomètre continue après la fin comme dans la source ; le temps terminal est enregistré séparément.

Le candidat `hybrid_tail2.pt`, à filtre local, reste disponible pour comparaison et peut perdre. Les expériences `hybrid_explored` avec planification plus forte ne sont pas retenues comme solution RL principale : leur réseau intervient trop peu.

## Installation explicite

Le venv déplacé a été recréé avec `uv` et Python 3.13 ; il est ignoré par Git. Sur une autre machine, depuis la racine :

```sh
UV_CACHE_DIR=chou/.uv-cache UV_PYTHON_INSTALL_DIR=chou/.uv-python uv venv --python 3.13 chou/.venv
UV_CACHE_DIR=chou/.uv-cache uv pip install --python chou/.venv/bin/python -r chou/requirements.txt
source chou/.venv/bin/activate
python --version
python -c "import pygame.font"
```

Pygame est fixé à 2.6.1 ; PyTorch et NumPy sont déclarés. Aucun Python ou pip global n’est utilisé. `.python-version` contient `3.13`. Tous les fichiers restent dans `chou/`. Les suppressions préexistantes des autres groupes sont préservées.

## Reproduction

```sh
# Le calendrier d’exploration dépend du budget total : conserver un million.
# Le modèle proposé est le checkpoint intermédiaire à 500 000 transitions.
python chou/scripts/train_hybrid.py --steps 1000000 --seed 731 --save-every 100000 --reward progress --output chou/runs/nouvelle_session_hybride

# Partie à 5 Hz jusqu’à son terme, identifiant de session inédit.
python chou/serpent-algo.py --evaluate --checkpoint chou/checkpoints/hybrid_223.pt --run-id nouvelle_evaluation --episodes 1 --seed 910001 --max-steps 0 --trace --display

# Nouvel apprentissage RL pur, sans écraser les poids livrés.
python chou/serpent-algo.py --train --encoder ordered --run-id nouveau_rl_pur --seed 31 --transitions 1000000

PYTHONPATH=chou python -m unittest discover -s chou/tests -v
python chou/serpent-algo.py --smoke-test --checkpoint chou/checkpoints/hybrid_223.pt
python chou/scripts/sprint_report.py
```

Une limite `--max-steps` non nulle est une interruption de mesure externe, signalée dans les résultats. Les durées ne départagent que des scores finaux identiques. Le barème officiel d’agrégation de plusieurs parties demeure inconnu.

## Architecture et provenance

- `env.py` : moteur vérifié contre `baseline/serpent_original.py`, conservé avec sa licence.
- `agent.py`, `replay.py`, `state.py` : RL pur, représentation ordonnée du corps, Double DQN, replay uniforme/PER et n-step.
- `hybrid_agent.py`, `cycle_shield.py` : réseau à valeurs d’action et cycle réorganisable ; aucun accès au RNG des pommes.
- `policy_io.py`, `evaluation.py`, `ui.py` : type de checkpoint explicite, inférence seule, journaux, cadence et interface.
- `scripts/`, `tests/`, `runs/` : expériences, vérifications et mesures brutes.

Récompense du modèle hybride livré : `−0,05 + progrès cyclique/15 + 2/pomme + 100/complétion`. Elle ne modifie pas les points du jeu. La démonstration initialise une estimation de distance à la pomme présente. Les cibles Double DQN séparent sélection en ligne et évaluation par le réseau cible, sans gradient, avec masque des actions admissibles.

Recherche : [RESEARCH.md](docs/RESEARCH.md), notamment Double DQN, DQfD et Safe RL via Shielding. Source complémentaire : [John Tapsell, cycles hamiltoniens pour Snake](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/). Les garanties locales sont justifiées séparément.
