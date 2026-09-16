# Chou — Snake Double DQN

> L’agent vise d’abord le score officiel maximal. Le temps intervient uniquement pour départager des scores identiques. Le projet n’optimise pas le ratio score/temps.

Le jeu fourni est un tore 15 × 15, sans murs mortels, à 5 déplacements par seconde. Le score compte les pommes. La croissance est différée d'un mouvement : le maximum exact est **223 points**, et non 222. Ces règles, y compris leurs cas limites, sont conservées. L'agent choisit ses actions avec un réseau réellement entraîné par reinforcement learning ; il n'est pas remplacé par un solveur lorsque ses poids manquent.

**Modèle livré :** Double DQN `ordered`, graine d'entraînement 23, 1 000 000 transitions et 249 751 mises à jour. Il a été sélectionné sur trois parties de validation distinctes du test final : scores 65, 66 et 61 à la borne expérimentale de 600 déplacements (trois interruptions, pas des victoires). Les cinq parties réservées ont ensuite donné **74, 69, 51, 64 et 47 points**, soit **61 de moyenne**, médiane 64, avec **cinq collisions et aucune complétion**. Le record de ce test est 74 points en 173,542 s. Le maximum de 223 et une première place ne sont donc pas démontrés. Les huit sessions d'entraînement totalisent 10 millions de transitions sans recompter les reprises ; 38 tests techniques passent.

## Lancement du professeur

Depuis la racine du dépôt **LeSerpent2026** :

```sh
source chou/.venv/bin/activate
python --version
python -c "import pygame.font"
python chou/serpent-algo.py
```

Python doit être **3.13.x**. Le lancement sans argument ouvre la fenêtre, charge `chou/checkpoints/selected.pt` sur CPU et joue à 5 Hz, sans téléchargement ni entraînement. Échap ou la fermeture de fenêtre quittent proprement ; Espace recommence après la fin. Les chemins sont résolus depuis le fichier, pas depuis le dossier courant. Le chronomètre d'affichage continue après la fin comme dans le moteur fourni ; la durée à la fin est aussi enregistrée.

Le répertoire initialement nommé `Chou` a été normalisé en `chou` pour que le contrat de lancement soit portable sur les systèmes sensibles à la casse. Les fichiers supprimés dans les autres groupes avant cette intervention n'ont pas été restaurés ni modifiés.

### Installation explicite sur une autre machine

Le venv est local et ignoré par Git. Il ne doit pas être copié depuis une ancienne position du projet. Avec `uv` disponible, depuis la racine du dépôt :

```sh
UV_CACHE_DIR=chou/.uv-cache UV_PYTHON_INSTALL_DIR=chou/.uv-python uv venv --python 3.13 chou/.venv
UV_CACHE_DIR=chou/.uv-cache uv pip install --python chou/.venv/bin/python -r chou/requirements.txt
source chou/.venv/bin/activate
python --version
python -c "import pygame.font"
python chou/serpent-algo.py
```

Aucun Python ou pip global n'est employé. `.python-version` contient `3.13`; Pygame est fixé à `2.6.1` et utilise une wheel compatible. PyTorch et NumPy sont requis ; un GPU ne l'est pas. `requirements.lock.txt` enregistre l'environnement mesuré. Tests et graphiques utilisent les dépendances optionnelles de `requirements-dev.txt`.

## Architecture

- `serpent-algo.py` : point d'entrée, vérification Python 3.13 et modes facultatifs.
- `snake_rl/env.py` : transitions identiques au moteur d'origine, score indépendant des récompenses.
- `state.py` : référence `classic11`, puis représentation `ordered` à 248 valeurs. Elle ajoute une grille 15 × 15 centrée sur la tête avec les rangs du corps, les deltas toriques de pomme, les dangers absolus, les rayons libres, la longueur et la croissance différée. Aucun prochain tirage ni état du générateur n'est observé. Les rangs donnent un ordre de libération, pas un délai garanti.
- `agent.py` et `replay.py` : MLP à deux couches cachées de 128 neurones, quatre actions absolues, Double DQN, replay uniforme ou PER, retours 1-step ou n-step. Réseau cible indépendant, cibles sans gradients, Huber loss, clipping des gradients et sauvegardes CPU avec signature des règles.
- `training.py` : entraînement accéléré et borné, explicitement autorisé par l'utilisateur. Sauvegardes de reprise avec replay, optimizer, RNG et compteurs.
- `evaluation.py`, `metrics.py`, `ui.py` : évaluation sans exploration ni mises à jour, cadence originale, classement lexicographique, interface.
- `baselines.py` : comparateurs **non appris**, jamais un remplacement implicite du modèle RL.

Aucun masque de sécurité, démonstrateur ou simulateur de recherche ne choisit les actions du réseau livré. Une demande de demi-tour est une action définie : le moteur l'ignore et continue, comme dans l'original. Le replay apprend cet effet de l'action demandée ; les traces distinguent demande et direction effective.

## Récompenses et limites d'apprentissage

Les précisions de l'utilisateur autorisent à modifier les récompenses d'entraînement, tout en conservant les règles et les points du jeu. Le réglage expérimenté donne +10 par pomme, −10 par collision, +100 à la victoire et −0,02 par déplacement ordinaire. Un shaping potentiel est ajouté uniquement à l'apprenant :

```text
Phi(s) = −0,2 × distance de Manhattan torique à la pomme
r_apprentissage = r_env + gamma × Phi(s_suivant) − Phi(s)
Phi(terminal) = 0 ; gamma = 0,99
```

Le retour actualisé n'est pas une preuve d'optimalité selon le classement officiel. La sélection utilise les points réellement obtenus. `rl_return` conserve le retour de l'environnement, `learning_return` celui reçu par l'apprenant dans les journaux d'entraînement. Les retours ne sont pas comparés entre configurations comme des scores officiels.

Les collectes sont interrompues après 2 000 pas ou 500 pas sans pomme. **Ces bornes ne sont pas des timeouts du jeu.** Elles ferment le segment de collecte tout en conservant le bootstrap non terminal et en vidant les suffixes n-step. Le jeu sans argument n'a pas cette limite. Une reprise commence une nouvelle partie avec une nouvelle graine ; le learner reprend ses poids, son optimizer, son replay et son exploration. Il s'agit d'une continuation statistique, pas d'une reproduction bit à bit de toute la session.

## Commandes supplémentaires

```sh
# Entraîner une nouvelle expérience, sans écraser le modèle livré
python chou/serpent-algo.py --train --encoder ordered --run-id mon_entrainement --seed 31 --transitions 1000000

# Reprendre le learner d'une expérience, depuis une nouvelle partie
python chou/serpent-algo.py --train --run-id ma_reprise --resume chou/runs/mon_entrainement/resume.pt --transitions 100000

# Évaluer à 5 Hz, sans fenêtre ; ajouter --display pour voir la partie
python chou/serpent-algo.py --evaluate --run-id mon_test --seed 30001 --episodes 5 --max-steps 2000 --trace

# Une partie évaluée sans interruption externe
python chou/serpent-algo.py --evaluate --run-id sans_borne --seed 30001 --episodes 1 --max-steps 0 --display

# Comparateur algorithmique explicite
python chou/serpent-algo.py --evaluate --baseline greedy --run-id comparaison_greedy --seed 30001 --episodes 5 --max-steps 2000

# Validation technique et coût CPU, qui n'est pas un résultat de jeu
python chou/serpent-algo.py --smoke-test
python chou/serpent-algo.py --benchmark
PYTHONPATH=chou python -m unittest discover -s chou/tests -v

# Graphiques et synthèses à partir des seuls fichiers effectivement produits
python chou/scripts/report.py
```

`--per` et `--n-step 3` permettent les ablations. Le fichier `resume.pt` est un checkpoint de travail volumineux ignoré par Git ; les checkpoints d'inférence sont autonomes. Les poids absents, corrompus ou incompatibles provoquent une erreur explicite, sans installation automatique ni remplacement par un algorithme.

## Évaluation et provenance

Les expériences, effectifs, graines, coûts, résultats individuels, modèle retenu et limites sont décrits dans [docs/RESULTS.md](docs/RESULTS.md). La provenance vérifiable du modèle par défaut figure aussi dans `checkpoints/selection.json`.

L'entraînement utilise des graines distinctes de la validation et du test final réservé. Les essais comparatifs utilisent les mêmes graines et le même budget. Les modalités officielles d'agrégation n'étant pas données, la convention locale compare moyenne des scores, médiane, quartile inférieur, puis complétions. Le temps intervient seulement en cas de résultats de score identiques, pour des parties naturellement terminées. Les scores de parties interrompues sont visibles comme mesures à budget fixé ; leur admissibilité au classement du professeur n'est pas présumée. Le meilleur record individuel ne sélectionne pas un modèle.

Les essais de validation peuvent s'exécuter dans des processus concurrents, chacun avec sa propre clock à 5 Hz. Les durées réellement mesurées et la cadence observée sont enregistrées ; le test final utilise cinq processus indépendants, chacun à 5 Hz, sans entraînement concurrent. Ce protocole est local et son coût de décision est enregistré. Aucun temps d'entraînement accéléré n'est présenté comme temps officiel de partie. Les CSV/JSON séparent score, retour RL, pommes, longueur, pas, cause de fin, coût d'entraînement et latence de décision.

L'admissibilité de poids préentraînés, d'observations enrichies, le nombre d'essais officiels et le relevé exact du temps restent à confirmer auprès du professeur. Le réseau à onze informations est conservé comme référence ; aucune première place n'est annoncée sans compétition comparable.

## Audit, recherche et historique

[Audit des règles](docs/RULES_AUDIT.md) · [Sources scientifiques](docs/RESEARCH.md) · [Changelog](CHANGELOG.md) · [Timeline](docs/TIMELINE.md)

Le fichier original est préservé sans modification dans `baseline/serpent_original.py`, accompagné de la licence Apache 2.0 du dépôt. Les articles et le dépôt pédagogique sont attribués dans RESEARCH.md. Aucune performance publiée n'est attribuée à ce projet.
