# CHANGELOG Navet

## 2026-09-16 20:25 — Étape 1 : baseline DQN Loeber (proposée, en attente test torch)
- Quoi : `game.py` (tore `% GRID_SIZE`, score+1, reward interne +10/-10/+100/+0.1 isolée), `model.py` (Linear 11-256-3, gamma 0.9, sans target), `agent.py` (état 11 bits tore-aware, eps 80-n_games), `train.py` + CSV + checkpoints.
- Pourquoi : courbe "avant" pour mesurer chaque fix ensuite.
- Flaws volontaires : eps meurt à 80, step +0.1 pousse au S, timeout=mort, pas de target.
- Résultat : pas encore chiffré (torch absent sur cette machine) — à lancer : `pip3 install -r requirements.txt && python3 train.py --episodes 200`.
## 2026-09-16 20:34 — Run baseline chiffré (seed0, 200 épisodes, .venv torch 2.8.0)
- Résultat : mean 22.27, max 57, last50 30.04, pas/pomme ~12.0, causes 199 corps / 1 famine. 27.5s. CSV `results/run_baseline_e0.csv`.
- Interprétation : apprend malgré eps=0 après 80 (tore pardonne), mais variance forte, pas de victoire.
## 2026-09-16 20:34 — Fix 1/4 : target network seul (sync 20 parties, seed0, 200 épisodes)
- Quoi : `QTrainer(target_model)`, `--use-target`, cible `maxQ_target(s')` détachée.
- Résultat : mean 20.3, max 55, last50 27.22, pas/pomme 11.4, 200 corps. CSV `results/run_target_e1.csv`. 13.9s.
- Interprétation : pas de gain significatif seul — attendu, exploration morte + gamma myope dominent. Stabilité à confirmer via loss/variance, pas via seul score. Prochain : fix eps plancher.
## 2026-09-16 20:50 — Fix 2/4 eps-decay + runs longs M5 (1000 parties)
- Quoi : `eps_mode=decay`, `eps=0.05+0.95*exp(-n/400)`, gardé target sync20, gamma toujours 0.9.
- Résultats : `run_eps_decay_e2.csv` mean 5.98 max 38 last100 13.64 last50 15.58 (climb final : 24pts à 850, 30pts à 1000). `run_baseline_1000.csv` contrôle mean 23.12 max 54 last100 22.71 plateau eps=0.
- Interprétation : exploration continue dégrade le score court (pas d'overfit chanceux) mais courbe montante après 800 parties vs baseline qui plafonne. Gamma 0.9 reste le goulot (horizon ~10 pas). Prochain Fix 3/4 : gamma 0.97.
## 2026-09-16 20:39 — Fix 3/4 gamma 0.97 + éval propre eps=0
- Quoi : `--gamma 0.97`, 1500 parties, target+eps-decay gardés. `run_gamma097_e3.csv` mean 9.1 max 37 last100 17.81 (train bruité par exploration).
- Éval 50 parties eps=0 : baseline1000 23.92 (max43), eps-decay 25.58 (max48), gamma0.97 29.42 (max51).
- Interprétation : le score train ment quand on explore ; l'éval dit que chaque fix gagne (+1.7 puis +3.8). Prochain Fix 4/4 : séparer truncation famine vs terminal mort + coût/pas -0.01 anti-S.
## 2026-09-16 20:46 — Fix 4/4 groupé (truncation bootstrap + coût/pas -0.01)
- Quoi : `play_step` retourne `terminated` vs `truncated`, cible Q bootstrappe sur faim (`r + gamma*maxQ(s')`), pas de bootstrap sur mort corps. `step_reward = -0.01` remplace `+0.1`. 2000 parties M5 en 98s (`run_fix4_grouped.csv`).
- Éval 50 parties eps=0 :
  - `model_fix4_last.pth` : mean 23.36, max 51, pas/pomme 11.58
  - `model_fix4_best.pth` : **mean 33.40**, **max 61**, pas/pomme 11.50
- Progression ablations (éval eps=0) : Baseline 23.92 -> Fix1+2 (target+eps) 25.58 -> Fix3 (+gamma) 29.42 -> Fix4 (+truncation+coût-0.01) **33.40 (max 61)**.
- Anti-S validé : `pas/pomme` reste à ~11.5 (bien loin des 225 de la grille). L'agent fonce sur la pomme au plus court sans serpentiner.

## 2026-09-16 21:04 — Visualisation : `play_visual.py`
- Quoi : rendu pygame du socle (tête orange / corps vert / pomme rouge, panneau score-remplissage-temps, `GAME_SPEED=5` par défaut) qui fait jouer un `.pth`. Réutilise `SnakeGame` et `Agent.get_state` tels quels, aucune règle redupliquée. Contrôles : Espace pause, ↑/↓ vitesse, Échap quitter. Affiche la décision du réseau (`Q: tout droit / droite / gauche`).
- Contrôle d'intégrité : sur seed 999, les 3 premières parties donnent score/steps/cause **identiques** à `evaluate.py` (35/404, 16/196, 48/638, toutes `corps`). Ce qui s'affiche est ce que mesure l'éval.
- Pourquoi : les CSV donnent le score, pas le comportement. Ajouter l'œil humain sur la politique apprise.

## 2026-09-16 21:10 — Diagnostic : le serpent est aveugle au tore, pas timide
- Observation humaine via la visu : « il traverse très peu les murs alors que c'est la mécanique intéressante ».
- Mesure (`diag_wrap.py`, model_fix4_best, 30 parties / 12280 pas / 1052 pommes) :
  - wraps réels : **1.0 %** des pas
  - situations où le chemin court passe par un mur : **26.8 %** des pas
  - ... dont l'état 11 bits pointe **à l'opposé** : **100.0 %** de ces cas
- Cause : les bits pomme comparaient des coordonnées brutes (`apple[0] < head[0]`). Tête en x=1, pomme en x=13 → l'état dit « à droite » alors que le chemin court fait 3 cases à gauche en traversant le mur.
- Conséquence méthodo : un training plus long n'aurait rien donné, on aurait entraîné plus longtemps sur une entrée qui ment. La visu a trouvé en une partie ce que 2000 parties de training n'avaient pas révélé.

## 2026-09-16 21:15 — Fix 5/5 : bits pomme en distance torique
- Quoi : `dx = (apple[0]-head[0]) % GRID_SIZE`, `food_right = dx < GRID_SIZE-dx` (idem y). Flag `--torus-food` sur `train.py`, `evaluate.py`, `play_visual.py`, `diag_wrap.py`.
- Défaut `False` : les modèles ≤ fix4 restent rejouables à l'identique (l'état doit matcher l'entraînement, sinon on évalue un modèle avec une entrée qu'il n'a jamais vue).
- Résultat : en attente du run 20k.

## 2026-09-16 21:20 — `train_parallel.py` : N seeds en parallèle
- Quoi : lance N `train.py` en processus séparés, `OMP_NUM_THREADS=1` chacun, puis résume last-10 % et **écart-type inter-seed**.
- Choix : parallélisme **entre** runs, pas dedans. Le réseau 11-256-3 ne sature pas un cœur, donc multi-worker sur un run ne gagnerait presque rien ; ce qui coûte c'est le nombre d'épisodes joués. Bonus : nos comparaisons de fixes étaient mono-seed, on ne savait pas si un écart était du signal ou du bruit — maintenant si.
- Mesure : ~790 épisodes/min par run une fois lancé, 8 runs simultanés sur 10 cœurs à ~100% CPU chacun. 20k épisodes ≈ 25 min en parallèle, contre ~3 h 20 en séquentiel. (Première estimation à 290/min fausse : mesurée sur la 1re minute, qui incluait l'import torch.)

## 2026-09-16 21:22 — Runs longs lancés (en cours)
- `fix5` : 5 seeds × 20000 épisodes, `--torus-food`, gamma 0.97, eps-decay 2000, eps-min 0.02, truncated, step -0.01.
- `ctrl20k` : 3 seeds × 20000 épisodes, **sans** `--torus-food`, tout le reste identique.
- Pourquoi le contrôle : durée et fix changent en même temps. Sans ctrl20k on ne pourrait pas dire lequel des deux a produit le gain.
- À mesurer à l'arrivée : éval eps=0 50 parties, et surtout `diag_wrap.py` — le taux de wrap doit décoller du 1.0 % si le fix mord.

## 2026-09-16 21:20 — Arrêt des runs à mi-parcours + éval honnête
- Runs stoppés à ~8600/20000 épisodes (fix5) et ~8000/20000 (ctrl20k) sur demande. Les `_best.pth` sont intacts (sauvegardés à chaque record).
- **Correction d'un bug de mesure dans `diag_wrap.py`** : le compteur « état pointe à l'opposé » était calculé sur les coordonnées brutes *quel que soit* le modèle, donc il affichait 100 % même pour un modèle torus-food dont l'entrée est correcte. Il dépend maintenant de `--torus-food`. Les 100 % du diagnostic de 20:59 restent valides (ils portaient bien sur fix4, sans le flag).
- **Wrap réel, fix4 vs fix5 s4** (10 parties) : 1.2 % → **7.1 %** des pas. Pas/pomme 12.1 → **8.8**. Le fix mord sur le comportement.
- **Éval eps=0, 50 parties** :
  - fix5 : s1 **38.9** (max 67), s2 23.9, s3 29.6 (max 72), s4 26.0, s5 15.4 → mean **26.8**, écart-type **7.7**
  - ctrl20k : s1 27.8, s2 22.6, s3 34.0 → mean **28.1**, écart-type **4.7**
  - rappel fix4 (2000 ép) : 33.4 (max 61)
- **Conclusion honnête : l'écart fix5 vs contrôle (-1.4) est plus petit que le bruit inter-seed (~7.7). Sur n=5 vs n=3, les deux sont indistinguables en score.** Le `38.9` de s1 est un tirage favorable, pas une victoire démontrée.
- Ce que le fix change de façon nette et reproductible : le **comportement** (wrap ×6, pas/pomme -27 %). Ce que le fix ne démontre pas encore : un gain de **score**.
- Leçon méthodo : sans les 5 seeds + le contrôle, on aurait publié « 38.9, record battu ». C'est précisément le piège que les runs mono-seed des étapes 1-4 nous tendaient.
- Alerte : fix5 s5 a un pas/pomme de **138** (contre ~9-12 ailleurs) — il tourne en rond et survit par famine tronquée. Le `-0.01`/pas ne suffit pas toujours à l'en dissuader.

## 2026-09-16 21:22 — Fix 6/6 : état enrichi (flood-fill) — la vraie cause du plafond
- **Diagnostic décisif** : en éval, **50 morts sur 50 sont `corps`**. Zéro famine, zéro autre cause. Les 3 bits de danger ne voient qu'**une case** devant ; à 70 pommes le serpent fait 73 cases et entre dans des poches sans issue qu'il ne peut pas voir.
- Quoi : état 11 → **16 bits**, flag `--rich-state`. Ajoute par direction l'**espace libre atteignable** (BFS borné à `2*len(body)`, normalisé), la **distance à la queue** et la **longueur relative**. Réseau `Linear_QNet(16,256,3)`.
- Coût : 12 µs/appel contre 2 µs (négligeable devant le pas d'optim torch).
- Défaut `False` : tous les modèles ≤ fix5 restent évaluables et rejouables.
- Contrôle de démarrage (1500 ép) : fix6 3.26 vs fix5 3.46 au **même stade** — courbes superposées, pas de régression. (J'avais d'abord cru à un ralentissement en comparant au 17 de fix5, qui était mesuré à 5000 ép, pas 1500.)
- Run lancé : 5 seeds × 20000 épisodes, `--torus-food --rich-state`.

## 2026-09-16 21:24 — Temps par score (pour le rendu)
- Demande : fournir le temps associé à chaque score.
- Choix de la métrique : **temps de jeu = `steps / GAME_SPEED`**, la durée réelle de la partie à la clock du socle (5 FPS). C'est ce que voit le prof, et c'est indépendant de la vitesse de notre machine. Le temps CPU est loggé à part (`temps_cpu_ms`), il mesure notre agent, pas la partie.
- `evaluate.py` : colonnes `temps_jeu_s`, `sec_par_pomme`, `temps_cpu_ms` + résumé imprimé (score mean/max/min, temps mean/max, s/pomme, meilleure partie, cpu/partie).
- `watch_training.py` : affiche « record en Xs de jeu ».
- `play_visual.py` : le panneau affiche le **temps de jeu** (steps/5) et non le temps mur, donc le chiffre reste juste même en accéléré (`--speed 40`).
- `bilan.py` : tableau récapitulatif multi-modèles score + temps pour le rendu.
- Exemple (`model_fix5_s1_best`, 30 parties) : score moy **39.8**, max **67**, temps moy **102 s**, temps du record **206 s**, **2.56 s/pomme**, 15 ms CPU/partie.

## 2026-09-16 21:26 — Réduction à 3 runs parallèles
- Quoi : `kill` des seeds 4 et 5 du run fix6 pour alléger la machine. 3 runs restants (seeds 1, 2, 3), 7 cœurs sur 10 libérés.
- Choix des seeds : les 5 étaient à égalité (~1480 épisodes), donc j'ai gardé les **numéros les plus bas**, pas les mieux classées. Choisir après coup les seeds qui performent le mieux serait exactement le biais de sélection qu'on dénonce depuis 21:25.
- Fichiers partiels des seeds 4-5 déplacés dans `results/interrompus/` pour ne pas polluer le bilan.
- **Conséquence à déclarer au prof** : n=3 au lieu de 5 pour fix6. L'écart-type inter-seed reste calculable mais sur 3 points seulement — moins fiable que les 5 de fix5. À garder en tête en comparant fix6 et fix5.

## 2026-09-16 21:54 — `courbes.py` : figures du rendu
- Quoi : 6 PNG dans `results/figures/` générés depuis `results/*.csv`. `matplotlib` était dans `requirements.txt` mais pas installé.
- Choix de lecture : **moyenne glissante 200** sur les runs d'entraînement (le score brut y est bruité par l'exploration, eps>0, la courbe brute ne dit rien) ; sur les évals eps=0, **un point noir par seed** + barre d'erreur = écart-type inter-seed.
- Figures : 1 progression étapes 1-4 · 2 runs multi-seed (variance) · 3 comparaison éval eps=0 · 4 score vs temps de jeu · 5 causes de mort · 6 exploitation du tore.
- Deux défauts de lisibilité corrigés après inspection visuelle des PNG (une note chevauchait la barre d'erreur en fig. 3, une annotation passait sur le titre en fig. 1). Générer sans erreur ne garantit pas lisible.
- Annotation ajoutée en fig. 1 : la baseline monte **plus vite** que les fixes, ce qui se lit à l'envers sans explication — c'est l'effet eps=0 dès 80 parties (elle exploite un optimum local et plafonne), alors que les fixes explorent, scorent moins en train et gagnent en éval.
- Fig. 3 : Fix6 absent (éval pas encore faite, run en cours).

## 2026-09-16 21:56 — Fix 6 à 14000/20000 : l'écart dépasse le bruit
- `last500` par seed : s1 **44.0**, s2 **41.7**, s3 **44.1**. **Max 100 / 99 / 101** (ancien record toutes étapes confondues : 72).
- Les 3 seeds sont resserrées (~2.4 points d'écart) alors que fix5 vs contrôle étaient indistinguables (écart 1.4 pour un bruit de 7.7).
- **C'est la première fois qu'un fix produit un écart clairement supérieur au bruit inter-seed.** Le diagnostic « 50 morts sur 50 = morsure, l'état ne voit qu'une case » était le bon.
- À confirmer en éval eps=0 en fin de run (le score train reste bruité par eps, ici ~0.02).

## 2026-09-16 22:12 — Fix 6 : résultat final (run arrêté à 18333/20000)
- **Éval eps=0, 50 parties par seed** :

| modèle | score moy | max | écart-type intra | temps moy | temps du record | s/pomme |
|---|---|---|---|---|---|---|
| `model_fix6_s1_best.pth` | **88.1** | **119** | 13.4 | 253s | 380s | 2.88s |
| `model_fix6_s2_best.pth` | 86.0 | 116 | 14.5 | 237s | 318s | 2.76s |
| `model_fix6_s3_best.pth` | 85.5 | 108 | 13.6 | 234s | 310s | 2.74s |

- **Inter-seed : mean 86.6, écart-type 1.1** (n=3).
- Comparaison : contrôle 28.1 (±4.7), fix5 26.8 (±7.7), fix4 33.4, baseline 23.9.
- **Écart fix6 vs contrôle : +58.4 pour un bruit de 4.7, soit 12× le bruit.** C'est le **premier fix statistiquement concluant** de la soirée : fix1 à fix5 étaient soit mono-seed (non mesurables), soit dans le bruit (fix5).
- Progression complète en éval eps=0 : 23.9 → 25.6 → 29.4 → 33.4 → (fix5 26.8, dans le bruit) → **86.6**.
- **Réserve honnête** : fix6 meurt encore à **100 % par morsure**, comme fix4. Le flood-fill repousse le plafond (33 → 88 pommes) mais n'élimine pas la cause. La prochaine piste serait un horizon plus long (gamma) ou un état qui voit si la queue reste accessible.

## 2026-09-16 22:13 — `main.py` : point d'entrée du rendu
- `python3 main.py` lance `model_fix6_s1_best.pth` avec `--torus-food --rich-state` en visualisation pygame.
- Les flags d'état sont déclarés **en un seul endroit** (`BEST_MODEL` / `BEST_FLAGS`) : un modèle doit être joué avec l'état sur lequel il a été entraîné. C'est exactement le piège de 21:14, où `main.py` sans arguments affichait fix4 et nous a fait croire que fix5 restait timide.
- Les arguments passés en ligne de commande restent prioritaires (`--speed`, `--episodes`, `--model`).
- Figures régénérées avec fix6 (`results/figures/`, 6 PNG).
- Panneau pygame corrigé : à 3 chiffres de score, « Remplissage » et « Temps jeu » se chevauchaient (mise en page prévue pour 2 chiffres). Repéré sur une capture à 70 pommes, pas dans le code.
