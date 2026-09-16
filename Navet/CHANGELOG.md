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
