"""
Bloc "Agent" du slide 4, en PPO.

ÉTAT — 14 features
------------------
Les 11 du slide 6, corrigées pour le tore, plus 3 features d'espace libre :

  [0:3]   Danger tout droit / droite / gauche
          -> uniquement collision avec le CORPS. Les bords ne tuent pas
             (plateau torique), contrairement au Snake classique.
  [3:7]   Direction courante (gauche, droite, haut, bas) — one-hot
  [7:11]  Position relative de la pomme (gauche, droite, haut, bas)
          -> calculée en distance TORIQUE : on compare |dx| et GRID-|dx|,
             car passer par le bord est souvent le chemin le plus court.
  [11:14] Espace libre après chaque action, normalisé [0,1]
          -> sans ça, le réseau ne distingue pas un couloir d'un cul-de-sac.

PPO
---
On-policy : on collecte un rollout, on l'exploite sur quelques époques,
puis on le jette. Pas de replay buffer (ce serait off-policy).

Le masque de sécurité est appliqué aux logits AVANT le softmax, donc
la politique n'échantillonne jamais un coup suicidaire.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from env_wrapper import SnakeEnv
from game_core import GRID_SIZE, CLOCKWISE, LEFT, RIGHT, UP, DOWN, turn
from model import ActorCritic
from safety import analyse, simulate_step

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- HYPERPARAMÈTRES PPO (valeurs canoniques) ---
class Config:
    gamma = 0.99           # facteur d'actualisation
    gae_lambda = 0.95      # lissage de l'avantage (GAE)
    clip_eps = 0.2         # borne du ratio PPO
    epochs = 4             # époques par rollout
    rollout_steps = 2048   # pas collectés avant chaque mise à jour
    minibatch = 256
    lr = 5e-4
    entropy_coef = 0.003   # bonus d'exploration (réduit : à 0.01 la
                           # politique restait plate et jouait tout droit)
    value_coef = 0.5
    max_grad_norm = 0.5
    safety_margin = 1.0    # espace libre requis = longueur * marge


def torus_delta(a, b):
    """
    Écart signé le plus court sur un axe torique.
    Si le serpent est en x=14 et la pomme en x=0, la réponse est +1
    (par le bord), pas -14.
    """
    d = (b - a) % GRID_SIZE
    if d > GRID_SIZE // 2:
        d -= GRID_SIZE
    return d


def observe(env, margin=Config.safety_margin):
    """
    Construit l'état (14 features) ET le masque de sécurité en UN SEUL
    passage de flood-fill.

    Avant, get_state() et get_mask() refaisaient chacun les mêmes 3
    flood-fill : 6 par pas de jeu au lieu de 3. Le profilage montrait
    69% du temps d'entraînement passé là. D'où cette fusion.
    """
    snake = env.snake
    body = snake.body
    direction = snake.direction
    head = body[0]
    apple = env.apple.position

    # [0:3] danger, masque de sécurité et [11:14] espace libre : un seul passage
    danger, mask, space = analyse(body, direction, margin)

    # [3:7] Direction courante (one-hot)
    dir_l = float(direction == LEFT)
    dir_r = float(direction == RIGHT)
    dir_u = float(direction == UP)
    dir_d = float(direction == DOWN)

    # [7:11] Pomme, en distance torique
    dx = torus_delta(head[0], apple[0])
    dy = torus_delta(head[1], apple[1])
    food_l = float(dx < 0)
    food_r = float(dx > 0)
    food_u = float(dy < 0)
    food_d = float(dy > 0)

    # [14:17] Distance à la pomme et longueur, normalisées.
    # Sans la DISTANCE (et pas seulement la direction), l'agent ne peut
    # pas savoir s'il se rapproche : il apprend alors à survivre sans
    # jamais manger. C'est ce qui s'est produit lors du premier
    # entraînement (score 0 en mode déterministe).
    half = GRID_SIZE / 2.0
    dist_x = abs(dx) / half
    dist_y = abs(dy) / half
    longueur = len(body) / (GRID_SIZE * GRID_SIZE)

    state = np.array(
        danger + [dir_l, dir_r, dir_u, dir_d]
        + [food_l, food_r, food_u, food_d] + space
        + [dist_x, dist_y, longueur],
        dtype=np.float32,
    )
    return state, np.array(mask, dtype=bool)


# --- Compatibilité : eval.py et play_visual.py utilisent ces deux noms ---

def get_state(env):
    return observe(env)[0]


def get_mask(env, margin=Config.safety_margin):
    return observe(env, margin)[1]


def compute_gae(rewards, values, dones, last_value, cfg):
    """Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + cfg.gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def train(total_steps=400_000, out_dir=".", seed=0, log_every=20):
    cfg = Config()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ActorCritic(n_features=17).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    env = SnakeEnv()
    state, mask = observe(env, cfg.safety_margin)

    episodes = []           # une entrée par essai : temps, score, ratio
    episode_index = 0
    steps_done = 0
    best_score = 0
    t_start = time.time()

    while steps_done < total_steps:
        # ---------- 1. Collecte d'un rollout (on-policy) ----------
        buf_states, buf_actions, buf_logprobs = [], [], []
        buf_rewards, buf_dones, buf_values, buf_masks = [], [], [], []

        for _ in range(cfg.rollout_steps):
            st = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mk = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                action, logprob, _, value = model.act(st, mk)

            a = int(action.item())
            reward, done, score = env.play_step(a)

            buf_states.append(state)
            buf_actions.append(a)
            buf_logprobs.append(float(logprob.item()))
            buf_values.append(float(value.item()))
            buf_rewards.append(reward)
            buf_dones.append(float(done))
            buf_masks.append(mask)

            steps_done += 1

            if done:
                stats = env.stats()
                episode_index += 1
                stats["essai"] = episode_index
                stats["steps_cumules"] = steps_done
                episodes.append(stats)
                best_score = max(best_score, stats["score"])

                if episode_index % log_every == 0:
                    recent = episodes[-log_every:]
                    avg = sum(e["score"] for e in recent) / len(recent)
                    print(
                        f"essai {episode_index:5d} | steps {steps_done:7d} "
                        f"| score moy {avg:6.2f} | record {best_score:3d} "
                        f"| {time.time()-t_start:6.1f}s",
                        flush=True,
                    )

                env.reset()

            state, mask = observe(env, cfg.safety_margin)

        # ---------- 2. Avantages (GAE) ----------
        with torch.no_grad():
            st = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mk = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
            _, last_value = model(st, mk)
            last_value = float(last_value.item())

        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones, last_value, cfg
        )

        b_states = torch.as_tensor(np.array(buf_states), dtype=torch.float32, device=DEVICE)
        b_actions = torch.as_tensor(buf_actions, dtype=torch.long, device=DEVICE)
        b_logprobs = torch.as_tensor(buf_logprobs, dtype=torch.float32, device=DEVICE)
        b_masks = torch.as_tensor(np.array(buf_masks), dtype=torch.bool, device=DEVICE)
        b_adv = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
        b_ret = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

        # ---------- 3. Époques PPO ----------
        n = len(buf_states)
        idx = np.arange(n)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch):
                mb = idx[start:start + cfg.minibatch]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=DEVICE)

                new_logprob, entropy, value = model.evaluate(
                    b_states[mb_t], b_actions[mb_t], b_masks[mb_t]
                )

                # Avantages normalisés par minibatch
                adv = b_adv[mb_t]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Ratio clippé — le cœur de PPO
                ratio = torch.exp(new_logprob - b_logprobs[mb_t])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(value, b_ret[mb_t])
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

    # ---------- 4. Sauvegardes ----------
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "ppo_snake.pth")
    torch.save(model.state_dict(), model_path)

    log = {
        "algo": "PPO (acteur-critique, action masking)",
        "total_steps": steps_done,
        "essais": len(episodes),
        "record": best_score,
        "duree_entrainement_s": round(time.time() - t_start, 1),
        "config": {
            k: v for k, v in vars(Config).items() if not k.startswith("_")
        },
        "episodes": episodes,
    }
    log_path = os.path.join(out_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nModèle  -> {model_path}")
    print(f"Log     -> {log_path}")
    print(f"Record  : {best_score} pommes | {len(episodes)} essais")
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--out", type=str, default=".")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(total_steps=args.steps, out_dir=args.out, seed=args.seed)
