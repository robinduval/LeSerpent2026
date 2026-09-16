# Train baseline : python3 train.py --episodes 500 --seed 0
# Log CSV : results/run_baseline.csv (n_game,score,steps,pas_pomme,fill_pct,victoire,cause,eps,loss)
import argparse
import csv
import json
import os
import random
import time

import numpy as np

from game import SnakeGame, GRID_SIZE
from agent import Agent


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=str, default="config_base.json")
    ap.add_argument("--out", type=str, default="results/run_baseline.csv")
    ap.add_argument("--use-target", action="store_true", help="Fix 1/4 : active target network")
    ap.add_argument("--target-sync", type=int, default=20, help="sync target toutes les N parties")
    ap.add_argument("--save-prefix", type=str, default="model_baseline", help="préfixe .pth")
    ap.add_argument("--eps-mode", type=str, default="loeber", choices=["loeber", "decay"], help="Fix 2/4")
    ap.add_argument("--eps-min", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=float, default=400)
    ap.add_argument("--gamma", type=float, default=None, help="override config gamma (Fix 3/4)")
    ap.add_argument("--step-reward", type=float, default=0.1, help="Fix 4/4 groupe : +0.1 flaw vs -0.01 anti-S")
    ap.add_argument("--hunger-mode", type=str, default="loeber", choices=["loeber", "truncated"], help="Fix 4/4 groupe")
    ap.add_argument("--hunger-k", type=float, default=2.0, help="limite faim = hunger_k * cases_libres")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    episodes = args.episodes or cfg.get("episodes", 500)
    set_seeds(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gamma = args.gamma if args.gamma is not None else cfg.get("gamma", 0.9)
    agent = Agent(gamma=gamma, use_target=args.use_target,
                  eps_mode=args.eps_mode, eps_min=args.eps_min, eps_decay=args.eps_decay)

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        print("ATTENTION : torch absent, mode smoke-test logique jeu uniquement (pas d'apprentissage).")
        print("Installe : pip3 install torch --index-url https://download.pytorch.org/whl/cpu")

    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_game", "score", "steps", "pas_pomme", "fill_pct", "victoire", "cause", "eps", "loss"])
        for n in range(1, episodes + 1):
            game = SnakeGame(seed=args.seed + n * 100000 if args.seed else None,
                             step_reward=args.step_reward, hunger_mode=args.hunger_mode, hunger_k=args.hunger_k)
            state = agent.get_state(game)
            total_loss, n_loss = 0.0, 0
            while not game.done:
                action = agent.get_action(state) if has_torch else [1, 0, 0]
                reward, done, score, info = game.play_step(action)
                truncated = info.get("truncated", False)
                state2 = agent.get_state(game)
                if has_torch:
                    loss = agent.train_short(state, action, reward, state2, done, truncated)
                    total_loss += loss
                    n_loss += 1
                    agent.remember(state, action, reward, state2, done, truncated)
                state = state2
            if has_torch:
                agent.train_long()
            agent.n_games += 1
            if has_torch and args.use_target and (n % args.target_sync == 0):
                agent.trainer.sync_target()
            pas_pomme = game.steps / max(1, game.score)
            fill = len(game.body) / (GRID_SIZE * GRID_SIZE) * 100
            w.writerow([n, game.score, game.steps, f"{pas_pomme:.2f}", f"{fill:.1f}",
                        int(game.victory), info.get("cause", "?"),
                        agent.get_epsilon(), f"{(total_loss / max(1, n_loss)):.4f}"])
            if n % 50 == 0:
                print(f"[{n}/{episodes}] score={game.score} steps={game.steps} cause={info.get('cause')} eps={agent.get_epsilon()}")
            if has_torch and game.score > getattr(main, "record", 0):
                main.record = game.score
                import torch
                torch.save(agent.model.state_dict(), f"{args.save_prefix}_best.pth")
    print(f"Fini en {time.time() - t0:.1f}s -> {args.out}")

    # checkpoints réguliers (fix vs Loeber qui ne garde que le record)
    if has_torch:
        import torch
        torch.save(agent.model.state_dict(), f"{args.save_prefix}_last.pth")


if __name__ == "__main__":
    main()
