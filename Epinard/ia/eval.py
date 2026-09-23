"""
Évaluation headless : joue N essais avec le modèle entraîné et produit
un JSON avec, pour chaque essai : temps, score, ratio score/temps.

    python3 eval.py --essais 50 --out eval_results.json
"""
import argparse
import json
import time

import numpy as np
import torch

from agent import Config, get_mask, get_state
from env_wrapper import SnakeEnv
from model import ActorCritic
from safety import fallback_action


def run(model_path="ppo_snake.pth", n_essais=50, stochastique=True, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ActorCritic()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    cfg = Config()
    env = SnakeEnv()
    resultats = []

    for i in range(1, n_essais + 1):
        env.reset()
        while not env.game_over:
            state = get_state(env)
            mask = get_mask(env, cfg.safety_margin)
            st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mk = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)

            with torch.no_grad():
                if stochastique:
                    action, _, _, _ = model.act(st, mk)
                    a = int(action.item())
                else:
                    a = int(model.greedy(st, mk).item())

            if not mask[a]:
                a = fallback_action(env.snake.body, env.snake.direction)

            env.play_step(a)

        s = env.stats()
        s["essai"] = i
        resultats.append(s)
        print(f"Essai {i:3d} | score {s['score']:3d} | temps {s['temps']:7.2f}s "
              f"| ratio {s['ratio_score_temps']:6.3f} pt/s | {s['fin']}")

    scores = [r["score"] for r in resultats]
    ratios = [r["ratio_score_temps"] for r in resultats]
    temps = [r["temps"] for r in resultats]

    resume = {
        "essais": len(resultats),
        "score_moyen": round(sum(scores) / len(scores), 2),
        "score_median": int(np.median(scores)),
        "record": max(scores),
        "score_min": min(scores),
        "temps_moyen_s": round(sum(temps) / len(temps), 2),
        "ratio_moyen": round(sum(ratios) / len(ratios), 3),
        "victoires": sum(1 for r in resultats if r["victoire"]),
    }

    print("\n--- Résumé ---")
    for k, v in resume.items():
        print(f"{k:16s}: {v}")

    return {"resume": resume, "essais": resultats}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo_snake.pth")
    p.add_argument("--essais", type=int, default=50)
    p.add_argument("--out", default="eval_results.json")
    p.add_argument("--greedy", action="store_true",
                   help="mode deterministe (deconseille : la politique PPO est stochastique)")
    args = p.parse_args()

    data = run(args.model, args.essais, not args.greedy)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nRésultats -> {args.out}")
