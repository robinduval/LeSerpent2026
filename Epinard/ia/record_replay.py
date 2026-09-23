"""
Enregistre des parties complètes (position du serpent + pomme à chaque pas)
dans un JSON, pour le replay animé dans le dashboard web.

    python3 record_replay.py --model ppo_snake.pth --essais 5 --out replay.json
"""
import argparse
import json

import numpy as np
import torch

from agent import Config, observe
from env_wrapper import SnakeEnv
from model import ActorCritic
from safety import fallback_action


def record(model_path, n_essais=5, max_frames=4000, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ActorCritic()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    cfg = Config()
    env = SnakeEnv()
    parties = []

    for i in range(1, n_essais + 1):
        env.reset()
        frames = [{
            "b": [list(c) for c in env.snake.body],
            "a": list(env.apple.position),
            "s": env.score,
        }]

        while not env.game_over and len(frames) < max_frames:
            state, mask = observe(env, cfg.safety_margin)
            st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mk = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = model.act(st, mk)
                a = int(action.item())
            if not mask[a]:
                a = fallback_action(env.snake.body, env.snake.direction)

            env.play_step(a)
            frames.append({
                "b": [list(c) for c in env.snake.body],
                "a": list(env.apple.position) if env.apple.position else None,
                "s": env.score,
            })

        s = env.stats()
        s["essai"] = i
        # Le temps mesuré ici est le temps de calcul, pas le temps de jeu.
        # Le temps de jeu réel = nb de pas / 5 FPS (la clock du jeu original).
        s["temps_jeu_s"] = round(s["steps"] / 5.0, 1)
        s["ratio_jeu"] = round(s["score"] / (s["steps"] / 5.0), 3) if s["steps"] else 0
        parties.append({"stats": s, "frames": frames})
        print(f"Essai {i}: score={s['score']} steps={s['steps']} "
              f"temps_jeu={s['temps_jeu_s']}s fin={s['fin']}")

    return parties


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo_snake.pth")
    p.add_argument("--essais", type=int, default=5)
    p.add_argument("--out", default="replay.json")
    p.add_argument("--max-frames", type=int, default=4000)
    args = p.parse_args()

    parties = record(args.model, args.essais, args.max_frames)
    with open(args.out, "w") as f:
        json.dump({"grid": 15, "parties": parties}, f)
    print(f"\nReplay -> {args.out}")
