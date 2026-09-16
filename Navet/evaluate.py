# Eval eps=0, clock intacte côté logique (pas de train, pas d'exploration).
import argparse, csv
import torch
from game import SnakeGame, GRID_SIZE
from agent import Agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--out", type=str, default="results/eval.csv")
    ap.add_argument("--use-target", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--step-reward", type=float, default=0.1)
    ap.add_argument("--hunger-mode", type=str, default="loeber", choices=["loeber", "truncated"])
    ap.add_argument("--hunger-k", type=float, default=2.0)
    args = ap.parse_args()

    agent = Agent(gamma=args.gamma, use_target=args.use_target)
    agent.model.load_state_dict(torch.load(args.model, map_location="cpu"))
    agent.model.eval()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "score", "steps", "pas_pomme", "cause"])
        for n in range(1, args.episodes + 1):
            game = SnakeGame(seed=args.seed + n, step_reward=args.step_reward,
                             hunger_mode=args.hunger_mode, hunger_k=args.hunger_k)
            state = agent.get_state(game)
            while not game.done:
                with torch.no_grad():
                    m = torch.argmax(agent.model(torch.tensor(state, dtype=torch.float))).item()
                move = [0, 0, 0]
                move[m] = 1
                _, _, _, info = game.play_step(move)
                state = agent.get_state(game)
            w.writerow([n, game.score, game.steps, f"{game.steps / max(1, game.score):.2f}", info.get("cause")])
    print(f"eval -> {args.out}")


if __name__ == "__main__":
    main()
