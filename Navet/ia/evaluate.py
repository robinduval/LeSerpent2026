# Eval eps=0, clock intacte côté logique (pas de train, pas d'exploration).
import argparse, csv, time
import torch
from game import SnakeGame, GRID_SIZE, GAME_SPEED
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
    ap.add_argument("--torus-food", action="store_true", help="Fix 5/5 : doit matcher l'entrainement du modele")
    ap.add_argument("--rich-state", action="store_true", help="Fix 6/6 : doit matcher l'entrainement du modele")
    args = ap.parse_args()

    agent = Agent(gamma=args.gamma, use_target=args.use_target, torus_food=args.torus_food,
                  rich_state=args.rich_state)
    agent.model.load_state_dict(torch.load(args.model, map_location="cpu"))
    agent.model.eval()

    rows_acc = []
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        # temps_jeu_s : duree reelle de la partie a la clock du socle (GAME_SPEED=5),
        # c'est ce que voit le prof. temps_cpu_ms : cout de calcul de notre agent.
        w.writerow(["n", "score", "steps", "pas_pomme", "cause",
                    "temps_jeu_s", "sec_par_pomme", "temps_cpu_ms"])
        for n in range(1, args.episodes + 1):
            game = SnakeGame(seed=args.seed + n, step_reward=args.step_reward,
                             hunger_mode=args.hunger_mode, hunger_k=args.hunger_k)
            state = agent.get_state(game)
            t_cpu = time.perf_counter()
            while not game.done:
                with torch.no_grad():
                    m = torch.argmax(agent.model(torch.tensor(state, dtype=torch.float))).item()
                move = [0, 0, 0]
                move[m] = 1
                _, _, _, info = game.play_step(move)
                state = agent.get_state(game)
            cpu_ms = (time.perf_counter() - t_cpu) * 1000
            temps_jeu = game.steps / GAME_SPEED
            w.writerow([n, game.score, game.steps, f"{game.steps / max(1, game.score):.2f}",
                        info.get("cause"), f"{temps_jeu:.1f}",
                        f"{temps_jeu / max(1, game.score):.2f}", f"{cpu_ms:.1f}"])
            rows_acc.append((game.score, temps_jeu, cpu_ms))
    # resume lisible : score ET temps, pour le rendu au prof
    if rows_acc:
        sc = [r[0] for r in rows_acc]
        tj = [r[1] for r in rows_acc]
        cp = [r[2] for r in rows_acc]
        n = len(sc)
        best_i = max(range(n), key=lambda i: sc[i])
        print(f"eval -> {args.out}")
        print(f"  parties        : {n}")
        print(f"  score          : mean {sum(sc)/n:.1f}  max {max(sc)}  min {min(sc)}")
        print(f"  temps de jeu   : mean {sum(tj)/n:.1f}s  max {max(tj):.1f}s  (clock GAME_SPEED={GAME_SPEED})")
        print(f"  sec par pomme  : {sum(tj)/max(1,sum(sc)):.2f}s")
        print(f"  meilleure part.: score {sc[best_i]} en {tj[best_i]:.1f}s de jeu")
        print(f"  cpu par partie : mean {sum(cp)/n:.0f} ms (cout de notre agent, hors clock)")


if __name__ == "__main__":
    main()
