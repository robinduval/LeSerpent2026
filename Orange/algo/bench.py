"""Banc d'essai : joue N parties sans affichage et trace un tableau de bord.

Exemples :
    python bench.py                      # 20 parties, stratégie "safe"
    python bench.py -n 50 --compare      # compare "safe" et "naive"
    python bench.py -n 10 --no-show      # enregistre le PNG sans ouvrir de fenêtre
"""
import argparse
import importlib.util
import os
import random
import time
from multiprocessing import Pool

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location("serpent_algo", os.path.join(HERE, "serpent-algo.py"))
game = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(game)

from solver import Solver  # noqa: E402


def play(args):
    """Joue une partie avec les classes du jeu. Retourne ses statistiques."""
    strategy, seed, max_idle, params = args
    random.seed(seed)  # position des pommes
    snake = game.Snake()
    apple = game.Apple(snake.body)
    solver = Solver(game.GRID_SIZE, strategy=strategy, seed=seed, **params)
    steps, idle, last_meal = 0, 0, 0
    lengths = []          # longueur après chaque coup
    apple_steps = []      # (longueur au moment du repas, coups pour l'atteindre)
    modes = {}
    outcome = "mort"
    t0 = time.perf_counter()
    while True:
        direction = solver.choose(snake.body, snake.grow_pending, apple.position)
        modes[solver.mode] = modes.get(solver.mode, 0) + 1
        snake.set_direction(direction)
        snake.move()
        steps += 1
        if solver.mode not in ("POMME", "REMPLIR", "ANTICIPE", "DETOUR", "NAIF"):
            idle += 1  # coup d'attente : aucun chemin sûr vers la pomme
        lengths.append(len(snake.body))
        if snake.is_game_over():
            break
        if snake.head_pos == list(apple.position):
            apple_steps.append((len(snake.body), steps - last_meal))
            idle, last_meal = 0, steps
            snake.grow()
            if not apple.relocate(snake.body):
                outcome = "victoire"
                lengths.append(len(snake.body) + 1)
                break
        if idle > (max_idle or len(snake.body)):  # par défaut : un tour complet sans manger
            outcome = "bloqué"
            break
    return {
        "strategy": strategy, "seed": seed, "outcome": outcome, "score": snake.score,
        "final_len": lengths[-1], "steps": steps, "lengths": lengths,
        "apple_steps": apple_steps, "modes": modes, "time": time.perf_counter() - t0,
        "game_time": steps / game.GAME_SPEED,  # durée de la partie à vitesse réelle (s)
    }


def run(strategy, n, seed0, max_idle, params=None):
    with Pool() as pool:
        return pool.map(play, [(strategy, seed0 + i, max_idle, params or {}) for i in range(n)])


def summary(results):
    total = game.GRID_SIZE ** 2
    n = len(results)
    wins = sum(r["outcome"] == "victoire" for r in results)
    deaths = sum(r["outcome"] == "mort" for r in results)
    stuck = n - wins - deaths
    fill = sum(r["final_len"] for r in results) / n / total * 100
    win_steps = [r["steps"] for r in results if r["outcome"] == "victoire"]
    avg_win = sum(win_steps) / len(win_steps) if win_steps else float("nan")
    game_time = sum(r["game_time"] for r in results) / n
    cpu_time = sum(r["time"] for r in results) / n
    modes = {}
    for r in results:
        for m, c in r["modes"].items():
            modes[m] = modes.get(m, 0) + c
    nsteps = sum(modes.values())
    mode_txt = ", ".join(f"{m} {c / nsteps:.1%}" for m, c in sorted(modes.items(), key=lambda x: -x[1]))
    print(f"[{results[0]['strategy']}] {n} parties | victoires {wins} ({wins / n:.0%}) | "
          f"morts {deaths} | bloqué {stuck} | remplissage moyen {fill:.1f}% | "
          f"coups moyens par victoire {avg_win:.0f}")
    print(f"    durée moyenne d'une partie à vitesse réelle ({game.GAME_SPEED} coups/s) : "
          f"{game_time / 60:.1f} min | calcul : {cpu_time:.1f}s (accélération x{game_time / max(cpu_time, 1e-9):.0f})")
    print(f"    modes : {mode_txt}")
    return wins / n, fill


def dashboard(all_results, path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = game.GRID_SIZE ** 2
    colors = {"safe": "#2a9d8f", "naive": "#e76f51"}
    outcome_colors = {"victoire": "#2a9d8f", "mort": "#e63946", "bloqué": "#f4a261"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Serpent {game.GRID_SIZE}×{game.GRID_SIZE} torique — bilan des parties", fontsize=14)

    # 1. Longueur du serpent au fil des coups, une courbe par partie
    ax = axes[0, 0]
    for strategy, results in all_results.items():
        for r in results:
            ax.plot(r["lengths"], color=colors.get(strategy, "gray"), alpha=0.35, lw=1)
            ax.plot(len(r["lengths"]) - 1, r["lengths"][-1], "o", ms=4,
                    color=outcome_colors[r["outcome"]])
    ax.axhline(total, color="black", ls="--", lw=0.8, label="grille pleine")
    ax.set(title="Longueur au fil des coups (point = fin de partie)", xlabel="coups", ylabel="longueur")
    ax.legend(loc="lower right")

    # 2. Issues des parties
    ax = axes[0, 1]
    names = list(all_results)
    bottom = [0] * len(names)
    for outcome, col in outcome_colors.items():
        vals = [sum(r["outcome"] == outcome for r in all_results[s]) for s in names]
        ax.bar(names, vals, bottom=bottom, color=col, label=outcome)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set(title="Issue des parties", ylabel="nombre de parties")
    ax.legend()

    # 3. Coups nécessaires pour chaque pomme selon la longueur du serpent
    ax = axes[1, 0]
    bucket = 10
    for strategy, results in all_results.items():
        sums, counts = {}, {}
        for r in results:
            for length, k in r["apple_steps"]:
                b = length // bucket * bucket
                sums[b] = sums.get(b, 0) + k
                counts[b] = counts.get(b, 0) + 1
        xs = sorted(sums)
        ax.plot(xs, [sums[x] / counts[x] for x in xs], "-o", ms=3,
                color=colors.get(strategy, "gray"), label=strategy)
    ax.set(title="Coups moyens pour atteindre une pomme", xlabel="longueur du serpent", ylabel="coups")
    ax.legend()

    # 4. Taux de remplissage final
    ax = axes[1, 1]
    for strategy, results in all_results.items():
        fills = [r["final_len"] / total * 100 for r in results]
        ax.hist(fills, bins=20, range=(0, 100), alpha=0.6, color=colors.get(strategy, "gray"), label=strategy)
    ax.set(title="Remplissage final de la grille", xlabel="% de la grille", ylabel="parties")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Tableau de bord enregistré : {path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=20, help="nombre de parties par stratégie")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", default="safe", choices=["safe", "naive"])
    parser.add_argument("--compare", action="store_true", help="compare safe et naive")
    parser.add_argument("--max-idle", type=int, default=0,
                        help="coups d'attente avant « bloqué » (0 = longueur du serpent : un tour complet)")
    parser.add_argument("--remplir", type=float, default=0.4,
                        help="mode remplissage dès que le serpent occupe cette fraction de la grille (0 = désactivé)")
    parser.add_argument("--anticipe", type=int, default=0,
                        help="anticipe la prochaine pomme sous N cases libres (0 = désactivé)")
    parser.add_argument("--out", default=os.path.join(HERE, "bench.png"))
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    strategies = ["safe", "naive"] if args.compare else [args.strategy]
    all_results = {}
    for s in strategies:
        all_results[s] = run(s, args.n, args.seed, args.max_idle, {"anticipate": args.anticipe, "fill_from": args.remplir})
        summary(all_results[s])
    dashboard(all_results, args.out, not args.no_show)


if __name__ == "__main__":
    main()
