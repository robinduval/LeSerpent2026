"""Benchmark headless du planner, sur les classes Snake/Apple du vrai moteur.

La partie s'arrête à score == 100 (--target). Métrique principale : % de parties
qui atteignent 100 ; puis, parmi elles seulement, ticks_to_100 (médiane,
moyenne, meilleur). Le temps est compté en ticks, jamais en secondes.

Exemples :
    python bench.py --games 100
    python bench.py --games 1000 --workers 8 --set max_planning_time_ms=5
    python bench.py --games 100 --sweep cycle_above=0.4,0.55,0.7
    python bench.py --games 100 --set max_planning_time_ms=0 --set max_nodes=3000   # déterministe
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from planner import PlannerConfig, SnakePlanner  # noqa: E402

HERE = Path(__file__).resolve().parent
_engine = None


def engine():
    """Charge serpent-algo.py (nom avec tiret => importlib)."""
    global _engine
    if _engine is None:
        spec = importlib.util.spec_from_file_location("serpent_algo", HERE / "serpent-algo.py")
        _engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_engine)
    return _engine


def play_game(args):
    """Rejoue exactement la mise à jour de main() du moteur, sans affichage."""
    seed, overrides, max_stall, target = args
    eng = engine()
    random.seed(seed)
    planner = SnakePlanner(eng.GRID_SIZE, eng.GRID_SIZE, PlannerConfig.from_overrides(overrides))
    snake = eng.Snake()
    apple = eng.Apple(snake.body)
    ticks = 0
    last_eat = 0
    max_len = len(snake.body)
    think = 0.0
    think_max = 0.0
    modes: dict[str, int] = {}
    outcome = "dead"
    while True:
        t0 = time.perf_counter()
        d = planner.choose_action(snake.body, snake.direction, apple.position,
                                  snake.score, snake.grow_pending)
        dt = time.perf_counter() - t0
        think += dt
        think_max = max(think_max, dt)
        m = planner.last_info.get("mode", "?")
        modes[m] = modes.get(m, 0) + 1

        snake.set_direction(d)
        snake.move()
        ticks += 1
        if snake.is_game_over():
            break
        if snake.head_pos == list(apple.position):
            snake.grow()
            last_eat = ticks
            if target and snake.score >= target:
                outcome = "win100"
                break
            if not apple.relocate(snake.body):
                outcome = "victory"
                break
        max_len = max(max_len, len(snake.body))
        if ticks - last_eat > max_stall:
            outcome = "stall"
            break
    return {
        "seed": seed, "score": snake.score, "ticks": ticks, "outcome": outcome,
        "reached_100": outcome == "win100",
        "ticks_to_100": ticks if outcome == "win100" else None,
        "max_len": max(max_len, len(snake.body)), "think_ms": 1000 * think / ticks,
        "think_max_ms": 1000 * think_max, "modes": modes,
    }


def summarize(results, label=""):
    scores = [r["score"] for r in results]
    ticks = [r["ticks"] for r in results]
    total_score, total_ticks = sum(scores), sum(ticks)
    outcomes = {k: sum(r["outcome"] == k for r in results) for k in ("win100", "victory", "dead", "stall")}
    wins = [r["ticks_to_100"] for r in results if r["reached_100"]]
    modes: dict[str, int] = {}
    for r in results:
        for k, v in r["modes"].items():
            modes[k] = modes.get(k, 0) + v
    tot_modes = sum(modes.values()) or 1
    n = len(results)
    print(f"=== {label or 'config'}  ({n} parties)")
    print(f"  atteint 100 {100 * len(wins) / n:6.2f} %  ({len(wins)}/{n})")
    if wins:
        print(f"  ticks_to_100 médiane {statistics.median(wins):7.1f} | moyenne {statistics.mean(wins):7.1f}"
              f" | meilleur {min(wins)} | pire {max(wins)}")
    print(f"  score      moyen {statistics.mean(scores):7.2f} | médian {statistics.median(scores):6.1f}"
          f" | min {min(scores)} | max {max(scores)}")
    print(f"  pommes/100 ticks {100 * total_score / total_ticks:6.3f} | ticks/pomme {total_ticks / max(1, total_score):6.2f}")
    print(f"  longueur max atteinte {max(r['max_len'] for r in results)}"
          f" (moyenne {statistics.mean(r['max_len'] for r in results):.1f})")
    print(f"  issues     100 {outcomes['win100']} | plateau plein {outcomes['victory']}"
          f" | mort {outcomes['dead']} | stagnation {outcomes['stall']}")
    fails = [f"{r['score']}@{r['seed']}" for r in results if not r["reached_100"]]
    if fails:
        print("  échecs (score@seed) " + ", ".join(fails))
    print(f"  réflexion  {statistics.mean(r['think_ms'] for r in results):.2f} ms/tick"
          f" (pire {max(r['think_max_ms'] for r in results):.1f} ms)")
    print("  modes      " + ", ".join(f"{k} {100 * v / tot_modes:.1f}%"
                                      for k, v in sorted(modes.items(), key=lambda kv: -kv[1])))


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def run(games, workers, overrides, seed0, max_stall, label, target=100):
    jobs = [(seed0 + i, overrides, max_stall, target) for i in range(games)]
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(play_game, jobs, chunksize=1)
    else:
        results = [play_game(j) for j in jobs]
    summarize(results, label)
    print(f"  (durée {time.time() - t0:.1f} s)")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-stall", type=int, default=3000,
                    help="arrêt si aucune pomme pendant N ticks (le moteur n'a pas de famine)")
    ap.add_argument("--target", type=int, default=100,
                    help="score qui termine la partie (0 = jouer jusqu'au plateau plein)")
    ap.add_argument("--set", action="append", metavar="PARAM=VAL", help="surcharge de PlannerConfig")
    ap.add_argument("--sweep", metavar="PARAM=V1,V2,...", help="compare plusieurs valeurs d'un paramètre")
    a = ap.parse_args()

    base = parse_kv(a.set)
    if a.sweep:
        k, vals = a.sweep.split("=", 1)
        for v in vals.split(","):
            run(a.games, a.workers, {**base, k: v}, a.seed, a.max_stall, f"{k}={v} {base or ''}", a.target)
    else:
        run(a.games, a.workers, base, a.seed, a.max_stall, str(base or "défaut"), a.target)


if __name__ == "__main__":
    main()
