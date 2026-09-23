"""Benchmark des stratégies : python bench.py [--games 50] [--only nom1,nom2] [--seed0 0]

Pour chaque stratégie : nombre de victoires / morts / blocages, score moyen,
nombre de coups moyen (1 coup = 1/5 s au rythme officiel du jeu) et pire temps
de calcul d'un coup."""
import argparse
import statistics
import sys
import time
from multiprocessing import Pool

import core


def make(name):
    import heuristic
    import hamilton
    table = {
        "heur-literal": lambda: heuristic.HeuristicEngine("literal"),
        "heur-corrected": lambda: heuristic.HeuristicEngine("corrected"),
        "heur-nofill": lambda: heuristic.HeuristicEngine("none"),
        "cycle": lambda: hamilton.FixedCycle(),
        "phc": lambda: hamilton.PHC(),
        "dyncycle-greedy": lambda: hamilton.DynamicCycleGreedy(),
        "dyncycle": lambda: hamilton.DynamicCycle(),
        "dyncycle-d1": lambda: hamilton.DynamicCycle(depth=1),
        "dyncycle-d2": lambda: hamilton.DynamicCycle(depth=2),
        "dyncycle-d4": lambda: hamilton.DynamicCycle(depth=4),
        "search": lambda: hamilton.DynamicCycleSearch(),
        "search-d1": lambda: hamilton.DynamicCycleSearch(depth=1),
        "search-d3": lambda: hamilton.DynamicCycleSearch(depth=3),
        "search-c200": lambda: hamilton.DynamicCycleSearch(cap=200),
        "search-c2000": lambda: hamilton.DynamicCycleSearch(cap=2000),
        "hybrid": lambda: hamilton.Hybrid(),
    }
    if name.startswith("search-d"):          # search-d3-c600
        d, c = name[len("search-d"):].split("-c")
        return hamilton.DynamicCycleSearch(depth=int(d), cap=int(c))
    return table[name]()


def one(args):
    name, seed = args
    return play_named(name, seed)


def play_named(name, seed):
    agent = make(name)
    return core.play(agent, seed, timing=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--only", default="heur-literal,heur-corrected,cycle,phc,dyncycle")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    names = a.only.split(",")
    print(f"{a.games} parties par stratégie, graines {a.seed0}..{a.seed0 + a.games - 1}")
    print(f"{'stratégie':<16}{'vict':>6}{'mort':>6}{'bloq':>6}{'score moy':>11}"
          f"{'coups moy':>11}{'min':>7}{'max':>7}{'pire coup ms':>14}{'cpu/partie s':>14}")
    for name in names:
        t0 = time.time()
        with Pool(a.procs) as p:
            res = p.map(one, [(name, s) for s in range(a.seed0, a.seed0 + a.games)])
        st = [r["status"] for r in res]
        won = [r for r in res if r["status"] == "won"]
        mv = [r["moves"] for r in won]
        print(f"{name:<16}{st.count('won'):>6}{st.count('dead'):>6}{st.count('stalled'):>6}"
              f"{statistics.mean(r['score'] for r in res):>11.1f}"
              f"{(statistics.mean(mv) if mv else float('nan')):>11.0f}"
              f"{(min(mv) if mv else 0):>7}{(max(mv) if mv else 0):>7}"
              f"{1000 * max(r['worst'] for r in res):>14.1f}"
              f"{statistics.mean(r['cpu'] for r in res):>14.2f}", flush=True)


if __name__ == "__main__":
    main()
