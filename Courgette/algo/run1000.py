"""Lance beaucoup de parties d'une stratégie (sans fenêtre) et convertit les coups
en temps de jeu réel (GAME_SPEED = 5 coups par seconde).

Exemples :
    python run1000.py search                 # 1000 parties de la stratégie jouée
    python run1000.py celltree --games 200
    python run1000.py --list                 # stratégies disponibles

Affiche : victoires / morts / blocages, coups et secondes (meilleure, pire,
moyenne sur les parties gagnées), et les « timeouts » : coups dont le calcul a
pris plus d'une frame (0,2 s), ce qui ralentirait l'horloge du vrai jeu.
Attention : lancer beaucoup de processus en parallèle ralentit chaque coup ;
--procs règle ce compromis (défaut : nb de coeurs - 2)."""
import argparse
import os
import statistics
import time
from multiprocessing import Pool

import bench

GAME_SPEED = 5          # coups par seconde, comme serpent-algo.py
NAMES = ["search", "search-plus", "search-plus-noreplan", "celltree", "dyncycle", "dyncycle-greedy", "phc", "cycle",
         "heur-literal", "heur-corrected", "heur-nofill"]


def fmt(moves):
    s = moves / GAME_SPEED
    return f"{moves:>7.0f} coups = {s:>7.1f} s ({s / 60:4.1f} min)"


def one(args):
    name, seed = args
    return bench.play_named(name, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategy", nargs="?", default="search", help="nom de la stratégie (--list)")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0, help="première graine")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--list", action="store_true", help="affiche les stratégies et quitte")
    a = ap.parse_args()
    if a.list:
        print("\n".join(NAMES))
        return
    bench.make(a.strategy)                    # erreur immédiate si le nom est inconnu

    print(f"{a.strategy} : {a.games} parties (graines {a.seed0}..{a.seed0 + a.games - 1}), {a.procs} processus")
    t0 = time.time()
    res = []
    with Pool(a.procs) as p:
        for i, r in enumerate(p.imap_unordered(one, [(a.strategy, s) for s in range(a.seed0, a.seed0 + a.games)]), 1):
            res.append(r)
            if i % max(1, a.games // 20) == 0 or i == a.games:
                print(f"  {i}/{a.games} parties ({time.time() - t0:.0f} s)", flush=True)

    st = [r["status"] for r in res]
    won = [r for r in res if r["status"] == "won"]
    print(f"\nvictoires {st.count('won')}  morts {st.count('dead')}  bloquées {st.count('stalled')}"
          f"  (score moyen {statistics.mean(r['score'] for r in res):.1f} / 223)")
    if won:
        mv = [r["moves"] for r in won]
        best = min(won, key=lambda r: r["moves"])
        worst = max(won, key=lambda r: r["moves"])
        print(f"meilleure : {fmt(best['moves'])}  (graine {best['seed']})")
        print(f"pire      : {fmt(worst['moves'])}  (graine {worst['seed']})")
        print(f"moyenne   : {fmt(statistics.mean(mv))}")
        print(f"médiane   : {fmt(statistics.median(mv))}")
    slow = sum(r["slow"] for r in res)
    print(f"timeouts (coups > 200 ms) : {slow} sur {sum(r['moves'] for r in res)} coups,"
          f" dans {sum(1 for r in res if r['slow'])} parties ; pire coup {1000 * max(r['worst'] for r in res):.0f} ms")


if __name__ == "__main__":
    main()
