# Mesure rapide (~2 s) : où part le temps ? Pour chaque pomme, pas réellement joués contre
# plus court chemin torique au moment où elle apparaît (borne optimiste : ne garantit pas
# la survie après la pomme). Par phase, et pour les cycles par tranche de remplissage.
#
#   python3 regret.py                 -> prudent, switch 60 %, seeds 1000..1009
#   python3 regret.py 0.5 prudent 20  -> seuil, mode, nombre de seeds
#   python3 regret.py 0.6 prudent 10 shortcut=4 a_first=True  -> options du joueur
import statistics
import sys
from multiprocessing import Pool

from player import *  # noqa: F401,F403

SWITCH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
MODE = sys.argv[2] if len(sys.argv) > 2 else "prudent"
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
# options supplémentaires du joueur : clé=valeur, ex. shortcut=4 a_first=True
KW = {k: eval(v) for k, v in (a.split("=", 1) for a in sys.argv[4:])}

def run(seed):
    g = new_game(seed); p = AlgoPlayer(SWITCH, MODE, **KW)
    out = []  # (phase, remplissage, pas réels, plus court chemin)
    cur = None
    while not g.done and g.steps_since_apple <= LOOP_LIMIT:
        if g.steps_since_apple == 0 or cur is None:
            body = [idx(x, y) for x, y in g.body]; ap = idx(*g.apple)
            dist, _, _ = dijkstra(body[0], free_times(body, 1 if g.grow_pending else 0), ap)
            cur = [p.hamilton, len(body) / N, 0, dist[ap]]
        g.play_step(p.get_action(g))
        cur[2] += 1
        if g.steps_since_apple == 0 or g.victory:
            out.append(tuple(cur)); cur = None
    return out, g.victory, g.steps

def main():
    with Pool(10) as pool:
        res = pool.map(run, range(1000, 1000 + SEEDS))
    rows = [r for o, _, _ in res for r in o]
    wins = [s for _, v, s in res if v]
    print(f"victoires {len(wins)}/{SEEDS}, pas moyens (victoires) "
          f"{statistics.mean(wins) if wins else float('nan'):.0f}")
    for name, sel in (("phase 1 (Dijkstra prudent)", lambda r: not r[0]), ("phase 2 (cycles)", lambda r: r[0])):
        rs = [r for r in rows if sel(r)]
        unreach = [r for r in rs if r[3] == float("inf")]
        print(f"{name}: {len(unreach)/SEEDS:.1f} pommes/partie inatteignables à l'apparition, {sum(r[2] for r in unreach)/max(1,len(unreach)):.1f} pas/pomme pour elles")
        rs = [r for r in rs if r[3] != float("inf")]
        tot, ideal = sum(r[2] for r in rs), sum(r[3] for r in rs)
        zero = sum(1 for r in rs if r[2] == r[3])
        print(f"{name}: {len(rs)/SEEDS:.0f} pommes/partie, {tot/SEEDS:.0f} pas/partie, {tot/len(rs):.1f} pas/pomme, "
              f"plus court {ideal/len(rs):.1f} -> surcoût {(tot-ideal)/SEEDS:.0f} pas/partie ({tot/ideal:.2f}x), "
              f"pommes sans détour {zero/len(rs):.0%}")
    # phase 2 par tranche de remplissage
    for lo, hi in ((0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        rs = [r for r in rows if r[0] and lo <= r[1] < hi and r[3] != float("inf")]
        if rs:
            print(f"  cycles {lo:.0%}-{min(hi,1):.0%}: {sum(r[2] for r in rs)/len(rs):.1f} pas/pomme vs plus court {sum(r[3] for r in rs)/len(rs):.1f}, "
                  f"surcoût {sum(r[2]-r[3] for r in rs)/SEEDS:.0f} pas/partie")


if __name__ == '__main__':
    main()
