# Benchmark du seuil de switch Dijkstra+flood-fill -> Hamilton.
# Chaque seuil est joué sur les mêmes seeds (comparaison appariée), en parallèle.
#
#   python3 sweep.py                          -> seuils par défaut, 30 seeds
#   python3 sweep.py --seeds 100 --thresholds 0.3 0.35 0.4
#
# Sortie : tableau trié (victoires puis vitesse) + CSV brut dans results/.
import argparse
import csv
import os
import statistics
import time
from multiprocessing import Pool

from player import CYCLE_MODES, NEVER, play_one

DEFAULT_THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, NEVER]


def _job(args):
    seed, thr, cycle, trap = args
    t = time.time()
    r = play_one(seed, thr, cycle, trap)
    r["threshold"] = thr
    r["cycle"] = cycle
    r["seconds"] = round(time.time() - t, 2)
    return r


def label(thr):
    return "jamais" if thr > 1 else f"{thr:.0%}"


def main():
    ap = argparse.ArgumentParser(description="Benchmark du seuil de switch vers Hamilton.")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                    help="taux de remplissage (0..1) ; > 1 = jamais de switch")
    ap.add_argument("--seeds", type=int, default=30, help="parties par seuil")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--cycle", default="reconstruit", choices=CYCLE_MODES,
                    help="helice = référence 20:14 ; reconstruit = option A (20:25) ; "
                         "temporel = option B (20:40)")
    ap.add_argument("--trap", action="store_true",
                    help="(temporel) switch aussi juste avant d'être coincé")
    args = ap.parse_args()

    jobs = [(args.seed0 + s, thr, args.cycle, args.trap) for thr in args.thresholds for s in range(args.seeds)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows.append(r)
            print(f"\r{i}/{len(jobs)} parties", end="", flush=True)
    print(f"\r{len(jobs)} parties en {time.time() - t0:.0f}s ({args.workers} workers)\n")

    os.makedirs("results", exist_ok=True)
    name = args.cycle + ("_piege" if args.trap else "")
    out = os.path.join("results", f"sweep_{name}_{time.strftime('%Y%m%d_%H%M')}.csv")
    keys = ["cycle", "threshold", "seed", "victory", "score", "steps", "cause", "fill", "switch_step",
            "switch_fill", "switch_by", "secours", "ordered_step", "transition_steps", "shortcuts", "rebuilds",
            "rebuild_kept", "rebuild_fails", "seconds"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["threshold"], r["seed"])))

    summary = []
    for thr in args.thresholds:
        rs = [r for r in rows if r["threshold"] == thr]
        wins = [r for r in rs if r["victory"]]
        switched = [r for r in rs if r["switch_step"] is not None]
        entered = [r for r in switched if r["ordered_step"] is not None]
        by_trap = [r for r in switched if r.get("switch_by") == "piege"]
        causes = {}
        for r in rs:
            if not r["victory"]:
                causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        summary.append({
            "thr": thr,
            "win": len(wins) / len(rs),
            "score": statistics.mean(r["score"] for r in rs),
            "steps_win": statistics.mean(r["steps"] for r in wins) if wins else None,
            "steps_med": statistics.median(r["steps"] for r in wins) if wins else None,
            "entered": f"{len(entered)}/{len(switched)}" if switched else "-",
            "trans": statistics.mean(r["transition_steps"] for r in entered) if entered else None,
            "sw_fill": statistics.mean(r["switch_fill"] for r in switched) if switched else None,
            "trap": f"{len(by_trap)}/{len(switched)}" if switched else "-",
            "by": " ".join(f"{k}:{v}" for k, v in sorted(
                {b: sum(1 for r in switched if r.get("switch_by") == b)
                 for b in {r.get("switch_by") for r in switched}}.items())) or "-",
            "causes": " ".join(f"{k}:{v}" for k, v in sorted(causes.items())) or "-",
        })

    # meilleur = plus de victoires, puis victoire la plus rapide (speedrun)
    summary.sort(key=lambda s: (-s["win"], s["steps_win"] if s["steps_win"] else float("inf")))
    fmt = "{:>7} {:>6} {:>7} {:>10} {:>9} {:>9} {:>8} {:>8} {:>7}  {}"
    print(fmt.format("seuil", "vict.", "score", "pas(vict)", "médiane", "entrée", "trans.",
                     "switch à", "piège", "défaites"))
    for s in summary:
        print(fmt.format(label(s["thr"]), f"{s['win']:.0%}", f"{s['score']:.1f}",
                         f"{s['steps_win']:.0f}" if s["steps_win"] else "-",
                         f"{s['steps_med']:.0f}" if s["steps_med"] else "-",
                         s["entered"], f"{s['trans']:.0f}" if s["trans"] is not None else "-",
                         f"{s['sw_fill']:.0%}" if s["sw_fill"] is not None else "-",
                         s["trap"], s["causes"]))
        print(f"{'':>7} switch par -> {s['by']}")
    print(f"\ncycle : {name} ; {args.seeds} seeds par seuil (seeds {args.seed0}..{args.seed0 + args.seeds - 1}, "
          f"identiques pour tous les seuils)")
    print("entrée = parties où le corps a pu rejoindre le cycle / parties ayant switché ; "
          "trans. = pas de transition moyens ; switch à = remplissage moyen au switch ; "
          "piège = parties switchées par le déclenchement piège")
    print(f"CSV : {out}")


if __name__ == "__main__":
    main()
