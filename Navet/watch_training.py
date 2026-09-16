# Suivi live des runs en cours : lit les CSV que train.py ecrit ligne a ligne.
# Aucune interaction avec les process d'entrainement, on ne fait que lire -> zero risque
# de perturber les runs. Ctrl+C pour sortir, l'entrainement continue.
import argparse
import csv
import glob
import os
import time

BAR_W = 24


def read_run(path, total):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except (FileNotFoundError, OSError):
        return None
    if not rows:
        return None
    scores = [int(r["score"]) for r in rows if r.get("score")]
    if not scores:
        return None
    tail = scores[-min(200, len(scores)):]
    # temps de jeu du record, a la clock du socle (GAME_SPEED=5) : ce que verra le prof
    best_steps = 0
    for r in rows:
        try:
            if int(r["score"]) == max(scores):
                best_steps = max(best_steps, int(r["steps"]))
        except (ValueError, KeyError, TypeError):
            pass
    return {
        "n": len(scores),
        "mean_tail": sum(tail) / len(tail),
        "max": max(scores),
        "eps": float(rows[-1].get("eps", 0) or 0) / 200.0,
        "pct": len(scores) / total if total else 0.0,
        "best_time": best_steps / 5.0,
    }


def spark(values, width=32):
    # mini-courbe ASCII de la progression (moyennes de tranches egales)
    if not values:
        return ""
    chars = " .:-=+*#%@"
    step = max(1, len(values) // width)
    buckets = [values[i:i + step] for i in range(0, len(values), step)][:width]
    means = [sum(b) / len(b) for b in buckets if b]
    if not means:
        return ""
    lo, hi = min(means), max(means)
    rng = (hi - lo) or 1.0
    return "".join(chars[min(len(chars) - 1, int((m - lo) / rng * (len(chars) - 1)))] for m in means)


def main():
    ap = argparse.ArgumentParser(description="Suivi live des trainings (lecture seule).")
    ap.add_argument("--tags", nargs="+", default=["fix5", "ctrl20k"])
    ap.add_argument("--total", type=int, default=20000, help="episodes vises par run")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="un seul affichage puis sortie")
    args = ap.parse_args()

    t0 = time.time()
    prev = {}
    try:
        while True:
            lines = []
            elapsed = time.time() - t0
            lines.append(f"  Navet — suivi training   (+{int(elapsed//60):02d}:{int(elapsed%60):02d}, maj {args.interval:.0f}s, Ctrl+C pour sortir)")
            lines.append("")
            all_done = True
            for tag in args.tags:
                paths = sorted(glob.glob(f"results/run_{tag}_s*.csv"))
                if not paths:
                    continue
                lines.append(f"  [{tag}]")
                tag_means = []
                for p in paths:
                    seed = os.path.basename(p).split("_s")[-1].split(".")[0]
                    st = read_run(p, args.total)
                    if st is None:
                        lines.append(f"    seed {seed}  (démarrage…)")
                        all_done = False
                        continue
                    tag_means.append(st["mean_tail"])
                    if st["n"] < args.total:
                        all_done = False
                    filled = int(st["pct"] * BAR_W)
                    bar = "█" * filled + "░" * (BAR_W - filled)
                    # vitesse et ETA a partir du delta depuis le tour precedent
                    eta = ""
                    if p in prev:
                        dn = st["n"] - prev[p][0]
                        dt = time.time() - prev[p][1]
                        if dn > 0 and dt > 0:
                            rate = dn / dt
                            left = args.total - st["n"]
                            if left > 0:
                                eta = f"  ETA {int(left/rate//60):2d}m{int(left/rate%60):02d}s"
                            else:
                                eta = "  terminé"
                    prev[p] = (st["n"], time.time())
                    lines.append(
                        f"    seed {seed} {bar} {st['n']:5d}/{args.total}"
                        f"  last200={st['mean_tail']:5.1f}  max={st['max']:3d}"
                        f"  eps={st['eps']:.2f}  record en {st['best_time']:.0f}s de jeu{eta}"
                    )
                if len(tag_means) > 1:
                    m = sum(tag_means) / len(tag_means)
                    var = (sum((x - m) ** 2 for x in tag_means) / len(tag_means)) ** 0.5
                    lines.append(f"    → inter-seed last200 : mean={m:5.1f}  écart-type={var:4.1f}")
                # courbe d'apprentissage du premier run du tag
                try:
                    with open(paths[0]) as f:
                        sc = [int(r["score"]) for r in csv.DictReader(f) if r.get("score")]
                    if sc:
                        lines.append(f"    courbe s{os.path.basename(paths[0]).split('_s')[-1].split('.')[0]} |{spark(sc)}| max {max(sc)}")
                except (OSError, ValueError):
                    pass
                lines.append("")

            # efface l'ecran sans dependre de $TERM (ANSI : home + clear)
            print("\033[H\033[J" + "\n".join(lines))
            if args.once or all_done:
                if all_done and not args.once:
                    print("  Tous les runs sont terminés.")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  (suivi arrêté — l'entraînement continue en arrière-plan)")


if __name__ == "__main__":
    main()
