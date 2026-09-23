# Tableau recapitulatif pour le rendu : score ET temps, par modele.
# Temps de jeu = steps / GAME_SPEED : la duree reelle de la partie a la clock du
# socle (5 FPS), independante de la vitesse de calcul de la machine.
import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys

from game import GAME_SPEED


def eval_model(path, py, episodes, flags):
    out = f"results/bilan_{os.path.basename(path).replace('.pth','')}.csv"
    cmd = [py, "evaluate.py", "--model", path, "--episodes", str(episodes),
           "--hunger-mode", "truncated", "--step-reward", "-0.01", "--out", out] + flags
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    with open(out) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    sc = [int(x["score"]) for x in rows]
    tj = [float(x["temps_jeu_s"]) for x in rows]
    best = max(range(len(sc)), key=lambda i: sc[i])
    return {"mean": statistics.mean(sc), "max": max(sc),
            "std": statistics.pstdev(sc),
            "t_mean": statistics.mean(tj), "t_best": tj[best],
            "s_pomme": sum(tj) / max(1, sum(sc))}


def main():
    ap = argparse.ArgumentParser(description="Tableau score+temps pour le rendu.")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--torus-food", action="store_true")
    ap.add_argument("--rich-state", action="store_true")
    args = ap.parse_args()

    models = args.models or sorted(glob.glob("model_*_best.pth"))
    flags = []
    if args.torus_food:
        flags.append("--torus-food")
    if args.rich_state:
        flags.append("--rich-state")

    print(f"Eval eps=0, {args.episodes} parties/modele, clock GAME_SPEED={GAME_SPEED}")
    print(f"flags: {' '.join(flags) or '(etat 11 bits historique)'}\n")
    hdr = f"{'modele':32s} {'score moy':>9s} {'max':>5s} {'ecart-t':>8s} {'t moy':>8s} {'t record':>9s} {'s/pomme':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        r = eval_model(m, sys.executable, args.episodes, flags)
        if r is None:
            print(f"{os.path.basename(m):32s} {'(incompatible avec ces flags)':>45s}")
            continue
        print(f"{os.path.basename(m):32s} {r['mean']:9.1f} {r['max']:5d} {r['std']:8.1f} "
              f"{r['t_mean']:7.0f}s {r['t_best']:8.0f}s {r['s_pomme']:7.2f}s")


if __name__ == "__main__":
    main()
