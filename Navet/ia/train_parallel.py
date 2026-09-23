# Training long parallelise : N runs independants (seeds differentes) en processus separes.
# Pourquoi multi-run et non multi-worker sur un run : le reseau 11-256-3 est minuscule,
# un pas d'optim ne sature pas un coeur. Ce qui coute, c'est le nombre d'episodes joues.
# Lancer 5 seeds en parallele donne 5x plus d'experience par minute ET une mesure de
# variance (nos runs mono-seed ne disaient pas si un ecart etait du signal ou du bruit).
# Chaque process est bride a 1 thread torch, sinon les runs se disputent les coeurs.
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time


def run_one(seed, args, py):
    prefix = f"model_{args.tag}_s{seed}"
    out = f"results/run_{args.tag}_s{seed}.csv"
    cmd = [py, "train.py",
           "--episodes", str(args.episodes),
           "--seed", str(seed),
           "--out", out,
           "--save-prefix", prefix,
           "--use-target",
           "--eps-mode", "decay",
           "--eps-min", str(args.eps_min),
           "--eps-decay", str(args.eps_decay),
           "--gamma", str(args.gamma),
           "--step-reward", str(args.step_reward),
           "--hunger-mode", "truncated"]
    if args.torus_food:
        cmd.append("--torus-food")
    if args.rich_state:
        cmd.append("--rich-state")
    env = dict(os.environ)
    # 1 thread par process : le parallelisme est entre les runs, pas dedans
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    log = open(f"results/log_{args.tag}_s{seed}.txt", "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return {"seed": seed, "proc": p, "log": log, "out": out, "prefix": prefix}


def main():
    ap = argparse.ArgumentParser(description="Lance N trainings en parallele (une seed par process).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--tag", type=str, default="fix5")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--eps-min", type=float, default=0.02)
    ap.add_argument("--eps-decay", type=float, default=2000)
    ap.add_argument("--step-reward", type=float, default=-0.01)
    ap.add_argument("--torus-food", action="store_true")
    ap.add_argument("--rich-state", action="store_true")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    py = sys.executable
    t0 = time.time()
    print(f"{len(args.seeds)} runs x {args.episodes} episodes, tag={args.tag}, "
          f"torus_food={args.torus_food}, rich_state={args.rich_state}")
    jobs = [run_one(s, args, py) for s in args.seeds]
    for j in jobs:
        j["proc"].wait()
        j["log"].close()
        print(f"  seed {j['seed']} fini ({time.time()-t0:.0f}s) -> {j['prefix']}_best.pth")

    # resume : dernier decile de chaque run, pour comparer sans le bruit du debut
    summary = []
    for j in jobs:
        try:
            with open(j["out"]) as f:
                rows = list(csv.DictReader(f))
        except FileNotFoundError:
            print(f"  seed {j['seed']} : pas de CSV (run echoue, voir results/log_{args.tag}_s{j['seed']}.txt)")
            continue
        if not rows:
            continue
        scores = [int(r["score"]) for r in rows]
        tail = scores[-max(1, len(scores)//10):]
        summary.append({"seed": j["seed"], "mean_tail": statistics.mean(tail),
                        "max": max(scores), "n": len(scores)})
        print(f"  seed {j['seed']}: last10%={statistics.mean(tail):.1f} max={max(scores)}")

    if len(summary) > 1:
        means = [s["mean_tail"] for s in summary]
        print(f"\ninter-seed : mean={statistics.mean(means):.1f} "
              f"ecart-type={statistics.pstdev(means):.1f} "
              f"(min {min(means):.1f} / max {max(means):.1f})")
    with open(f"results/summary_{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s -> results/summary_{args.tag}.json")


if __name__ == "__main__":
    main()
