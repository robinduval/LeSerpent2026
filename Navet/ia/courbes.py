# Genere les figures du rendu a partir des CSV results/.
# Choix de lecture : les runs d'entrainement sont bruites par l'exploration (eps>0),
# donc on trace une moyenne glissante en plus du nuage brut ; sinon la courbe ne dit rien.
# Les evals (eps=0) sont la vraie mesure de performance.
import argparse
import csv
import glob
import os
import statistics

import matplotlib
matplotlib.use("Agg")  # pas de fenetre, on ecrit des PNG
import matplotlib.pyplot as plt

FIGDIR = "results/figures"
# palette stable : une couleur par famille, reutilisee d'une figure a l'autre
COL = {
    "baseline": "#8c8c8c",
    "target": "#6b9ac4",
    "eps": "#4a7c59",
    "gamma": "#c9a227",
    "fix4": "#d96c3f",
    "fix5": "#a4508b",
    "fix6": "#2e6f95",
    "ctrl20k": "#b0b0b0",
}


def read_run(path):
    """CSV d'entrainement -> liste de scores."""
    try:
        with open(path) as f:
            return [int(r["score"]) for r in csv.DictReader(f) if r.get("score")]
    except (OSError, ValueError, KeyError):
        return []


def read_eval(path):
    """CSV d'eval -> (scores, temps_jeu_s ou None si ancien format)."""
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return [], []
    sc = [int(r["score"]) for r in rows if r.get("score")]
    tj = [float(r["temps_jeu_s"]) for r in rows if r.get("temps_jeu_s")]
    return sc, tj


def moving_avg(v, w):
    if len(v) < w:
        w = max(1, len(v) // 10 or 1)
    out, acc = [], 0.0
    from collections import deque
    win = deque()
    for x in v:
        win.append(x)
        acc += x
        if len(win) > w:
            acc -= win.popleft()
        out.append(acc / len(win))
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_progression(window):
    """Fig 1 : courbes d'apprentissage des etapes 1-4 (runs mono-seed historiques)."""
    runs = [("run_baseline_1000.csv", "Baseline (eps=80-n)", "baseline"),
            ("run_target_e1.csv", "Fix1 target network", "target"),
            ("run_eps_decay_e2.csv", "Fix2 eps-decay", "eps"),
            ("run_gamma097_e3.csv", "Fix3 gamma 0.97", "gamma"),
            ("run_fix4_grouped.csv", "Fix4 truncation + step -0.01", "fix4")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for fn, label, key in runs:
        sc = read_run(f"results/{fn}")
        if not sc:
            continue
        ax.plot(moving_avg(sc, window), label=f"{label}  (n={len(sc)})",
                color=COL[key], linewidth=1.8)
    style(ax, f"Etapes 1-4 : apprentissage (moyenne glissante {window} parties)",
          "partie", "score")
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.99, 0.02, "score pendant l'entrainement : bruite par l'exploration (eps>0)",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic", alpha=0.7)
    # piege de lecture : la baseline monte vite car eps=0 des 80 parties (elle n'explore
    # plus et exploite un optimum local) ; les fixes explorent donc scorent moins EN
    # TRAIN mais gagnent en eval. Sans cette note la figure se lit a l'envers.
    ax.annotate("eps=0 des 80 parties :\nmonte vite puis plafonne\n(exploite un optimum local)",
                xy=(420, 25.0), xytext=(560, 21.0), fontsize=8.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#999", linewidth=1))
    ax.text(0.5, 0.06, "lire cette figure avec la fig. 3 : en EVAL (eps=0) chaque fix gagne",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#a33", alpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/1_progression_etapes.png", dpi=150)
    plt.close(fig)
    return "1_progression_etapes.png"


def fig_seeds(window):
    """Fig 2 : fix5 / fix6 / controle multi-seed -> montre la variance inter-seed."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tag, key, label in [("ctrl20k", "ctrl20k", "Controle (sans tore)"),
                            ("fix5", "fix5", "Fix5 tore"),
                            ("fix6", "fix6", "Fix6 etat enrichi")]:
        paths = sorted(glob.glob(f"results/run_{tag}_s*.csv"))
        for i, p in enumerate(paths):
            sc = read_run(p)
            if not sc:
                continue
            seed = os.path.basename(p).split("_s")[-1].split(".")[0]
            ax.plot(moving_avg(sc, window), color=COL[key], linewidth=1.4,
                    alpha=0.85 if i == 0 else 0.45,
                    label=f"{label} ({len(paths)} seeds)" if i == 0 else None)
    style(ax, f"Runs multi-seed : chaque seed tracee separement (moy. glissante {window})",
          "partie", "score")
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.99, 0.02, "l'ecart entre seeds d'une meme couleur = le bruit a battre",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic", alpha=0.7)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/2_seeds_variance.png", dpi=150)
    plt.close(fig)
    return "2_seeds_variance.png"


def fig_eval_bars():
    """Fig 3 : eval eps=0, barres + points par seed + barre d'erreur inter-seed."""
    groups = [
        ("Baseline", ["eval_baseline1000.csv"], "baseline"),
        ("Fix1+2 eps", ["eval_eps.csv"], "eps"),
        ("Fix3 gamma", ["eval_gamma.csv"], "gamma"),
        ("Fix4", ["eval_fix4_best.csv"], "fix4"),
        ("Controle 8k", sorted(glob.glob("results/eval_ctrl20k_s*.csv")), "ctrl20k"),
        ("Fix5 tore 8k", sorted(glob.glob("results/eval_fix5_s*.csv")), "fix5"),
        ("Fix6 etat 20k", sorted(glob.glob("results/eval_fix6_s*.csv")), "fix6"),
    ]
    labels, means, errs, cols, clouds = [], [], [], [], []
    for label, files, key in groups:
        per_seed = []
        for f in files:
            p = f if f.startswith("results/") else f"results/{f}"
            sc, _ = read_eval(p)
            if sc:
                per_seed.append(statistics.mean(sc))
        if not per_seed:
            continue
        labels.append(f"{label}\n(n={len(per_seed)})")
        means.append(statistics.mean(per_seed))
        errs.append(statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0)
        cols.append(COL[key])
        clouds.append(per_seed)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(labels))
    ax.bar(x, means, yerr=errs, capsize=5, color=cols, alpha=0.75,
           error_kw={"linewidth": 1.2, "ecolor": "#333"})
    for i, pts in enumerate(clouds):
        if len(pts) > 1:
            ax.scatter([i] * len(pts), pts, color="#222", s=18, zorder=3, alpha=0.8)
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(i, m + e + 0.8, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    style(ax, "Evaluation eps=0 : score moyen (barre d'erreur = ecart-type inter-seed)",
          "", "score moyen sur 50 parties")
    ax.set_ylim(0, max(m + e for m, e in zip(means, errs)) * 1.28)
    ax.text(0.01, 0.97, "points noirs = une seed chacun ; n=1 -> pas de barre d'erreur possible",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            style="italic", alpha=0.7)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/3_eval_comparaison.png", dpi=150)
    plt.close(fig)
    return "3_eval_comparaison.png"


def fig_score_temps():
    """Fig 4 : score vs temps de jeu (la metrique demandee pour le rendu)."""
    files = sorted(glob.glob("results/eval_fix5_s*.csv")) + \
            sorted(glob.glob("results/eval_fix6_s*.csv")) + \
            sorted(glob.glob("results/bilan_*.csv"))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    seen = set()
    any_pt = False
    for p in files:
        sc, tj = read_eval(p)
        if not sc or not tj or len(sc) != len(tj):
            continue  # ancien format sans colonne temps
        key = "fix6" if "fix6" in p else "fix5"
        lbl = {"fix5": "Fix5 tore", "fix6": "Fix6 etat enrichi"}[key]
        ax.scatter(tj, sc, s=22, alpha=0.55, color=COL[key],
                   label=lbl if key not in seen else None, edgecolors="none")
        seen.add(key)
        any_pt = True
    if not any_pt:
        plt.close(fig)
        return None
    style(ax, "Score vs temps de jeu (clock du socle, GAME_SPEED=5)",
          "temps de jeu de la partie (s)", "score (pommes)")
    ax.legend(fontsize=9, frameon=False)
    ax.text(0.99, 0.02, "une partie = un point ; pente = pommes par seconde",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic", alpha=0.7)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/4_score_vs_temps.png", dpi=150)
    plt.close(fig)
    return "4_score_vs_temps.png"


def fig_causes():
    """Fig 5 : causes de mort en eval -> justifie Fix6 (100% morsures)."""
    groups = [("Fix4", ["results/eval_fix4_best.csv"], "fix4"),
              ("Fix5", sorted(glob.glob("results/eval_fix5_s*.csv")), "fix5"),
              ("Fix6", sorted(glob.glob("results/eval_fix6_s*.csv")), "fix6")]
    labels, corps, autres, cols = [], [], [], []
    for label, files, key in groups:
        c = a = 0
        for p in files:
            try:
                with open(p) as f:
                    for r in csv.DictReader(f):
                        if r.get("cause") == "corps":
                            c += 1
                        elif r.get("cause"):
                            a += 1
            except OSError:
                pass
        if c + a == 0:
            continue
        tot = c + a
        labels.append(label)
        corps.append(100 * c / tot)
        autres.append(100 * a / tot)
        cols.append(COL[key])
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(labels))
    ax.bar(x, corps, color=cols, alpha=0.8, label="morsure (corps)")
    ax.bar(x, autres, bottom=corps, color="#dcdcdc", label="famine / autre")
    for i, v in enumerate(corps):
        ax.text(i, v / 2, f"{v:.0f}%", ha="center", fontsize=10,
                color="white", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    style(ax, "Causes de mort en evaluation (eps=0)", "", "% des parties")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/5_causes_mort.png", dpi=150)
    plt.close(fig)
    return "5_causes_mort.png"


def fig_tore():
    """Fig 6 : taux de wrap mesure par diag_wrap (chiffres du journal)."""
    # valeurs mesurees et consignees dans CHANGELOG.md (diag_wrap.py, 10 parties)
    data = [("Fix4\n(coord. brutes)", 1.2, 12.1, COL["fix4"]),
            ("Fix5\n(dist. torique)", 7.1, 8.8, COL["fix5"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = [d[0] for d in data]
    x = range(len(labels))
    a1.bar(x, [d[1] for d in data], color=[d[3] for d in data], alpha=0.8)
    for i, d in enumerate(data):
        a1.text(i, d[1] + 0.15, f"{d[1]}%", ha="center", fontsize=10, fontweight="bold")
    a1.set_xticks(list(x)); a1.set_xticklabels(labels, fontsize=9)
    style(a1, "Traversees de mur", "", "% des pas")
    a1.set_ylim(0, 9)
    a2.bar(x, [d[2] for d in data], color=[d[3] for d in data], alpha=0.8)
    for i, d in enumerate(data):
        a2.text(i, d[2] + 0.2, f"{d[2]}", ha="center", fontsize=10, fontweight="bold")
    a2.set_xticks(list(x)); a2.set_xticklabels(labels, fontsize=9)
    style(a2, "Efficacite du trajet", "", "pas par pomme (moins = mieux)")
    a2.set_ylim(0, 14)
    fig.suptitle("Fix5 : exploitation du tore (diag_wrap.py, 10 parties)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/6_exploitation_tore.png", dpi=150)
    plt.close(fig)
    return "6_exploitation_tore.png"


def main():
    ap = argparse.ArgumentParser(description="Genere les figures du rendu depuis results/*.csv")
    ap.add_argument("--window", type=int, default=200,
                    help="fenetre de la moyenne glissante sur les runs")
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    made = []
    for fn in (lambda: fig_progression(args.window),
               lambda: fig_seeds(args.window),
               fig_eval_bars,
               fig_score_temps,
               fig_causes,
               fig_tore):
        try:
            r = fn()
            if r:
                made.append(r)
        except Exception as e:  # une figure ratee ne doit pas bloquer les autres
            print(f"  ! figure ignoree : {type(e).__name__}: {e}")
    for m in made:
        print(f"  {FIGDIR}/{m}")
    print(f"\n{len(made)} figures -> {FIGDIR}/")


if __name__ == "__main__":
    main()
