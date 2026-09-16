"""Courbes d'apprentissage, générées depuis les métriques d'un run.

Matplotlib est configuré en backend non interactif : tracer des courbes ne
doit jamais ouvrir de fenêtre ni ralentir une grid search. On dessine à la
fin d'un run, à partir du fichier JSONL, plutôt qu'à chaque épisode.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def read_metrics(run_dir):
    """Relit metrics.jsonl et sépare les lignes d'entraînement et d'évaluation."""
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return (
        [row for row in rows if row.get("phase") == "train"],
        [row for row in rows if row.get("phase") == "eval"],
    )


def rolling_mean(values, window):
    """Moyenne glissante causale, tronquée au début de la série."""
    out, total = [], 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def plot_run(run_dir, path=None, title=None):
    """Produit le tableau de bord d'un run en PNG. Retourne le chemin."""
    train_rows, eval_rows = read_metrics(run_dir)
    if not train_rows:
        return None

    episodes = [row["episode"] for row in train_rows]
    scores = [row["train_score"] for row in train_rows]

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.suptitle(title or os.path.basename(run_dir.rstrip("/")), fontsize=14)

    # 1. Score d'entraînement et moyennes glissantes
    ax = axes[0][0]
    ax.plot(episodes, scores, linewidth=0.5, alpha=0.35, label="score par épisode")
    ax.plot(episodes, rolling_mean(scores, 50), linewidth=1.6, label="moyenne 50")
    ax.plot(episodes, rolling_mean(scores, 100), linewidth=1.6, label="moyenne 100")
    ax.set_title("Entraînement (avec exploration)")
    ax.set_xlabel("épisode")
    ax.set_ylabel("score officiel")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Évaluation : c'est la courbe qui décide du champion
    ax = axes[0][1]
    if eval_rows:
        eval_episodes = [row["episode"] for row in eval_rows]
        means = [row["eval_mean_score"] for row in eval_rows]
        p10 = [row["eval_p10_score"] for row in eval_rows]
        p90 = [row["eval_p90_score"] for row in eval_rows]
        ax.plot(eval_episodes, means, marker="o", linewidth=1.8, label="score moyen")
        ax.fill_between(eval_episodes, p10, p90, alpha=0.2, label="p10 - p90")
        ax.plot(
            eval_episodes,
            [row["eval_record"] for row in eval_rows],
            linestyle="--",
            linewidth=1.0,
            label="record du bloc",
        )
        ax.legend()
    ax.set_title("Évaluation (epsilon = 0, seeds fixées)")
    ax.set_xlabel("épisode")
    ax.set_ylabel("score officiel")
    ax.grid(alpha=0.3)

    # 3. Exploration
    ax = axes[1][0]
    ax.plot(episodes, [row["epsilon"] for row in train_rows], color="tab:orange")
    ax.set_title("Epsilon")
    ax.set_xlabel("épisode")
    ax.set_ylabel("epsilon")
    ax.grid(alpha=0.3)

    # 4. Loss et valeur Q moyenne
    ax = axes[1][1]
    loss_points = [
        (row["episode"], row["loss_mean"])
        for row in train_rows
        if row.get("loss_mean") is not None
    ]
    if loss_points:
        ax.plot(*zip(*loss_points), color="tab:red", linewidth=0.9, label="loss")
        ax.set_yscale("log")
    q_points = [
        (row["episode"], row["q_mean"])
        for row in train_rows
        if row.get("q_mean") is not None
    ]
    if q_points:
        twin = ax.twinx()
        twin.plot(*zip(*q_points), color="tab:blue", linewidth=0.9, label="Q moyen")
        twin.set_ylabel("Q moyen", color="tab:blue")
    ax.set_title("Loss et valeur Q moyenne")
    ax.set_xlabel("épisode")
    ax.set_ylabel("loss (échelle log)", color="tab:red")
    ax.grid(alpha=0.3)

    figure.tight_layout()
    path = path or os.path.join(run_dir, "training_curve.png")
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def plot_comparison(run_dirs, path, labels=None, title="Comparaison des configurations"):
    """Superpose les courbes d'évaluation de plusieurs runs.

    C'est la figure honnête pour comparer deux algorithmes : on trace la
    mesure à epsilon nul sur les mêmes seeds, pas le score d'entraînement.
    """
    figure, ax = plt.subplots(figsize=(10, 6))
    for i, run_dir in enumerate(run_dirs):
        _, eval_rows = read_metrics(run_dir)
        if not eval_rows:
            continue
        label = labels[i] if labels else os.path.basename(run_dir.rstrip("/"))
        ax.plot(
            [row["episode"] for row in eval_rows],
            [row["eval_mean_score"] for row in eval_rows],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("épisode d'entraînement")
    ax.set_ylabel("score moyen d'évaluation (epsilon = 0)")
    ax.legend()
    ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path
