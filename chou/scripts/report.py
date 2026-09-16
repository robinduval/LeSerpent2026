#!/usr/bin/env python3
"""Rebuild transparent plots from existing logs; never run or score a policy.

Run from any directory with Python 3.13:
    python chou/scripts/report.py
Optional dependencies: chou/requirements-dev.txt.
All generated reports and plotting caches stay in chou/runs/report/.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / "runs"
OUTPUT = RUNS / "report"
WINDOW = 100


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path):
    """Read a single snapshot; a currently unwritten final line is explicit."""
    if not path.exists():
        return [], None
    contents = path.read_bytes()
    rows, skipped_partial = [], 0
    lines = contents.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not contents.endswith(b"\n"):
                skipped_partial += 1
            else:
                raise ValueError(f"Malformed JSON record: {path}:{index + 1}")
    source = {"path": str(path.relative_to(BASE)), "sha256": hashlib.sha256(contents).hexdigest(),
              "bytes_read": len(contents), "rows_read": len(rows),
              "incomplete_final_lines_ignored": skipped_partial}
    return rows, source


def numeric_summary(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "std_population": None,
                "min": None, "max": None}
    if not all(math.isfinite(value) for value in values):
        raise ValueError("A report source contains non-finite metric values")
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "std_population": statistics.pstdev(values),
            "min": min(values), "max": max(values)}


def write_json(name, data):
    path = OUTPUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def gather():
    training, validation = [], []
    for folder in sorted(RUNS.iterdir()):
        if not folder.is_dir() or folder == OUTPUT:
            continue
        config = read_json(folder / "config.json")
        episodes, episode_source = read_jsonl(folder / "episodes.jsonl")
        learning, learning_source = read_jsonl(folder / "learning.jsonl")
        if episode_source or learning_source:
            scores = [row["official_score"] for row in episodes]
            counters = [row.get("training_transitions", 0) for row in episodes]
            counters.extend(row.get("transitions", 0) for row in learning)
            training.append({
                "run_id": folder.name, "kind": "accelerated exploratory training, not evaluation",
                "status": "snapshot of observed records; completion not inferred",
                "config": config, "sources": [s for s in (episode_source, learning_source) if s],
                "logged_episodes": len(episodes), "logged_learning_rows": len(learning),
                "training_transitions_observed": max(counters, default=0),
                "scores_all_logged_episodes": numeric_summary(scores),
                "scores_last_100_logged_episodes": numeric_summary(scores[-WINDOW:]),
                "completed_logged_episodes": sum(bool(row["completed"]) for row in episodes),
                "termination_reasons": dict(Counter(row["termination_reason"] for row in episodes)),
                "last_logged_learning_metrics": learning[-1] if learning else None,
                "resume_source": config.get("resume") if config else None,
                "_episodes": episodes, "_learning": learning,
            })
        # Keep the held-out test separate from model-selection validation.
        # RESULTS.md reports final_test independently; do not pool its larger
        # budget or its reserved seeds into validation charts and summaries.
        if not folder.name.startswith("validation_"):
            continue
        evaluations, evaluation_source = read_jsonl(folder / "evaluation.jsonl")
        # An actions trace may precede the first finished evaluation episode.
        if not evaluation_source and not (folder / "actions.jsonl").exists():
            continue
        complete_summary = read_json(folder / "summary.json")
        expected_episodes = config.get("episodes") if config and config.get("evaluate") else None
        groups = defaultdict(list)
        for row in evaluations:
            groups[(row["model_id"], row.get("evaluation_step_budget"), row.get("clock_hz"))].append(row)
        if not groups:
            groups[(None, None, None)] = []
        for (model_id, budget, hz), rows in groups.items():
            scores = [row["official_score"] for row in rows]
            reasons = Counter(row["termination_reason"] for row in rows)
            exact_times = defaultdict(list)
            for row in rows:
                # Compare only naturally terminated games of exactly equal score.
                if row["termination_reason"] in ("self_collision", "completed"):
                    elapsed = row.get("official_time_seconds")
                    if elapsed is not None:
                        exact_times[str(row["official_score"])].append(elapsed)
            status = "invocation summary present" if complete_summary is not None else "in progress or unfinished"
            if expected_episodes is not None and len(rows) < expected_episodes:
                status += "; fewer observed episodes than planned"
            validation.append({
                "run_id": folder.name, "model_id": model_id,
                "status": status, "observed_episodes": len(rows),
                "planned_episodes": expected_episodes,
                "planned_episode_count_known": expected_episodes is not None,
                "sources": [evaluation_source] if evaluation_source else [],
                "seeds": [row["seed"] for row in rows],
                "evaluation_step_budget": budget, "clock_hz": hz,
                "scores": numeric_summary(scores), "termination_reasons": dict(reasons),
                "completed_episodes": sum(bool(row["completed"]) for row in rows),
                "completion_rate": sum(bool(row["completed"]) for row in rows) / len(rows) if rows else None,
                "external_cutoffs": reasons.get("external_step_limit", 0),
                "collision_rate": reasons.get("self_collision", 0) / len(rows) if rows else None,
                "target_223_rate": sum(score == 223 for score in scores) / len(rows) if rows else None,
                "natural_terminal_times_by_exact_score": {score: numeric_summary(times)
                    for score, times in sorted(exact_times.items(), key=lambda item: int(item[0]))},
                "time_comparison_scope": "natural terminal events only; separately for each exact score",
                "_rows": rows,
            })
    return training, validation


def rolling(values, window=WINDOW):
    if len(values) < window:
        return []
    total = sum(values[:window])
    means = [total / window]
    for index in range(window, len(values)):
        total += values[index] - values[index-window]
        means.append(total / window)
    return means


def public(records):
    return [{key: value for key, value in record.items() if not key.startswith("_")}
            for record in records]


def plots(training, validation):
    os.environ["MPLCONFIGDIR"] = str(OUTPUT / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none",
                         "savefig.facecolor": "white", "axes.grid": True,
                         "grid.alpha": .2, "figure.dpi": 150})
    colors = plt.get_cmap("tab20")

    def save(fig, filename):
        fig.savefig(OUTPUT / f"{filename}.svg", bbox_inches="tight")
        fig.savefig(OUTPUT / f"{filename}.png", bbox_inches="tight")
        plt.close(fig)

    def training_plot(filename, title, ylabel, value_key, smoothing):
        fig, ax = plt.subplots(figsize=(11, 5.4))
        rendered = 0
        for index, run in enumerate(training):
            rows = run["_episodes"] if value_key == "official_score" else run["_learning"]
            pairs = [(row.get("training_transitions", row.get("transitions")), row[value_key])
                     for row in rows if value_key in row]
            if len(pairs) < smoothing:
                continue
            xs, ys = zip(*pairs)
            ax.plot(xs[smoothing-1:], rolling(ys, smoothing), label=run["run_id"],
                    color=colors(index % 20), linewidth=1.6, alpha=.9)
            rendered += 1
        ax.set_title(title, loc="left", pad=28)
        ax.text(0, 1.015, "Entraînement exploratoire accéléré — ce graphe n'est pas une évaluation.",
                transform=ax.transAxes, fontsize=9, color="#5b6470")
        ax.set_xlabel("Transitions d'entraînement cumulées du learner")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1_000_000:g} M"))
        footer_y = -.06
        if rendered:
            legend = ax.legend(loc="upper left", bbox_to_anchor=(0, -.2), ncol=2,
                               fontsize=8, frameon=False)
            fig.canvas.draw()
            footer_y = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0 - .045
        else:
            ax.text(.5, .5, "Pas encore assez de données pour la fenêtre annoncée.",
                    ha="center", transform=ax.transAxes)
        fig.text(.08, footer_y, "Sources : episodes.jsonl / learning.jsonl ; reprises et configurations dans training_summary.json.",
                 fontsize=8, color="#5b6470")
        save(fig, filename)

    training_plot("training_scores", "Scores observés à l'entraînement · moyenne mobile de 100 épisodes",
                  "Score officiel par épisode (pommes)", "official_score", 100)
    training_plot("training_loss", "Loss DDQN · moyenne mobile de 25 valeurs journalisées",
                  "Huber loss (pondérée si PER)", "loss", 25)
    training_plot("training_q", "Valeur Q moyenne · moyenne mobile de 25 valeurs journalisées",
                  "Valeur Q prédite, unités de récompense d'apprentissage", "q_mean", 25)

    fig, ax = plt.subplots(figsize=(11, max(3.5, len(validation) * .62 + 1.6)))
    labels = []
    for index, run in enumerate(validation):
        sample = run["scores"]
        pending = " · en cours/inachevé" if "in progress" in run["status"] else ""
        labels.append(f"{run['run_id']} · n={sample['n']}{pending}")
        if sample["n"]:
            ax.errorbar(sample["mean"], index, xerr=sample["std_population"], fmt="o",
                        capsize=4, color=colors(index % 20), markersize=6)
            # Every observed episode remains visible, including external cutoffs.
            ax.scatter([row["official_score"] for row in run["_rows"]],
                       [index] * sample["n"], s=17, color=colors(index % 20), alpha=.45)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_title("Évaluations à 5 Hz · scores actuellement observés", loc="left", pad=26)
    ax.text(0, 1.025, "Point : moyenne ; barre : écart-type population (pas un intervalle de confiance).",
            transform=ax.transAxes, fontsize=9, color="#5b6470")
    ax.set_xlabel("Score officiel (pommes)")
    fig.text(.02, -.06, "Instantané, pas un classement : n, graines et budgets peuvent différer.\n"
             "Les scores aux interruptions externes sont inclus ; leur admissibilité officielle reste inconnue.",
             fontsize=8, color="#5b6470")
    if not labels:
        ax.text(.5, .5, "Aucune évaluation observée.", ha="center", transform=ax.transAxes)
    save(fig, "validation_scores")

    comparisons = defaultdict(list)
    for run in validation:
        for score, summary in run["natural_terminal_times_by_exact_score"].items():
            comparisons[int(score)].append((run["run_id"], summary))
    # Display only score strata containing at least two candidate runs.
    comparable = {score: rows for score, rows in comparisons.items() if len(rows) >= 2}
    fig, ax = plt.subplots(figsize=(11, max(3.5, sum(map(len, comparable.values())) * .5 + 1.8)))
    labels, position = [], 0
    for score, candidates in sorted(comparable.items()):
        for name, summary in candidates:
            labels.append(f"Score {score} · {name} · n={summary['n']}")
            ax.errorbar(summary["mean"], position, xerr=summary["std_population"], fmt="o",
                        capsize=4, color=colors(position % 20))
            position += 1
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_title("Temps de fins naturelles · comparaisons à score strictement égal", loc="left", pad=22)
    ax.set_xlabel("Temps observé à l'événement terminal, secondes (moyenne ± écart-type)")
    if not comparable:
        ax.text(.5, .5, "Aucun score terminal commun à au moins deux candidats pour l'instant.",
                ha="center", wrap=True, transform=ax.transAxes)
    fig.text(.02, -.06, "Interruptions externes exclues. Comparer uniquement les lignes du même score.\n"
             "Les échecs et interruptions de chaque candidat restent visibles dans validation_summary.json.",
             fontsize=8, color="#5b6470")
    save(fig, "equal_score_times")


def main():
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Use Python 3.13: chou/.venv/bin/python chou/scripts/report.py")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    training, validation = gather()
    generated = datetime.now(timezone.utc).isoformat()
    write_json("training_summary.json", {"generated_at_utc": generated,
        "notice": "Accelerated exploratory training episodes are not evaluations or official times.",
        "smoothing_episode_window": WINDOW, "runs": public(training)})
    write_json("validation_summary.json", {"generated_at_utc": generated,
        "notice": "Snapshot of validation episodes only. No ranking is inferred from unequal samples.",
        "scope": "validation_* directories only; held-out final_test is deliberately excluded",
        "score_error_bars": "population standard deviation, not a confidence interval",
        "partial_runs": "Missing summary.json means in progress or unfinished; planned n may be unknown.",
        "time_rule": "Only compare natural terminal times separately at each identical exact score.",
        "candidates": public(validation)})
    plots(training, validation)
    print(json.dumps({"output": str(OUTPUT), "training_runs": len(training),
                      "validation_groups": len(validation),
                      "observed_validation_episodes": sum(row["observed_episodes"] for row in validation),
                      "svg_figures": 5}, ensure_ascii=False))


if __name__ == "__main__":
    main()
