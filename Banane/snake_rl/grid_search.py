"""Recherche d'hyperparamètres reproductible.

    python -m snake_rl.grid_search search_spaces/algorithms.json

Un espace de recherche est un fichier JSON :

    {
      "name": "algorithmes",
      "base": {"episodes": 1500, "eval_episodes": 20},
      "grid": {"algorithm": ["dqn", "ddqn"], "learning_rate": [0.001, 0.0003]},
      "seeds": [0, 1],
      "workers": 4
    }

Chaque combinaison de `grid` est croisée avec chaque seed, ce qui donne un
trial par entraînement. Aucune constante n'est modifiée dans le code : relancer
une campagne, c'est relancer le même fichier.

Les trials tournent en parallèle sur CPU. Les réseaux sont minuscules, donc
plusieurs processus CPU battent largement un seul processus GPU : le coût est
dominé par la simulation du jeu en Python, pas par les multiplications
matricielles.
"""

import argparse
import csv
import itertools
import multiprocessing
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from .config import Config
from .metrics import environment_info
from .rules import RULESET


def expand_grid(grid):
    """Produit toutes les combinaisons d'un dictionnaire de listes."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def build_trials(space):
    """Construit la liste des trials : combinaisons croisées avec les seeds."""
    base = Config(**space.get("base", {}))
    combinations = expand_grid(space.get("grid", {}))
    seeds = space.get("seeds", [base.seed])

    trials = []
    for index, overrides in enumerate(combinations, start=1):
        for seed in seeds:
            label = "_".join(
                f"{key}-{value}" for key, value in sorted(overrides.items())
            )
            trials.append(
                {
                    "trial_id": f"trial_{index:04d}_seed{seed}",
                    "group": label or "base",
                    "overrides": dict(overrides),
                    "config": base.replace(
                        **overrides,
                        seed=seed,
                        run_id=f"trial_{index:04d}_seed{seed}",
                        # Une fenêtre Pygame par trial serait ingérable, et
                        # les replays restent enregistrés sur disque.
                        show_replay_window=False,
                    ),
                }
            )
    return trials


def run_trial(payload):
    """Exécute un trial. Fonction de niveau module, pour être picklable."""
    from .train import Trainer

    config = Config.from_dict(payload["config"])
    started = time.perf_counter()
    try:
        summary = Trainer(
            config, run_dir=payload["run_dir"], show_window=False
        ).train(verbose=False)
        summary["status"] = "ok"
    except Exception as error:  # un trial raté ne doit pas tuer la campagne
        summary = {
            "run_id": config.run_id,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "config": config.to_dict(),
        }
    summary["trial_id"] = payload["trial_id"]
    summary["group"] = payload["group"]
    summary["overrides"] = payload["overrides"]
    summary["seconds"] = round(time.perf_counter() - started, 2)
    return summary


def ranking_key(summary):
    """Classement du cadrage §36 : moyenne, p10, médiane, puis variance faible.

    Volontairement explicable : quatre critères lisibles dans cet ordre, pas
    une note composite opaque qu'on ne saurait pas justifier en soutenance.
    """
    if not eligible(summary):
        return (-1e9, -1e9, -1e9, -1e9)
    return (
        summary["best_eval_mean_score"],
        summary.get("best_eval_p10_score") or 0.0,
        summary.get("best_eval_median_score") or 0.0,
        -(summary.get("best_eval_std_score") or 0.0),
    )


def eligible(summary):
    """Aucun résultat mural ou sans provenance ne participe au classement."""
    return (summary.get("ruleset") == RULESET
            and summary.get("status") == "ok"
            and summary.get("best_eval_mean_score") is not None)


def aggregate_groups(summaries):
    """Agrège les seeds d'une même configuration.

    Une configuration se juge sur sa moyenne à travers les seeds, pas sur sa
    meilleure seed : sinon on sélectionne de la chance, pas une méthode.
    """
    groups = {}
    for summary in summaries:
        if not eligible(summary):
            continue
        groups.setdefault(summary["group"], []).append(summary)

    rows = []
    for group, members in groups.items():
        scores = [
            member["best_eval_mean_score"]
            for member in members
            if member.get("best_eval_mean_score") is not None
        ]
        if not scores:
            continue
        rows.append(
            {
                "group": group,
                "seeds": len(members),
                "overrides": members[0]["overrides"],
                "mean_of_eval_means": statistics.fmean(scores),
                "worst_seed": min(scores),
                "best_seed": max(scores),
                "spread": max(scores) - min(scores),
                "mean_p10": statistics.fmean(
                    member.get("best_eval_p10_score") or 0.0 for member in members
                ),
                "mean_record": statistics.fmean(
                    member.get("best_eval_record") or 0 for member in members
                ),
                "mean_seconds": statistics.fmean(member["seconds"] for member in members),
            }
        )
    rows.sort(key=lambda row: (row["mean_of_eval_means"], row["worst_seed"]), reverse=True)
    return rows


def write_reports(search_dir, summaries, groups, elapsed, space):
    """Écrit search_summary.csv et search_report.md."""
    csv_path = os.path.join(search_dir, "search_summary.csv")
    columns = [
        "trial_id", "group", "status", "ruleset", "eligible", "algorithm", "seed",
        "best_eval_mean_score", "best_eval_median_score", "best_eval_p10_score",
        "best_eval_p90_score", "best_eval_std_score", "best_eval_record",
        "best_eval_win_rate", "best_eval_truncation_rate", "record_train",
        "episodes", "seconds",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for summary in sorted(summaries, key=ranking_key, reverse=True):
            row = dict(summary)
            row["eligible"] = eligible(summary)
            row["seed"] = summary.get("config", {}).get("seed")
            writer.writerow(row)

    ranked = sorted([s for s in summaries if eligible(s)], key=ranking_key, reverse=True)
    # Ne pas faire confiance à une agrégation fournie depuis un ancien run.
    groups = aggregate_groups(summaries)
    lines = [
        f"# Rapport de recherche — {space.get('name', 'sans nom')}",
        "",
        f"- Trials : {len(summaries)} ({sum(s['status'] == 'ok' for s in summaries)} réussis)",
        f"- Temps total : {elapsed / 60:.1f} min",
        f"- Règles : {RULESET}; {len(summaries) - len(ranked)} trials exclus du classement",
        f"- Grille : `{json.dumps(space.get('grid', {}), ensure_ascii=False)}`",
        f"- Seeds d'entraînement : {space.get('seeds')}",
        f"- Épisodes par trial : {space.get('base', {}).get('episodes')}",
        "",
        "Le classement suit le score moyen d'évaluation à epsilon nul, puis le",
        "percentile 10, puis la médiane, puis la variance. Le record isolé est",
        "affiché mais ne décide de rien.",
        "",
        "## Configurations, moyennées sur les seeds",
        "",
        "| Configuration | Seeds | Moyenne | Pire seed | Meilleure seed | Écart | p10 moyen |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in groups:
        lines.append(
            f"| `{row['group']}` | {row['seeds']} | **{row['mean_of_eval_means']:.2f}** "
            f"| {row['worst_seed']:.2f} | {row['best_seed']:.2f} "
            f"| {row['spread']:.2f} | {row['mean_p10']:.2f} |"
        )

    lines += ["", "## Top 10 des trials individuels", "",
              "| Trial | Moyenne éval | Médiane | p10 | Record | Troncatures | Temps |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for summary in ranked[:10]:
        if summary["status"] != "ok":
            continue
        lines.append(
            f"| `{summary['trial_id']}` | **{summary['best_eval_mean_score']:.2f}** "
            f"| {summary['best_eval_median_score']:.1f} "
            f"| {summary['best_eval_p10_score']:.1f} "
            f"| {summary['best_eval_record']} "
            f"| {summary.get('best_eval_truncation_rate', 0):.0%} "
            f"| {summary['seconds']:.0f} s |"
        )

    failed = [s for s in summaries if s["status"] != "ok"]
    if failed:
        lines += ["", "## Trials en échec", ""]
        lines += [f"- `{s['trial_id']}` : {s['error']}" for s in failed]

    if groups:
        best = groups[0]
        lines += [
            "",
            "## Décision pour le round suivant",
            "",
            f"Meilleure configuration : `{best['group']}` "
            f"({json.dumps(best['overrides'], ensure_ascii=False)}), "
            f"score moyen d'évaluation {best['mean_of_eval_means']:.2f} "
            f"sur {best['seeds']} seed(s), pire seed {best['worst_seed']:.2f}.",
        ]

    report_path = os.path.join(search_dir, "search_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return csv_path, report_path


def run_search(space, output_dir=None, workers=None):
    """Exécute une campagne complète et écrit ses rapports."""
    output_dir = output_dir or space.get("output_dir", "runs")
    search_dir = os.path.join(output_dir, space.get("name", "search"))
    if os.path.isdir(search_dir) and os.listdir(search_dir):
        raise ValueError("campagne existante (potentiellement legacy) : choisir un nouveau nom")
    os.makedirs(search_dir, exist_ok=True)

    trials = build_trials(space)
    workers = workers or space.get("workers") or max(1, (os.cpu_count() or 2) - 1)

    payloads = [
        {
            "trial_id": trial["trial_id"],
            "group": trial["group"],
            "overrides": trial["overrides"],
            "config": trial["config"].to_dict(),
            "run_dir": os.path.join(search_dir, trial["trial_id"]),
        }
        for trial in trials
    ]

    with open(os.path.join(search_dir, "search_space.json"), "w", encoding="utf-8") as h:
        json.dump(
            {"space": space, "environment": environment_info()},
            h, indent=2, ensure_ascii=False,
        )

    print(f"{len(payloads)} trials, {workers} processus en parallèle", flush=True)
    started = time.perf_counter()
    summaries = []

    if workers == 1:
        for payload in payloads:
            summaries.append(run_trial(payload))
            report_progress(summaries[-1], len(summaries), len(payloads), started)
    else:
        # `spawn` et non `fork` : un contexte CUDA déjà initialisé dans le
        # processus parent ne survit pas à un fork, et chaque trial échouait
        # sur « CUDA error: initialization error », même en device cpu.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for summary in pool.map(run_trial, payloads):
                summaries.append(summary)
                report_progress(summaries[-1], len(summaries), len(payloads), started)

    elapsed = time.perf_counter() - started
    groups = aggregate_groups([s for s in summaries if s["status"] == "ok"])
    csv_path, report_path = write_reports(search_dir, summaries, groups, elapsed, space)

    print(f"\nTerminé en {elapsed / 60:.1f} min")
    print(f"  {csv_path}\n  {report_path}")
    if groups:
        print(
            f"Meilleure configuration : {groups[0]['group']} "
            f"-> {groups[0]['mean_of_eval_means']:.2f} de moyenne d'évaluation"
        )
    return {"summaries": summaries, "groups": groups, "search_dir": search_dir}


def report_progress(summary, done, total, started):
    elapsed = time.perf_counter() - started
    if summary["status"] == "ok":
        detail = f"moyenne éval {summary['best_eval_mean_score']:.2f}"
    else:
        detail = f"ÉCHEC {summary['error']}"
    print(
        f"[{done:3d}/{total}] {summary['trial_id']:32s} {detail}  "
        f"({elapsed / 60:.1f} min écoulées)",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recherche d'hyperparamètres")
    parser.add_argument("space", help="fichier JSON décrivant l'espace de recherche")
    parser.add_argument("--output-dir")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)

    with open(args.space, encoding="utf-8") as handle:
        space = json.load(handle)
    return run_search(space, output_dir=args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
