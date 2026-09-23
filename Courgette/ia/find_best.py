"""
find_best.py — Parcourt tous les CSV d'entraînement (générés par snake-ia.py
train, colonnes : episode,score,steps,time,score_per_time,...) et trouve la
meilleure partie (score/temps maximal) parmi celles où score > 10 (l'objectif
minimum du projet : une partie qui n'a pas atteint 10 points n'est pas
considérée comme "compétitive", même si son ratio brut est élevé).

Usage :
    python find_best.py                 # cherche dans results/ et results_compare/
    python find_best.py autre_dossier/   # cherche dans un ou plusieurs dossiers donnés
"""

import csv
import glob
import os
import sys


def find_training_csvs(directories):
    """Liste tous les CSV d'entraînement (exclut les eval_*.csv et summary.csv,
    qui n'ont pas les mêmes colonnes)."""
    paths = []
    for d in directories:
        for path in glob.glob(os.path.join(d, "*.csv")):
            name = os.path.basename(path)
            if name.startswith("eval_") or name == "summary.csv":
                continue
            paths.append(path)
    return sorted(paths)


def best_row_above_10(path):
    """Retourne la ligne (dict) au meilleur score/temps parmi celles où
    score > 10 dans ce CSV, ou None s'il n'y en a aucune."""
    best = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["score"])
                ratio = float(row["score_per_time"])
            except (KeyError, ValueError):
                continue
            if score <= 10:
                continue
            if best is None or ratio > float(best["score_per_time"]):
                best = row
    return best


def main():
    directories = sys.argv[1:] or ["results", "results_compare"]
    directories = [d for d in directories if os.path.isdir(d)]
    if not directories:
        print("Aucun dossier trouvé (results/, results_compare/ ?).")
        return

    csv_paths = find_training_csvs(directories)
    if not csv_paths:
        print("Aucun CSV d'entraînement trouvé.")
        return

    overall_best = None
    overall_best_path = None

    for path in csv_paths:
        row = best_row_above_10(path)
        if row is None:
            continue
        ratio = float(row["score_per_time"])
        print(f"{path}: meilleur score/temps avec score>10 -> "
              f"episode={row['episode']} score={row['score']} "
              f"temps={row['time']}s ratio={ratio:.4f}")
        if overall_best is None or ratio > float(overall_best["score_per_time"]):
            overall_best = row
            overall_best_path = path

    print()
    if overall_best is None:
        print("Aucune partie avec score > 10 trouvée dans ces CSV.")
    else:
        print("=== Meilleur score/temps global (score > 10) ===")
        print(f"Fichier : {overall_best_path}")
        print(f"Episode : {overall_best['episode']}")
        print(f"Score   : {overall_best['score']}")
        print(f"Temps   : {overall_best['time']}s")
        print(f"Ratio   : {float(overall_best['score_per_time']):.4f}")


if __name__ == "__main__":
    main()
