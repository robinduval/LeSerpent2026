"""Module de suivi et de visualisation pour l'entraînement Snake RL (D3QN + PER).

Fournit :
- TrainingLogger : écrit un CSV (une ligne par partie).
- load_log : recharge le CSV en dict de np.ndarray.
- moving_average : moyenne mobile tronquée au début (pas de NaN).
- plot_training : trace score et ratio score/temps en fonction du nombre de parties.
- summarize : calcule et imprime les statistiques clés (record, victoires, meilleur ratio).

Interface figée (colonnes CSV, ordre exact) :
    episode,score,steps,time_s,ratio,total_reward,epsilon,won,truncated,wall_time_s
"""

from __future__ import annotations

import csv
import os
import tempfile

import numpy as np

# Colonnes du CSV, dans l'ordre exact attendu par les autres modules du projet.
FIELDNAMES = [
    "episode",
    "score",
    "steps",
    "time_s",
    "ratio",
    "total_reward",
    "epsilon",
    "won",
    "truncated",
    "wall_time_s",
]

# Colonnes entières / booléennes (écrites comme int, chargées avec le bon dtype).
_INT_COLS = {"episode", "score", "steps"}
_BOOL_COLS = {"won", "truncated"}


def _fmt(value) -> str:
    """Formate une valeur pour le CSV : entiers sans décimales, floats à 4 max sans zéros inutiles.

    Args:
        value: Valeur à formater (int, float, bool, etc.).

    Returns:
        str: Chaîne formatée pour écriture CSV.
    """
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    # Float : jusqu'à 4 décimales, sans zéros inutiles.
    formatted = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


class TrainingLogger:
    """Écrit un CSV d'entraînement (une ligne par partie, flush immédiat).

    Ouvre le CSV en mode 'w' (écrase tout fichier existant), écrit l'en-tête FIELDNAMES,
    puis chaque log() ajoute une ligne et flush immédiatement. Context manager supporté.
    """

    def __init__(self, csv_path):
        """Initialise le logger CSV.

        Crée le dossier parent s'il n'existe pas, ouvre csv_path en 'w' (écrase),
        écrit l'en-tête FIELDNAMES et flush.

        Args:
            csv_path: Chemin du fichier CSV de destination.
        """
        self.csv_path = csv_path
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # "w" : écrase un éventuel fichier existant.
        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(FIELDNAMES)
        self._file.flush()

    def log(self, episode, score, steps, time_s, ratio, total_reward, epsilon, won, truncated, wall_time_s):
        """Écrit une ligne (une partie entraînée) et flush immédiatement sur disque.

        Args:
            episode: Numéro d'épisode (séquentiel).
            score: Score affiché (nombre de pommes mangées).
            steps: Nombre de pas de jeu.
            time_s: Temps nominal du jeu à la vitesse GAME_SPEED (steps / GAME_SPEED).
            ratio: Ratio score/temps (score / time_s).
            total_reward: Récompense RL totale accumulée.
            epsilon: Epsilon courant (exploration).
            won: Booléen, victoire (grille remplie).
            truncated: Booléen, troncature (timeout).
            wall_time_s: Temps réel d'exécution de l'épisode.
        """
        row = [
            _fmt(episode),
            _fmt(score),
            _fmt(steps),
            _fmt(time_s),
            _fmt(ratio),
            _fmt(total_reward),
            _fmt(epsilon),
            _fmt(bool(won)),
            _fmt(bool(truncated)),
            _fmt(wall_time_s),
        ]
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """Ferme le fichier CSV."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        """Context manager : retourne self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager : ferme le fichier."""
        self.close()
        return False


def load_log(csv_path) -> dict:
    """Charge le CSV d'entraînement en dict {nom_colonne: np.ndarray}.

    Adapte les dtypes : int pour episode/score/steps, bool pour won/truncated, float sinon.
    Gère les CSVs vides (en-tête seul) avec des tableaux vides.

    Args:
        csv_path: Chemin du fichier CSV.

    Returns:
        dict: {nom_colonne (str): np.ndarray}.
    """
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    if header is None:
        header = list(FIELDNAMES)

    columns = {name: [] for name in header}
    for row in rows:
        for name, value in zip(header, row):
            columns[name].append(value)

    result = {}
    for name in header:
        raw = columns[name]
        if name in _BOOL_COLS:
            result[name] = np.array([bool(int(float(v))) for v in raw], dtype=bool)
        elif name in _INT_COLS:
            result[name] = np.array([int(float(v)) for v in raw], dtype=int)
        else:
            result[name] = np.array([float(v) for v in raw], dtype=float)
    return result


def moving_average(values, window: int = 50) -> np.ndarray:
    """Moyenne mobile tronquée (pas de NaN au début).

    Calcule la moyenne en fenêtre glissante, mais truncature au début : les premiers
    éléments ont une fenêtre plus petite (1 élément, 2 éléments, ..., window éléments).

    Args:
        values: Tableau de valeurs.
        window: Taille de la fenêtre mobile (défaut 50).

    Returns:
        np.ndarray: Moyennes mobiles, même longueur que values, sans NaN.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    window = max(1, int(window))
    cumsum = np.cumsum(values, dtype=float)
    result = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        total = cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0.0)
        count = i - lo + 1
        result[i] = total / count
    return result


def plot_training(csv_path, png_path, window: int = 50) -> str:
    """Génère et sauvegarde un graphe d'entraînement (score et ratio score/temps).

    Produit un PNG à 2 sous-graphes : score par partie + moyenne mobile, et ratio score/temps.
    Marque les victoires (won=True) par des étoiles oranges. Annote le record de score.
    Gère les CSVs vides (affiche "Aucune donnée").

    Args:
        csv_path: Chemin du CSV d'entraînement.
        png_path: Chemin de sortie du PNG (crée le dossier parent si nécessaire).
        window: Taille de la fenêtre moyenne mobile (défaut 50).

    Returns:
        str: png_path (invariant d'interface).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_log(csv_path)
    n = len(data.get("episode", np.array([])))

    parent = os.path.dirname(png_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Palette sobre : brut en gris-bleu clair, moyenne mobile en bleu foncé, accent orange (record / victoires).
    COLOR_RAW = "#9fb3c8"
    COLOR_MA = "#1f4e79"
    COLOR_ACCENT = "#d9730d"

    fig, (ax_score, ax_ratio) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Snake RL — D3QN + PER (gamma=0.95)")

    if n == 0:
        for ax, title in ((ax_score, "Score par partie"), (ax_ratio, "Ratio score/temps (score/s)")):
            ax.set_title(title)
            ax.text(0.5, 0.5, "Aucune donnée", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.3)
        ax_ratio.set_xlabel("Partie n°")
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        return png_path

    episodes = data["episode"]
    scores = data["score"].astype(float)
    ratios = data["ratio"].astype(float)
    won = data.get("won", np.zeros(n, dtype=bool))

    ma_score = moving_average(scores, window=window)
    ma_ratio = moving_average(ratios, window=window)

    # --- Sous-graphe du haut : score ---
    ax_score.plot(episodes, scores, color=COLOR_RAW, linewidth=0.8, alpha=0.5, marker="o", markersize=2, label="Score")
    ax_score.plot(episodes, ma_score, color=COLOR_MA, linewidth=2.0, label=f"Moyenne mobile ({window})")
    best_score = float(np.max(scores))
    ax_score.axhline(best_score, color=COLOR_ACCENT, linestyle="--", linewidth=1.2)
    best_idx = int(np.argmax(scores))
    ax_score.annotate(
        f"record = {int(best_score)}",
        xy=(episodes[best_idx], best_score),
        xytext=(0, 6),
        textcoords="offset points",
        color=COLOR_ACCENT,
        fontsize=9,
    )
    ax_score.set_ylabel("Score")
    ax_score.set_title("Score par partie")
    ax_score.grid(alpha=0.3)
    ax_score.legend(loc="upper left")

    # --- Sous-graphe du bas : ratio score/temps ---
    ax_ratio.plot(episodes, ratios, color=COLOR_RAW, linewidth=0.8, alpha=0.5, marker="o", markersize=2, label="Ratio score/s")
    ax_ratio.plot(episodes, ma_ratio, color=COLOR_MA, linewidth=2.0, label=f"Moyenne mobile ({window})")
    if np.any(won):
        ax_ratio.scatter(
            episodes[won],
            ratios[won],
            marker="*",
            s=90,
            color=COLOR_ACCENT,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label="Partie gagnée",
        )
    ax_ratio.set_ylabel("Score / temps (s⁻¹)")
    ax_ratio.set_xlabel("Partie n°")
    ax_ratio.set_title("Ratio score/temps par partie")
    ax_ratio.grid(alpha=0.3)
    ax_ratio.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path


def summarize(csv_path, print_report: bool = True) -> dict:
    """Calcule et imprime les statistiques clés d'entraînement en français.

    Extrait : record (meilleur score et épisode), victoires (grille remplie),
    meilleur ratio score/temps, moyennes (50 derniers épisodes), troncatures.

    Args:
        csv_path: Chemin du CSV d'entraînement.
        print_report: Si True, affiche le rapport en français (défaut True).

    Returns:
        dict: Statistiques avec keys : n_episodes, best_score, best_score_episode,
              best_score_time_s, n_won, won_episodes, won_times_s, fastest_win_time_s,
              best_ratio, best_ratio_episode, best_ratio_score, best_ratio_time_s,
              mean_score_last_50, mean_ratio_last_50, n_truncated.
    """
    data = load_log(csv_path)
    n = len(data.get("episode", np.array([])))

    if n == 0:
        report = {
            "n_episodes": 0,
            "best_score": 0,
            "best_score_episode": None,
            "best_score_time_s": None,
            "n_won": 0,
            "won_episodes": [],
            "won_times_s": [],
            "fastest_win_time_s": None,
            "best_ratio": 0.0,
            "best_ratio_episode": None,
            "best_ratio_score": None,
            "best_ratio_time_s": None,
            "mean_score_last_50": 0.0,
            "mean_ratio_last_50": 0.0,
            "n_truncated": 0,
        }
        if print_report:
            print("Aucune donnée disponible dans le journal d'entraînement.")
        return report

    episodes = data["episode"]
    scores = data["score"].astype(float)
    time_s = data["time_s"].astype(float)
    ratios = data["ratio"].astype(float)
    won = data["won"]
    truncated = data["truncated"]

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_score_episode = int(episodes[best_idx])
    best_score_time_s = float(time_s[best_idx])

    won_mask = won.astype(bool)
    won_episodes = [int(e) for e in episodes[won_mask]]
    won_times_s = [float(t) for t in time_s[won_mask]]
    n_won = int(np.sum(won_mask))
    fastest_win_time_s = float(np.min(time_s[won_mask])) if n_won > 0 else None

    best_ratio_idx = int(np.argmax(ratios))
    best_ratio = float(ratios[best_ratio_idx])
    best_ratio_episode = int(episodes[best_ratio_idx])
    best_ratio_score = float(scores[best_ratio_idx])
    best_ratio_time_s = float(time_s[best_ratio_idx])

    last = min(50, n)
    mean_score_last_50 = float(np.mean(scores[-last:]))
    mean_ratio_last_50 = float(np.mean(ratios[-last:]))
    n_truncated = int(np.sum(truncated.astype(bool)))

    report = {
        "n_episodes": n,
        "best_score": best_score,
        "best_score_episode": best_score_episode,
        "best_score_time_s": best_score_time_s,
        "n_won": n_won,
        "won_episodes": won_episodes,
        "won_times_s": won_times_s,
        "fastest_win_time_s": fastest_win_time_s,
        "best_ratio": best_ratio,
        "best_ratio_episode": best_ratio_episode,
        "best_ratio_score": best_ratio_score,
        "best_ratio_time_s": best_ratio_time_s,
        "mean_score_last_50": mean_score_last_50,
        "mean_ratio_last_50": mean_ratio_last_50,
        "n_truncated": n_truncated,
    }

    if print_report:
        print("=== Rapport d'entraînement Snake RL ===")
        print(f"Parties jouées : {n}")
        print(f"Meilleur score : {best_score:g} (partie n°{best_score_episode}, en {best_score_time_s:.2f} s)")
        if n_won > 0:
            print(f"Parties gagnées (grille remplie) : {n_won} — épisodes {won_episodes}")
            print(f"  Temps de victoire : {[round(t, 2) for t in won_times_s]} s")
            print(f"  Victoire la plus rapide : {fastest_win_time_s:.2f} s")
        else:
            print("Parties gagnées (grille remplie) : aucune")
        print(
            f"Meilleur ratio score/temps : {best_ratio:.4f} score/s "
            f"(partie n°{best_ratio_episode}, score={best_ratio_score:g}, temps={best_ratio_time_s:.2f} s)"
        )
        print(f"Score moyen (50 dernières parties) : {mean_score_last_50:.2f}")
        print(f"Ratio moyen (50 dernières parties) : {mean_ratio_last_50:.4f}")
        print(f"Parties tronquées (timeout) : {n_truncated}")

    return report


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "training.csv")
        png_path = os.path.join(tmp_dir, "training.png")
        empty_csv_path = os.path.join(tmp_dir, "empty.csv")

        n_games = 300
        game_speed = 5  # FPS nominal du jeu

        # --- Génération d'un CSV synthétique : score croissant bruité, une victoire artificielle ---
        with TrainingLogger(csv_path) as logger:
            for ep in range(1, n_games + 1):
                trend = min(60.0, ep / n_games * 60.0)
                score = max(0, int(round(trend + rng.normal(0, 4))))
                steps = 50 + score * 20 + int(rng.integers(0, 30))
                time_s = steps / game_speed
                ratio = score / time_s if time_s > 0 else 0.0
                total_reward = score * 10.0 - steps * 0.01
                epsilon = max(0.05, 1.0 - ep / n_games)
                won = False
                truncated = bool(rng.random() < 0.05)
                wall_time_s = steps * 0.01

                logger.log(ep, score, steps, time_s, ratio, total_reward, epsilon, won, truncated, wall_time_s)

            # Partie gagnée artificielle : grille remplie, score max connu, plutôt rapide.
            win_episode = n_games + 1
            win_score = 224  # 15x15 - 1 (grille GRID_SIZE=15 remplie)
            win_steps = 1200
            win_time_s = win_steps / game_speed
            win_ratio = win_score / win_time_s
            logger.log(win_episode, win_score, win_steps, win_time_s, win_ratio, win_score * 10.0, 0.05, True, False, win_steps * 0.01)

        total_games = n_games + 1

        # --- Vérification de l'en-tête exact ---
        with open(csv_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
        expected_header = ",".join(FIELDNAMES)
        assert header_line == expected_header, f"En-tête inattendu : {header_line!r}"
        print("OK: en-tête CSV exact.")

        # --- Vérification de load_log ---
        data = load_log(csv_path)
        assert len(data["episode"]) == total_games, "Nombre de lignes chargées incorrect."
        assert data["episode"].dtype.kind in ("i", "u"), "episode doit être un entier."
        assert data["score"].dtype.kind in ("i", "u"), "score doit être un entier."
        assert data["won"].dtype == bool, "won doit être booléen."
        assert data["truncated"].dtype == bool, "truncated doit être booléen."
        assert data["ratio"].dtype.kind == "f", "ratio doit être un float."
        print(f"OK: load_log — {total_games} lignes, dtypes corrects.")

        # --- Vérification de moving_average ---
        ma = moving_average(data["score"], window=50)
        assert len(ma) == len(data["score"]), "moving_average doit préserver la longueur."
        assert not np.any(np.isnan(ma)), "moving_average ne doit jamais produire de NaN."
        print("OK: moving_average — longueur préservée, pas de NaN.")

        # --- Vérification de plot_training ---
        out_png = plot_training(csv_path, png_path, window=50)
        assert out_png == png_path
        assert os.path.exists(png_path) and os.path.getsize(png_path) > 0, "Le PNG doit être non vide."
        print(f"OK: plot_training — PNG généré ({os.path.getsize(png_path)} octets).")

        # --- Vérification de summarize ---
        report = summarize(csv_path, print_report=True)
        # Tolérance à 1e-3 : la colonne "ratio" du CSV est arrondie à 4 décimales à l'écriture.
        expected_best_ratio = float(np.max(data["score"].astype(float) / data["time_s"].astype(float)))
        assert report["best_score"] == float(win_score) or report["best_score"] == max(
            float(win_score), float(np.max(data["score"][:-1]))
        ), "Le record de score est incorrect."
        assert report["n_won"] == 1, "Il doit y avoir exactement une partie gagnée."
        assert win_episode in report["won_episodes"], "L'épisode gagné doit apparaître dans won_episodes."
        assert abs(report["best_ratio"] - expected_best_ratio) < 1e-3, "Le meilleur ratio est incohérent."
        print("OK: summarize — record, victoire et meilleur ratio cohérents.")

        # --- Cas CSV vide (en-tête seule) ---
        with TrainingLogger(empty_csv_path) as empty_logger:
            pass
        empty_data = load_log(empty_csv_path)
        assert len(empty_data["episode"]) == 0, "Le CSV vide doit donner des tableaux vides."
        empty_png = os.path.join(tmp_dir, "empty.png")
        out_empty_png = plot_training(empty_csv_path, empty_png, window=50)
        assert os.path.exists(out_empty_png) and os.path.getsize(out_empty_png) > 0, "Le PNG (cas vide) doit être non vide."
        empty_report = summarize(empty_csv_path, print_report=True)
        assert empty_report["n_episodes"] == 0
        assert empty_report["best_score"] == 0
        assert empty_report["n_won"] == 0
        print("OK: cas CSV vide géré sans erreur (load_log, plot_training, summarize).")

        print("\nTous les tests sont passés avec succès.")
