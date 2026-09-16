"""snake-ia.py - Point d'entrée CLI du projet Snake RL (D3QN + PER).

Nom avec tiret : ce fichier n'est jamais importé, seulement exécuté.
Fournit 4 sous-commandes (via argparse) : train, play, summary, plot.

Exemples d'usage :
  python snake-ia.py train --episodes 600 --seed 42
  python snake-ia.py play --model results/best_model.pth --games 5
  python snake-ia.py summary --csv results/training_log.csv
  python snake-ia.py plot --csv results/training_log.csv --out results/training_plot.png
  python snake-ia.py --help
"""

import argparse
import os
import sys

# --- Rendre le dossier du script importable, quel que soit le cwd d'appel ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from agent import train, play, TRAINING_FPS  # noqa: E402


def _default_results_dir():
    """Retourne le chemin par défaut du dossier de résultats.

    Returns:
        str: <dossier du script>/results
    """
    return os.path.join(SCRIPT_DIR, "results")


def cmd_train(args):
    """Handler de la sous-commande 'train' : entraîne l'agent D3QN.

    Args:
        args: Arguments argparse (episodes, render, results_dir, seed).
    """
    train(
        n_episodes=args.episodes,
        render=args.render,
        results_dir=args.results_dir,
        seed=args.seed,
        # Sans --fps : clock acceleree TRAINING_FPS (entrainement seulement).
        fps=TRAINING_FPS if args.fps is None else args.fps,
    )


def cmd_play(args):
    """Handler de la sous-commande 'play' : joue des parties en greedy avec un modèle.

    Args:
        args: Arguments argparse (model, games, no_render).
    """
    play(
        model_path=args.model,
        n_games=args.games,
        render=not args.no_render,
    )


def cmd_summary(args):
    """Handler de la sous-commande 'summary' : affiche le résumé d'entraînement (helper.py).

    Args:
        args: Arguments argparse (csv).
    """
    from helper import summarize
    summarize(args.csv)


def cmd_plot(args):
    """Handler de la sous-commande 'plot' : génère le graphe d'entraînement (helper.py).

    Args:
        args: Arguments argparse (csv, out).
    """
    from helper import plot_training
    plot_training(args.csv, args.out)


def build_parser():
    """Construit le parser argparse avec 4 sous-commandes (train, play, summary, plot).

    Returns:
        argparse.ArgumentParser: Parser configuré avec tous les arguments et handlers.
    """
    parser = argparse.ArgumentParser(
        prog="snake-ia.py",
        description="Snake RL - agent D3QN (Dueling Double DQN + PER).",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_train = subparsers.add_parser("train", help="Entraine l'agent D3QN.")
    p_train.add_argument("--episodes", type=int, default=600, help="Nombre d'episodes (defaut: 600).")
    p_train.add_argument("--render", action="store_true", help="Affiche la partie pendant l'entrainement.")
    p_train.add_argument("--seed", type=int, default=None, help="Graine aleatoire pour la reproductibilite.")
    p_train.add_argument(
        "--fps", type=int, default=None,
        help="Cadence d'affichage avec --render (defaut: TRAINING_FPS=60, clock acceleree ; 0 = illimite). "
             "Le mode play reste toujours a la clock de base GAME_SPEED.",
    )
    p_train.add_argument(
        "--results-dir", type=str, default=None,
        help="Dossier de sortie (defaut: <dossier du script>/results).",
    )
    p_train.set_defaults(func=cmd_train)

    p_play = subparsers.add_parser("play", help="Joue des parties en greedy avec un modele entraine.")
    p_play.add_argument(
        "--model", type=str, default=os.path.join(_default_results_dir(), "best_model.pth"),
        help="Chemin du modele a charger (defaut: results/best_model.pth).",
    )
    p_play.add_argument("--games", type=int, default=5, help="Nombre de parties a jouer (defaut: 5).")
    p_play.add_argument("--no-render", action="store_true", help="Desactive l'affichage graphique.")
    p_play.set_defaults(func=cmd_play)

    p_summary = subparsers.add_parser("summary", help="Affiche le resume d'un entrainement (via helper.py).")
    p_summary.add_argument(
        "--csv", type=str, default=os.path.join(_default_results_dir(), "training_log.csv"),
        help="Chemin du CSV de log (defaut: results/training_log.csv).",
    )
    p_summary.set_defaults(func=cmd_summary)

    p_plot = subparsers.add_parser("plot", help="Genere le graphe d'entrainement (via helper.py).")
    p_plot.add_argument(
        "--csv", type=str, default=os.path.join(_default_results_dir(), "training_log.csv"),
        help="Chemin du CSV de log (defaut: results/training_log.csv).",
    )
    p_plot.add_argument(
        "--out", type=str, default=os.path.join(_default_results_dir(), "training_plot.png"),
        help="Chemin de sortie du graphe (defaut: results/training_plot.png).",
    )
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main():
    """Point d'entrée principal : parse les arguments et exécute la sous-commande.

    Affiche l'aide si aucune sous-commande n'est fournie.
    """
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
