"""Cherche la victoire la plus rapide en jouant des milliers de parties en parallèle.

Chaque partie est identifiée par une graine : une victoire trouvée ici se rejoue à l'identique
avec « python Fraise/algo/snake-algo.py rapide 20 --algo <algo> --graine <graine> ».

Usage : python Fraise/algo/chercher.py --algo audacieux --parties 2000 [--controle espace] [--risque 0.5]
"""

import argparse
import csv
import importlib.util
import multiprocessing
import os
import time

DOSSIER = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("snake_algo", os.path.join(DOSSIER, "snake-algo.py"))
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

_config = {}


def _init(config):
    global _config, _game, _algo
    _config = config
    _game = sa.SnakeGameAlgo(affichage=False)
    _algo = _construire()


def _construire():
    nom = _config["algo"]
    if nom == "audacieux":
        return sa.Audacieux(controle=_config["controle"], risque=_config["risque"], survie=_config["survie"])
    if nom == "glouton":
        return sa.Glouton(seuil_longueur=_config["seuil"], patience=_config["patience"],
                          survie=_config["survie"])
    return sa.ALGORITHMES[nom]()


def _jouer(graine):
    score, temps, fin, coups = sa.jouer_partie(_game, _algo, graine=graine)
    return graine, score, round(temps, 1), fin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recherche de la victoire la plus rapide")
    parser.add_argument("--algo", default="audacieux")
    parser.add_argument("--parties", type=int, default=1000)
    parser.add_argument("--depart", type=int, default=0, help="première graine")
    parser.add_argument("--controle", default="espace", choices=("aucun", "espace", "queue"))
    parser.add_argument("--risque", type=float, default=0.5)
    parser.add_argument("--survie", default="mur", choices=("espace", "mur", "loin_queue"))
    parser.add_argument("--seuil", type=float, default=0.6, help="glouton : seuil de longueur")
    parser.add_argument("--patience", type=int, default=0, help="glouton : coups sans pomme avant de retenter")
    parser.add_argument("--cible", type=float, default=None,
                        help="temps de jeu visé : on s'arrête dès qu'une victoire fait mieux")
    parser.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    config = {"algo": args.algo, "controle": args.controle, "risque": args.risque, "survie": args.survie,
              "seuil": args.seuil, "patience": args.patience}
    graines = range(args.depart, args.depart + args.parties)
    debut = time.time()
    resultats = []
    with multiprocessing.Pool(args.procs, initializer=_init, initargs=(config,)) as pool:
        for resultat in pool.imap_unordered(_jouer, graines, chunksize=4):
            resultats.append(resultat)
            if resultat[3] == "victoire":
                print(f"  victoire : score {resultat[1]} en {resultat[2]:.1f}s  ->  --graine {resultat[0]}"
                      + ("   *** CIBLE ATTEINTE ***" if args.cible and resultat[2] <= args.cible else ""), flush=True)
                if args.cible and resultat[2] <= args.cible:
                    pool.terminate()
                    break
            if len(resultats) % 500 == 0:
                print(f"  ... {len(resultats)} parties jouées", flush=True)
    duree = time.time() - debut

    victoires = sorted([r for r in resultats if r[3] == "victoire"], key=lambda r: r[2])
    scores = [r[1] for r in resultats]
    print(f"{len(resultats)} parties en {duree:.0f}s ({args.procs} processus) | algo {args.algo}"
          + (f" contrôle {args.controle} survie {args.survie}" if args.algo == "audacieux" else "")
          + (f" seuil {args.seuil} patience {args.patience} survie {args.survie}" if args.algo == "glouton" else ""))
    print(f"  victoires : {len(victoires)}/{len(resultats)} | score moyen {sum(scores)/len(scores):.1f} | record {max(scores)}")
    for graine, score, temps, _ in victoires[:5]:
        print(f"  VICTOIRE  score {score} en {temps:.1f}s de jeu  ->  --graine {graine}")

    chemin = os.path.join(DOSSIER, "recherche.csv")
    nouveau = not os.path.exists(chemin)
    with open(chemin, "a", newline="") as fichier:
        writer = csv.writer(fichier)
        if nouveau:
            writer.writerow(["algo", "controle", "survie", "graine", "score", "temps_s", "fin"])
        for graine, score, temps, fin in resultats:
            writer.writerow([args.algo, args.controle, args.survie, graine, score, temps, fin])
    print(f"  détails ajoutés à {chemin}")
