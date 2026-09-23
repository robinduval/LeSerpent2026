"""Trace l'évolution des algorithmes au cours d'une partie (groupe Fraise).

Produit deux fichiers dans ce dossier :
  - mesures.csv    : score et temps de chaque partie jouée
  - evolution.svg  : le graphique (SVG écrit à la main, aucune bibliothèque externe)

Usage :  python Fraise/algo/graphique.py [--parties 10] [--seed 1]
"""

import argparse
import csv
import importlib.util
import os
import random

DOSSIER = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("snake_algo", os.path.join(DOSSIER, "snake-algo.py"))
algo_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(algo_module)

COULEURS = {"dijkstra": "#1f77b4", "gbfs": "#d62728", "sur": "#ff7f0e", "hybride": "#2ca02c"}
LIBELLES = {"dijkstra": "Dijkstra (cours)", "gbfs": "GBFS (cours)",
            "sur": "Glouton sûr", "hybride": "Hybride (le nôtre)"}


def jouer_avec_trace(game, algo):
    """Joue une partie et retourne la courbe [(temps de jeu, score)] et la fin de partie."""
    game.reset()
    courbe = [(0.0, 0)]
    fini, score = False, 0
    while not fini:
        fini, nouveau = game.play_step(algo.choisir(game))
        if nouveau != score:
            score = nouveau
            courbe.append((game.coups / algo_module.serpent.GAME_SPEED, score))
        if game.coups_sans_pomme > algo_module.LIMITE_BOUCLE:
            return courbe, "boucle"
    courbe.append((game.coups / algo_module.serpent.GAME_SPEED, score))
    return courbe, ("victoire" if game.victoire else "mort")


def mesurer(nb_parties, seed):
    """Joue nb_parties par algorithme. Retourne les stats et la meilleure partie de chacun."""
    game = algo_module.SnakeGameAlgo(affichage=False)
    mesures, resume = [], {}
    for nom in COULEURS:
        algo = algo_module.ALGORITHMES[nom]()
        random.seed(seed)
        parties, meilleure = [], None
        for numero in range(1, nb_parties + 1):
            courbe, fin = jouer_avec_trace(game, algo)
            score, temps = courbe[-1][1], courbe[-1][0]
            parties.append((score, temps, fin))
            mesures.append({"algo": nom, "partie": numero, "score": score,
                            "temps_s": round(temps, 1), "fin": fin})
            # classement du cours : meilleur score, puis temps le plus court
            if meilleure is None or (score, -temps) > (meilleure[0][-1][1], -meilleure[0][-1][0]):
                meilleure = (courbe, fin)
            print(f"  {nom:9s} partie {numero:2d} : score {score:3d} en {temps:7.1f}s ({fin})", flush=True)
        scores = [p[0] for p in parties]
        resume[nom] = {
            "courbe": meilleure[0],
            "record": max(scores),
            "moyenne": sum(scores) / len(scores),
            "victoires": sum(p[2] == "victoire" for p in parties),
            "parties": len(parties),
            "temps_record": min(p[1] for p in parties if p[0] == max(scores)),
        }
    return mesures, resume


def svg(resume, chemin):
    """Écrit le graphique : courbes score/temps en haut, scores moyens en barres en bas."""
    L, H = 920, 660
    gx, gy, gw, gh = 70, 60, 800, 330          # cadre du graphique du haut
    bx, by, bw, bh = 220, 470, 600, 150        # cadre des barres du bas
    tmax = max(c["courbe"][-1][0] for c in resume.values()) * 1.05
    smax = max(c["record"] for c in resume.values()) * 1.08
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {L} {H}" width="{L}" height="{H}" '
           'font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{L}" height="{H}" fill="#ffffff"/>',
           f'<text x="{L/2}" y="30" text-anchor="middle" font-size="19" font-weight="bold">'
           'Snake algorithmique - groupe Fraise</text>',
           f'<text x="{L/2}" y="50" text-anchor="middle" font-size="13" fill="#555">'
           'Score au fil du temps de jeu (meilleure partie de chaque algorithme)</text>']

    def px(t):
        return gx + gw * t / tmax

    def py(s):
        return gy + gh - gh * s / smax

    # grille et axes
    for i in range(6):
        s = smax * i / 5
        y = py(s)
        out.append(f'<line x1="{gx}" y1="{y:.1f}" x2="{gx + gw}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        out.append(f'<text x="{gx - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#555">{s:.0f}</text>')
    for i in range(7):
        t = tmax * i / 6
        x = px(t)
        out.append(f'<line x1="{x:.1f}" y1="{gy}" x2="{x:.1f}" y2="{gy + gh}" stroke="#f0f0f0"/>')
        out.append(f'<text x="{x:.1f}" y="{gy + gh + 18}" text-anchor="middle" font-size="11" fill="#555">'
                   f'{t/60:.0f} min</text>')
    out.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" fill="none" stroke="#999"/>')
    out.append(f'<text x="{gx - 45}" y="{gy + gh/2}" font-size="12" fill="#333" '
               f'transform="rotate(-90 {gx - 45} {gy + gh/2})" text-anchor="middle">Score (pommes)</text>')
    out.append(f'<text x="{gx + gw/2}" y="{gy + gh + 40}" text-anchor="middle" font-size="12" fill="#333">'
               'Temps de jeu (clock d\'origine : 5 images/seconde)</text>')

    # courbes
    for i, (nom, infos) in enumerate(resume.items()):
        points = " ".join(f"{px(t):.1f},{py(s):.1f}" for t, s in infos["courbe"])
        out.append(f'<polyline points="{points}" fill="none" stroke="{COULEURS[nom]}" stroke-width="2.2"/>')
        tf, sf = infos["courbe"][-1]
        out.append(f'<circle cx="{px(tf):.1f}" cy="{py(sf):.1f}" r="4" fill="{COULEURS[nom]}"/>')
        ly = gy + 16 + i * 19
        out.append(f'<line x1="{gx + gw - 250}" y1="{ly - 4}" x2="{gx + gw - 225}" y2="{ly - 4}" '
                   f'stroke="{COULEURS[nom]}" stroke-width="2.2"/>')
        out.append(f'<text x="{gx + gw - 218}" y="{ly}" font-size="12" fill="#333">'
                   f'{LIBELLES[nom]} : {sf} en {tf/60:.0f} min</text>')

    # barres : score moyen
    parties = next(iter(resume.values()))["parties"]
    out.append(f'<text x="{L/2}" y="{by - 25}" text-anchor="middle" font-size="13" fill="#555">'
               f'Score moyen sur {parties} parties (225 cases : le maximum est une victoire)</text>')
    largeur = bh / len(resume) - 10
    for i, (nom, infos) in enumerate(resume.items()):
        y = by + i * (bh / len(resume))
        w = bw * infos["moyenne"] / smax
        out.append(f'<rect x="{bx}" y="{y:.1f}" width="{w:.1f}" height="{largeur:.1f}" fill="{COULEURS[nom]}" '
                   'opacity="0.85"/>')
        out.append(f'<text x="{bx - 8}" y="{y + largeur*0.7:.1f}" text-anchor="end" font-size="12" fill="#333">'
                   f'{LIBELLES[nom]}</text>')
        detail = f'{infos["moyenne"]:.1f}'
        if infos["victoires"]:
            detail += f'  ({infos["victoires"]}/{infos["parties"]} victoires)'
        out.append(f'<text x="{bx + w + 8:.1f}" y="{y + largeur*0.7:.1f}" font-size="12" fill="#333">{detail}</text>')

    out.append('</svg>')
    with open(chemin, "w") as fichier:
        fichier.write("\n".join(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphique de suivi des algorithmes (groupe Fraise)")
    parser.add_argument("--parties", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    mesures, resume = mesurer(args.parties, args.seed)
    chemin_csv = os.path.join(DOSSIER, "mesures.csv")
    with open(chemin_csv, "w", newline="") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=["algo", "partie", "score", "temps_s", "fin"])
        writer.writeheader()
        writer.writerows(mesures)
    chemin_svg = os.path.join(DOSSIER, "evolution.svg")
    svg(resume, chemin_svg)
    print(f"\nÉcrit : {chemin_csv} et {chemin_svg}")
