"""
snake-algo.py — Snake piloté par un algorithme déterministe, sans apprentissage.

Groupe ABRICOT — Projet STP 2026 — rendu algorithmique.

RÈGLES DU JEU — le fichier de base fait foi (serpent-algo.py, copié à côté,
seule l'injection de prompt en a été retirée). Il fait « % GRID_SIZE » dans
move() : la grille est un TORE, le serpent traverse les bords et seule
l'auto-morsure tue. On n'imite pas le jeu : on importe ses classes Snake et
Apple, et on rejoue sa boucle principale coup pour coup.

CONTRAINTES RESPECTÉES — GRID_SIZE = 15, GAME_SPEED = 5 et le scoring
(+1 par pomme) ne sont jamais modifiés.

POURQUOI LE TORE CHANGE TOUT — sur une grille à murs 15x15, aucun cycle
hamiltonien n'existe (225 cases, nombre impair, graphe biparti). Sur le tore,
il en existe un : 14 pas à droite, 1 pas en bas, 15 fois. Le suivre remplit
la grille à coup sûr, et la position de départ du serpent est déjà dessus.

QUATRE ALGORITHMES (--algo) :
  dijkstra   plus court chemin vers la pomme ; s'il n'y en a pas, le coup qui
             laisse le plus d'espace. Optimal par pomme, myope sur la partie.
  glouton    Greedy Best-First, heuristique Manhattan torique. Rapide, se piège.
  cycle      suit le cycle hamiltonien. Victoire garantie, mais ~12 500 pas.
  tapsell    (défaut) le cycle, plus les raccourcis de John Tapsell : sauter en
             avant dans l'ordre du cycle, sans jamais dépasser la pomme ni
             rattraper la queue. Le corps reste rangé dans l'ordre du cycle,
             donc le serpent peut toujours reprendre le cycle : victoire
             garantie, en ~6 800 pas.

Usage :
    python snake-algo.py play                     # tapsell, à la vitesse du jeu
    python snake-algo.py play --algo cycle --speed 60
    python snake-algo.py bench --games 30         # compare les quatre algos
"""

import argparse
import heapq
import importlib.util
import os
import random
import statistics
import sys
import time
from collections import deque

ICI = os.path.dirname(os.path.abspath(__file__))


def _assurer_environnement():
    """Relance le script avec le Python du venv du groupe si pygame manque."""
    try:
        import pygame  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    venv = os.path.join(os.path.dirname(ICI), ".venv", "bin", "python")
    if os.path.exists(venv) and os.path.realpath(sys.executable) != os.path.realpath(venv):
        os.execv(venv, [venv, os.path.abspath(__file__)] + sys.argv[1:])
    raise SystemExit("pygame est introuvable, et Abricot/.venv/ aussi.")


_assurer_environnement()
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Le jeu de base s'appelle « serpent-algo.py » : le tiret interdit un import
# ordinaire, on passe par importlib.
_spec = importlib.util.spec_from_file_location("serpent_base", os.path.join(ICI, "serpent-algo.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
pygame = base.pygame

N = base.GRID_SIZE
CELLS = N * N
DIRS = [base.UP, base.DOWN, base.LEFT, base.RIGHT]

# Les raccourcis ne sont pris que tant que le serpent occupe moins de la
# moitié de la grille. Mesuré sur 30 parties : 6 803 pas avec cette coupure,
# 10 235 sans. Les sauts tardifs tassent le corps derrière la tête et bloquent
# ensuite tout saut pendant des centaines de pas.
SEUIL_RACCOURCIS = 0.5
# Le serpent grandit UN PAS APRÈS avoir mangé (grow_pending) : la queue reste
# alors en place un coup de plus. Deux cases de marge devant elle suffisent.
MARGE_QUEUE = 2


def voisin(c, d):
    return ((c[0] + d[0]) % N, (c[1] + d[1]) % N)


def direction_vers(a, b):
    for d in DIRS:
        if voisin(a, d) == b:
            return d
    raise ValueError(f"{a} et {b} ne sont pas voisins")


def demi_tour(d, e):
    return d[0] == -e[0] and d[1] == -e[1]


# ---------------------------------------------------------------- partie
class Partie:
    """La boucle de main() du jeu de base, sans fenêtre ni clock.

    Un appel à jouer() = un tour de boucle : set_direction, move, test de
    collision, pomme mangée ou non. Même ordre, mêmes objets.
    """

    def __init__(self):
        self.snake = base.Snake()
        self.apple = base.Apple(self.snake.body)
        self.pas = 0
        self.fin = None                 # None, "mort" ou "victoire"

    def corps(self):
        return [tuple(c) for c in self.snake.body]

    def pomme(self):
        return tuple(self.apple.position)

    def jouer(self, d):
        self.snake.set_direction(d)
        self.snake.move()
        self.pas += 1
        if self.snake.is_game_over():
            self.fin = "mort"
        elif self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            if not self.apple.relocate(self.snake.body):
                self.fin = "victoire"


# ---------------------------------------------------------------- cycle
def cycle_helicoidal():
    """14 pas à droite, 1 en bas, 15 fois : passe une fois par chaque case
    et revient au départ. N'existe que parce que la grille est un tore."""
    seq, x, y = [], 0, 0
    for _ in range(N):
        for _ in range(N):
            seq.append((x, y))
            x = (x + 1) % N
        x = (x - 1) % N
        y = (y + 1) % N
    assert len(set(seq)) == CELLS
    assert all(b in [voisin(a, d) for d in DIRS] for a, b in zip(seq, seq[1:] + seq[:1]))
    return seq


CYCLE = cycle_helicoidal()
ORDRE = {c: i for i, c in enumerate(CYCLE)}


def dist_cycle(a, b):
    """Nombre de pas de a à b en suivant le cycle."""
    return (ORDRE[b] - ORDRE[a]) % CELLS


def algo_cycle(p):
    h = p.corps()[0]
    return direction_vers(h, CYCLE[(ORDRE[h] + 1) % CELLS])


def algo_tapsell(p):
    corps = p.corps()
    h, queue, pomme = corps[0], corps[-1], p.pomme()
    if len(corps) + 1 > SEUIL_RACCOURCIS * CELLS:
        return algo_cycle(p)
    occupe = set(corps)
    d_queue, d_pomme = dist_cycle(h, queue), dist_cycle(h, pomme)
    meilleur, saut_max = None, 0
    for d in DIRS:
        if demi_tour(d, p.snake.direction):
            continue
        n = voisin(h, d)
        if n in occupe:
            continue
        saut = dist_cycle(h, n)
        # Ne pas rattraper la queue : c'est ce qui garde le corps rangé dans
        # l'ordre du cycle, donc la victoire garantie.
        if saut >= d_queue - MARGE_QUEUE:
            continue
        # Ne pas dépasser la pomme : il faudrait refaire le tour pour elle.
        if saut > d_pomme:
            continue
        if saut > saut_max:
            meilleur, saut_max = d, saut
    return meilleur if meilleur is not None else algo_cycle(p)


# ---------------------------------------------------------------- recherches
def espace_libre(depart, bloque):
    """Flood-fill : nombre de cases atteignables depuis `depart`."""
    vu, pile = {depart}, [depart]
    while pile:
        c = pile.pop()
        for d in DIRS:
            n = voisin(c, d)
            if n not in vu and n not in bloque:
                vu.add(n)
                pile.append(n)
    return len(vu)


def coup_de_survie(p):
    """Aucun chemin vers la pomme : le coup qui laisse le plus d'espace."""
    corps = p.corps()
    bloque = set(corps[:-1])                  # la queue part au prochain pas
    meilleur, place_max = p.snake.direction, -1
    for d in DIRS:
        if demi_tour(d, p.snake.direction):
            continue
        n = voisin(corps[0], d)
        if n in bloque:
            continue
        place = espace_libre(n, bloque | {n})
        if place > place_max:
            meilleur, place_max = d, place
    return meilleur


def manhattan_torique(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return min(dx, N - dx) + min(dy, N - dy)


def _recherche(p, priorite):
    """File de priorité commune à Dijkstra et au glouton ; seule la clé change."""
    corps = p.corps()
    h, pomme = corps[0], p.pomme()
    bloque = set(corps[:-1])
    ouverts = [(priorite(0, h), 0, h)]
    parent = {h: None}
    while ouverts:
        _, g, c = heapq.heappop(ouverts)
        if c == pomme:
            while parent[c] != h:
                c = parent[c]
            return direction_vers(h, c)
        for d in DIRS:
            n = voisin(c, d)
            if n in parent or n in bloque:
                continue
            if c == h and demi_tour(d, p.snake.direction):
                continue
            parent[n] = c
            heapq.heappush(ouverts, (priorite(g + 1, n), g + 1, n))
    return coup_de_survie(p)


def algo_dijkstra(p):
    return _recherche(p, lambda g, n: g)


def algo_glouton(p):
    pomme = p.pomme()
    return _recherche(p, lambda g, n: manhattan_torique(n, pomme))


ALGOS = {"dijkstra": algo_dijkstra, "glouton": algo_glouton,
         "cycle": algo_cycle, "tapsell": algo_tapsell}


# ---------------------------------------------------------------- modes
def partie_sans_fenetre(algo, graine, max_pas=200_000):
    random.seed(graine)
    p = Partie()
    t_max = 0.0
    while p.fin is None and p.pas < max_pas:
        t0 = time.perf_counter()
        d = algo(p)
        t_max = max(t_max, time.perf_counter() - t0)
        p.jouer(d)
    return p.fin or "limite", p.snake.score, p.pas, t_max


def bench(noms, n_parties):
    print(f"{n_parties} parties par algorithme, graines 0 à {n_parties - 1}\n")
    print(f"{'algo':10s} {'victoires':>10s} {'score moy':>10s} {'min':>5s} {'pas moy':>9s}"
          f" {'pas/pomme':>10s} {'à 5 pas/s':>10s} {'calcul max':>11s}")
    for nom in noms:
        res = [partie_sans_fenetre(ALGOS[nom], g) for g in range(n_parties)]
        vict = sum(r[0] == "victoire" for r in res)
        scores = [r[1] for r in res]
        pas = statistics.mean(r[2] for r in res)
        minutes = pas / base.GAME_SPEED / 60
        print(f"{nom:10s} {vict:>6d}/{n_parties:<3d} {statistics.mean(scores):10.1f}"
              f" {min(scores):5d} {pas:9.0f} {pas / max(1, statistics.mean(scores)):10.1f}"
              f" {minutes:7.1f} min {max(r[3] for r in res) * 1000:8.1f} ms", flush=True)


def dessiner_cycle(surface):
    """Tracé discret du cycle hamiltonien, pour voir ce que suit le serpent."""
    couleur, c = (40, 60, 55), base.CELL_SIZE
    for a, b in zip(CYCLE, CYCLE[1:] + CYCLE[:1]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            continue                              # passage par le bord
        pa = (a[0] * c + c // 2, a[1] * c + c // 2 + base.SCORE_PANEL_HEIGHT)
        pb = (b[0] * c + c // 2, b[1] * c + c // 2 + base.SCORE_PANEL_HEIGHT)
        pygame.draw.line(surface, couleur, pa, pb, 3)


def play(nom, speed, trace):
    """La boucle de main() du jeu de base, l'algorithme à la place du clavier."""
    algo = ALGOS[nom]
    pygame.init()
    screen = pygame.display.set_mode((base.SCREEN_WIDTH, base.SCREEN_HEIGHT))
    pygame.display.set_caption(f"Snake algo ({nom}) - groupe Abricot")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    p, debut = Partie(), time.time()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and p.fin:
                p, debut = Partie(), time.time()
        if p.fin is None:
            p.jouer(algo(p))
            if p.fin:
                print(f"{nom} : {p.fin}, score {p.snake.score} en {p.pas} pas "
                      f"({p.pas / base.GAME_SPEED / 60:.1f} min au rythme du jeu)")

        screen.fill(base.GRIS_FOND)
        pygame.draw.rect(screen, base.NOIR, pygame.Rect(0, base.SCORE_PANEL_HEIGHT,
                                                         base.SCREEN_WIDTH, base.SCREEN_WIDTH))
        base.draw_grid(screen)
        if trace:
            dessiner_cycle(screen)
        p.apple.draw(screen)
        p.snake.draw(screen)
        base.display_info(screen, font_main, p.snake, debut)
        if p.fin:
            victoire = p.fin == "victoire"
            base.display_message(screen, font_game_over, "VICTOIRE !" if victoire else "GAME OVER",
                                 base.VERT if victoire else base.ROUGE)
            base.display_message(screen, font_main, f"{p.pas} pas - ESPACE pour rejouer.",
                                 base.BLANC, y_offset=100)
        pygame.display.flip()
        clock.tick(speed)


def main():
    parser = argparse.ArgumentParser(description="Snake torique résolu par algorithme.")
    sub = parser.add_subparsers(dest="mode", required=True)
    p_play = sub.add_parser("play", help="regarder une partie")
    p_play.add_argument("--algo", choices=ALGOS, default="tapsell")
    p_play.add_argument("--speed", type=int, default=base.GAME_SPEED,
                        help=f"images/s à l'affichage (défaut {base.GAME_SPEED}, la vitesse du jeu ; "
                             "seule la cadence d'affichage change, pas la partie)")
    p_play.add_argument("--trace", action="store_true", help="dessiner le cycle hamiltonien")
    p_bench = sub.add_parser("bench", help="mesurer sans fenêtre")
    p_bench.add_argument("--algo", choices=ALGOS, nargs="+", default=list(ALGOS))
    p_bench.add_argument("--games", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "play":
        play(args.algo, args.speed, args.trace)
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        bench(args.algo, args.games)


if __name__ == "__main__":
    main()
