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

CINQ ALGORITHMES (--algo) :
  dijkstra   plus court chemin vers la pomme ; s'il n'y en a pas, le coup qui
             laisse le plus d'espace. Optimal par pomme, myope sur la partie.
  glouton    Greedy Best-First, heuristique Manhattan torique. Rapide, se piège.
  cycle      suit le cycle hamiltonien. Victoire certaine, mais ~12 500 pas.
  tapsell    le cycle, plus les raccourcis de John Tapsell : sauter en avant
             dans l'ordre du cycle, sans dépasser la pomme ni rattraper la
             queue. ~6 800 pas. PAS strictement sûr : les cases sautées restent
             derrière la tête, et une série de pommes juste devant elle peut
             lui faire rattraper sa queue (rare, mais observé en fin de partie
             quand les raccourcis restent actifs).
  recompose  (défaut) le cycle n'est plus fixe. À chaque pas, on réécrit la
             partie du cycle qui est devant la tête : chemin court vers la
             pomme, puis un parcours de toutes les autres cases libres qui
             revient à la queue (rotations de Pósa). Le corps n'est jamais
             touché et aucune case n'est sautée : victoire certaine, ~3 900 pas.

Usage :
    python snake-algo.py play                     # recompose, à la vitesse du jeu
    python snake-algo.py play --algo cycle --speed 60 --trace
    python snake-algo.py bench --games 30         # compare les cinq algos
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


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
_assurer_environnement()

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


# ---------------------------------------------------------------- recompose
# Le cycle se découpe en deux arcs : [queue .. tête], le corps, auquel on ne
# touche jamais ; et F, les cases libres devant la tête. Suivre n'importe quel
# ordre de F qui part d'une voisine de la tête et finit sur une voisine de la
# queue donne encore un cycle hamiltonien qui contient le corps : la victoire
# reste certaine. On choisit donc, à chaque pas, un ordre de F où la pomme
# vient le plus tôt possible.

def carte_distances(src, permis):
    d = {src: 0}
    file = deque([src])
    while file:
        c = file.popleft()
        for dd in DIRS:
            n = voisin(c, dd)
            if n in permis and n not in d:
                d[n] = d[c] + 1
                file.append(n)
    return d


def reste_couvrable(reste, depuis, voisines_queue):
    """Test rapide : le reste de F peut-il encore être parcouru en entier,
    en partant d'une voisine de `depuis` et en finissant près de la queue ?
    D'un seul tenant, et au plus un cul-de-sac (qui sera alors la fin)."""
    if not reste:
        return depuis in voisines_queue
    depart = next(iter(reste))
    vu, pile = {depart}, [depart]
    while pile:
        c = pile.pop()
        for dd in DIRS:
            n = voisin(c, dd)
            if n in reste and n not in vu:
                vu.add(n)
                pile.append(n)
    if len(vu) != len(reste):
        return False                          # le chemin a coupé F en deux
    autour = {voisin(depuis, dd) for dd in DIRS}
    if not (autour & reste) or not (voisines_queue & reste):
        return False
    impasses = 0
    for c in reste:
        degre = sum(1 for dd in DIRS if voisin(c, dd) in reste) + (c in autour)
        if degre == 0:
            return False
        if degre == 1 and c not in voisines_queue:
            impasses += 1
            if impasses > 1:
                return False
    return True


def chemins_vers_pomme(h, pomme, F, voisines_queue, marge, budget):
    """Chemins tête -> pomme dans F, du plus court au plus long (plus court +
    marge), qui longent les obstacles et ne coupent pas le reste de F.
    Le plus court chemin brut coupe F en deux une fois sur deux : c'est ce
    test qui rend la recomposition possible."""
    dist = carte_distances(pomme, F | {h})
    if h not in dist:
        return
    compteur = [0]
    chemin, dedans = [], set()

    def colle(c):          # voisins déjà hors de F : on longe les obstacles
        return sum(1 for dd in DIRS if voisin(c, dd) not in F or voisin(c, dd) in dedans)

    def dfs(c, reste_pas):
        compteur[0] += 1
        if compteur[0] > budget:
            return
        if c == pomme:
            if reste_couvrable(F - dedans, pomme, voisines_queue):
                yield list(chemin)
            return
        suivants = [voisin(c, dd) for dd in DIRS]
        suivants = [n for n in suivants
                    if n in F and n not in dedans and n in dist and dist[n] <= reste_pas - 1]
        suivants.sort(key=lambda n: (dist[n], -colle(n)))
        for n in suivants:
            chemin.append(n)
            dedans.add(n)
            yield from dfs(n, reste_pas - 1)
            chemin.pop()
            dedans.discard(n)
            if compteur[0] > budget:
                return

    # longueurs de même parité d'abord (les plus naturelles sur la grille)
    for extra in list(range(0, marge + 1, 2)) + list(range(1, marge + 1, 2)):
        for c in dfs(h, dist[h] + extra):
            yield c
            return


def rotations_posa(prefixe, F, queue, rng, max_iter):
    """Complète `prefixe` en un chemin qui passe par toutes les cases de F et
    finit sur une voisine de la queue. Extension gloutonne (vers la case la
    moins accessible, règle de Warnsdorff) ; bloqué, on fait une rotation de
    Pósa : la fin touche une case i du chemin, on renverse tout ce qui suit i.
    Le préfixe, qui mène à la pomme, n'est jamais retourné."""
    m = len(prefixe)
    chemin = list(prefixe)
    rang = {c: i for i, c in enumerate(chemin)}
    voisines_queue = {voisin(queue, dd) for dd in DIRS}
    for _ in range(max_iter):
        fin = chemin[-1]
        autour = [voisin(fin, dd) for dd in DIRS]
        if len(chemin) == len(F):
            if fin in voisines_queue:
                return chemin
        else:
            libres = [n for n in autour if n in F and n not in rang]
            if libres:
                n = min(libres, key=lambda n: (
                    sum(1 for dd in DIRS if voisin(n, dd) in F and voisin(n, dd) not in rang),
                    rng.random()))
                rang[n] = len(chemin)
                chemin.append(n)
                continue
        pivots = [rang[w] for w in autour if w in rang and m - 1 <= rang[w] < len(chemin) - 2]
        if not pivots:
            return None
        i = rng.choice(pivots)
        chemin[i + 1:] = chemin[i + 1:][::-1]
        for j in range(i + 1, len(chemin)):
            rang[chemin[j]] = j
    return None


class Recompose:
    """Cycle hamiltonien réécrit devant la tête à chaque pas."""

    MARGE = 6           # pas de détour tolérés pour ne pas couper F
    BUDGET = 3000       # nœuds explorés pour trouver ce chemin
    MAX_ROTATIONS = 3000

    def __init__(self, graine=None):
        self.seq = list(CYCLE)
        self.rang = dict(ORDRE)
        self.rng = random.Random(graine)   # jamais le random global : c'est celui des pommes

    def rel(self, h, c):
        return (self.rang[c] - self.rang[h]) % CELLS

    def recomposer(self, p):
        corps = p.corps()
        h, queue, pomme = corps[0], corps[-1], p.pomme()
        T = self.rel(h, queue)
        base_h = self.rang[h]
        devant = [self.seq[(base_h + i) % CELLS] for i in range(1, T)]
        F = set(devant)
        if pomme not in F:
            return
        actuel = self.rel(h, pomme)
        # Déjà au plus court : rien à gagner, on s'épargne la recherche.
        if carte_distances(pomme, F | {h}).get(h, 0) >= actuel:
            return
        voisines_queue = {voisin(queue, dd) for dd in DIRS}
        for prefixe in chemins_vers_pomme(h, pomme, F, voisines_queue, self.MARGE, self.BUDGET):
            if len(prefixe) >= actuel:
                return
            nouveau = rotations_posa(prefixe, F, queue, self.rng, self.MAX_ROTATIONS)
            if nouveau is None:
                return
            arriere = [self.seq[(base_h + i) % CELLS] for i in range(T, CELLS)]
            seq = arriere + [h] + nouveau
            # Vérification complète avant adoption : si elle échoue, on garde
            # l'ancien cycle. La garantie ne dépend pas de la recherche.
            if len(set(seq)) == CELLS and all(
                    b in [voisin(a, dd) for dd in DIRS] for a, b in zip(seq, seq[1:] + seq[:1])):
                self.seq = seq
                self.rang = {c: i for i, c in enumerate(seq)}
            return

    def __call__(self, p):
        self.recomposer(p)
        h = p.corps()[0]
        return direction_vers(h, self.seq[(self.rang[h] + 1) % CELLS])


# Chaque partie reçoit une instance neuve : recompose garde son cycle en mémoire.
ALGOS = {"dijkstra": lambda graine=None: algo_dijkstra,
         "glouton": lambda graine=None: algo_glouton,
         "cycle": lambda graine=None: algo_cycle,
         "tapsell": lambda graine=None: algo_tapsell,
         "recompose": Recompose}


# ---------------------------------------------------------------- modes
def partie_sans_fenetre(nom, graine, max_pas=200_000):
    random.seed(graine)
    algo = ALGOS[nom](graine)
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
        res = [partie_sans_fenetre(nom, g) for g in range(n_parties)]
        vict = sum(r[0] == "victoire" for r in res)
        scores = [r[1] for r in res]
        pas = statistics.mean(r[2] for r in res)
        minutes = pas / base.GAME_SPEED / 60
        print(f"{nom:10s} {vict:>6d}/{n_parties:<3d} {statistics.mean(scores):10.1f}"
              f" {min(scores):5d} {pas:9.0f} {pas / max(1, statistics.mean(scores)):10.1f}"
              f" {minutes:7.1f} min {max(r[3] for r in res) * 1000:8.1f} ms", flush=True)


def dessiner_cycle(surface, seq):
    """Tracé discret du cycle hamiltonien, pour voir ce que suit le serpent."""
    couleur, c = (40, 60, 55), base.CELL_SIZE
    for a, b in zip(seq, seq[1:] + seq[:1]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            continue                              # passage par le bord
        pa = (a[0] * c + c // 2, a[1] * c + c // 2 + base.SCORE_PANEL_HEIGHT)
        pb = (b[0] * c + c // 2, b[1] * c + c // 2 + base.SCORE_PANEL_HEIGHT)
        pygame.draw.line(surface, couleur, pa, pb, 3)


def play(nom, speed, trace):
    """La boucle de main() du jeu de base, l'algorithme à la place du clavier."""
    algo = ALGOS[nom]()
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
                p, debut, algo = Partie(), time.time(), ALGOS[nom]()
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
            dessiner_cycle(screen, getattr(algo, "seq", CYCLE))
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
    sub = parser.add_subparsers(dest="mode")
    p_play = sub.add_parser("play", help="regarder une partie")
    p_play.add_argument("--algo", choices=ALGOS, default="recompose")
    p_play.add_argument("--speed", type=int, default=base.GAME_SPEED,
                        help=f"images/s à l'affichage (défaut {base.GAME_SPEED}, la vitesse du jeu ; "
                             "seule la cadence d'affichage change, pas la partie)")
    p_play.add_argument("--trace", action="store_true", help="dessiner le cycle hamiltonien")
    p_bench = sub.add_parser("bench", help="mesurer sans fenêtre")
    p_bench.add_argument("--algo", choices=ALGOS, nargs="+", default=list(ALGOS))
    p_bench.add_argument("--games", type=int, default=30)
    # Sans argument : démonstration de l'algo retenu, comme snake-ia.py.
    args = parser.parse_args(sys.argv[1:] or ["play"])

    if args.mode == "play":
        play(args.algo, args.speed, args.trace)
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        bench(args.algo, args.games)


if __name__ == "__main__":
    main()
