"""Simulateur sans graphique du serpent (15x15, torique) + utilitaires de grille.

Les règles sont celles de serpent-algo.py, à l'identique (voir test_sim.py qui
compare pas à pas ce simulateur avec les vraies classes Snake/Apple) :
  - une case = un entier c = y * N + x ; les bords s'enroulent ;
  - on ne peut pas faire demi-tour (set_direction ignore l'inverse) ;
  - la queue est retirée AVANT le test de collision : entrer dans la case de la
    queue est légal, sauf juste après avoir mangé (la queue ne bouge pas) ;
  - victoire quand plus aucune case libre pour la pomme.
"""
import random
from collections import deque

N = 15
NN = N * N

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3          # index de direction
VEC = [(0, -1), (0, 1), (-1, 0), (1, 0)]
OPP = [DOWN, UP, RIGHT, LEFT]


def cid(x, y):
    return (y % N) * N + (x % N)


# NB[c][d] = case voisine de c dans la direction d (torique)
NB = [tuple(cid(c % N + dx, c // N + dy) for dx, dy in VEC) for c in range(NN)]
# DIR_OF[a][b] = direction pour aller de a à b (voisines), sinon -1
DIR_OF = [[-1] * NN for _ in range(NN)]
for _c in range(NN):
    for _d in range(4):
        DIR_OF[_c][NB[_c][_d]] = _d


def torus_dist(a, b):
    """Distance de Manhattan torique entre deux cases."""
    dx = abs(a % N - b % N)
    dy = abs(a // N - b // N)
    return min(dx, N - dx) + min(dy, N - dy)


START_HEAD = cid(N // 4, N // 2)
START_BODY = (START_HEAD, cid(N // 4 - 1, N // 2), cid(N // 4 - 2, N // 2))
START_DIR = RIGHT


class Game:
    """État du jeu. game.body : deque de cases, tête en premier."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.body = deque(START_BODY)
        self.occ = bytearray(NN)
        for c in self.body:
            self.occ[c] = 1
        self.dir = START_DIR
        self.grow = False          # croissance en attente (on vient de manger)
        self.score = 0
        self.moves = 0
        self.alive = True
        self.won = False
        self.apple = self._spawn()

    def _spawn(self):
        free = [c for c in range(NN) if not self.occ[c]]
        return self.rng.choice(free) if free else None

    @property
    def head(self):
        return self.body[0]

    @property
    def tail(self):
        return self.body[-1]

    @property
    def over(self):
        return not self.alive or self.won

    def step(self, d):
        """Joue un coup dans la direction d (l'inverse est ignoré, comme le jeu)."""
        if d != OPP[self.dir]:
            self.dir = d
        nh = NB[self.body[0]][self.dir]
        if self.grow:
            self.grow = False
        else:
            self.occ[self.body.pop()] = 0
        self.moves += 1
        if self.occ[nh]:
            self.alive = False
            return
        self.body.appendleft(nh)
        self.occ[nh] = 1
        if nh == self.apple:
            self.grow = True
            self.score += 1
            self.apple = self._spawn()
            if self.apple is None:
                self.won = True


def build_cycle():
    """Cycle hamiltonien du tore 15x15 orienté comme le serpent au départ.

    Lignes en boustrophédon sur les colonnes 1..N-1, puis retour par la colonne
    0 (le passage 14 -> 0 utilise l'enroulement). Le serpent initial (tête en
    (3,7), vers la droite) est posé sur ce cycle."""
    seq = []
    for y in range(N):
        xs = range(1, N) if y % 2 == 0 else range(N - 1, 0, -1)
        seq.extend(cid(x, y) for x in xs)
    seq.extend(cid(0, y) for y in range(N - 1, -1, -1))
    assert len(seq) == NN and len(set(seq)) == NN
    for a, b in zip(seq, seq[1:] + seq[:1]):
        assert DIR_OF[a][b] >= 0
    i = seq.index(START_HEAD)
    if seq[(i + 1) % NN] != NB[START_HEAD][START_DIR]:
        seq.reverse()
    i = seq.index(START_HEAD)
    assert seq[(i + 1) % NN] == NB[START_HEAD][START_DIR]
    assert seq[(i - 1) % NN] == START_BODY[1] and seq[(i - 2) % NN] == START_BODY[2]
    return seq


def play(agent, seed, max_stall=None, timing=False):
    """Joue une partie complète. Retourne un dict de résultats.

    agent.choose(game) -> index de direction. max_stall : coups sans pomme
    avant de déclarer la partie bloquée (0 = jamais ; None = 20 * NN)."""
    import time
    if max_stall is None:
        max_stall = 20 * NN
    g = Game(seed)
    if hasattr(agent, "reset"):
        agent.reset(g)
    since = 0
    last = 0
    worst = 0.0
    slow = 0                  # coups dont le calcul dépasse une frame (1/5 s)
    t0 = time.perf_counter()
    while not g.over:
        if timing:
            t1 = time.perf_counter()
        d = agent.choose(g)
        if timing:
            dt = time.perf_counter() - t1
            worst = max(worst, dt)
            slow += dt > 0.2
        g.step(d)
        if g.score != last:
            last = g.score
            since = 0
        else:
            since += 1
            if max_stall and since > max_stall:
                break
    status = "won" if g.won else ("dead" if not g.alive else "stalled")
    return dict(seed=seed, status=status, score=g.score, moves=g.moves,
                cpu=time.perf_counter() - t0, worst=worst, slow=slow)
