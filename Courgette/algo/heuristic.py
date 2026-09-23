"""Stratégie 1 — « Moteur heuristique » (A* + simulation virtuelle + flood fill).

C'est la stratégie décrite dans snake_algorithmic_strategy_summary.md :
  1. A* torique vers la pomme (avec prise en compte du temps : une case du corps
     redevient libre quand la queue l'a quittée) ;
  2. filtre 1 : on « joue » le chemin sur un serpent virtuel, puis un BFS vérifie
     que la tête peut encore rejoindre sa queue après avoir mangé ;
  3. filtre 2 : flood fill de l'espace libre restant.
Si aucun chemin ne passe les filtres, on survit en s'éloignant de la queue
(mouvement de repli) en attendant que la situation se débloque.

area_rule règle le filtre 2 :
  "literal"   : rejeter si espace atteignable < longueur du corps (règle du doc,
                telle qu'écrite : elle interdit tout coup dès que la moitié de la
                grille est remplie) ;
  "corrected" : espace atteignable >= min(longueur, cases libres) c'est-à-dire
                « ne pas couper l'espace libre en îlots » en fin de partie ;
  "none"      : pas de filtre 2 (filtre 1 seul).
"""
import heapq
from collections import deque

from core import N, NN, NB, OPP, DIR_OF, torus_dist

TD = [[torus_dist(a, b) for b in range(NN)] for a in range(NN)]


def vacate_times(g):
    """vac[c] = numéro du coup à partir duquel la case c est libre (0 = déjà libre).
    Le corps[i] (0 = tête) est libre au coup k >= L - i (+1 si la queue est figée
    par une croissance en attente)."""
    L = len(g.body)
    extra = 1 if g.grow else 0
    vac = [0] * NN
    for i, c in enumerate(g.body):
        vac[c] = L - i + extra
    return vac


def astar(vac, sources, goal):
    """A* torique à obstacles temporels. sources : cases atteintes au coup 1.
    Retourne la liste de cases [premier pas, ..., goal] ou None."""
    td = TD[goal]
    best = {}
    parent = {}
    heap = []
    for s in sources:
        if vac[s] <= 1:
            best[s] = 1
            parent[s] = -1
            heapq.heappush(heap, (1 + td[s], 1, s))
    while heap:
        f, k, c = heapq.heappop(heap)
        if c == goal:
            path = [c]
            while parent[path[-1]] != -1:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        if best.get(c, 1 << 30) < k:
            continue
        k1 = k + 1
        for n in NB[c]:
            if vac[n] <= k1 and best.get(n, 1 << 30) > k1:
                best[n] = k1
                parent[n] = c
                heapq.heappush(heap, (k1 + td[n], k1, n))
    return None


def follow(g, path):
    """Serpent virtuel après avoir suivi `path` (mange la pomme au dernier pas)."""
    body = deque(g.body)
    occ = bytearray(g.occ)
    grow = g.grow
    for c in path:
        if grow:
            grow = False
        else:
            occ[body.pop()] = 0
        body.appendleft(c)
        occ[c] = 1
    if path and path[-1] == g.apple:
        grow = True
    return body, occ, grow


def analyze(body, occ):
    """BFS depuis la tête sur les cases libres.
    Retourne (aire libre atteignable, distance jusqu'à une case voisine de la
    queue ou -1 si la queue est injoignable)."""
    head = body[0]
    tail = body[-1]
    seen = bytearray(occ)
    seen[head] = 1
    frontier = [head]
    area = 0
    tdist = 0 if tail in NB[head] else -1
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for c in frontier:
            for n in NB[c]:
                if not seen[n]:
                    seen[n] = 1
                    nxt.append(n)
                    area += 1
                    if tdist < 0 and tail in NB[n]:
                        tdist = depth
        frontier = nxt
    return area, tdist


class HeuristicEngine:
    name = "heuristique"

    def __init__(self, area_rule="corrected"):
        self.area_rule = area_rule
        self.name = f"heuristique-{area_rule}"

    def _safe(self, g, path):
        body, occ, _ = follow(g, path)
        area, tdist = analyze(body, occ)
        if tdist < 0:
            return False
        free = NN - len(body)
        if self.area_rule == "literal":
            return area >= len(body)
        if self.area_rule == "corrected":
            return area >= min(len(body), free)
        return True

    def _legal_firsts(self, g, vac):
        back = OPP[g.dir]
        return [NB[g.head][d] for d in range(4) if d != back and vac[NB[g.head][d]] <= 1]

    def choose(self, g):
        vac = vacate_times(g)
        firsts = self._legal_firsts(g, vac)
        if not firsts:
            return g.dir
        path = astar(vac, firsts, g.apple)
        if path and self._safe(g, path):
            return DIR_OF[g.head][path[0]]
        # A* refusé par les filtres : on essaie les autres premiers pas, du plus court au plus long
        alts = []
        for f in firsts:
            if path and f == path[0]:
                continue
            p = astar(vac, [f], g.apple)
            if p:
                alts.append(p)
        alts.sort(key=len)
        for p in alts:
            if self._safe(g, p):
                return DIR_OF[g.head][p[0]]
        return self._fallback(g, firsts)

    def _fallback(self, g, firsts):
        """Aucun chemin sûr vers la pomme : on s'éloigne de la queue (on tourne)
        sans jamais entrer dans une impasse."""
        best = None
        for f in firsts:
            body, occ, _ = follow(g, [f])
            area, tdist = analyze(body, occ)
            key = (tdist >= 0, tdist, area)
            if best is None or key > best[0]:
                best = (key, f)
        return DIR_OF[g.head][best[1]]
