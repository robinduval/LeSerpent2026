"""Stratégie 3 — Cell-tree (arbre de cellules 2x2) adapté au tore 15x15.

Principe (twanvl/snake, « cell ») : on découpe la grille en cellules 2x2 ; un arbre
couvrant de ces cellules définit un cycle hamiltonien (le tour de l'arbre) :
  - le côté d'une cellule n'a son arête interne dans le cycle que s'il n'y a PAS
    de branche de l'arbre de ce côté ;
  - une branche entre deux cellules voisines ajoute les 2 arêtes qui les traversent.
Échanger une branche de l'arbre contre une autre change le cycle de façon
« compound » (4 arêtes d'un coup), ce qu'une réparation 2-opt ne sait pas faire.
Tant que les arêtes du corps du serpent restent dans le cycle, le corps reste un
segment contigu du cycle : on peut toujours le suivre => victoire garantie.

Adaptation à 15 x 15 (impair, donc impossible à paver en 2x2) :
  - bloc 14x14 = 7x7 cellules (l'arbre a 48 branches) ;
  - les 29 cases restantes (colonne x=14 et ligne y=14) sont branchées en
    permanence sur des arêtes qui sont TOUJOURS dans le cycle, quel que soit
    l'arbre (côtés extérieurs du bloc, sans branche possible) :
      * colonne : 7 détours de 2 cases (14,2j)-(14,2j+1) sur le côté droit des
        cellules de la dernière colonne ;
      * ligne : un seul détour de 15 cases qui fait le tour du tore en x
        (0,14) -> (14,14) -> (13,14) -> ... -> (1,14), sur le côté bas de la
        cellule en bas à gauche. Un nombre impair de cases n'est possible que
        grâce à l'enroulement, d'où ce détour unique et long.
    Ces 29 cases n'ont donc que leurs voisines de détour comme mouvements possibles.

Le planificateur est celui de DHCR : plus court chemin A* (sans les arêtes
impossibles) vers la pomme, puis on « rejoue » chaque pas en échangeant des
branches de l'arbre si l'arête voulue n'est pas encore dans le cycle. Un échange
n'est légal que s'il ne retire aucune arête du corps. Si aucun chemin n'est
réalisable, on suit le cycle tel quel (toujours sûr).
"""
import heapq
import time

from core import N, NN, NB, OPP, DIR_OF, cid
from heuristic import TD

B = 7                                   # cellules par côté
NC = B * B
NE = 2 * B * (B - 1)                    # 84 branches possibles
NEVER, ALWAYS, CROSS, INNER = 1, 0, 2, 3


# ----------------------------------------------------------------- géométrie
def _quad(I, J):
    """(TL, TR, BL, BR) : les 4 cases de la cellule (I, J)."""
    return (cid(2 * I, 2 * J), cid(2 * I + 1, 2 * J),
            cid(2 * I, 2 * J + 1), cid(2 * I + 1, 2 * J + 1))


EDGE_CELLS = [None] * NE               # branche -> (cellule A, cellule B)
CROSSING = [None] * NE                 # branche -> 2 arêtes ajoutées quand la branche existe
INNER_PAIR = [None] * NE               # branche -> 2 arêtes internes retirées quand elle existe
CADJ = [[] for _ in range(NC)]         # cellule -> [(cellule voisine, id de branche)]

for J in range(B):
    for I in range(B - 1):
        e = J * (B - 1) + I
        a, b = _quad(I, J), _quad(I + 1, J)
        EDGE_CELLS[e] = (J * B + I, J * B + I + 1)
        CROSSING[e] = ((a[1], b[0]), (a[3], b[2]))
        INNER_PAIR[e] = ((a[1], a[3]), (b[0], b[2]))
for J in range(B - 1):
    for I in range(B):
        e = B * (B - 1) + J * B + I
        a, b = _quad(I, J), _quad(I, J + 1)
        EDGE_CELLS[e] = (J * B + I, (J + 1) * B + I)
        CROSSING[e] = ((a[2], b[0]), (a[3], b[1]))
        INNER_PAIR[e] = ((a[2], a[3]), (b[0], b[1]))
for e, (ka, kb) in enumerate(EDGE_CELLS):
    CADJ[ka].append((kb, e))
    CADJ[kb].append((ka, e))

# KIND[a*NN+b] : nature de l'arête entre deux cases voisines a, b
#   ALWAYS toujours dans le cycle ; NEVER jamais ; CROSS dans le cycle ssi la
#   branche EID existe ; INNER dans le cycle ssi la branche EID n'existe pas.
KIND = [-1] * (NN * NN)
EID = [-1] * (NN * NN)
for a in range(NN):
    for b in NB[a]:
        KIND[a * NN + b] = NEVER


def _set(a, b, kind, e=-1):
    KIND[a * NN + b] = KIND[b * NN + a] = kind
    EID[a * NN + b] = EID[b * NN + a] = e


for e in range(NE):
    for a, b in CROSSING[e]:
        _set(a, b, CROSS, e)
    for a, b in INNER_PAIR[e]:
        _set(a, b, INNER, e)
for I in range(B):                      # côtés extérieurs (aucune branche possible)
    tl, tr, bl, br = _quad(I, 0)
    _set(tl, tr, ALWAYS)                # haut
    tl, tr, bl, br = _quad(I, B - 1)
    if I != 0:
        _set(bl, br, ALWAYS)            # bas (sauf (0,13)-(1,13), remplacé par le détour de ligne)
for J in range(B):
    tl, tr, bl, br = _quad(0, J)
    _set(tl, bl, ALWAYS)                # gauche
# colonne x = 14 : 7 détours de 2 cases (côté droit des cellules de la dernière colonne)
for j in range(B):
    _set(cid(13, 2 * j), cid(14, 2 * j), ALWAYS)
    _set(cid(14, 2 * j), cid(14, 2 * j + 1), ALWAYS)
    _set(cid(14, 2 * j + 1), cid(13, 2 * j + 1), ALWAYS)
# ligne y = 14 : un détour de 15 cases (0,14) (14,14) (13,14) ... (1,14)
_set(cid(0, 13), cid(0, 14), ALWAYS)
_set(cid(0, 14), cid(14, 14), ALWAYS)
for x in range(14, 1, -1):
    _set(cid(x, 14), cid(x - 1, 14), ALWAYS)
_set(cid(1, 14), cid(1, 13), ALWAYS)

# voisins utilisables (on écarte les adjacences qui ne sont jamais dans le cycle)
NBX = [tuple(n for n in NB[c] if KIND[c * NN + n] != NEVER) for c in range(NN)]


def comb_tree():
    """Arbre de départ : toutes les branches horizontales + la colonne 0 en vertical.
    Contient le serpent initial (tête (3,7) vers la droite)."""
    t = bytearray(NE)
    for J in range(B):
        for I in range(B - 1):
            t[J * (B - 1) + I] = 1
    for J in range(B - 1):
        t[B * (B - 1) + J * B + 0] = 1
    return t


def present(tree, a, b):
    k = KIND[a * NN + b]
    if k == ALWAYS:
        return True
    if k == NEVER:
        return False
    return bool(tree[EID[a * NN + b]]) if k == CROSS else not tree[EID[a * NN + b]]


def cycle_nbrs(tree, c):
    return [n for n in NB[c] if present(tree, c, n)]


def cycle_order(tree, start):
    """Le cycle complet (liste de 225 cases) ; lève une erreur s'il n'est pas hamiltonien."""
    order = [start]
    prev, cur = -1, start
    while True:
        nb = cycle_nbrs(tree, cur)
        assert len(nb) == 2, f"case {cur} de degré {len(nb)}"
        nxt = nb[0] if nb[0] != prev else nb[1]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        assert len(order) <= NN
    assert len(order) == NN, f"cycle de longueur {len(order)}"
    return order


# ---------------------------------------------------------------- recherche
def astar_ct(vac, sources, goal):
    """Comme heuristic.astar, mais sans les arêtes qui ne peuvent jamais exister."""
    td = TD[goal]
    best, parent, heap = {}, {}, []
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
        for n in NBX[c]:
            if vac[n] <= k + 1 and best.get(n, 1 << 30) > k + 1:
                best[n] = k + 1
                parent[n] = c
                heapq.heappush(heap, (k + 1 + td[n], k + 1, n))
    return None


class CellTree:
    name = "celltree"

    def __init__(self, budget=0.15, keep=3, slack=8):
        self.budget = budget            # secondes max par recherche (un coup = 0,2 s)
        self.keep = keep                # nb d'échanges gardés (les mieux classés) à chaque pas
        self.slack = slack              # longueur max = plus court chemin + slack

    def reset(self, g):
        self.tree = comb_tree()
        self.plan = []
        self.plan_apple = None
        order = cycle_order(self.tree, g.head)      # vérifie l'arbre de départ
        assert order[1] in (g.body[1], NB[g.head][g.dir]) and set(cycle_nbrs(self.tree, g.head)) == {g.body[1], NB[g.head][g.dir]}

    # -------- outils sur l'arbre
    def _tree_path(self, tree, A, Bc):
        prev = [-2] * NC
        prev[A] = -1
        via = [-1] * NC
        stack = [A]
        while stack:
            c = stack.pop()
            for c2, e in CADJ[c]:
                if tree[e] and prev[c2] == -2:
                    prev[c2] = c
                    via[c2] = e
                    stack.append(c2)
        path = []
        c = Bc
        while c != A:
            path.append(via[c])
            c = prev[c]
        return path

    def _side(self, tree, A, skip):
        """Cellules atteignables depuis A sans passer par la branche `skip`."""
        mark = bytearray(NC)
        mark[A] = 1
        stack = [A]
        while stack:
            c = stack.pop()
            for c2, e in CADJ[c]:
                if e != skip and tree[e] and not mark[c2]:
                    mark[c2] = 1
                    stack.append(c2)
        return mark

    @staticmethod
    def _legal(r, a, st, th, tt):
        """Retirer la branche r et ajouter a est légal si aucune arête retirée du
        cycle n'est une arête du corps (cases consécutives dans le corps)."""
        for x, y in CROSSING[r] + INNER_PAIR[a]:
            sx, sy = st[x], st[y]
            if tt <= sx <= th and tt <= sy <= th and (sx - sy == 1 or sy - sx == 1):
                return False
        return True

    def _edits(self, tree, h, n, st, th, tt):
        """Échanges (retirer r, ajouter a) légaux qui mettent l'arête (h, n) dans le cycle."""
        k = KIND[h * NN + n]
        e = EID[h * NN + n]
        out = []
        if k == CROSS:                            # il faut ajouter la branche e
            A, Bc = EDGE_CELLS[e]
            for r in self._tree_path(tree, A, Bc):
                if self._legal(r, e, st, th, tt):
                    out.append((r, e))
        elif k == INNER:                          # il faut retirer la branche e
            mark = self._side(tree, EDGE_CELLS[e][0], e)
            for f in range(NE):
                if f != e and not tree[f]:
                    X, Y = EDGE_CELLS[f]
                    if mark[X] != mark[Y] and self._legal(e, f, st, th, tt):
                        out.append((e, f))
        return out

    # -------- recherche du plus court chemin RÉALISABLE (IDA*)
    def _ida(self, h, i, bound, st, th, tt, grow):
        """Cherche jusqu'à la pomme en <= bound - i coups depuis h, en échangeant des
        branches si l'arête voulue n'est pas dans le cycle. Retourne
        [(cible, échange ou None), ...] ou None."""
        self._nodes += 1
        if self._nodes & 15 == 0 and time.perf_counter() > self._deadline:
            self._out = True
        if self._out:
            return None
        tree, goal, td = self.tree, self._goal, self._td
        tta = tt if grow else tt + 1              # après la sortie de la queue
        cands = []
        for n in NBX[h]:
            if i + 1 + td[n] > bound:
                continue
            if tta <= st[n] <= th:                # occupé (corps ou chemin déjà parcouru)
                continue
            cands.append((0 if present(tree, h, n) else 1, td[n], n))
        cands.sort()
        for _, _, n in cands:
            if present(tree, h, n):
                edits = [None]
            else:
                edits = self._edits(tree, h, n, st, th, tt)
                if len(edits) > 1:                # garder les plus prometteurs
                    scored = []
                    for r, a in edits:
                        tree[r], tree[a] = 0, 1
                        good = sum(1 for m in NBX[n] if td[m] < td[n] and present(tree, n, m))
                        tree[r], tree[a] = 1, 0
                        scored.append((-good, (r, a)))
                    scored.sort(key=lambda t: t[0])
                    edits = [e for _, e in scored[:self.keep]]
            for edit in edits:
                if edit is not None:
                    tree[edit[0]], tree[edit[1]] = 0, 1
                if n == goal:
                    rest = []
                else:
                    old = st[n]
                    st[n] = th + 1
                    rest = self._ida(n, i + 1, bound, st, th + 1, tta, False)
                    st[n] = old
                if edit is not None:
                    tree[edit[0]], tree[edit[1]] = 1, 0
                if rest is not None:
                    return [(n, edit)] + rest
                if self._out:
                    return None
        return None

    def _plan(self, g):
        h = g.head
        L = len(g.body)
        st = [-10] * NN
        vac = [0] * NN
        extra = 1 if g.grow else 0
        for i, c in enumerate(g.body):
            st[c] = L - 1 - i
            vac[c] = L - i + extra + 1
        firsts = [n for n in NBX[h] if vac[n] == 0 and n != g.body[1]]
        if not firsts:
            return None
        p = astar_ct(vac, firsts, g.apple)        # longueur du plus court chemin possible
        if not p:
            return None
        self._goal, self._td = g.apple, TD[g.apple]
        self._deadline = time.perf_counter() + self.budget
        self._out = False
        self._nodes = 0
        for bound in range(len(p), len(p) + self.slack + 1):
            r = self._ida(h, 0, bound, st, L - 1, 0, g.grow)
            if r is not None:
                return r
            if self._out:
                break
        return None

    def choose(self, g):
        h = g.head
        if not self.plan or self.plan_apple != g.apple:
            self.plan_apple = g.apple
            self.plan = self._plan(g) or []
            if not self.plan:                     # rien de réalisable : suivre le cycle
                nxt = [n for n in cycle_nbrs(self.tree, h) if n != g.body[1]]
                return DIR_OF[h][nxt[0]]
        target, edit = self.plan.pop(0)
        if edit is not None:
            self.tree[edit[0]], self.tree[edit[1]] = 0, 1
        assert present(self.tree, h, target)
        return DIR_OF[h][target]
