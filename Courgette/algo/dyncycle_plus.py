"""Stratégie 2+ — version améliorée de DynamicCycleSearch (hamilton.py).

Même principe et même garantie de victoire : le corps reste un segment contigu
d'un cycle hamiltonien, réparé à la volée par des « backbites », et une
recherche A* dans l'espace (cycle, tête) trouve la suite de coups qui mange la
pomme le plus vite. Mesures sur la version d'origine : en milieu de partie, seules
27 à 58 % des recherches atteignent la pomme avant la limite de temps, et 65 % du
temps de calcul part dans des réparations d'essai (inverser un morceau du tableau
du cycle puis l'annuler). Améliorations :

1. Réparations virtuelles. Toutes les inversions d'un même pas commencent juste
   après la tête (indice relatif 1) : ce sont des inversions de PRÉFIXE du chemin
   libre. La case à l'indice relatif r après les inversions J1..Jk se calcule en
   O(k) (r -> J - r si 1 <= r <= J - 1), sans toucher au tableau. On n'inverse
   réellement que dans la copie du noeud créé.
2. Clé de doublon compacte : bytes(order) (les cases < 256 tiennent sur un octet)
   au lieu de tuple(order).
3. Recherche « anytime » : tant que le plan en cours n'atteint pas la pomme, on
   relance une recherche tous les `replan_every` coups et on ne remplace le plan
   que si le nouveau est strictement meilleur (coups + distance restante sur le
   cycle). Cette mesure décroît à chaque coup : pas d'oscillation possible.
"""
import gc
import heapq
import time

from core import NN, NB, DIR_OF, build_cycle
from heuristic import TD


def _rev(order, pos, start, count):
    lo, hi = start, start + count - 1
    for _ in range(count // 2):
        a, b = order[lo % NN], order[hi % NN]
        order[lo % NN], pos[b] = b, lo % NN
        order[hi % NN], pos[a] = a, hi % NN
        lo += 1
        hi -= 1


class DynamicCycleSearchPlus:
    name = "search-plus"

    def __init__(self, depth=5, cap=3000, budget=0.15, replan_every=4, weight=1.0, dedup="cells"):
        self.depth = depth
        self.dedup = dedup                   # "cells" : (tête, cases visitées) ; "order" : cycle exact
        self.weight = weight                 # A* pondéré : f = g + weight * h (1 = optimal)
        self.cap = cap
        self.budget = budget                 # secondes max par recherche (1 coup = 0,2 s)
        self.replan_every = replan_every     # 0 = jamais (une recherche par pomme, comme l'original)

    def reset(self, g):
        self.order = build_cycle()
        self.pos = [0] * NN
        for i, c in enumerate(self.order):
            self.pos[c] = i
        self.plan = []
        self.plan_apple = None
        self.plan_e = 0                      # coups restants + distance sur le cycle à la fin du plan
        self.plan_reached = False
        self.since = 0

    # ------------------------------------------------------------ réparations
    def _make_start(self, order, pos, ih, dt, target, depth, revs):
        """Liste de J (inversions de préfixe [1, J-1]) qui rendent `target`
        successeur de la tête, ou None. Rien n'est modifié."""
        # case à l'indice relatif 1 après les inversions `revs`
        r = 1
        for J in reversed(revs):
            if r <= J - 1:
                r = J - r
        s = order[(ih + r) % NN]
        if s == target:
            return revs
        if depth == 0:
            return None
        cands = []
        for y in NB[s]:
            ry = (pos[y] - ih) % NN
            for J in revs:
                if 1 <= ry <= J - 1:
                    ry = J - ry
            if 3 <= ry <= dt - 1:
                r = ry - 1                          # nouvelle case en 1 = ancienne case en ry - 1
                for J in reversed(revs):
                    if 1 <= r <= J - 1:
                        r = J - r
                ns = order[(ih + r) % NN]
                if depth == 1 and ns != target:
                    continue
                cands.append((TD[ns][target], ry))
        cands.sort()
        for _, J in cands:
            res = self._make_start(order, pos, ih, dt, target, depth - 1, revs + [J])
            if res is not None:
                return res
        return None

    # --------------------------------------------------------------- recherche
    def _search(self, g):
        """Retourne (plan, e, atteint) : plan = [(cible, [(début, nb), ...]), ...]."""
        apple = g.apple
        order0, pos0 = self.order[:], self.pos[:]
        ih0, it0 = pos0[g.head], pos0[g.tail]
        # noeud : [order, pos, ih, it, grow, steps, parent, cible, inversions]
        nodes = [[order0, pos0, ih0, it0, g.grow, 0, -1, None, None, 0]]
        heap = [(TD[g.head][apple], 0, 0)]
        seen = set()
        best_e, best_id = (pos0[apple] - ih0) % NN, 0
        tick = 1
        expanded = 0
        goal_id = -1
        cells_key = self.dedup == "cells"
        deadline = time.perf_counter() + self.budget
        while heap and expanded < self.cap:
            if time.perf_counter() > deadline:
                break
            _, _, nid = heapq.heappop(heap)
            node = nodes[nid]
            if node[7] == apple:
                goal_id = nid
                break
            order, pos, ih, it, grow, steps = node[:6]
            mask = node[9]
            expanded += 1
            dt = (it - ih) % NN
            h = order[ih]
            nih = (ih + 1) % NN
            nit = it if grow else (it + 1) % NN
            s2 = steps + 1
            for n in NB[h]:
                J = (pos[n] - ih) % NN
                if not 1 <= J <= dt - 1:
                    continue
                m2 = mask | (1 << n)
                if cells_key:
                    key = (n, m2)
                    if key in seen:
                        continue
                revs = self._make_start(order, pos, ih, dt, n, self.depth, [])
                if revs is None:
                    continue
                o2, p2 = order[:], pos[:]
                ops = [(ih + 1, Jr - 1) for Jr in revs]
                for st, cnt in ops:
                    _rev(o2, p2, st, cnt)
                if not cells_key:
                    key = (nih, nit, bytes(o2))
                    if key in seen:
                        continue
                seen.add(key)
                nodes.append([o2, p2, nih, nit, False, s2, nid, n, ops, m2])
                cid_ = len(nodes) - 1
                if n == apple:
                    heapq.heappush(heap, (s2, tick, cid_))
                else:
                    e = s2 + (p2[apple] - nih) % NN
                    if e < best_e:
                        best_e, best_id = e, cid_
                    heapq.heappush(heap, (s2 + self.weight * TD[n][apple], tick, cid_))
                tick += 1
        reached = goal_id >= 0
        if reached:
            best_e = nodes[goal_id][5]
        else:
            goal_id = best_id
        plan = []
        while goal_id > 0:
            nd = nodes[goal_id]
            plan.append((nd[7], nd[8]))
            goal_id = nd[6]
        plan.reverse()
        return plan, best_e, reached

    def choose(self, g):
        h = g.head
        new_apple = self.plan_apple != g.apple
        want = (new_apple or not self.plan
                or (not self.plan_reached and self.replan_every
                    and self.since >= self.replan_every))
        if want:
            gc_was = gc.isenabled()
            gc.disable()                             # pas de pause du ramasse-miettes pendant la recherche
            try:
                plan, e, reached = self._search(g)
            finally:
                if gc_was:
                    gc.enable()
            self.since = 0
            keep_old = (not new_apple and self.plan and not reached
                        and e >= self.plan_e)
            if not keep_old:
                self.plan, self.plan_e, self.plan_reached = plan, e, reached
                self.plan_apple = g.apple
            if not self.plan:                        # rien de mieux : suivre le cycle
                self.plan = [(self.order[(self.pos[h] + 1) % NN], [])]
                self.plan_e = (self.pos[g.apple] - self.pos[h]) % NN
                self.plan_reached = False
        target, ops = self.plan.pop(0)
        self.plan_e -= 1
        self.since += 1
        for st, cnt in ops:
            _rev(self.order, self.pos, st, cnt)
        assert self.order[(self.pos[h] + 1) % NN] == target
        return DIR_OF[h][target]
