"""Stratégie 2 — Cycles hamiltoniens (victoire garantie) avec raccourcis.

Les classes, de la plus simple à la plus rapide :

FixedCycle           suit un cycle fixe passant par les 225 cases (~ NN^2/4 coups).
PHC                  « Perturbed Hamiltonian Cycle » (Tapsell) : raccourcis vers un
                     voisin plus loin sur le cycle, sans dépasser pomme ni queue.
                     Les cases sautées restent libres derrière la tête : d'où une
                     marge de sécurité (sans elle, 1 partie sur 20 était perdue).
DynamicCycleGreedy   le cycle est réparé à la volée (2-opt) pour que le corps reste
                     un segment contigu ; un seul coup d'avance.
DynamicCycle         idem, en rejouant un chemin A* complet vers la pomme.
DynamicCycleSearch   idem, avec une recherche A* dans l'espace (cycle, tête) : c'est
                     la stratégie jouée par snake-algo.py.
"""
import heapq
import time

from core import NN, NB, OPP, DIR_OF, build_cycle
from heuristic import TD, astar, vacate_times


class _CycleBase:
    def reset(self, g):
        self.order = build_cycle()
        self.pos = [0] * NN
        for i, c in enumerate(self.order):
            self.pos[c] = i

    def _succ(self, c):
        return self.order[(self.pos[c] + 1) % NN]


class FixedCycle(_CycleBase):
    name = "cycle"

    def choose(self, g):
        return DIR_OF[g.head][self._succ(g.head)]


class PHC(_CycleBase):
    """Attention : les cases sautées restent libres DERRIÈRE la tête ; si on en laisse
    trop, quelques pommes mangées d'affilée (la queue est figée à chaque
    croissance) peuvent coincer la tête contre sa propre queue. D'où `margin` :
    on exige au moins `margin` cases libres devant la tête après un raccourci."""
    name = "phc"

    def __init__(self, margin=8):
        self.margin = margin

    def choose(self, g):
        h, apple = g.head, g.apple
        pos = self.pos
        i = pos[h]
        dt = (pos[g.tail] - i) % NN          # distance (sur le cycle) tête -> queue
        da = (pos[apple] - i) % NN           # distance tête -> pomme
        best_k, best_n = 1, self._succ(h)
        for n in NB[h]:
            k = (pos[n] - i) % NN
            if k <= best_k or k > da:
                continue
            # ne pas atteindre/dépasser la queue ; après avoir mangé la queue ne
            # bouge pas au coup suivant : garder une case libre devant la tête
            if k > dt - 1 - self.margin - (1 if n == apple else 0):
                continue
            best_k, best_n = k, n
        return DIR_OF[h][best_n]


class DynamicCycleGreedy(_CycleBase):
    """Un seul coup d'avance : prend le voisin qui réduit le plus la distance
    (sur le cycle réparé) jusqu'à la pomme."""
    name = "dyncycle-greedy"

    def _reverse(self, start, count):
        """Inverse `count` éléments du cycle à partir de l'indice `start` (circulaire)."""
        order, pos = self.order, self.pos
        lo, hi = start, start + count - 1
        for _ in range(count // 2):
            a, b = order[lo % NN], order[hi % NN]
            order[lo % NN], pos[b] = b, lo % NN
            order[hi % NN], pos[a] = a, hi % NN
            lo += 1
            hi -= 1

    def choose(self, g):
        h, apple = g.head, g.apple
        order, pos = self.order, self.pos
        i = pos[h]
        s = order[(i + 1) % NN]
        dt = (pos[g.tail] - i) % NN
        a = (pos[apple] - i) % NN
        best_d, best_n, best_k = a - 1, s, 1
        for n in NB[h]:
            k = (pos[n] - i) % NN
            if k < 2 or k > a:
                continue
            if k > dt - 1 - (1 if n == apple else 0):
                continue
            # 2-opt : remplacer les arêtes (h,s) et (n,y) par (h,n) et (s,y), ce
            # qui inverse le segment s..n. Valable seulement si s et y sont voisines.
            y = order[(i + k + 1) % NN]
            if DIR_OF[s][y] < 0:
                continue
            d_after = k - a                  # distance sur le NOUVEAU cycle jusqu'à la pomme
            if d_after < best_d:
                best_d, best_n, best_k = d_after, n, k
        if best_k >= 2:
            self._reverse(i + 1, best_k)
        return DIR_OF[h][best_n]


class DynamicCycle(_CycleBase):
    """Cycle dynamique planifié (idée de DHCR / Haidet, en version 2-opt en chaîne).

    Invariant : le corps est TOUJOURS un segment contigu du cycle. Les cases libres
    forment donc un chemin hamiltonien Q (de la case qui suit la tête jusqu'à celle
    qui précède la queue), et suivre le cycle est toujours possible : victoire
    garantie, comme pour le cycle fixe.

    Chaque coup :
      1. A* (avec obstacles temporels) donne le plus court chemin P vers la pomme ;
      2. on « rejoue » P sur le cycle : pour que la tête puisse aller sur P[i], cette
         case doit devenir la successeuse de la tête dans le cycle. On y arrive par
         des « backbites » (2-opt sur l'extrémité de Q : relier Q[0] à un voisin
         Q[j] et couper l'arête (Q[j-1], Q[j])), en chaîne jusqu'à `depth` ;
      3. si tout P est réalisable, on suit P (les réparations du 1er pas sont
         appliquées pour de vrai) ; sinon on tente les autres 1ers pas ; en dernier
         recours on suit le cycle tel quel (toujours sûr).
    Le plan restant est mémorisé pour ne pas osciller d'un coup à l'autre."""
    name = "dyncycle"

    def __init__(self, depth=3):
        self.depth = depth

    def reset(self, g):
        super().reset(g)
        self.plan = []
        self.plan_apple = None

    def _reverse(self, start, count):
        order, pos = self.order, self.pos
        lo, hi = start, start + count - 1
        for _ in range(count // 2):
            a, b = order[lo % NN], order[hi % NN]
            order[lo % NN], pos[b] = b, lo % NN
            order[hi % NN], pos[a] = a, hi % NN
            lo += 1
            hi -= 1

    def _make_start(self, ih, dt, target, depth):
        """Fait de `target` la case qui suit la tête dans le cycle. Retourne la liste
        des inversions appliquées (état modifié en place) ou None (état intact)."""
        order, pos = self.order, self.pos
        s = order[(ih + 1) % NN]
        if s == target:
            return []
        if depth == 0:
            return None
        cands = []
        for y in NB[s]:
            J = (pos[y] - ih) % NN
            if 3 <= J <= dt - 1:
                ns = order[(ih + J - 1) % NN]
                if depth == 1 and ns != target:
                    continue
                cands.append((TD[ns][target], J))
        cands.sort()
        for _, J in cands:
            self._reverse(ih + 1, J - 1)
            r = self._make_start(ih, dt, target, depth - 1)
            if r is not None:
                return [(ih + 1, J - 1)] + r
            self._reverse(ih + 1, J - 1)
        return None

    def _realize(self, g, path):
        """Rejoue `path` sur le cycle. Retourne les inversions du 1er pas (elles
        restent appliquées) ou None (cycle inchangé)."""
        pos = self.pos
        ih, it, grow = pos[g.head], pos[g.tail], g.grow
        done = []
        first = None
        for step, target in enumerate(path):
            dt = (it - ih) % NN
            ops = self._make_start(ih, dt, target, self.depth)
            if ops is None:
                for st, cnt in reversed(done):
                    self._reverse(st, cnt)
                return None
            done.extend(ops)
            if step == 0:
                first = list(ops)
            ih = (ih + 1) % NN
            if grow:
                grow = False
            else:
                it = (it + 1) % NN
        # on ne garde que les réparations du premier pas
        for st, cnt in reversed(done):
            self._reverse(st, cnt)
        for st, cnt in first:
            self._reverse(st, cnt)
        return first

    def choose(self, g):
        h = g.head
        vac = vacate_times(g)
        for c in g.body:
            vac[c] += 1                       # la case de la queue n'est pas « réparable » : +1
        back = OPP[g.dir]
        firsts = [NB[h][d] for d in range(4) if d != back and vac[NB[h][d]] == 0]
        # 1) plan mémorisé
        if self.plan and self.plan_apple == g.apple and self.plan[0] in NB[h]:
            if self._realize(g, self.plan) is not None:
                n = self.plan.pop(0)
                return DIR_OF[h][n]
        self.plan = []
        self.plan_apple = g.apple
        # 2) nouveau plan : A* puis les autres premiers pas
        tried = []
        p = astar(vac, firsts, g.apple)
        if p:
            tried.append(p)
            alts = []
            for f in firsts:
                if f != p[0]:
                    q = astar(vac, [f], g.apple)
                    if q:
                        alts.append(q)
            alts.sort(key=len)
            tried.extend(alts)
        for q in tried:
            if self._realize(g, q) is not None:
                self.plan = q[1:]
                return DIR_OF[h][q[0]]
        # 3) suivre le cycle tel quel
        return DIR_OF[h][self._succ(h)]


class DynamicCycleSearch(_CycleBase):
    """Cycle dynamique + recherche A* dans l'espace des (cycle, position de la tête).

    Même invariant que DynamicCycle (corps = segment contigu du cycle => victoire
    garantie), mais au lieu de rejouer UN chemin A*, on cherche directement, parmi
    tous les enchaînements « aller sur un voisin libre, en réparant le cycle si
    nécessaire », celui qui mange la pomme le plus vite.

      - noeud  = (cycle réparé, tête, queue, nb de coups)
      - arête  = aller sur un voisin n de la tête ; si n n'est pas la successeuse de
                 la tête dans le cycle, `_make_start` répare le cycle (backbites)
      - coût   = coups joués ; heuristique = distance torique jusqu'à la pomme
    Si `cap` noeuds ne suffisent pas, on prend le noeud minimisant
    (coups joués + distance SUR LE CYCLE jusqu'à la pomme) : c'est toujours au
    moins aussi bon que de suivre le cycle, donc on progresse à chaque plan.
    Le plan est mémorisé : une recherche par pomme, pas une par coup."""
    name = "dyncycle-search"

    def __init__(self, depth=3, cap=600, budget=0.18):
        self.depth = depth
        self.cap = cap
        self.budget = budget                 # secondes max par recherche (un coup = 1/GAME_SPEED = 0.2 s)

    def reset(self, g):
        super().reset(g)
        self.plan = []
        self.plan_apple = None

    @staticmethod
    def _rev(order, pos, start, count):
        lo, hi = start, start + count - 1
        for _ in range(count // 2):
            a, b = order[lo % NN], order[hi % NN]
            order[lo % NN], pos[b] = b, lo % NN
            order[hi % NN], pos[a] = a, hi % NN
            lo += 1
            hi -= 1

    def _make_start(self, order, pos, ih, dt, target, depth):
        """Rend `target` successeur de la tête. Retourne la liste des inversions
        appliquées EN PLACE, ou None (tableaux intacts)."""
        s = order[(ih + 1) % NN]
        if s == target:
            return []
        if depth == 0:
            return None
        cands = []
        for y in NB[s]:
            J = (pos[y] - ih) % NN
            if 3 <= J <= dt - 1:
                ns = order[(ih + J - 1) % NN]
                if depth == 1 and ns != target:
                    continue
                cands.append((TD[ns][target], J))
        cands.sort()
        for _, J in cands:
            self._rev(order, pos, ih + 1, J - 1)
            r = self._make_start(order, pos, ih, dt, target, depth - 1)
            if r is not None:
                return [(ih + 1, J - 1)] + r
            self._rev(order, pos, ih + 1, J - 1)
        return None

    def _search(self, g):
        apple = g.apple
        order0, pos0 = self.order[:], self.pos[:]
        ih0, it0 = pos0[g.head], pos0[g.tail]
        # noeud : [order, pos, ih, it, grow, steps, parent, target, ops]
        nodes = [[order0, pos0, ih0, it0, g.grow, 0, -1, None, None]]
        heap = [(TD[g.head][apple], 0, 0)]
        seen = set()
        cyc0 = (pos0[apple] - ih0) % NN
        best_e, best_id = cyc0, 0
        tick = 1
        expanded = 0
        goal_id = -1
        deadline = time.perf_counter() + self.budget
        while heap and expanded < self.cap:
            if expanded & 3 == 3 and time.perf_counter() > deadline:
                break
            f, _, nid = heapq.heappop(heap)
            node = nodes[nid]
            if node[7] == apple:                      # but atteint (donc optimal parmi l'exploré)
                goal_id = nid
                break
            order, pos, ih, it, grow, steps = node[:6]
            expanded += 1
            dt = (it - ih) % NN
            h = order[ih % NN]
            nih = (ih + 1) % NN
            nit = it if grow else (it + 1) % NN
            for n in NB[h]:
                J = (pos[n] - ih) % NN
                if not 1 <= J <= dt - 1:
                    continue
                ops = self._make_start(order, pos, ih, dt, n, self.depth)
                if ops is None:
                    continue
                o2, p2 = order[:], pos[:]
                for st, cnt in reversed(ops):        # restaure le noeud parent
                    self._rev(order, pos, st, cnt)
                key = (nih, nit, tuple(o2))
                if key in seen:
                    continue
                seen.add(key)
                s2 = steps + 1
                nodes.append([o2, p2, nih, nit, False, s2, nid, n, ops])
                cid_ = len(nodes) - 1
                if n == apple:
                    heapq.heappush(heap, (s2, tick, cid_))
                else:
                    e = s2 + (p2[apple] - nih) % NN
                    if e < best_e:
                        best_e, best_id = e, cid_
                    heapq.heappush(heap, (s2 + TD[n][apple], tick, cid_))
                tick += 1
        if goal_id < 0:
            goal_id = best_id
        plan = []
        while goal_id > 0:
            nd = nodes[goal_id]
            plan.append((nd[7], nd[8]))
            goal_id = nd[6]
        plan.reverse()
        return plan

    def choose(self, g):
        h = g.head
        if not self.plan or self.plan_apple != g.apple:
            self.plan = self._search(g)
            self.plan_apple = g.apple
            if not self.plan:                       # rien de mieux : suivre le cycle
                self.plan = [(self._succ(h), [])]
        target, ops = self.plan.pop(0)
        for st, cnt in ops:
            self._rev(self.order, self.pos, st, cnt)
        assert self._succ(h) == target
        return DIR_OF[h][target]
