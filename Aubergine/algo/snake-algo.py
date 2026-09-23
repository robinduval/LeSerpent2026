"""
Snake-algo : bot « cycle hamiltonien + raccourcis » pour le vrai jeu (serpent-algo.py).

Le fichier du jeu n'est PAS modifié ni recopié : on l'importe et on lance sa vraie
fonction main(). Le bot s'y branche par sous-classes de Snake et d'Apple (le jeu les
cherche dans ses globals) : à chaque déplacement, le bot choisit la direction avec le
set_direction() du jeu, puis Snake.move() du jeu s'exécute tel quel.

Algorithme (en couches) :
  1. Cycle hamiltonien sur le tore : idx[case] dans [0, N), rel(c) = (idx[c] - idx[tête]) mod N.
  2. Invariant : en avançant depuis la queue le long du cycle, on croise le corps dans l'ordre.
     Toutes les cases de rel 1 à rel(queue)-1 sont donc vides. On n'ose que rel <= limite.
  3. Plus court chemin vers la pomme dans le graphe sans cycle « rel croissant », par
     programmation dynamique en O(N), recalculé à chaque coup.
  4. (option --dynamic) Cycle dynamique : à chaque nouvelle pomme, la partie libre du cycle
     (entre la tête et la queue) est remplacée par un autre chemin hamiltonien de ces cases
     qui place la pomme plus tôt. Validation stricte, sinon l'ancien cycle est gardé.
  5. (option --libre) Raccourcis à travers toutes les cases libres de la zone : un plan de la tête à la
     pomme, plus court que celui de la couche 3, n'est suivi que si un cycle hamiltonien valide (corps
     dans l'ordre) a pu être reconstruit pour l'état final. Les cases sautées ne deviennent pas des trous.

Usage :
  python snake-algo.py                      # fenêtre, vitesse 30 images/s
  python snake-algo.py --fps 5              # vitesse d'origine du jeu
  python snake-algo.py --fps 0              # aussi vite que possible
  python snake-algo.py --direct             # mode risqué : plus court chemin réel, sans garantie
  python snake-algo.py --headless --games 20 --jobs 8 --debug   # vrai jeu sans fenêtre
"""
import argparse
import heapq
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

INF = 10 ** 9


class _BudgetExceeded(Exception):
    """Budget de calcul de la couche 4 dépassé : on garde l'ancien cycle."""


# --- COUCHE 1 : CYCLE HAMILTONIEN ---

def _zigzag(width, height):
    """Colonne 0 réservée au retour, aller-retours sur les colonnes 1..width-1.
    Sans passage de mur si height est pair ; si height est impair (et width impair),
    la dernière ligne finit en (width-1, height-1) et se referme par un mur."""
    cycle = [(x, 0) for x in range(width)]
    for y in range(1, height):
        xs = range(width - 1, 0, -1) if y % 2 == 1 else range(1, width)
        cycle += [(x, y) for x in xs]
    cycle += [(0, y) for y in range(height - 1, 0, -1)]
    return cycle


def build_cycle(width, height, wrap=True):
    """Construit un cycle hamiltonien de la grille width x height (liste de (x, y))."""
    if width < 3 or height < 3:
        raise ValueError("Grille trop petite (minimum 3x3).")
    if height % 2 == 0:
        cycle = _zigzag(width, height)
    elif width % 2 == 0:
        cycle = [(x, y) for (y, x) in _zigzag(height, width)]  # transposée
    elif wrap:
        cycle = _zigzag(width, height)  # deux dimensions impaires : le mur sert d'arête
    else:
        raise ValueError(
            f"Grille {width}x{height} : deux dimensions impaires, pas de cycle hamiltonien "
            "sans traverser les murs (nombre de cases impair)."
        )
    validate_cycle(cycle, width, height, wrap)
    return cycle


def _adjacent(a, b, width, height, wrap):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    if wrap:
        dx, dy = min(dx, width - dx), min(dy, height - dy)
    return dx + dy == 1


def validate_cycle(cycle, width, height, wrap=True):
    """Chaque case une seule fois, cases consécutives voisines (dernière -> première incluse)."""
    if len(cycle) != width * height or len(set(cycle)) != width * height:
        raise ValueError("Cycle invalide : cases manquantes ou en double.")
    for i, cell in enumerate(cycle):
        if not _adjacent(cell, cycle[(i + 1) % len(cycle)], width, height, wrap):
            raise ValueError(f"Cycle invalide : {cell} et {cycle[(i + 1) % len(cycle)]} ne sont pas voisines.")


def _orient_cycle(cycle, width, height, body):
    """Choisit la symétrie/sens du cycle dans lequel le corps de départ (queue -> tête)
    est déjà rangé dans l'ordre du cycle (invariant vrai dès le premier coup)."""
    tail_to_head = [tuple(p) for p in reversed(body)]
    for flip_x in (False, True):
        for flip_y in (False, True):
            base = [((width - 1 - x) if flip_x else x, (height - 1 - y) if flip_y else y) for x, y in cycle]
            for candidate in (base, base[::-1]):
                pos = {c: i for i, c in enumerate(candidate)}
                if all(c in pos for c in tail_to_head) and all(
                    (pos[b] - pos[a]) % len(candidate) == 1 for a, b in zip(tail_to_head, tail_to_head[1:])
                ):
                    return candidate
    raise ValueError("Aucune orientation du cycle ne contient le serpent de départ dans l'ordre.")


# --- COUCHES 2 ET 3 : SÉCURITÉ ET PLUS COURT CHEMIN CONTRAINT ---

class CycleBot:
    """Choisit la direction à chaque coup. Toutes les cases sont des entiers x + y * width."""

    def __init__(self, width, height, initial_body, wrap=True, growth=1, margin=2, debug=False,
                 dynamic=False, dyn_fill=0.0, dyn_budget=20000, tie_holes=True, libre=False, plan_budget=5000, l3=True):
        self.width, self.height, self.wrap = width, height, wrap
        self.n = width * height
        self.growth = growth
        self.margin = margin
        self.debug = debug
        cycle = _orient_cycle(build_cycle(width, height, wrap), width, height, initial_body)
        self.cyc = [x + y * width for x, y in cycle]
        self.idx = [0] * self.n
        self._reindex()
        self.nbrs = [self._neighbors(c) for c in range(self.n)]
        self.nbset = [set(v) for v in self.nbrs]
        self.fallbacks = 0  # nombre de fois où le filet de sécurité final a dû intervenir
        self.violations = []  # messages (mode debug) : invariant rompu, coup illégal ou sans issue
        # couche 4 : cycle dynamique
        self.dynamic = dynamic
        self.dyn_fill = dyn_fill
        self.dyn_budget = dyn_budget
        self.last_apple = None
        self.l4 = {"essais": 0, "acceptes": 0, "gain": 0, "temps": 0.0, "sans_gain": 0, "echec": 0, "budget": 0, "hors_zone": 0}
        self._ops = 0
        self._budget = dyn_budget
        # couche 3 : départage des chemins de même longueur par le nombre de sauts (trous)
        self.tie_holes = tie_holes
        self.l3 = l3  # False : la couche 3 ne fait plus de raccourcis (suit le cycle), seule la couche 5 en fait
        # couche 5 : plans de raccourci à travers les cases libres, cycle reconstruit et validé
        self.libre = libre
        self.plan_budget = plan_budget
        self.plan = None
        self.last_plan_apple = None
        self.l5 = {"essais": 0, "acceptes": 0, "gain": 0, "temps": 0.0, "sans_gain": 0, "echec": 0,
                   "budget": 0, "hors_zone": 0, "abandons": 0}

    def _reindex(self):
        for i, c in enumerate(self.cyc):
            self.idx[c] = i

    def _neighbors(self, c):
        x, y = c % self.width, c // self.width
        out = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if self.wrap:
                nx, ny = nx % self.width, ny % self.height
            elif not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            out.append(nx + ny * self.width)
        return out

    def cell(self, pos):
        return pos[0] + pos[1] * self.width

    def check_invariant(self, body):
        """Mode debug : corps rangé dans l'ordre du cycle, sans collision, longueur == cases occupées."""
        ih = self.idx[self.cell(body[0])]
        rels = [(self.idx[self.cell(p)] - ih) % self.n for p in body]
        assert len({self.cell(p) for p in body}) == len(body), "collision ou case comptée deux fois"
        assert all(rels[i] > rels[i + 1] for i in range(1, len(rels) - 1)), "invariant du cycle rompu"
        validate_cycle([(c % self.width, c // self.width) for c in self.cyc], self.width, self.height, self.wrap)
        assert all(self.idx[c] == i for i, c in enumerate(self.cyc)), "idx désynchronisé du cycle"

    def choose(self, body, pending, apple):
        """Retourne la direction (dx, dy) à donner au serpent."""
        n, idx, cyc, nbrs = self.n, self.idx, self.cyc, self.nbrs
        h = self.cell(body[0])
        ih = idx[h]
        tail_rel = n if len(body) == 1 else (idx[self.cell(body[-1])] - ih) % n
        free_ahead = tail_rel - 1                 # cases rel 1..tail_rel-1 : toutes vides
        if self.plan is not None:                 # plan de la couche 5 en cours d'exécution
            step = self._follow_plan(body, pending)
            if step is not None:
                return step
        if (self.dynamic and apple is not None and apple != self.last_apple
                and len(body) / n >= self.dyn_fill):
            self.last_apple = apple               # nouvelle pomme : on tente de réorganiser le cycle
            self._reorganize(ih, free_ahead, h, self.cell(body[-1]), self.cell(apple))
        p = 1 if pending else 0                   # croissance en attente
        limit_free = free_ahead - p - self.margin
        limit_apple = limit_free - self.growth    # manger ajoute une croissance en attente
        if not self.l3:
            limit_free = limit_apple = 0
        apple_cell = self.cell(apple) if apple else None
        apple_rel = (idx[apple_cell] - ih) % n if apple else None

        if self.libre and apple is not None and apple != self.last_plan_apple:
            self.last_plan_apple = apple          # nouvelle pomme : on cherche un plan de raccourci
            if self._make_plan(body, p, h, ih, free_ahead, apple_cell, apple_rel, limit_apple):
                step = self._follow_plan(body, pending)
                if step is not None:
                    return step

        target = None
        if apple_rel is not None and 1 <= apple_rel <= limit_apple:
            target = self._first_step(h, ih, apple_rel)[0]
        if target is None:
            # pomme trop loin / trop risquée : voisin autorisé au plus grand rel, sinon case suivante du cycle
            best_rel = 0
            for v in nbrs[h]:
                r = (idx[v] - ih) % n
                if 1 <= r <= limit_free and v != apple_cell and r > best_rel:
                    target, best_rel = v, r
        if target is None:
            target = cyc[(ih + 1) % n]

        if not self._is_ok(target, body, p, apple_cell):
            # filet de sécurité (ne doit jamais arriver si l'invariant tient) : coup illégal ou
            # après lequel plus aucun coup ne serait légal (cases vides « trous » derrière la queue)
            self.fallbacks += 1
            if self.debug and len(self.violations) < 5:
                self.violations.append(f"coup refusé par le filet : len={len(body)} pending={p} "
                                       f"free_ahead={free_ahead} rel_pomme={apple_rel} cible_rel={(idx[target] - ih) % n}")
            alternatives = [v for v in nbrs[h] if self._is_ok(v, body, p, apple_cell)]
            if alternatives:
                target = max(alternatives, key=lambda v: (idx[v] - ih) % n)
        return self._direction(body[0], target)

    def _is_ok(self, cell, body, pending, apple_cell):
        """Le coup est légal ET laisse au moins un coup légal ensuite (ou gagne la partie)."""
        if not self._is_legal(cell, body, pending):
            return False
        eaten = cell == apple_cell
        new_body = [[cell % self.width, cell // self.width]] + (body if pending else body[:-1])
        if eaten and len(new_body) == self.n:
            return True  # dernière pomme : victoire
        return any(self._is_legal(v, new_body, 1 if eaten else 0) for v in self.nbrs[cell])

    # --- COUCHE 4 : CYCLE DYNAMIQUE ---

    def _reorganize(self, ih, free_ahead, head, tail, apple):
        """Remplace la partie libre du cycle (cases rel 1..free_ahead, toutes vides) par un autre
        chemin hamiltonien de ces cases qui place la pomme plus tôt. La partie fixe n'est jamais
        touchée. Toute anomalie, échec de validation ou dépassement de budget : ancien cycle gardé."""
        t0 = time.perf_counter()
        n, idx, cyc = self.n, self.idx, self.cyc
        old_rel = (idx[apple] - ih) % n
        if old_rel > free_ahead:
            self.l4["hors_zone"] += 1  # pomme dans un trou de la partie fixe : intouchable
            return
        if free_ahead < 3 or old_rel <= 1:
            return
        free = {cyc[(ih + k) % n] for k in range(1, free_ahead + 1)}
        self.l4["essais"] += 1
        self._ops, self._budget = 0, self.dyn_budget
        try:
            first = self._shortest(head, apple, free, 0)
            if first is None or len(first) >= old_rel:
                self.l4["sans_gain"] += 1
                return  # pas de gain possible : les détours ne font qu'allonger le chemin
            for order in range(4):  # 4 ordres de voisinage : 4 chemins candidats, le premier valide gagne
                path = self._build_path(head, tail, apple, free, order)
                if path is not None and (path.index(apple) + 1) < old_rel:
                    for k, c in enumerate(path, start=1):
                        cyc[(ih + k) % n] = c
                    self._reindex()
                    self.l4["acceptes"] += 1
                    self.l4["gain"] += old_rel - (path.index(apple) + 1)
                    return
            self.l4["echec"] += 1  # aucun des 4 chemins candidats n'a passé la validation
        except _BudgetExceeded:
            self.l4["budget"] += 1
        finally:
            self.l4["temps"] += time.perf_counter() - t0

    # --- COUCHE 5 : PLAN DE RACCOURCI À TRAVERS LES CASES LIBRES ---

    def _make_plan(self, body, p, h, ih, free_ahead, apple_cell, apple_rel, limit_apple):
        """Cherche un chemin S de la tête à la pomme à travers n'importe quelles cases de la zone libre
        (pas seulement en « rel » croissant), plus court que celui de la couche 3. Les cases sautées ne
        deviennent PAS des trous : on reconstruit un cycle complet pour l'état final et on ne garde le plan
        que si ce cycle passe la validation stricte. Sinon, rien n'est modifié."""
        t0 = time.perf_counter()
        n, idx, cyc = self.n, self.idx, self.cyc
        if not 1 <= apple_rel <= free_ahead:
            self.l5["hors_zone"] += 1  # pomme dans un trou de la partie fixe : intouchable
            return False
        self.l5["essais"] += 1
        self._ops, self._budget = 0, self.plan_budget
        try:
            zone = {cyc[(ih + k) % n] for k in range(1, free_ahead + 1)}
            baseline = self._first_step(h, ih, apple_rel)[1] if apple_rel <= limit_apple else INF
            baseline = min(baseline, apple_rel)      # suivre le cycle atteint toujours la pomme en apple_rel coups
            first = self._shortest(h, apple_cell, zone, 0)
            if first is None or len(first) >= baseline:
                self.l5["sans_gain"] += 1
                return False
            length = len(body)
            for order in range(4):  # 4 chemins plus courts équivalents, le premier reconstructible gagne
                steps = self._shortest(h, apple_cell, zone, order)
                if steps is None or len(steps) >= baseline:
                    continue
                new_cyc = self._rebuild(body, p, steps, ih, length)
                if new_cyc is not None:
                    self.plan = {"steps": steps, "i": 0, "cyc": new_cyc}
                    self.l5["acceptes"] += 1
                    self.l5["gain"] += baseline - len(steps)
                    return True
            self.l5["echec"] += 1
        except _BudgetExceeded:
            self.l5["budget"] += 1
        finally:
            self.l5["temps"] += time.perf_counter() - t0
        return False

    def _rebuild(self, body, p, steps, ih, length):
        """Cycle de l'état final (après avoir suivi `steps` et mangé la pomme) ou None.
        L'état final : corps = steps (tête en dernier) + corps actuel privé des `m - p` cases de queue
        qui auront quitté leur place. Cycle final = [partie fixe restante] [tête] steps [chemin libre P'].
        P' recouvre toutes les cases entre la tête actuelle et la nouvelle queue, sauf celles de steps :
        on recolle les tronçons restants (dans les deux sens) par une recherche en profondeur."""
        n, idx, cyc = self.n, self.idx, self.cyc
        m = len(steps)
        pops = m - p
        if pops > length - 1:
            return None  # toute la queue actuelle partirait : cas non traité
        new_tail = self.cell(body[length - 1 - pops])
        tail_rel = (idx[new_tail] - ih) % n
        span = tail_rel - 1                          # cases de rel 1..tail_rel-1 : steps + P'
        if span - m < 1 + self.margin:
            return None                              # trop peu de cases libres après le plan : prudence
        segment = [cyc[(ih + j) % n] for j in range(1, span + 1)]
        taken = set(steps)
        chains, current = [], []
        for c in segment:
            if c in taken:
                if current:
                    chains.append(current)
                    current = []
            else:
                current.append(c)
        if current:
            chains.append(current)
        order = self._relink(chains, steps[-1], new_tail)
        if order is None:
            return None
        free_path = []
        for i, reverse in order:
            free_path += chains[i][::-1] if reverse else chains[i]
        new_cyc = list(cyc)
        for j, c in enumerate(steps + free_path):
            new_cyc[(ih + 1 + j) % n] = c
        return new_cyc if self._state_ok(new_cyc, body, steps, pops, length) else None

    def _relink(self, chains, head_new, tail_new):
        """Ordre et sens des tronçons formant un chemin de voisin de head_new à voisin de tail_new."""
        r = len(chains)
        starts = {}
        for i, c in enumerate(chains):
            starts.setdefault(c[0], []).append((i, False))
            if len(c) > 1:
                starts.setdefault(c[-1], []).append((i, True))
        used = [False] * r
        order = []

        def dfs(current, done):
            self._spend()
            if done == r:
                return tail_new in self.nbset[current]
            for v in self.nbrs[current]:
                for i, reverse in starts.get(v, ()):
                    if not used[i]:
                        used[i] = True
                        order.append((i, reverse))
                        c = chains[i]
                        if dfs(c[0] if reverse else c[-1], done + 1):
                            return True
                        order.pop()
                        used[i] = False
            return False

        return list(order) if dfs(head_new, 0) else None

    def _state_ok(self, new_cyc, body, steps, pops, length):
        """Validation stricte de l'état final : cycle hamiltonien valide (toutes les cases une fois, cases
        consécutives voisines, dernière -> première) et corps rangé dans l'ordre du cycle."""
        n, nbset = self.n, self.nbset
        if len(set(new_cyc)) != n or any(new_cyc[(i + 1) % n] not in nbset[c] for i, c in enumerate(new_cyc)):
            return False
        pos = [0] * n
        for i, c in enumerate(new_cyc):
            pos[c] = i
        final_body = steps[::-1] + [self.cell(b) for b in body[:length - pops]]  # tête d'abord
        ih = pos[final_body[0]]
        rels = [(pos[c] - ih) % n for c in final_body]
        return len(set(final_body)) == len(final_body) and all(rels[i] > rels[i + 1] for i in range(1, len(rels) - 1))

    def _follow_plan(self, body, pending):
        """Prochain pas du plan (vérifié légal dans le vrai jeu) ; au dernier pas, le nouveau cycle est installé."""
        plan = self.plan
        target = plan["steps"][plan["i"]]
        if not self._is_legal(target, body, 1 if pending else 0):
            self.plan = None
            self.l5["abandons"] += 1
            if self.debug and len(self.violations) < 5:
                self.violations.append("plan abandonné : pas illégal")
            return None
        plan["i"] += 1
        if plan["i"] == len(plan["steps"]):
            self.cyc = plan["cyc"]
            self._reindex()
            self.plan = None
        return self._direction(body[0], target)

    def _spend(self, cost=1):
        self._ops += cost
        if self._ops > self._budget:
            raise _BudgetExceeded

    def _shortest(self, start, goal, allowed, order):
        """BFS (plus court chemin) de start (exclu) à goal (inclus) sur des cases de `allowed`."""
        parent = {start: None}
        queue = [start]
        for c in queue:
            self._spend()
            if c == goal:
                break
            nb = self.nbrs[c]
            for v in nb[order % len(nb):] + nb[:order % len(nb)]:
                if v not in parent and v in allowed:
                    parent[v] = c
                    queue.append(v)
        if goal not in parent:
            return None
        path = []
        while goal != start:
            path.append(goal)
            goal = parent[goal]
        return path[::-1]

    def _extend(self, apple, tail, free, used, need_parity, order):
        """Suite du chemin : de la pomme jusqu'à une case libre voisine de la queue, sans réutiliser
        de case, avec un nombre de cases de parité `need_parity` (les détours ajoutent 2 cases à la fois).
        BFS sur (case, parité). Retourne la liste de cases (éventuellement vide) ou None."""
        tail_adjacent = self.nbset[tail] & free
        if need_parity == 0 and apple in tail_adjacent:
            return []
        parent = {}
        queue = []
        for v in self.nbrs[apple]:
            if v in free and v not in used:
                parent[(v, 1)] = None
                queue.append((v, 1))
        for state in queue:
            self._spend()
            c, par = state
            if par == need_parity and c in tail_adjacent:
                path = []
                while state is not None:
                    path.append(state[0])
                    state = parent[state]
                return path[::-1] if len(set(path)) == len(path) else None  # cases répétées : abandon
            nb = self.nbrs[c]
            for v in nb[order % len(nb):] + nb[:order % len(nb)]:
                nxt = (v, par ^ 1)
                if nxt not in parent and v in free and v not in used:
                    parent[nxt] = state
                    queue.append(nxt)
        return None

    def _build_path(self, head, tail, apple, free, order):
        """Chemin tête -> pomme -> queue, complété par des détours ; validé strictement."""
        first = self._shortest(head, apple, free, order)
        if first is None:
            return None
        need_parity = (len(free) - len(first)) % 2
        rest = self._extend(apple, tail, free, set(first), need_parity, order)
        if rest is None:
            return None
        seq = [head] + first + rest + [tail]  # les extrémités fixes permettent aussi des détours
        uncovered = free - set(first) - set(rest)
        if uncovered:
            self._absorb(seq, uncovered, apple, after_apple=True)  # de préférence après la pomme
        if uncovered:
            self._absorb(seq, uncovered, apple, after_apple=False)
        path = seq[1:-1]
        return path if self._valid_path(path, free, head, tail) else None

    def _absorb(self, seq, uncovered, apple, after_apple):
        """Détours : x -> y devient x -> x' -> y' -> y quand x' et y' sont libres, non couvertes,
        et forment un carré avec x et y. Modifie seq et uncovered en place."""
        nbrs, nbset = self.nbrs, self.nbset
        changed = True
        while uncovered and changed:
            changed = False
            pa = seq.index(apple)
            positions = range(len(seq) - 2, pa - 1, -1) if after_apple else range(pa - 1, -1, -1)
            for i in positions:  # ordre décroissant : une insertion ne décale pas les positions restantes
                self._spend()
                x, y = seq[i], seq[i + 1]
                done = False
                for xp in nbrs[x]:
                    if xp in uncovered:
                        for yp in nbrs[y]:
                            if yp != xp and yp in uncovered and yp in nbset[xp]:
                                seq[i + 1:i + 1] = [xp, yp]
                                uncovered.discard(xp)
                                uncovered.discard(yp)
                                changed = done = True
                                break
                    if done:
                        break
                if not uncovered:
                    return

    def _valid_path(self, path, free, head, tail):
        """Exactement les cases libres, une fois chacune, pas entre voisines, raccordé à la tête et à la queue."""
        if len(path) != len(free) or set(path) != free:
            return False
        if path[0] not in self.nbset[head] or path[-1] not in self.nbset[tail]:
            return False
        return all(b in self.nbset[a] for a, b in zip(path, path[1:]))

    def _first_step(self, h, ih, apple_rel):
        """Plus court chemin tête -> pomme dans le graphe « rel croissant » (DP à rebours).
        Retourne (premier pas, longueur). À longueur égale, le nombre total de cases sautées est le
        même ; on départage par le nombre de sauts (moins de trous éparpillés), puis par le plus petit rel."""
        n, idx, cyc, nbrs = self.n, self.idx, self.cyc, self.nbrs
        weigh = 1 if self.tie_holes else 0
        dist = [INF] * (apple_rel + 1)
        jumps = [0] * (apple_rel + 1)
        dist[apple_rel] = 0
        for k in range(apple_rel - 1, 0, -1):
            best = (INF, 0)
            for v in nbrs[cyc[(ih + k) % n]]:
                r = (idx[v] - ih) % n
                if k < r <= apple_rel and dist[r] < INF:
                    cand = (dist[r], jumps[r] + (weigh if r - k > 1 else 0))
                    if cand < best:
                        best = cand
            if best[0] < INF:
                dist[k], jumps[k] = best[0] + 1, best[1]
        target, key, steps = None, (INF, INF, INF), INF
        for v in nbrs[h]:
            r = (idx[v] - ih) % n
            if 1 <= r <= apple_rel and dist[r] < INF:
                cand = (dist[r], jumps[r] + (weigh if r > 1 else 0), r)
                if cand < key:
                    target, key, steps = v, cand, dist[r] + 1
        return target, steps

    def _is_legal(self, cell, body, pending):
        """Vraie règle du jeu : la case ne doit pas être dans le corps une fois la queue retirée."""
        occupied = body[:-1] if not pending else body
        return all(self.cell(p) != cell for p in occupied[1:]) if len(body) > 1 else True

    def _direction(self, head, cell):
        dx, dy = cell % self.width - head[0], cell // self.width - head[1]
        if self.wrap:
            if dx in (self.width - 1, -(self.width - 1)):
                dx = -1 if dx > 0 else 1
            if dy in (self.height - 1, -(self.height - 1)):
                dy = -1 if dy > 0 else 1
        return (dx, dy)


class DirectBot(CycleBot):
    """Mode risqué (--direct) : plus court chemin RÉEL vers la pomme à travers toutes les cases libres du
    tore, sans la contrainte du cycle hamiltonien. Aucune garantie de victoire : le cycle n'est plus
    respecté. Sécurité réglable : 0 = aucune, 1 = après avoir mangé, la tête doit encore pouvoir
    rejoindre la queue. Si aucun chemin sûr n'existe, on se met en sécurité (espace libre maximal) ;
    après `stall` x N coups sans pomme, on prend le chemin même s'il est jugé dangereux."""

    DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # haut, bas, gauche, droite : l'inverse de d est d ^ 1

    def __init__(self, *args, safety=1, stall=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety = safety
        self.stall = stall * self.n
        self.since_apple = 0
        self.seen_apple = None
        self.step_to = [[self._wrap_step(c, d) for d in range(4)] for c in range(self.n)]
        self.risky = 0  # coups pris malgré un chemin jugé dangereux

    def _wrap_step(self, c, d):
        dx, dy = self.DIRS[d]
        return (c % self.width + dx) % self.width + ((c // self.width + dy) % self.height) * self.width

    def check_invariant(self, body):
        pass  # pas de cycle à vérifier dans ce mode

    def choose(self, body, pending, apple):
        p = 1 if pending else 0
        cells = [self.cell(b) for b in body]
        head, apple_cell = cells[0], self.cell(apple)
        if apple != self.seen_apple:
            self.seen_apple, self.since_apple = apple, 0
        self.since_apple += 1

        path = self._path(cells, p, apple_cell, body)
        if path is not None:
            if self.safety == 0 or self._safe_after(cells, path, p):
                return self._direction(body[0], path[0])
            if self.since_apple > self.stall:
                self.risky += 1
                return self._direction(body[0], path[0])
        return self._direction(body[0], self._safest_move(cells, p, apple_cell))

    def _path(self, cells, p, apple_cell, body):
        """Plus court chemin (en coups, puis en virages : trajet droit et cohérent) de la tête à la pomme.
        Une case du corps est utilisable dès que la queue l'aura quittée (coup d + 1 + p, d = rang depuis la queue)."""
        length = len(cells)
        free_time = {c: (length - 1 - i) + 1 + p for i, c in enumerate(cells)}
        head = cells[0]
        d0 = self.DIRS.index(self._direction(body[1], head)) if length > 1 else 3
        start = (head, d0)
        best = {start: (0, 0)}
        parent = {}
        heap = [(0, 0, 0, head, d0)]
        counter = 0
        while heap:
            steps, turns, _, c, d = heapq.heappop(heap)
            if best.get((c, d)) != (steps, turns):
                continue
            if c == apple_cell:
                path, state = [], (c, d)
                while state != start:
                    path.append(state[0])
                    state = parent[state]
                path.reverse()
                return path if len(set(path)) == len(path) else None
            for nd in range(4):
                if nd == d ^ 1:
                    continue
                v = self.step_to[c][nd]
                if free_time.get(v, 0) > steps + 1:
                    continue
                key, cost = (v, nd), (steps + 1, turns + (nd != d))
                if key not in best or cost < best[key]:
                    best[key] = cost
                    parent[key] = (c, d)
                    counter += 1
                    heapq.heappush(heap, (cost[0], cost[1], counter, v, nd))
        return None

    def _final_body(self, cells, path, p):
        full = path[::-1] + cells
        return full[:len(full) - (len(path) - p)]

    def _safe_after(self, cells, path, p):
        final = self._final_body(cells, path, p)
        if len(final) == self.n:
            return True  # dernière pomme : victoire
        return self._tail_reachable(final, 1)[0]

    def _tail_reachable(self, final, pending):
        """(la tête peut rejoindre la queue par des cases libres, taille de l'espace libre atteint)."""
        occupied = set(final)
        head, tail = final[0], final[-1]
        queue = [v for v in self.nbrs[head] if v not in occupied]
        seen = set(queue)
        ok = pending == 0 and tail in self.nbset[head]
        near_tail = self.nbset[tail]
        for c in queue:
            if c in near_tail:
                ok = True
            for v in self.nbrs[c]:
                if v not in occupied and v not in seen:
                    seen.add(v)
                    queue.append(v)
        return ok, len(seen)

    def _distance(self, a, b):
        dx = abs(a % self.width - b % self.width)
        dy = abs(a // self.width - b // self.width)
        return min(dx, self.width - dx) + min(dy, self.height - dy)

    def _safest_move(self, cells, p, apple_cell):
        """Repli : coup légal qui garde la queue accessible, avec le plus d'espace libre, puis le plus près de la pomme."""
        best, best_key = None, None
        for v in self.nbrs[cells[0]]:
            occupied = cells if p else cells[:-1]
            if v in occupied[1:]:
                continue
            final = ([v] + cells) if p else ([v] + cells[:-1])
            eaten = v == apple_cell
            ok, area = self._tail_reachable(final, 1 if eaten else 0)
            key = (ok, area, -self._distance(v, apple_cell))
            if best_key is None or key > best_key:
                best, best_key = v, key
        if best is None:  # aucun coup légal : la partie est perdue quoi qu'on fasse
            best = self.nbrs[cells[0]][0]
        return best


# --- BRANCHEMENT SUR LE VRAI JEU ---

def load_game():
    """Importe serpent-algo.py tel quel (le nom contient un tiret : import par chemin)."""
    path = Path(__file__).with_name("serpent-algo.py")
    spec = importlib.util.spec_from_file_location("serpent_algo", path)
    game = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game)
    return game


class Run:
    """État de la partie en cours (partagé entre Snake, Apple et l'horloge)."""
    def __init__(self):
        self.snake = None
        self.apple = None
        self.victory = False
        self.reported = False


def install_bot(game, args, results):
    """Remplace Snake et Apple du jeu par des sous-classes pilotées par le bot, et l'horloge
    par une horloge qui règle la vitesse et met fin à la partie (mode --headless)."""
    import pygame
    run = Run()
    cap = 10 * game.GRID_SIZE ** 2 * game.GRID_SIZE ** 2  # limite de coups : détecte une boucle infinie
    real_clock = pygame.time.Clock

    class BotSnake(game.Snake):
        def __init__(self):
            super().__init__()
            run.snake, run.victory, run.reported = self, False, False
            self.moves = 0
            self.think = 0.0  # temps de calcul du bot
            options = dict(wrap=True, growth=1, margin=args.margin, debug=args.debug, dynamic=args.dynamic,
                           dyn_fill=args.dyn_fill, dyn_budget=args.dyn_budget, tie_holes=args.tie_holes,
                           libre=args.libre, plan_budget=args.plan_budget, l3=args.l3)
            if args.direct:
                self.bot = DirectBot(game.GRID_SIZE, game.GRID_SIZE, self.body, safety=args.safety,
                                     stall=args.stall, **options)
            else:
                self.bot = CycleBot(game.GRID_SIZE, game.GRID_SIZE, self.body, **options)

        def _debug_check(self):
            if self.bot.plan is not None:
                return  # en plein plan de la couche 5 : le cycle n'est valide qu'à l'état final du plan
            try:
                self.bot.check_invariant(self.body)
            except AssertionError as error:  # consigné, la partie continue
                if len(self.bot.violations) < 5:
                    self.bot.violations.append(f"coup {self.moves} : {error}")

        def move(self):
            if args.debug:
                self._debug_check()
            t0 = time.perf_counter()
            direction = self.bot.choose(self.body, self.grow_pending, run.apple.position)
            self.think += time.perf_counter() - t0
            self.set_direction(direction)
            super().move()
            self.moves += 1
            if args.debug and not self.is_game_over():
                self._debug_check()

    class BotApple(game.Apple):
        def __init__(self, snake_body):
            super().__init__(snake_body)
            run.apple = self

        def relocate(self, snake_body):
            moved = super().relocate(snake_body)
            if not moved:
                run.victory = True
            return moved

    class BotClock:
        def __init__(self):
            self._clock = real_clock()

        def tick(self, _fps=0):
            s = run.snake
            if s is not None and not run.reported:
                outcome = "victoire" if run.victory else ("mort" if s.is_game_over() else None)
                if outcome is None and s.moves > cap:
                    outcome = "boucle"
                if outcome:
                    run.reported = True
                    results.append({"issue": outcome, "score": s.score, "coups": s.moves,
                                    "longueur": len(s.body), "replis": s.bot.fallbacks,
                                    "violations": list(s.bot.violations), "calcul": s.think, "l4": dict(s.bot.l4),
                                    "l5": dict(s.bot.l5)})
                    if not args.headless:
                        print(f"{outcome.upper()} : score {s.score}, {s.moves} coups, longueur {len(s.body)}")
                    if args.headless:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
            return self._clock.tick(args.fps)

    game.Snake, game.Apple = BotSnake, BotApple
    pygame.time.Clock = BotClock


def play(args, games=1, seed=None):
    """Lance `games` fois la vraie fonction main() du jeu ; retourne la liste des résultats."""
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    game = load_game()
    results = []
    install_bot(game, args, results)
    for i in range(games):
        if seed is not None:
            random.seed(seed + i)
        game.main()
    return results


def _worker(job):
    args, games, seed = job
    return play(args, games, seed)


def summarize(results, elapsed):
    n = len(results)
    wins = sum(r["issue"] == "victoire" for r in results)
    print(f"\n{n} parties du vrai jeu en {elapsed:.1f}s : {wins} victoires ({100 * wins / n:.1f} %), "
          f"{sum(r['issue'] == 'mort' for r in results)} morts, {sum(r['issue'] == 'boucle' for r in results)} boucles, "
          f"{sum(r['replis'] for r in results)} replis de sécurité")
    scores = [r["score"] for r in results]
    coups = [r["coups"] for r in results]
    print(f"score  : min {min(scores)}, moyenne {sum(scores) / n:.1f}, max {max(scores)}")
    print(f"coups  : min {min(coups)}, moyenne {sum(coups) / n:.0f}, max {max(coups)}")
    print(f"coups par pomme (moyenne) : {sum(coups) / max(1, sum(scores)):.1f}")
    print(f"calcul du bot : {1000 * sum(r['calcul'] for r in results) / max(1, sum(coups)):.3f} ms par coup")
    for nom, cle in (("couche 4", "l4"), ("couche 5", "l5")):
        c = {k: sum(r[cle][k] for r in results) for k in results[0][cle]}
        if c["essais"] or c["hors_zone"]:
            print(f"{nom} : {c['acceptes']}/{c['essais']} plans acceptés, gain moyen "
                  f"{c['gain'] / max(1, c['acceptes']):.1f}, {1000 * c['temps'] / max(1, c['essais']):.2f} ms par essai "
                  f"(sans gain possible {c['sans_gain']}, échec {c['echec']}, budget dépassé {c['budget']}, "
                  f"pomme hors zone libre {c['hors_zone']}" + (f", abandons {c['abandons']}" if cle == "l5" else "") + ")")
    for i, r in enumerate(results):
        for message in r["violations"]:
            print(f"  [debug] partie {i} : {message}")


def main():
    parser = argparse.ArgumentParser(description="Bot hamiltonien + raccourcis sur le vrai jeu Snake.")
    parser.add_argument("--fps", type=int, default=30, help="images/s (5 = vitesse d'origine, 0 = illimité)")
    parser.add_argument("--margin", type=int, default=2, help="marge de sécurité de la limite de raccourci")
    parser.add_argument("--headless", action="store_true", help="sans fenêtre, enchaîne --games parties puis résume")
    parser.add_argument("--games", type=int, default=1, help="nombre de parties (avec --headless)")
    parser.add_argument("--jobs", type=int, default=1, help="processus parallèles (avec --headless)")
    parser.add_argument("--seed", type=int, default=None, help="graine aléatoire de la première partie")
    parser.add_argument("--debug", action="store_true", help="vérifie cycle et invariant à chaque coup")
    parser.add_argument("--dynamic", action="store_true", help="active la couche 4 (cycle dynamique)")
    parser.add_argument("--dyn-fill", type=float, default=0.0, help="taux de remplissage minimal (0 à 1) de la couche 4")
    parser.add_argument("--dyn-budget", type=int, default=20000, help="budget d'opérations de la couche 4 par pomme")
    parser.add_argument("--tie-holes", action=argparse.BooleanOptionalAction, default=True,
                        help="départage les chemins de même longueur par le moins de sauts (trous)")
    parser.add_argument("--l3", action=argparse.BooleanOptionalAction, default=True,
                        help="raccourcis de la couche 3 (--no-l3 : la couche 3 suit le cycle sans raccourci)")
    parser.add_argument("--direct", action="store_true",
                        help="mode RISQUÉ : plus court chemin réel, sans cycle hamiltonien ni garantie de victoire")
    parser.add_argument("--safety", type=int, default=1, choices=(0, 1),
                        help="sécurité du mode --direct : 0 aucune, 1 la tête doit pouvoir rejoindre la queue")
    parser.add_argument("--stall", type=int, default=3,
                        help="mode --direct : après STALL x N coups sans pomme, on prend le chemin même dangereux")
    parser.add_argument("--libre", action="store_true", help="active la couche 5 (raccourcis à travers les cases libres)")
    parser.add_argument("--plan-budget", type=int, default=5000, help="budget d'opérations de la couche 5 par pomme")
    args = parser.parse_args()

    if not args.headless:
        play(args)
        return
    args.fps = 0
    start = time.time()
    seed = 0 if args.seed is None else args.seed
    if args.jobs <= 1:
        results = play(args, args.games, seed)
    else:
        import multiprocessing
        chunks = [len(range(j, args.games, args.jobs)) for j in range(args.jobs)]
        offsets = [sum(chunks[:j]) for j in range(args.jobs)]
        with multiprocessing.Pool(args.jobs) as pool:
            parts = pool.map(_worker, [(args, c, seed + o) for c, o in zip(chunks, offsets) if c])
        results = [r for part in parts for r in part]
    summarize(results, time.time() - start)
    sys.exit(0 if all(r["issue"] == "victoire" for r in results) else 1)


if __name__ == "__main__":
    main()
