"""Solveur du Serpent : plus court chemin (Dijkstra temporel) + certificat de survie.

Règles modélisées (identiques à serpent-algo.py) :
  - la grille est un tore (les murs se traversent) ;
  - la queue est retirée AVANT le test de collision : entrer dans la case de la
    queue est légal, sauf si le serpent vient de manger (grow_pending) ;
  - la seule mort possible est l'auto-morsure.

Idées clés :
  1. Temps de libération : le segment body[i] quitte sa case au coup L - i + g
     (L = longueur, g = 1 si le serpent doit encore grandir). Une case du corps
     devient donc franchissable si on y arrive assez tard. Le Dijkstra en tient
     compte, ce qui ouvre des chemins qu'un BFS « corps = mur » ne voit pas.
  2. Certificat de survie (is_safe) : un état est sûr s'il existe un chemin par
     des cases libres jusqu'à un segment body[j] atteint au coup d >= L - j + g.
     Le serpent peut alors boucler indéfiniment derrière son propre corps.
     Cette propriété se conserve d'un coup à l'autre en suivant le chemin : on
     ne joue que des coups qui mènent à un état certifié, donc on ne meurt pas.
  3. On ne mange la pomme que si l'état APRÈS le repas est certifié. Sinon on
     cherche un détour, et à défaut on temporise en restant collé au corps.
"""
from collections import deque
import heapq
import random

UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)

MODES = ("POMME", "DETOUR", "RECHERCHE", "CLOTURE", "QUEUE", "SURVIE", "NAIF")


class Solver:
    def __init__(self, grid_size, strategy="safe", seed=None,
                 search_budget=300, search_cooldown=6, stall_noise=0.2, eat_margin=1,
                 close_max_free=30):
        self.n = grid_size
        self.strategy = strategy  # "safe" (notre algo) ou "naive" (plus court chemin seul)
        self.rng = random.Random(seed)
        self.search_budget = search_budget      # états explorés au maximum par recherche
        self.search_cooldown = search_cooldown  # coups d'attente après une recherche vaine
        self.stall_noise = stall_noise          # proba. d'un coup sûr au hasard en temporisation
        self.eat_margin = eat_margin            # marge de survie exigée après un repas
        self.close_max_free = close_max_free    # clôture de fin de partie si <= N cases libres (0 = jamais)
        self._cycle = None      # circuit de clôture : cellule -> cellule suivante
        self.plan = []   # cases prévues (pour l'affichage)
        self.mode = ""
        self._pending = []      # plan de recherche en cours
        self._pending_apple = None
        self._cooldown = 0
        # Voisins toriques précalculés : cellule -> [(voisin, direction)]
        self.nbrs = {}
        for x in range(grid_size):
            for y in range(grid_size):
                self.nbrs[(x, y)] = [(((x + dx) % grid_size, (y + dy) % grid_size), (dx, dy))
                                     for dx, dy in DIRECTIONS]

    # --- Simulation exacte des règles du jeu ---

    def step(self, body, g, cell, apple):
        """Joue un coup vers `cell`. Retourne (body, g) ou None si le serpent meurt."""
        new_body = [cell] + (body if g else body[:-1])
        if cell in new_body[1:]:
            return None
        return new_body, cell == apple

    def simulate(self, body, g, path, apple):
        for cell in path:
            state = self.step(body, g, cell, apple)
            if state is None:
                return None
            body, g = state
        return body, g

    # --- Recherche de chemin ---

    def path_to(self, body, g, goal, fast=False):
        """Dijkstra temporel. Coût lexicographique (nb de coups, ouverture) :
        le chemin est le plus court possible, et à longueur égale on préfère
        longer le corps (moins de trous laissés derrière soi).
        fast=True : simple BFS temporel (même longueur, sans départage)."""
        if goal is None:
            return None
        L = len(body)
        free = {c: L - i + g for i, c in enumerate(body)}
        start = body[0]
        if fast:
            parent = {start: None}
            q = deque([(start, 0)])
            while q:
                c, t = q.popleft()
                if c == goal:
                    path = []
                    while c != start:
                        path.append(c)
                        c = parent[c]
                    return path[::-1]
                for nb, _ in self.nbrs[c]:
                    if nb not in parent and free.get(nb, 0) <= t + 1:
                        parent[nb] = c
                        q.append((nb, t + 1))
            return None
        best = {start: (0, 0)}
        parent = {}
        heap = [(0, 0, start)]
        while heap:
            t, pen, c = heapq.heappop(heap)
            if best[c] != (t, pen):
                continue
            if c == goal:
                path = []
                while c != start:
                    path.append(c)
                    c = parent[c]
                return path[::-1]
            for nb, _ in self.nbrs[c]:
                if free.get(nb, 0) > t + 1:
                    continue
                # Ouverture = voisins encore libres au moment du passage
                opening = sum(1 for v, _ in self.nbrs[nb] if free.get(v, 0) <= t + 1)
                cost = (t + 1, pen + opening)
                if nb not in best or cost < best[nb]:
                    best[nb] = cost
                    parent[nb] = c
                    heapq.heappush(heap, (cost[0], cost[1], nb))
        return None

    def is_safe(self, body, g, margin=0):
        """Certificat de survie (voir docstring du module).

        BFS temporel : on peut traverser un segment body[i] déjà libéré. L'état
        est sûr si l'on atteint un segment body[j] (j >= 1) par un chemin qui
        n'emprunte aucun segment d'indice < j : le serpent peut alors suivre
        body[j-1] ... body[0] puis son propre chemin, indéfiniment.
        `margin` ajoute des coups de retard (pommes mangées sur la boucle).
        """
        L = len(body)
        if L >= self.n * self.n:
            return True
        g += margin
        index = {c: i for i, c in enumerate(body)}
        head = body[0]
        # low[c] = plus petit indice de segment traversé pour arriver en c
        dist = {head: 0}
        low = {head: L}
        q = deque([head])
        while q:
            c = q.popleft()
            d, lo = dist[c], low[c]
            for nb, _ in self.nbrs[c]:
                j = index.get(nb)
                if j is not None:
                    if j == 0 or d + 1 < L - j + g:
                        continue  # segment pas encore libéré
                    if lo > j:
                        return True
                    nlo = j
                else:
                    nlo = lo
                if nb not in dist:
                    dist[nb] = d + 1
                    low[nb] = nlo
                    q.append(nb)
                elif dist[nb] == d + 1 and nlo > low[nb]:
                    low[nb] = nlo
        return False

    def area(self, body, g):
        """Nombre de cases atteignables (la queue compte comme libre si g == 0)."""
        blocked = set(body if g else body[:-1])
        seen = {body[0]}
        q = deque([body[0]])
        while q:
            c = q.popleft()
            for nb, _ in self.nbrs[c]:
                if nb not in seen and nb not in blocked:
                    seen.add(nb)
                    q.append(nb)
        return len(seen)

    def hug(self, body, cell):
        """Nombre de voisins de `cell` occupés par le corps (coller au corps)."""
        occupied = set(body)
        return sum(1 for v, _ in self.nbrs[cell] if v in occupied)

    # --- Décision ---

    def choose(self, body, grow_pending, apple):
        """Retourne la direction (dx, dy) du prochain coup."""
        body = [tuple(c) for c in body]
        apple = tuple(apple) if apple else None
        g = bool(grow_pending)
        head = body[0]
        dir_of = {nb: d for nb, d in self.nbrs[head]}

        if self.strategy == "naive":
            return self._choose_naive(body, g, apple, dir_of)

        # 1. Chemin le plus court vers la pomme, si l'état après le repas est sûr
        path = self._safe_apple_path(body, g, apple)
        if path:
            self._pending = []
            self._cycle = None
            return self._go("POMME", path, dir_of)

        # Coups qui mènent à un état certifié sûr, de préférence avec une marge
        # d'un coup (une pomme peut réapparaître sur la boucle de survie)
        states = [(nb, self.step(body, g, nb, apple)) for nb, _ in self.nbrs[head]]
        states = [(nb, s) for nb, s in states if s]
        candidates = [(nb, s) for nb, s in states if self.is_safe(*s, margin=1)]
        if not candidates:
            # Sans marge, on s'interdit de manger : la pomme suivante pourrait
            # tomber sur la boucle de survie et la rallonger d'un coup de trop.
            candidates = [(nb, s) for nb, s in states if nb != apple and self.is_safe(*s)]
        if not candidates:
            candidates = [(nb, s) for nb, s in states if self.is_safe(*s)]

        if candidates:
            # 2. Détour : un pas de côté puis le plus court chemin sûr vers la pomme
            best = None
            for nb, (b2, g2) in candidates:
                p = self._safe_apple_path(b2, g2, apple)
                if p:
                    key = (len(p), -self.hug(body, nb), self.rng.random())
                    if best is None or key < best[0]:
                        best = (key, [nb] + p)
            if best:
                return self._go("DETOUR", best[1], dir_of)

            safe_cells = {nb for nb, _ in candidates}

            # Clôture en cours : on suit le circuit
            if self._cycle and self._cycle.get(head) in safe_cells:
                return self._go("CLOTURE", self._cycle_ahead(head), dir_of)
            self._cycle = None

            # 3. Recherche : suite de coups sûrs qui rend la pomme accessible
            if self._pending and self._pending_apple == apple and self._pending[0] in safe_cells:
                return self._go("RECHERCHE", self._follow_pending(), dir_of)
            self._pending = []
            if self._cooldown > 0:
                self._cooldown -= 1
            else:
                plan = self._search(body, g, apple)
                if plan:
                    self._pending, self._pending_apple = plan, apple
                    return self._go("RECHERCHE", self._follow_pending(), dir_of)
                self._cooldown = self.search_cooldown

            # 4. Clôture de fin de partie : chemin tête -> toutes les cases libres
            #    -> queue. Corps + chemin = circuit qui couvre toute la grille.
            if self.n * self.n - len(body) <= self.close_max_free:
                self._cycle = self._close_cycle(body, g)
                if self._cycle and self._cycle[head] in safe_cells:
                    return self._go("CLOTURE", self._cycle_ahead(head), dir_of)
                self._cycle = None

            # 5. Temporiser : rester collé au corps, loin de sa queue. Un peu
            #    d'aléatoire casse les boucles où la forme du serpent se répète.
            if self.rng.random() < self.stall_noise:
                nb, _ = self.rng.choice(candidates)
            else:
                def stall_key(item):
                    nb, (b2, g2) = item
                    return (self.hug(body, nb), self._dist(b2, g2, b2[-1]), self.rng.random())
                nb, _ = max(candidates, key=stall_key)
            return self._go("QUEUE", [nb], dir_of)

        # 6. Survie : aucun coup certifié, on maximise l'espace disponible
        return self._survive(body, g, apple, dir_of)

    def _safe_apple_path(self, body, g, apple, fast=False):
        """Plus court chemin vers la pomme dont l'état d'arrivée est certifié."""
        p = self.path_to(body, g, apple, fast)
        if p and self.is_safe(self._after_path(body, g, p), True, margin=self.eat_margin):
            return p
        return None

    @staticmethod
    def _after_path(body, g, path):
        """Corps après avoir suivi un chemin valide (sans pomme en route).
        Équivaut à simulate(), en O(L) au lieu de O(L * len(path))."""
        return (path[::-1] + body)[:len(body) + g]

    def _search(self, body, g, apple):
        """BFS dans l'espace des états (serpent entier), limité à search_budget
        états. On ne traverse que des états certifiés ; le but est un état d'où
        un chemin sûr vers la pomme existe. Retourne la suite de cases."""
        if apple is None:
            return None
        seen = {tuple(body)}
        frontier = deque([(body, g, [])])
        expanded = 0
        while frontier and expanded < self.search_budget:
            b, gg, moves = frontier.popleft()
            expanded += 1
            for nb, _ in self.nbrs[b[0]]:
                s = self.step(b, gg, nb, apple)
                if not s:
                    continue
                b2, g2 = s
                if nb == apple:
                    if self.is_safe(b2, g2, margin=self.eat_margin):
                        return moves + [nb]
                    continue
                key = tuple(b2)
                if key in seen or not self.is_safe(b2, g2):
                    continue
                seen.add(key)
                p = self._safe_apple_path(b2, g2, apple, fast=True) if moves else None
                if p:
                    return moves + [nb] + p
                frontier.append((b2, g2, moves + [nb]))
        return None

    def _close_cycle(self, body, g, budget=100000, margin=1):
        """DFS : chemin de la tête qui visite toutes les cases libres et les
        derniers segments de la queue (déjà libérés au passage), puis rejoint
        un segment body[j] alors que tous les segments d'indice > j sont visités.
        Chemin + body[j..0] = circuit qui couvre toute la grille.
        Retourne le circuit {cellule: suivante} ou None."""
        L = len(body)
        head = body[0]
        index = {c: i for i, c in enumerate(body)}
        free = [c for c in self.nbrs if c not in index]
        if not free:
            return None
        path, visited = [], set()
        count = [0]
        state = {"free_left": len(free)}

        def enterable(v, t):
            j = index.get(v)
            if j is None:
                return v not in visited
            return j > 0 and v not in visited and t >= L - j + g + margin

        def closes(v):
            # v = body[j] : tous les segments après j déjà parcourus ?
            j = index[v]
            return state["free_left"] == 0 and all(body[k] in visited for k in range(j + 1, L))

        def dfs(c, t):
            count[0] += 1
            if count[0] > budget:
                return False
            options = [v for v, _ in self.nbrs[c] if enterable(v, t + 1)]
            for v in options:
                if v in index and closes(v):
                    path.append(v)
                    return True
            # Warnsdorff : d'abord les cases qui ont le moins de sorties
            options.sort(key=lambda v: sum(1 for u, _ in self.nbrs[v] if enterable(u, t + 2)))
            for v in options:
                is_free = v not in index
                visited.add(v)
                path.append(v)
                if is_free:
                    state["free_left"] -= 1
                if dfs(v, t + 1):
                    return True
                if is_free:
                    state["free_left"] += 1
                visited.discard(v)
                path.pop()
            return False

        if not dfs(head, 0):
            return None
        j = index[path[-1]]
        order = path + body[j - 1::-1]  # ... -> body[j] -> body[j-1] -> ... -> tête
        return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}

    def _cycle_ahead(self, head, k=30):
        """Les k prochaines cases du circuit (pour l'affichage)."""
        out, c = [], head
        for _ in range(k):
            c = self._cycle[c]
            out.append(c)
        return out

    def _follow_pending(self):
        path = self._pending
        self._pending = path[1:]
        return path

    def _choose_naive(self, body, g, apple, dir_of):
        path = self.path_to(body, g, apple)
        if path:
            return self._go("NAIF", path, dir_of)
        return self._survive(body, g, apple, dir_of)

    def _survive(self, body, g, apple, dir_of):
        options = []
        for nb, d in self.nbrs[body[0]]:
            state = self.step(body, g, nb, apple)
            if state:
                options.append((self.area(*state), self.rng.random(), nb))
        if not options:  # mort inévitable : on garde n'importe quelle direction
            nb = self.nbrs[body[0]][0][0]
            return self._go("SURVIE", [nb], dir_of)
        _, _, nb = max(options)
        return self._go("SURVIE", [nb], dir_of)

    def _dist(self, body, g, goal):
        """Distance du plus court chemin temporel vers `goal` (grande si inaccessible)."""
        p = self.path_to(body, g, goal, fast=True)
        return len(p) if p else self.n * self.n

    def _go(self, mode, path, dir_of):
        self.mode = mode
        self.plan = path
        return dir_of[path[0]]
