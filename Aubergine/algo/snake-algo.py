"""
Snake-algo : bot pour le vrai jeu (serpent-algo.py), sans trou et avec garantie de victoire.

Le fichier du jeu n'est PAS modifié ni recopié : on l'importe et on lance sa vraie fonction main().
Le bot s'y branche par des sous-classes de Snake et d'Apple (le jeu les cherche dans ses globals) :
à chaque déplacement le bot choisit la direction avec le set_direction() du jeu, puis le Snake.move()
du jeu s'exécute tel quel.

IDÉE CENTRALE
La pomme apparaît au hasard parmi les cases libres. Si les cases libres forment un seul bloc compact
près de la tête, la pomme est toujours proche. Si elles sont éparpillées (« trous »), la pomme tombe
souvent dans un recoin inaccessible et il faut attendre un tour complet. Le bot maintient donc en
permanence un CERTIFICAT : un chemin hamiltonien des cases libres, qui part d'un voisin de la tête et
finit sur un voisin de la queue. Tant qu'il existe, suivre ce chemin est toujours possible et sûr, et
la pomme est toujours sur ce chemin : victoire garantie. Toutes les couches ci-dessous sont toujours actives.

COUCHES
  1. Cycle hamiltonien du tore : donne le certificat de départ (le corps de départ est dans son ordre).
  2. Certificat mis à jour à chaque pas : on retire la case franchie, on ajoute la case quittée par la queue.
  3. Candidats de raccourci : plusieurs plus courts chemins réels de la tête à la pomme à travers les cases
     libres (Dijkstra : virages, contact avec le corps, bruit) au lieu de suivre le cycle.
  4. Preuve : pour chaque raccourci, l'état final (après avoir mangé) doit avoir un certificat. On recolle
     d'abord les tronçons de l'ancien certificat, sinon un solveur hamiltonien (recherche en profondeur avec
     ordre de Warnsdorff et élagage des culs-de-sac et des cases isolées) en construit un autre.
  5. Choix par « espace libre futur » : coups du raccourci + distance moyenne de la tête aux cases libres
     restantes (= coût attendu de la pomme suivante). Les raccourcis qui laissent un bloc libre compact
     près de la tête gagnent ; ils remplissent en priorité les cases qui éloigneraient la prochaine pomme.
  6. Filet de sécurité : si un pas était illégal ou si tout échoue, on garde le certificat courant.

Usage :
  python snake-algo.py                      # fenêtre, 30 images/s
  python snake-algo.py --fps 5              # vitesse d'origine du jeu
  python snake-algo.py --fps 0              # aussi vite que possible
  python snake-algo.py --headless --games 6 --jobs 6 --debug   # vrai jeu sans fenêtre
"""
import argparse
import heapq
import importlib.util
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.setrecursionlimit(5000)


class _BudgetExceeded(Exception):
    """Budget de calcul dépassé : le candidat est abandonné, le certificat courant est gardé."""


# --- COUCHE 1 : CYCLE HAMILTONIEN (certificat de départ) ---

def _zigzag(width, height):
    """Colonne 0 réservée au retour, aller-retours sur les colonnes 1..width-1. Sans passage de mur si
    height est pair ; si height et width sont impairs, la dernière ligne se referme par un mur."""
    cycle = [(x, 0) for x in range(width)]
    for y in range(1, height):
        xs = range(width - 1, 0, -1) if y % 2 == 1 else range(1, width)
        cycle += [(x, y) for x in xs]
    cycle += [(0, y) for y in range(height - 1, 0, -1)]
    return cycle


def build_cycle(width, height, wrap=True):
    """Cycle hamiltonien de la grille width x height (liste de (x, y))."""
    if width < 3 or height < 3:
        raise ValueError("Grille trop petite (minimum 3x3).")
    if height % 2 == 0:
        cycle = _zigzag(width, height)
    elif width % 2 == 0:
        cycle = [(x, y) for (y, x) in _zigzag(height, width)]
    elif wrap:
        cycle = _zigzag(width, height)
    else:
        raise ValueError(f"Grille {width}x{height} : deux dimensions impaires, pas de cycle hamiltonien "
                         "sans traverser les murs (nombre de cases impair).")
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
    """Symétrie/sens du cycle dans lequel le corps de départ (queue -> tête) est rangé dans l'ordre."""
    tail_to_head = [tuple(p) for p in reversed(body)]
    for flip_x in (False, True):
        for flip_y in (False, True):
            base = [((width - 1 - x) if flip_x else x, (height - 1 - y) if flip_y else y) for x, y in cycle]
            for candidate in (base, base[::-1]):
                pos = {c: i for i, c in enumerate(candidate)}
                if all(c in pos for c in tail_to_head) and all(
                        (pos[b] - pos[a]) % len(candidate) == 1 for a, b in zip(tail_to_head, tail_to_head[1:])):
                    return candidate
    raise ValueError("Aucune orientation du cycle ne contient le serpent de départ dans l'ordre.")


# --- LE BOT ---

class SnakeBot:
    """Toutes les cases sont des entiers x + y * width. `path` est le certificat : chemin hamiltonien des
    cases libres, d'un voisin de la tête à un voisin de la queue."""

    DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # haut, bas, gauche, droite : l'inverse de d est d ^ 1
    FUTURE_WEIGHT = 1.0        # poids de la distance moyenne aux cases libres restantes
    RETRY_EVERY = 10           # coups entre deux recherches de raccourci sur la même pomme
    SOLVER_NODES = 4000        # budget du solveur hamiltonien par candidat
    RELINK_NODES = 1500        # budget du recollage de tronçons par candidat
    RESTARTS = 6               # recherches courtes du solveur (le budget SOLVER_NODES est partagé)
    MAX_TRIES = 4              # candidats meilleurs que le cycle qu'on tente de prouver par pomme
    STEP_LIMIT = 80            # au-delà de ce nombre de coups sur une même pomme : plus de déviation, on suit
                               # le certificat (garantit que la pomme est mangée en au plus N coups)
    STEP_VARIANTS = 8          # variantes de chemin essayées pour les plans courts
    STEP_MAX = 4               # longueur maximale d'un plan court (preuve à la fin seulement)
    SHORT_FIRST = False        # essayer d'abord les plans les plus courts (plus faciles à prouver)
    STEP_TRIES = 6             # plans courts candidats qu'on tente de prouver à chaque fois
    # (poids de virage, poids de contact avec les cases libres, bruit) des variantes de plus court chemin
    VARIANTS = ((0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, -0.03, 0.0),
                (0.03, 0.02, 0.0), (0.03, -0.02, 0.0), (0.0, 0.0, 0.10), (0.0, 0.0, 0.10),
                (0.0, 0.0, 0.10), (0.0, 0.0, 0.35), (0.0, 0.0, 0.35), (0.0, 0.0, 0.35))

    def __init__(self, width, height, initial_body, wrap=True, debug=False):
        self.width, self.height, self.wrap = width, height, wrap
        self.n = width * height
        self.debug = debug
        self.nbrs = [self._neighbors(c) for c in range(self.n)]
        self.nbset = [set(v) for v in self.nbrs]
        self.step_to = [[self._step(c, d) for d in range(4)] for c in range(self.n)]
        self.rng = random.Random(12345)  # générateur à part : ne touche pas au hasard du jeu
        # certificat de départ : les cases du cycle entre la tête et la queue
        cycle = _orient_cycle(build_cycle(width, height, wrap), width, height, initial_body)
        order = [x + y * width for x, y in cycle]
        head_pos = order.index(self.cell(initial_body[0]))
        self.path = deque(order[(head_pos + k) % self.n] for k in range(1, self.n - len(initial_body) + 1))
        self.plan = None
        self.last_apple = None
        self.since_try = 0
        self.apple_moves = 0
        self._ops = self._budget = 0
        self.fallbacks = 0
        self.violations = []
        self.stats = {"pommes": 0, "raccourcis": 0, "recollage": 0, "solveur": 0, "echec": 0, "gain": 0,
                      "temps": 0.0, "pas": 0, "pas_1": 0, "pas_2": 0, "pas_3": 0, "pas_4": 0}

    # --- outils ---

    def _neighbors(self, c):
        x, y = c % self.width, c // self.width
        out = []
        for dx, dy in self.DIRS:
            nx, ny = x + dx, y + dy
            if self.wrap:
                nx, ny = nx % self.width, ny % self.height
            elif not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            out.append(nx + ny * self.width)
        return out

    def _step(self, c, d):
        dx, dy = self.DIRS[d]
        return (c % self.width + dx) % self.width + ((c // self.width + dy) % self.height) * self.width

    def cell(self, pos):
        return pos[0] + pos[1] * self.width

    def _direction(self, head_pos, cell):
        dx, dy = cell % self.width - head_pos[0], cell // self.width - head_pos[1]
        if self.wrap:
            if dx in (self.width - 1, -(self.width - 1)):
                dx = -1 if dx > 0 else 1
            if dy in (self.height - 1, -(self.height - 1)):
                dy = -1 if dy > 0 else 1
        return (dx, dy)

    def _is_legal(self, cell, cells, pending):
        """Vraie règle du jeu : la case ne doit pas être dans le corps une fois la queue retirée."""
        return cell not in (cells[1:] if pending else cells[1:-1])

    def _spend(self, cost=1):
        self._ops += cost
        if self._ops > self._budget:
            raise _BudgetExceeded

    def _final_body(self, cells, steps, pending):
        """Corps (tête d'abord) après avoir suivi `steps` et mangé la pomme au dernier pas."""
        full = steps[::-1] + cells
        return full[:len(full) - (len(steps) - pending)]

    def _valid_certificate(self, path, region, head, tail):
        """Exactement les cases de la région, une fois chacune, pas entre voisines, raccordé tête et queue."""
        if len(path) != len(region):
            return False
        if not path:
            return True
        if set(path) != region or path[0] not in self.nbset[head] or path[-1] not in self.nbset[tail]:
            return False
        return all(b in self.nbset[a] for a, b in zip(path, path[1:]))

    # --- décision à chaque coup ---

    def choose(self, body, pending, apple):
        """Retourne la direction (dx, dy) à donner au serpent."""
        p = 1 if pending else 0
        cells = [self.cell(b) for b in body]
        apple_cell = self.cell(apple)

        if self.plan is None:
            if self.debug:
                self._check_certificate(cells)
            self.since_try += 1
            self.apple_moves += 1
            if apple != self.last_apple:
                self.apple_moves = 0
            if self.apple_moves <= self.STEP_LIMIT and (apple != self.last_apple or self.since_try >= self.RETRY_EVERY):
                self.since_try = 0
                self.last_apple = apple
                self._plan(cells, p, apple_cell)
            self.last_apple = apple
            if self.plan is None and self.apple_moves <= self.STEP_LIMIT:
                self._short_plans(cells, p, apple_cell)  # crée éventuellement un plan de 1 à STEP_MAX pas

        if self.plan is not None:
            step = self.plan["steps"][self.plan["i"]]
            if not self._is_legal(step, cells, p):
                self.plan = None  # ne doit jamais arriver : on abandonne le plan
                self._note("plan abandonné : pas illégal")
            else:
                self.plan["i"] += 1
                if self.plan["i"] == len(self.plan["steps"]):
                    self.path = deque(self.plan["path"])  # état final atteint : son certificat devient le courant
                    self.plan = None
                return self._direction(body[0], step)

        # sans raccourci : on suit le certificat (retirer la case franchie, ajouter la case quittée par la queue)
        target = self.path[0] if self.path else None
        if target is None or not self._is_legal(target, cells, p):
            self.fallbacks += 1
            self._note("certificat inutilisable : coup de secours")
            legal = [v for v in self.nbrs[cells[0]] if self._is_legal(v, cells, p)]
            return self._direction(body[0], legal[0] if legal else self.nbrs[cells[0]][0])
        self.path.popleft()
        if not p:
            self.path.append(cells[-1])
        return self._direction(body[0], target)

    def _note(self, message):
        if self.debug and len(self.violations) < 5:
            self.violations.append(message)

    def _check_certificate(self, cells):
        """Mode debug : le certificat couvre exactement les cases libres, raccordé à la tête et à la queue."""
        free = set(range(self.n)) - set(cells)
        assert len(set(cells)) == len(cells), "collision"
        assert self._valid_certificate(list(self.path), free, cells[0], cells[-1]), "certificat invalide"

    # --- COUCHES 3 à 5 : raccourcis prouvés et choisis par espace libre futur ---

    def _plan(self, cells, p, apple_cell):
        t0 = time.perf_counter()
        try:
            self._plan_inner(cells, p, apple_cell)
        finally:
            self.stats["temps"] += time.perf_counter() - t0

    def _plan_inner(self, cells, p, apple_cell):
        n = self.n
        path = list(self.path)
        if apple_cell not in path:
            return
        self.stats["pommes"] += 1
        default_len = path.index(apple_cell) + 1
        if default_len <= 1:
            return  # pomme juste devant : rien à gagner
        free = set(range(n)) - set(cells)
        # référence : suivre le certificat jusqu'à la pomme
        default_final = self._final_body(cells, path[:default_len], p)
        default_future = self._future_distance(default_final)
        if default_future is None:
            return
        best_score = default_len + self.FUTURE_WEIGHT * default_future

        candidates = []
        seen = set()
        head_dir = self.DIRS.index(self._direction([cells[1] % self.width, cells[1] // self.width], cells[0])) \
            if len(cells) > 1 else 3
        free_degree = {c: sum(1 for v in self.nbrs[c] if v in free) for c in free}
        for turn_w, hug_w, noise in self.VARIANTS:
            steps = self._route(free, free_degree, cells[0], head_dir, apple_cell, turn_w, hug_w, noise)
            if steps is None or len(steps) >= default_len or tuple(steps) in seen:
                continue
            seen.add(tuple(steps))
            final = self._final_body(cells, steps, p)
            future = self._future_distance(final)
            if future is None:
                continue  # cases libres coupées en plusieurs morceaux : aucun chemin hamiltonien possible
            score = len(steps) + self.FUTURE_WEIGHT * future
            if score < best_score:
                candidates.append((score, steps, final))
        candidates.sort(key=lambda c: c[0])

        for score, steps, final in candidates[:self.MAX_TRIES]:
            certificate = self._prove(cells, p, steps, final, path)
            if certificate is not None:
                self.plan = {"steps": steps, "i": 0, "path": certificate}
                self.stats["raccourcis"] += 1
                self.stats["gain"] += default_len - len(steps)
                return
        if candidates:
            self.stats["echec"] += 1

    def _short_plans(self, cells, p, apple_cell):
        """Plans courts : on quitte le certificat pour 1 à STEP_MAX pas le long d'un plus court chemin réel vers
        la pomme. La preuve (certificat de l'état obtenu) ne porte que sur la FIN du plan : les pas intermédiaires
        n'ont pas besoin d'en avoir un. Un petit changement se prouve bien plus souvent qu'un trajet entier, et
        un plan de plusieurs pas passe là où le pas isolé échoue. Recalculé à chaque fois qu'un plan se termine."""
        path = list(self.path)
        if apple_cell not in path:
            return
        default_len = path.index(apple_cell) + 1
        if default_len <= 1:
            return
        free = set(range(self.n)) - set(cells)
        free_degree = {c: sum(1 for v in self.nbrs[c] if v in free) for c in free}
        head_dir = self.DIRS.index(self._direction([cells[1] % self.width, cells[1] // self.width], cells[0])) \
            if len(cells) > 1 else 3
        w = self.FUTURE_WEIGHT
        references = {}   # k -> coût si on suit le certificat pendant k pas : longueur + espace libre laissé
        candidates = {}   # préfixe (tuple) -> (score, longueur du chemin complet, état final)
        for turn_w, hug_w, noise in self.VARIANTS[:self.STEP_VARIANTS]:
            steps = self._route(free, free_degree, cells[0], head_dir, apple_cell, turn_w, hug_w, noise)
            if steps is None or len(steps) >= default_len:
                continue
            for k in range(1, min(self.STEP_MAX, len(steps)) + 1):
                prefix = tuple(steps[:k])
                if prefix in candidates or list(prefix) == path[:k]:
                    continue  # déjà vu, ou identique au suivi du certificat : rien à gagner
                if k not in references:
                    ref_future = self._future_distance(self._final_body(cells, path[:k], p))
                    references[k] = default_len + w * (ref_future if ref_future is not None else 0.0)
                final = self._final_body(cells, list(prefix), p)
                future = self._future_distance(final)
                if future is None:
                    continue  # cases libres coupées en plusieurs morceaux : aucun certificat possible
                score = len(steps) + w * future
                if score < references[k]:
                    candidates[prefix] = (score, len(steps), final)
        ranked = sorted(candidates.items(),
                        key=lambda item: (len(item[0]) if self.SHORT_FIRST else 0, item[1][0], -len(item[0])))
        for prefix, (score, length, final) in ranked[:self.STEP_TRIES]:
            certificate = self._prove(cells, p, list(prefix), final, path)
            if certificate is not None:
                self.plan = {"steps": list(prefix), "i": 0, "path": certificate}
                self.apple_moves += len(prefix) - 1
                self.stats["pas"] += 1
                self.stats["pas_" + str(len(prefix))] += 1
                return

    def _route(self, free, free_degree, head, head_dir, goal, turn_w, hug_w, noise):
        """Dijkstra sur des états (case, direction) à travers les cases libres : coût 1 par pas, plus un
        petit poids par virage, par nombre de voisines libres (contact avec le corps), plus du bruit."""
        rng = self.rng
        start = (head, head_dir)
        best = {start: 0.0}
        parent = {}
        heap = [(0.0, 0, head, head_dir)]
        counter = 0
        while heap:
            cost, _, c, d = heapq.heappop(heap)
            if best.get((c, d), 1e18) < cost:
                continue
            if c == goal:
                steps, state = [], (c, d)
                while state != start:
                    steps.append(state[0])
                    state = parent[state]
                steps.reverse()
                return steps if len(set(steps)) == len(steps) else None
            for nd in range(4):
                if nd == d ^ 1:
                    continue
                v = self.step_to[c][nd]
                if v not in free:
                    continue
                step = 1.0 + (turn_w if nd != d else 0.0) + hug_w * free_degree[v] \
                    + (noise * rng.random() if noise else 0.0)
                new_cost = cost + step
                key = (v, nd)
                if new_cost < best.get(key, 1e18):
                    best[key] = new_cost
                    parent[key] = (c, d)
                    counter += 1
                    heapq.heappush(heap, (new_cost, counter, v, nd))
        return None

    def _future_distance(self, final):
        """Distance moyenne (en pas) de la tête aux cases libres restantes, ou None si elles ne sont pas
        toutes atteignables d'un seul tenant. C'est le coût attendu de la pomme suivante."""
        occupied = set(final)
        remaining = self.n - len(occupied)
        if remaining == 0:
            return 0.0
        queue = [v for v in self.nbrs[final[0]] if v not in occupied]
        dist = {v: 1 for v in queue}
        for c in queue:
            for v in self.nbrs[c]:
                if v not in occupied and v not in dist:
                    dist[v] = dist[c] + 1
                    queue.append(v)
        if len(dist) != remaining:
            return None
        return sum(dist.values()) / remaining

    # --- COUCHE 4 : preuve (certificat de l'état final) ---

    def _prove(self, cells, p, steps, final, path):
        """Certificat de l'état final ou None. D'abord recoller les tronçons de l'ancien certificat, puis
        le solveur hamiltonien. Toujours revalidé strictement."""
        region = set(range(self.n)) - set(final)
        if not region:
            return []
        head, tail = final[0], final[-1]
        starts, ends = self.nbset[head] & region, self.nbset[tail] & region
        if not starts or not ends or not self._degrees_possible(region, starts, ends):
            return None
        pops = len(steps) - p
        guide = None
        if pops <= len(cells) - 1:
            extended = path + [cells[len(cells) - 1 - i] for i in range(pops)]
            guide = {c: i for i, c in enumerate(extended)}  # ordre de l'ancien certificat, guide du solveur
            self._ops, self._budget = 0, self.RELINK_NODES
            try:
                certificate = self._relink_old(cells, steps, pops, path, head, tail)
            except _BudgetExceeded:
                certificate = None
            if certificate is not None and self._valid_certificate(certificate, region, head, tail):
                self.stats["recollage"] += 1
                return certificate
        self._ops, self._budget = 0, self.SOLVER_NODES
        try:
            certificate = self._ham_path(region, starts, ends, guide)
        except _BudgetExceeded:
            certificate = None
        if certificate is not None and self._valid_certificate(certificate, region, head, tail):
            self.stats["solveur"] += 1
            return certificate
        return None

    def _degrees_possible(self, region, starts, ends):
        """Nécessaire : une case à 0 voisine libre est impossible ; une case à 1 voisine libre est une
        extrémité du chemin, donc voisine de la tête (départ) ou de la queue (arrivée) ; au plus 2 extrémités."""
        dead = 0
        for c in region:
            d = len(self.nbset[c] & region)
            if d == 0 and len(region) > 1:
                return False
            if d <= 1:
                dead += 1
                if c not in starts and c not in ends:
                    return False
        return dead <= 2 or len(region) <= 2

    def _relink_old(self, cells, steps, pops, path, head, tail):
        """Ancien certificat prolongé des cases quittées par la queue, privé du raccourci : tronçons recollés."""
        extended = path + [cells[len(cells) - 1 - i] for i in range(pops)]
        taken = set(steps)
        chains, current = [], []
        for c in extended:
            if c in taken:
                if current:
                    chains.append(current)
                    current = []
            else:
                current.append(c)
        if current:
            chains.append(current)
        starts = {}
        for i, chain in enumerate(chains):
            starts.setdefault(chain[0], []).append((i, False))
            if len(chain) > 1:
                starts.setdefault(chain[-1], []).append((i, True))
        used = [False] * len(chains)
        order = []

        def dfs(current_cell, done):
            self._spend()
            if done == len(chains):
                return tail in self.nbset[current_cell]
            for v in self.nbrs[current_cell]:
                for i, reverse in starts.get(v, ()):
                    if not used[i]:
                        used[i] = True
                        order.append((i, reverse))
                        chain = chains[i]
                        if dfs(chain[0] if reverse else chain[-1], done + 1):
                            return True
                        order.pop()
                        used[i] = False
            return False

        if not dfs(head, 0):
            return None
        result = []
        for i, reverse in order:
            result += chains[i][::-1] if reverse else chains[i]
        return result

    def _ham_path(self, region, start_cells, end_cells, guide=None):
        """Chemin hamiltonien de `region` d'une case de start_cells à une case de end_cells, ou None.
        Plusieurs recherches courtes plutôt qu'une seule qui s'enlise : d'abord guidée par l'ancien certificat
        (on le suit tant que c'est possible), puis Warnsdorff, puis Warnsdorff au départage aléatoire.
        Une recherche qui se termine sans solution prouve l'impossibilité."""
        adj = {c: [v for v in self.nbrs[c] if v in region] for c in region}
        per_try = max(50, self._budget // self.RESTARTS)
        modes = (["guide"] if guide else []) + ["warnsdorff"] + ["random"] * self.RESTARTS
        for mode in modes[:self.RESTARTS]:
            outcome = self._ham_attempt(region, adj, start_cells, set(end_cells), per_try, mode, guide)
            if outcome is False:
                return None      # exploration complète : pas de chemin
            if outcome is not None:
                return outcome
        return None

    def _ham_attempt(self, region, adj, start_cells, ends, limit, mode, guide):
        """Une recherche en profondeur, voisin de plus petit degré d'abord (Warnsdorff). Élagage : case isolée,
        plus d'une extrémité forcée, extrémité forcée qui n'est pas une case d'arrivée, cases restantes coupées
        en plusieurs morceaux. Retourne le chemin, False (aucun chemin) ou None (budget épuisé)."""
        k = len(region)
        if k == 1:
            c = next(iter(region))
            return [c] if c in start_cells and c in ends else False
        rng = self.rng
        deg = {c: len(adj[c]) for c in region}
        visited, path = set(), []
        counters = {"low0": 0, "low1": 0, "bad1": 0, "ends_left": len(ends), "nodes": 0}

        def classify(c, sign):
            d = deg[c]
            if d == 0:
                counters["low0"] += sign
            elif d == 1:
                counters["low1"] += sign
                if c not in ends:
                    counters["bad1"] += sign

        for c in region:
            classify(c, +1)

        def visit(c):
            classify(c, -1)
            visited.add(c)
            path.append(c)
            if c in ends:
                counters["ends_left"] -= 1
            for v in adj[c]:
                if v not in visited:
                    classify(v, -1)
                    deg[v] -= 1
                    classify(v, +1)

        def unvisit(c):
            for v in adj[c]:
                if v not in visited:
                    classify(v, -1)
                    deg[v] += 1
                    classify(v, +1)
            visited.discard(c)
            path.pop()
            if c in ends:
                counters["ends_left"] += 1
            classify(c, +1)

        def feasible(c):
            remaining = k - len(path)
            if remaining == 0:
                return c in ends
            if counters["ends_left"] == 0:
                return False
            near0 = near1 = near_bad1 = 0
            frontier = []
            for u in adj[c]:
                if u not in visited:
                    frontier.append(u)
                    if deg[u] == 0:
                        near0 += 1
                    elif deg[u] == 1:
                        near1 += 1
                        if u not in ends:
                            near_bad1 += 1
            if near0:
                return remaining == 1  # case reliée seulement à c : doit être la dernière
            if counters["low0"] or not frontier:
                return False
            if counters["low1"] - near1 > 1 or counters["bad1"] - near_bad1 != 0:
                return False
            seen = set(frontier)  # les cases restantes doivent rester d'un seul tenant
            stack = list(frontier)
            while stack:
                x = stack.pop()
                for v in adj[x]:
                    if v not in visited and v not in seen:
                        seen.add(v)
                        stack.append(v)
            return len(seen) == remaining

        def dfs(c):
            counters["nodes"] += 1
            if counters["nodes"] > limit:
                raise _BudgetExceeded
            if not feasible(c):
                return False
            if len(path) == k:
                return True
            nxt = [u for u in adj[c] if u not in visited]
            if mode == "guide":
                successor = guide[c] + 1
                nxt.sort(key=lambda u: (guide[u] != successor, deg[u]))
            elif mode == "random":
                nxt.sort(key=lambda u: (deg[u], u in ends, rng.random()))
            else:
                nxt.sort(key=lambda u: (deg[u], u in ends))
            for u in nxt:
                visit(u)
                if dfs(u):
                    return True
                unvisit(u)
            return False

        try:
            starts_order = sorted((c for c in start_cells if c in region),
                                  key=(lambda c: guide[c]) if mode == "guide" else (lambda c: (deg[c], rng.random())))
            for s0 in starts_order:
                visit(s0)
                if dfs(s0):
                    return list(path)
                unvisit(s0)
        except _BudgetExceeded:
            return None
        return False


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
    """Remplace Snake et Apple du jeu par des sous-classes pilotées par le bot, et l'horloge par une
    horloge qui règle la vitesse et met fin à la partie (mode --headless)."""
    import pygame
    run = Run()
    cap = 10 * game.GRID_SIZE ** 4  # limite de coups : détecte une boucle infinie
    real_clock = pygame.time.Clock

    class BotSnake(game.Snake):
        def __init__(self):
            super().__init__()
            run.snake, run.victory, run.reported = self, False, False
            self.moves = 0
            self.think = 0.0  # temps de calcul du bot
            self.bot = SnakeBot(game.GRID_SIZE, game.GRID_SIZE, self.body, wrap=True, debug=args.debug)

        def move(self):
            t0 = time.perf_counter()
            try:
                direction = self.bot.choose(self.body, self.grow_pending, run.apple.position)
            except AssertionError as error:  # mode debug : consigné, la partie continue
                self.bot._note(f"coup {self.moves} : {error}")
                self.bot.path = deque()
                direction = self.direction
            self.think += time.perf_counter() - t0
            self.set_direction(direction)
            super().move()
            self.moves += 1

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
                    results.append({"issue": outcome, "score": s.score, "coups": s.moves, "longueur": len(s.body),
                                    "replis": s.bot.fallbacks, "violations": list(s.bot.violations),
                                    "calcul": s.think, "stats": dict(s.bot.stats)})
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
          f"{sum(r['replis'] for r in results)} coups de secours")
    scores = [r["score"] for r in results]
    coups = [r["coups"] for r in results]
    print(f"score  : min {min(scores)}, moyenne {sum(scores) / n:.1f}, max {max(scores)}")
    print(f"coups  : min {min(coups)}, moyenne {sum(coups) / n:.0f}, max {max(coups)}")
    print(f"coups par pomme (moyenne) : {sum(coups) / max(1, sum(scores)):.1f}")
    print(f"calcul du bot : {1000 * sum(r['calcul'] for r in results) / max(1, sum(coups)):.3f} ms par coup")
    st = {k: sum(r["stats"][k] for r in results) for k in results[0]["stats"]}
    print(f"raccourcis prouvés : {st['raccourcis']}/{st['pommes']} pommes (recollage {st['recollage']}, solveur "
          f"{st['solveur']}, sans preuve {st['echec']}), plans courts prouvés {st['pas']} (1 pas : {st['pas_1']}, 2 : {st['pas_2']}, 3 : {st['pas_3']}, 4 : {st['pas_4']}), gain moyen {st['gain'] / max(1, st['raccourcis']):.1f} coups, "
          f"{1000 * st['temps'] / max(1, st['pommes']):.1f} ms de planification par pomme")
    for i, r in enumerate(results):
        for message in r["violations"]:
            print(f"  [debug] partie {i} : {message}")


def main():
    parser = argparse.ArgumentParser(description="Bot sans trou avec certificat hamiltonien sur le vrai jeu Snake.")
    parser.add_argument("--fps", type=int, default=30, help="images/s (5 = vitesse d'origine, 0 = illimité)")
    parser.add_argument("--headless", action="store_true", help="sans fenêtre, enchaîne --games parties puis résume")
    parser.add_argument("--games", type=int, default=1, help="nombre de parties (avec --headless)")
    parser.add_argument("--jobs", type=int, default=1, help="processus parallèles (avec --headless)")
    parser.add_argument("--seed", type=int, default=None, help="graine aléatoire de la première partie")
    parser.add_argument("--debug", action="store_true", help="vérifie le certificat à chaque coup")
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
