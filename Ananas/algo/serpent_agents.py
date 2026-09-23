"""Agents algorithmiques pour le Snake torique de serpent-algo.py.

Aucun apprentissage : uniquement recherche de chemin (BFS), simulation et
flood-fill. Les agents ne dépendent pas de pygame ; ils reçoivent l'état du
jeu sous forme de tuples et renvoient une direction.

Modèle d'une étape, reproduit fidèlement depuis serpent-algo.py :
  1. set_direction() ignore un demi-tour ;
  2. move() : nouvelle tête = (tête + direction) % GRID_SIZE, insérée en
     tête du corps ; la queue est retirée SAUF si grow_pending (qui est
     alors remis à False) ;
  3. collision si la tête est dans body[1:] (après le retrait de la queue :
     entrer dans la case que la queue quitte est donc autorisé) ;
  4. si la tête est sur la pomme : grow() -> grow_pending = True ; la
     croissance n'a lieu qu'au déplacement SUIVANT.
"""
import time
from collections import deque

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)


def opposite(d):
    return (-d[0], -d[1])


def neighbor(cell, d, n):
    return ((cell[0] + d[0]) % n, (cell[1] + d[1]) % n)


def legal_moves(direction):
    """Directions acceptées par Snake.set_direction (pas de demi-tour)."""
    return [d for d in DIRECTIONS if d != opposite(direction)]


def step(body, grow, d, n):
    """Un déplacement, dans l'ordre de Snake.move puis check_self_collision.

    Renvoie (nouveau_corps, mort). grow_pending est consommé par ce pas.
    """
    head = neighbor(body[0], d, n)
    new_body = (head,) + (body if grow else body[:-1])
    return new_body, head in new_body[1:]


def simulate(body, grow, direction, apple, moves, n):
    """Joue `moves` jusqu'à la pomme incluse.

    Renvoie (corps, grow_pending, direction, pomme_mangée) ou None si un
    mouvement est un demi-tour ou provoque une collision.
    """
    for m in moves:
        if m == opposite(direction):
            return None
        body, dead = step(body, grow, m, n)
        grow = False
        direction = m
        if dead:
            return None
        if body[0] == apple:
            return body, True, direction, True
    return body, grow, direction, False


def bfs_path(start, goal, blocked, n):
    """Plus court chemin (liste de directions) sur la grille torique."""
    if start == goal:
        return []
    parent = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for d in DIRECTIONS:
            nxt = neighbor(cell, d, n)
            if nxt in parent or nxt in blocked:
                continue
            parent[nxt] = (cell, d)
            if nxt == goal:
                path = []
                while parent[nxt] is not None:
                    nxt, d = parent[nxt]
                    path.append(d)
                return path[::-1]
            queue.append(nxt)
    return None


def flood_area(start, blocked, n):
    """Nombre de cases libres atteignables depuis `start` (exclu)."""
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for d in DIRECTIONS:
            nxt = neighbor(cell, d, n)
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) - 1


# --------------------------------------------------------------------------
# Étape 1 : BFS + simulation + repli sûr
# --------------------------------------------------------------------------

class BFSAgent:
    """Version 1 (conservée pour comparaison).

    1. BFS vers la pomme, corps considéré comme statique (seule la queue
       est libre si le serpent ne grandit pas) ;
    2. simulation du chemin complet ;
    3. chemin accepté si pas de collision et si, après la pomme, un chemin
       statique tête -> queue existe ;
    4. sinon, plus court chemin sûr vers la queue ;
    5. sinon, mouvement légal qui garde le plus d'espace accessible.
    """

    name = "BFS v1"

    def reset(self):
        pass

    @staticmethod
    def tail_reachable(body, grow, n):
        # Obstacles statiques : tout le corps sauf tête et queue. Si le
        # serpent grandit au prochain pas, la queue ne bouge pas tout de
        # suite : un chemin de longueur 1 vers elle serait mortel.
        path = bfs_path(body[0], body[-1], set(body[1:-1]), n)
        return path is not None and (len(path) >= 2 or not grow)

    def decide(self, body, grow, direction, apple, n):
        body = tuple(body)
        head, tail = body[0], body[-1]

        # 1-3. Plus court chemin vers la pomme, validé par simulation.
        blocked = set(body) if grow else set(body[:-1])
        path = bfs_path(head, apple, blocked, n)
        if path:
            sim = simulate(body, grow, direction, apple, path, n)
            if sim and sim[3] and self.tail_reachable(sim[0], sim[1], n):
                return path[0]

        # 4. Repli : plus court chemin vers la queue, premier pas vérifié.
        path = bfs_path(head, tail, set(body[1:-1]), n)
        if path and path[0] != opposite(direction):
            nb, dead = step(body, grow, path[0], n)
            if not dead:
                g = nb[0] == apple
                if self.tail_reachable(nb, g, n):
                    return path[0]

        # 5. Dernier recours : maximiser l'espace accessible.
        best, best_area = None, -1
        for d in legal_moves(direction):
            nb, dead = step(body, grow, d, n)
            if dead:
                continue
            area = flood_area(nb[0], set(nb[:-1]), n)
            if area > best_area:
                best, best_area = d, area
        return best if best is not None else direction


# --------------------------------------------------------------------------
# Étape 2 : SafePath (version améliorée)
# --------------------------------------------------------------------------

def free_times(body, grow):
    """Pas t à partir duquel la tête peut entrer dans chaque case du corps.

    Sans croissance, body[i] est libéré après L - i retraits de queue ; le
    retrait a lieu avant le test de collision, donc la tête peut y entrer au
    pas t = L - i. Une croissance en attente retarde tout d'un pas.
    """
    L = len(body)
    g = 1 if grow else 0
    return {c: L - i + g for i, c in enumerate(body)}


def timed_bfs(body, grow, goal, n, avoid=None):
    """BFS spatio-temporel : une case du corps devient traversable dès que
    la queue l'a quittée au moment où la tête y arrive.

    Renvoie (chemin_vers_goal ou None, échappatoire, cases_libres_atteintes)
    où « échappatoire » signifie que la tête peut rejoindre une case du corps
    au moment où elle se libère : le serpent peut alors suivre sa propre
    trace indéfiniment (survie garantie tant qu'il ne mange pas). `avoid`
    (la pomme) est exclue, car la manger retarderait la queue d'un pas.
    """
    ft = free_times(body, grow)
    start = body[0]
    parent = {start: None}
    queue = deque([(start, 0)])
    escape = False
    path = None
    free = 0
    while queue:
        cell, t = queue.popleft()
        for d in DIRECTIONS:
            nxt = neighbor(cell, d, n)
            if nxt in parent or (nxt == avoid and nxt != goal):
                continue
            f = ft.get(nxt)
            if f is not None:
                if t + 1 < f:
                    continue
                escape = True
            else:
                free += 1
            parent[nxt] = (cell, d)
            if nxt == goal and path is None:
                path, c = [], nxt
                while parent[c] is not None:
                    c, dd = parent[c]
                    path.append(dd)
                path.reverse()
            queue.append((nxt, t + 1))
    return path, escape, free


def deep_escape(body, grow, direction, apple, n, depth, memo=None):
    """Échappatoire avec anticipation : vraie si une suite d'au plus `depth`
    coups réellement simulés mène à un état qui a une échappatoire.

    Plus juste que timed_bfs seul, qui ne considère que l'arrivée au plus
    tôt et ignore qu'on peut « gagner du temps » dans une poche libre.
    """
    if memo is None:
        memo = set()
    if timed_bfs(body, grow, None, n, avoid=apple)[1]:
        return True
    if depth == 0 or (body, direction) in memo:
        return False
    memo.add((body, direction))
    for d in legal_moves(direction):
        nb, dead = step(body, grow, d, n)
        if dead:
            continue
        if nb[0] == apple:
            if timed_bfs(nb, True, None, n)[1]:
                return True
            continue
        if deep_escape(nb, False, d, apple, n, depth - 1, memo):
            return True
    return False


class SafePathAgent:
    """Version 2 « SafePath ».

    Changements par rapport à BFSAgent :
      A. BFS spatio-temporel : une case du corps est traversable si la queue
         l'aura libérée à l'instant d'arrivée (au lieu d'un corps figé) ;
      B. plusieurs candidats : un plus court chemin par premier mouvement
         légal, chacun simulé ; on garde le plus court qui est sûr, à
         égalité celui qui laisse le plus d'espace libre ;
      C. critère de sécurité après la pomme plus juste : échappatoire
         temporelle (rejoindre n'importe quelle case du corps au moment où
         elle se libère) au lieu d'un chemin statique vers la queue ;
      D. détection d'impasse : dans les replis, un mouvement sans
         échappatoire n'est pris que si rien d'autre n'existe, et la pomme
         est exclue de l'échappatoire (la manger retarde la queue) ;
      E. anti-boucle : les états (corps, direction) visités depuis la
         dernière pomme sont comptés ; les replis préfèrent les moins
         visités et le chemin le plus long vers la queue. Si un état se
         répète (cycle certain, le jeu étant déterministe), le test de
         sécurité passe à `deep_escape`, plus permissif mais toujours
         prouvé par simulation, pour sortir du cycle.
    """

    name = "SafePath"
    LOOP_DEPTH = 6

    def __init__(self):
        self.visits = {}
        self._last_apple = None

    def reset(self):
        self.visits = {}
        self._last_apple = None

    def decide(self, body, grow, direction, apple, n):
        body = tuple(body)
        if apple != self._last_apple:
            self.visits.clear()
            self._last_apple = apple
        key = (body, direction)
        self.visits[key] = self.visits.get(key, 0) + 1
        looping = self.visits[key] > 1  # E

        # B. Un candidat par premier mouvement légal.
        options = []
        for d in legal_moves(direction):
            nb, dead = step(body, grow, d, n)
            if dead:
                continue
            if nb[0] == apple:
                path = [d]
            else:
                # grow_pending vient d'être consommé par ce premier pas.
                rest = timed_bfs(nb, False, apple, n)[0]  # A
                path = None if rest is None else [d] + rest
            options.append((d, nb, path))

        best = None
        for d, nb, path in options:
            if path is None:
                continue
            sim = simulate(body, grow, direction, apple, path, n)
            if not sim or not sim[3]:
                continue
            _, escape, area = timed_bfs(sim[0], sim[1], None, n)  # C
            if not escape and looping:
                escape = deep_escape(sim[0], sim[1], sim[2], None, n,
                                     self.LOOP_DEPTH)
            if not escape:
                continue
            rank = (len(path), -area)
            if best is None or rank < best[0]:
                best = (rank, d)
        if best is not None:
            return best[1]

        # Repli : survivre en prenant de la place, sans tourner en rond.
        best = None
        for d, nb, _ in options:
            g = nb[0] == apple
            _, escape, area = timed_bfs(nb, g, None, n, avoid=apple)  # D
            if not escape and looping and not g:
                escape = deep_escape(nb, False, d, apple, n, self.LOOP_DEPTH)
            seen = self.visits.get((nb, d), 0)
            tail_path = bfs_path(nb[0], nb[-1], set(nb[1:-1]), n)
            tail_dist = len(tail_path) if tail_path is not None else -1
            rank = (escape, -seen, area, tail_dist)
            if best is None or rank > best[0]:
                best = (rank, d)
        return best[1] if best is not None else direction


# --------------------------------------------------------------------------
# Circuit dynamique (algorithme par défaut)
# --------------------------------------------------------------------------

def torus_cycle(n):
    """Circuit passant une fois par chaque case de la grille torique n x n.

    Lignes parcourues en zigzag sur les colonnes 1..n-1 (ligne paire de
    droite à gauche, impaire de gauche à droite), puis retour par la
    colonne 0 ; le bord torique (0,0)-(n-1,0) ferme le circuit. Avec cette
    orientation, le serpent de départ du jeu (ligne n//2 impaire, vers la
    droite) est déjà posé dans l'ordre du circuit. Sert de circuit initial.
    """
    order = []
    for y in range(n):
        xs = range(n - 1, 0, -1) if y % 2 == 0 else range(1, n)
        order.extend((x, y) for x in xs)
    order.extend((0, y) for y in range(n - 1, -1, -1))
    return order


def hamilton_completion(start, unvisited, targets, nbrs, budget, deadline=None,
                        ring=None):
    """Chemin partant de `start` (exclu) qui passe par toutes les cases de
    `unvisited` exactement une fois et finit sur une case de `targets`.

    Profondeur d'abord avec règle de Warnsdorff (case la plus contrainte
    d'abord, puis la plus éloignée de l'arrivée pour que le chemin y
    termine naturellement). Élagages : case sans issue, plus aucune case
    d'arrivée disponible, zone libre coupée en deux (test global lancé
    seulement si le coup peut couper, d'après les 8 cases voisines).
    `budget` borne le nombre de nœuds et `deadline` (time.perf_counter) la
    durée. Renvoie la liste des cases ou None. `unvisited` est modifié
    pendant la recherche puis restauré.
    """
    U = unvisited
    if not U:
        return [] if start in targets else None
    path = []
    count = [budget]
    live_targets = [sum(1 for t in targets if t in U)]

    # Distance (dans les cases libres) à l'arrivée, pour départager.
    dist = {t: 0 for t in targets if t in U}
    queue = deque(dist)
    while queue:
        c = queue.popleft()
        for m in nbrs[c]:
            if m in U and m not in dist:
                dist[m] = dist[c] + 1
                queue.append(m)

    def free_deg(c):
        return sum(1 for m in nbrs[c] if m in U)

    def may_split(c):
        # Retirer c peut-il couper U ? On compte les groupes de cases de U
        # autour de c (anneau des 8 voisins, les diagonales reliant les
        # voisins orthogonaux). Un seul groupe : pas de coupure possible.
        if ring is None:
            return True
        cells = ring[c]
        inside = [x in U for x in cells]
        groups = 0
        for i in range(0, 8, 2):  # voisins orthogonaux : indices pairs
            if inside[i] and not (inside[i - 1] and inside[i - 2]):
                groups += 1
        return groups > 1

    def connected(cur):
        seeds = [m for m in nbrs[cur] if m in U]
        if not seeds:
            return False
        seen = {seeds[0]}
        stack = [seeds[0]]
        while stack:
            c = stack.pop()
            for m in nbrs[c]:
                if m in U and m not in seen:
                    seen.add(m)
                    stack.append(m)
        return len(seen) == len(U)

    def dfs(cur):
        count[0] -= 1
        if count[0] < 0:
            return False
        if deadline is not None and count[0] % 16 == 0 and time.perf_counter() > deadline:
            count[0] = -1
            return False
        cand = [m for m in nbrs[cur] if m in U]
        cand.sort(key=lambda m: (free_deg(m), -dist.get(m, 0)))
        for m in cand:
            U.discard(m)
            path.append(m)
            is_t = m in targets
            if is_t:
                live_targets[0] -= 1
            if not U:
                if is_t:
                    return True
            elif live_targets[0] > 0:
                ok = True
                for u in nbrs[cur]:
                    if u in U:
                        a = free_deg(u) + (1 if u in nbrs[m] else 0)
                        if a == 0 or (a == 1 and u not in targets):
                            ok = False
                            break
                if ok and free_deg(m) == 0:
                    ok = False
                if ok and may_split(m) and not connected(m):
                    ok = False
                if ok and dfs(m):
                    return True
            if is_t:
                live_targets[0] += 1
            path.pop()
            U.add(m)
        return False

    return path if dfs(start) else None


def splice_completion(start, old_path, removed, targets, nbrs, budget, deadline=None):
    """Complétion rapide en recousant l'ancien circuit.

    `old_path` couvre déjà toutes les cases libres. En retirant les cases du
    nouveau chemin vers la pomme (`removed`), il reste des tronçons intacts
    (suites de cases consécutives de l'ancien chemin). On cherche un ordre
    et un sens de parcours de ces tronçons tel que chaque tronçon commence
    à côté de la fin du précédent, en partant de `start` et en finissant
    sur une case de `targets`. Quelques dizaines de tronçons au lieu de
    centaines de cases : la recherche est beaucoup plus rapide.
    """
    segs = []
    cur_seg = []
    for c in old_path:
        if c in removed:
            if cur_seg:
                segs.append(cur_seg)
                cur_seg = []
        else:
            cur_seg.append(c)
    if cur_seg:
        segs.append(cur_seg)
    if not segs:
        return [] if start in targets else None
    adj = {c: set(nbrs[c]) for c in nbrs}
    remaining = set(range(len(segs)))
    order = []
    count = [budget]

    def options(cur):
        out = []
        for i in remaining:
            sg = segs[i]
            if sg[0] in adj[cur]:
                out.append((i, False))
            if len(sg) > 1 and sg[-1] in adj[cur]:
                out.append((i, True))
        return out

    def dfs(cur):
        if not remaining:
            return cur in targets
        count[0] -= 1
        if count[0] < 0:
            return False
        if deadline is not None and count[0] % 32 == 0 and time.perf_counter() > deadline:
            count[0] = -1
            return False
        opts = options(cur)
        # Tronçon le plus contraint d'abord (le moins d'accroches possibles).
        opts.sort(key=lambda o: len(options(segs[o[0]][0 if o[1] else -1])))
        for i, rev in opts:
            remaining.discard(i)
            order.append((i, rev))
            end = segs[i][0] if rev else segs[i][-1]
            if dfs(end):
                return True
            order.pop()
            remaining.add(i)
        return False

    if not dfs(start):
        return None
    out = []
    for i, rev in order:
        out.extend(reversed(segs[i]) if rev else segs[i])
    return out


class DynamicCycleAgent:
    """Circuit hamiltonien dynamique.

    Le serpent suit toujours un chemin `self.path` qui part de la tête,
    passe par TOUTES les cases libres et finit à côté de la queue : avec le
    corps, il forme un circuit couvrant la grille. La case suivante est donc
    toujours libre (sûreté garantie), et la case libérée par la queue
    s'ajoute en bout de chemin.

    À chaque pas où la pomme n'est pas atteinte au plus court, on tente de
    reconstruire le circuit : plus court chemin (BFS, cases libres) jusqu'à
    la pomme, puis complétion hamiltonienne des autres cases libres jusqu'à
    un voisin de la queue. Si la recherche échoue (budget), on garde le
    circuit actuel, toujours valide. Départ : circuit en zigzag fixe.
    """

    name = "Circuit dynamique"
    BUDGET = 3000     # nœuds de recherche par tentative
    PREFIXES = 3      # plus courts chemins différents essayés vers la pomme
    TIME_LIMIT = 0.12 # s : la décision tient dans un tick (200 ms) du jeu

    def __init__(self):
        self.path = None
        self.n = None

    def reset(self):
        self.path = None

    def _init(self, body, n):
        self.n = n
        self.nbrs = {(x, y): [neighbor((x, y), d, n) for d in DIRECTIONS]
                     for x in range(n) for y in range(n)}
        # 8 voisins dans l'ordre circulaire, orthogonaux aux indices pairs.
        around = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        self.ring = {(x, y): [((x + dx) % n, (y + dy) % n) for dx, dy in around]
                     for x in range(n) for y in range(n)}
        order = torus_cycle(n)
        idx = {c: i for i, c in enumerate(order)}
        N = n * n
        i = (idx[body[0]] + 1) % N
        path = []
        while order[i] != body[-1]:
            path.append(order[i])
            i = (i + 1) % N
        self.path = path

    def _sync(self, body, n):
        """Ajoute en bout de chemin la case que la queue vient de libérer."""
        free = self.n * self.n - len(body)
        if len(self.path) < free:
            # La case libérée est celle qui suit le dernier élément du chemin
            # dans l'ancien circuit, c'est-à-dire l'ancienne queue ; on la
            # retrouve comme case libre absente du chemin.
            on = set(self.path) | set(body)
            missing = [c for c in self.nbrs if c not in on]
            self.path.extend(missing)

    def _shortest_prefixes(self, head, apple, blocked):
        """Jusqu'à PREFIXES plus courts chemins distincts head -> apple."""
        dist = {head: 0}
        queue = deque([head])
        while queue:
            c = queue.popleft()
            if c == apple:
                break
            for m in self.nbrs[c]:
                if m not in dist and m not in blocked:
                    dist[m] = dist[c] + 1
                    queue.append(m)
        if apple not in dist:
            return []
        # Remonte depuis la pomme en variant le choix du prédécesseur.
        out = []
        for variant in range(self.PREFIXES):
            p = [apple]
            c = apple
            k = variant
            while c != head:
                preds = [m for m in self.nbrs[c] if dist.get(m) == dist[c] - 1]
                c = preds[k % len(preds)]
                k //= max(1, len(preds))
                p.append(c)
            p = p[-2::-1]  # sans la tête, de la tête vers la pomme
            if p not in out:
                out.append(p)
        return out

    def _rebuild(self, body, apple, n):
        deadline = time.perf_counter() + self.TIME_LIMIT
        head, tail = body[0], body[-1]
        body_set = set(body)
        free = set(self.path)
        targets = {c for c in self.nbrs[tail] if c in free}
        prefixes = self._shortest_prefixes(head, apple, body_set)
        # 1) Recoudre les tronçons de l'ancien circuit (rapide).
        for prefix in prefixes:
            rest = splice_completion(prefix[-1], self.path, set(prefix), targets,
                                     self.nbrs, self.BUDGET, deadline)
            if rest is not None:
                return prefix + rest
        # 2) Sinon, recherche case par case (plus lente).
        for prefix in prefixes:
            if time.perf_counter() > deadline:
                break
            U = free - set(prefix)
            rest = hamilton_completion(prefix[-1], U, targets, self.nbrs, self.BUDGET,
                                       deadline, self.ring)
            if rest is not None:
                return prefix + rest
        return None

    def decide(self, body, grow, direction, apple, n):
        body = tuple(body)
        if self.path is None or self.n != n:
            self._init(body, n)
        else:
            self._sync(body, n)
        if apple in self.path:
            k = self.path.index(apple)
            hx, hy = body[0]
            ax, ay = apple
            dx, dy = abs(hx - ax), abs(hy - ay)
            manhattan = min(dx, n - dx) + min(dy, n - dy)
            if k + 1 > manhattan:
                new = self._rebuild(body, apple, n)
                if new is not None and new.index(apple) < k:
                    self.path = new
        nxt = self.path.pop(0)
        for d in DIRECTIONS:
            if neighbor(body[0], d, n) == nxt:
                return d
        return direction  # ne devrait pas arriver
