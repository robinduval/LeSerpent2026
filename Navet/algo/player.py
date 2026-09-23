# Navet — Étape 2 : cerveau algorithmique (Dijkstra torique + flood-fill, puis switch Hamilton).
# Même interface que l'agent RL : on lit SnakeGame (ia/game.py, réutilisé tel quel, aucune
# logique de jeu dupliquée) et on renvoie l'action relative one-hot [tout droit, droite, gauche].
#
# Cellules en index plat p = y * G + x ; voisins précalculés avec wrap torique.
# Obstacles "temporels" : le segment i du corps (0 = tête, L-1 = queue) se libère au pas
# L - i + pending. Une cellule du corps est donc franchissable si on y arrive assez tard :
# c'est la généralisation exacte de "la queue exclue si elle va se libérer".
import heapq
import random
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ia"))
from game import GRID_SIZE, RIGHT, DOWN, LEFT, UP, SnakeGame  # noqa: E402

G = GRID_SIZE
N = G * G
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]  # même ordre que game.play_step
INF = float("inf")
NEVER = 2.0  # switch_fill > 1 : on ne passe jamais en Hamilton
SHORTCUT_MARGIN = 4  # marge fixe exigée pour un raccourci Hamilton (pommes futures inconnues)


def idx(x, y):
    return y * G + x


def xy(p):
    return p % G, p // G


# NB[p][d] = voisin de p dans la direction CLOCKWISE[d] (tore)
NB = [[idx((p % G + dx) % G, (p // G + dy) % G) for (dx, dy) in CLOCKWISE] for p in range(N)]


class Cycle:
    """Cycle hamiltonien : ordre de parcours des N cases et position de chaque case."""

    def __init__(self, order):
        self.order = order
        self.pos = [0] * N
        for i, p in enumerate(order):
            self.pos[p] = i

    def dist(self, a, b):
        """Distance le long du cycle, de a vers b (sens de parcours)."""
        return (self.pos[b] - self.pos[a]) % N

    def succ(self, p):
        return self.order[(self.pos[p] + 1) % N]


def build_helix():
    """Cycle hamiltonien du tore G x G, en hélice : la colonne k est descendue en partant
    de y = -k (mod G). Fin de colonne k = (k, -k-1) -> début de colonne k+1 = (k+1, -k-1),
    voisins horizontaux ; après G colonnes le décalage vaut 0 et on revient sur (0, 0).
    (Un boustrophédon classique ne se ferme pas sur une grille impaire, l'hélice si.)"""
    order = []
    for k in range(G):
        y0 = (-k) % G
        order += [idx(k, (y0 + j) % G) for j in range(G)]
    assert sorted(order) == list(range(N))
    assert all(order[(i + 1) % N] in NB[order[i]] for i in range(N)), "cycle non contigu"
    return Cycle(order)


HELIX = build_helix()


def toric_dist(a, b):
    dx = abs(a % G - b % G)
    dy = abs(a // G - b // G)
    return min(dx, G - dx) + min(dy, G - dy)


COMPLETE_BUDGET = 4000  # noeuds max explorés par tentative de construction de cycle
def complete_cycle(body, apple, budget=COMPLETE_BUDGET):
    """Deux essais : viser la pomme d'abord (budget court — une mauvaise première branche
    ne se rattrape pas en profondeur), sinon Warnsdorff pur (la pomme où elle tombe)."""
    if apple is not None:
        cyc = _complete(body, apple, budget // 4)
        if cyc is not None:
            return cyc
    return _complete(body, None, budget)


def complete_candidates(body, apple, k, budget=COMPLETE_BUDGET):
    """k cycles d'A différents (le 1er = complete_cycle, les autres avec égalités tirées au
    hasard) ; sert à choisir celui qui laisse l'espace libre le plus large."""
    out = []
    first = complete_cycle(body, apple, budget)
    if first is not None:
        out.append(first)
    for r in range(1, k):
        cyc = _complete(body, apple, budget // 4, random.Random(apple * 131 + r))
        if cyc is not None:
            out.append(cyc)
    return out


def free_width(body):
    """Largeur de l'espace libre : nombre moyen de voisins libres par case libre
    (≈ 2 dans un couloir, → 4 dans un espace large)."""
    occ = [False] * N
    for c in body:
        occ[c] = True
    free = [c for c in range(N) if not occ[c]]
    if not free:
        return 0.0
    return sum(1 for c in free for w in NB[c] if not occ[w]) / len(free)


def _complete(body, apple, budget, rng=None):
    """Cycle construit à partir du corps : chemin hamiltonien tête -> toutes les cases
    libres -> queue. Corps + chemin = cycle où le corps est déjà rangé (tête devant,
    queue derrière). Recherche en profondeur bornée : on vise d'abord la pomme (distance
    torique), puis Warnsdorff (case la plus contrainte d'abord) pour ne pas laisser de trous.
    Élagage : une case libre doit garder >= 2 issues (sauf passage forcé), la queue >= 1,
    et les cases restantes doivent rester connexes. Renvoie un Cycle ou None."""
    head, tail = body[0], body[-1]
    unvisited = [True] * N
    for c in body:
        unvisited[c] = False
    remaining = N - len(body)
    if remaining == 0:
        return None
    # deg[u] = issues de u parmi (cases libres non visitées) + queue
    deg = [0] * N
    for u in range(N):
        if unvisited[u]:
            deg[u] = sum(1 for w in NB[u] if unvisited[w] or w == tail)
    tail_deg = [sum(1 for w in NB[tail] if unvisited[w])]
    path = []
    nodes = [0]
    apple_seen = [apple is None]

    def visit(v):
        unvisited[v] = False
        for w in NB[v]:
            deg[w] -= 1
        if v in NB[tail]:
            tail_deg[0] -= NB[tail].count(v)

    def unvisit(v):
        unvisited[v] = True
        for w in NB[v]:
            deg[w] += 1
        if v in NB[tail]:
            tail_deg[0] += NB[tail].count(v)

    def connected(cur, left):
        # les cases restantes sont-elles toutes atteignables depuis cur ?
        seen = {cur}
        stack = [cur]
        count = 0
        while stack:
            u = stack.pop()
            for w in NB[u]:
                if unvisited[w] and w not in seen:
                    seen.add(w)
                    count += 1
                    stack.append(w)
        return count == left

    def dfs(cur, left):
        if left == 0:
            return tail in NB[cur]
        nodes[0] += 1
        if nodes[0] > budget:
            return False
        if tail_deg[0] == 0:
            return False  # plus aucune case libre ne peut rejoindre la queue
        cands = [w for w in NB[cur] if unvisited[w]]
        forced = [w for w in cands if deg[w] <= 1]  # cur est sa dernière issue possible
        if len(forced) > 1:
            return False
        if forced:
            cands = forced
        elif not apple_seen[0]:
            cands.sort(key=lambda w: (toric_dist(w, apple), deg[w], rng.random() if rng else 0))
        else:
            cands.sort(key=lambda w: (deg[w], rng.random() if rng else 0))
        for w in cands:
            if deg[w] == 0 and left > 1:
                continue  # impasse : w n'aurait plus de sortie
            visit(w)
            if w == apple:
                apple_seen[0] = True
            path.append(w)
            # les autres voisins de cur perdent cur comme issue : il leur en faut encore 2
            ok = all(deg[u] + (1 if u in NB[w] else 0) >= 2 for u in NB[cur] if unvisited[u])
            if ok and (left - 1 < 3 or connected(w, left - 1)) and dfs(w, left - 1):
                return True
            path.pop()
            if w == apple:
                apple_seen[0] = False
            unvisit(w)
            if nodes[0] > budget:
                return False
        return False

    if not dfs(head, remaining):
        return None
    return Cycle(body[::-1] + path)  # queue ... tête, puis cases libres, puis retour queue


TIMED_BUDGET = 3000  # noeuds max explorés par tentative de construction B
TRAP_BUDGET, TRAP_TRIES = 20000, 40  # avant de se déclarer coincé : recherche plus poussée


def timed_cycle(body, pending, apple, margin=0, budget=TIMED_BUDGET, tries=6, seed=0):
    """Relances : ce type de recherche a des échecs à queue lourde (une mauvaise première
    branche ne se rattrape pas) ; plusieurs essais courts avec un peu d'aléa dans l'ordre
    des voisins valent mieux qu'un long. Le premier essai vise la pomme, les autres non."""
    rng = random.Random(seed)
    per = max(1, budget // tries)
    for t in range(tries):
        cyc = _timed(body, pending, apple if t == 0 else None, margin, per,
                     None if t == 0 else rng)
        if cyc is not None:
            return cyc
    return None


def timed_need(body, pending, margin):
    """Position minimale de chaque case du corps dans un cycle B (voir _timed)."""
    L = len(body)
    need = [0] * N
    for i in range(1, L):
        need[body[i]] = min(L - i + pending + 1 + margin, N - i)
    return need


def _timed(body, pending, apple, margin, budget, rng):
    """Option B : cycle hamiltonien de TOUTE la grille, parcouru depuis la tête, où chaque
    case du corps n'est atteinte qu'après sa libération. Le segment i (0 = tête) doit être à
    une position k >= L - i + pending + 1 (+1 : la pomme connue peut être mangée avant)
    + margin (pommes futures), plafonnée à N - i (position d'un corps parfaitement rangé).
    L'espace libre n'a pas besoin d'être d'un seul tenant : une poche fermée par le corps
    s'ouvre quand la queue avance. Par construction, cycle_safe() est vrai sur le résultat.
    Recherche en profondeur bornée, même élagage que _complete (issues, connexité)."""
    head = body[0]
    need = timed_need(body, pending, margin)
    unvisited = [True] * N
    unvisited[head] = False
    # deg[u] = issues de u parmi (cases non visitées) + tête (fermeture du cycle)
    deg = [len(NB[u]) for u in range(N)]
    head_deg = [sum(1 for w in NB[head] if unvisited[w])]
    path = []
    nodes = [0]
    apple_seen = [apple is None]

    def visit(v):
        unvisited[v] = False
        for w in NB[v]:
            deg[w] -= 1
        if v in NB[head]:
            head_deg[0] -= 1

    def unvisit(v):
        unvisited[v] = True
        for w in NB[v]:
            deg[w] += 1
        if v in NB[head]:
            head_deg[0] += 1

    def connected(cur, left):
        seen = {cur}
        stack = [cur]
        count = 0
        while stack:
            u = stack.pop()
            for w in NB[u]:
                if unvisited[w] and w not in seen:
                    seen.add(w)
                    count += 1
                    stack.append(w)
        return count == left

    def dfs(cur, k, left):
        # k = position (offset depuis la tête) de la prochaine case
        if left == 0:
            return head in NB[cur]
        nodes[0] += 1
        if nodes[0] > budget or head_deg[0] == 0:
            return False
        cands = [w for w in NB[cur] if unvisited[w]]
        forced = [w for w in cands if deg[w] <= 1]
        if len(forced) > 1:
            return False
        if forced:
            cands = forced
        cands = [w for w in cands if k >= need[w]]  # pas encore libérée : interdite ici
        if not apple_seen[0]:
            cands.sort(key=lambda w: (toric_dist(w, apple), deg[w]))
        elif rng is None:
            cands.sort(key=lambda w: (deg[w], -need[w]))
        else:
            cands.sort(key=lambda w: (deg[w], rng.random()))
        for w in cands:
            if deg[w] == 0 and left > 1:
                continue
            visit(w)
            if w == apple:
                apple_seen[0] = True
            path.append(w)
            ok = all(deg[u] + (1 if u in NB[w] else 0) >= 2 for u in NB[cur] if unvisited[u])
            if ok and (left - 1 < 3 or connected(w, left - 1)) and dfs(w, k + 1, left - 1):
                return True
            path.pop()
            if w == apple:
                apple_seen[0] = False
            unvisit(w)
            if nodes[0] > budget:
                return False
        return False

    if not dfs(head, 1, N - 1):
        return None
    return Cycle([head] + path)


# --- primitives -----------------------------------------------------------------

def free_times(body, pending):
    ft = [0] * N
    L = len(body)
    for i, c in enumerate(body):
        ft[c] = L - i + pending
    return ft


def dijkstra(start, ft, target=-1):
    """Dijkstra torique depuis start (coût 1 par pas, voisins avec wrap). Une cellule n'est
    acceptée que si on y arrive au plus tôt quand elle est libre (ft). S'arrête sur target
    si donné ; renvoie (dist, prev, ordre de visite)."""
    dist = [INF] * N
    prev = [-1] * N
    dist[start] = 0
    heap = [(0, start)]
    order = []
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        order.append(u)
        if u == target:
            break
        nd = d + 1
        for v in NB[u]:
            if nd < ft[v] or nd >= dist[v]:
                continue
            dist[v] = nd
            prev[v] = u
            heapq.heappush(heap, (nd, v))
    return dist, prev, order


def path_to(prev, start, target):
    if target != start and prev[target] < 0:
        return None
    path = []
    while target != start:
        path.append(target)
        target = prev[target]
    return path[::-1]


def flood(start, ft, target=-1):
    """Flood-fill temporel depuis start : (nb de cellules atteignables, distance à target ou -1)."""
    seen = [False] * N
    seen[start] = True
    q = deque([(start, 0)])
    count, tdist = 0, -1
    while q:
        u, d = q.popleft()
        nd = d + 1
        for v in NB[u]:
            if seen[v] or nd < ft[v]:
                continue
            seen[v] = True
            count += 1
            if v == target:
                tdist = nd
            q.append((v, nd))
    return count, tdist


def advance(body, pending, path, apple):
    """Serpent virtuel après avoir suivi path (même mécanique que play_step :
    insertion tête, pop sauf pending, la pomme mangée arme pending).
    La pomme ne peut être qu'en fin de chemin (Dijkstra s'y arrête, _find_entry l'évite)."""
    new_len = len(body) + (pending if path else 0)
    new_body = (path[::-1] + body)[:new_len]
    new_pending = 1 if path and path[-1] == apple else 0
    return new_body, new_pending


def cycle_safe(cyc, body, pending, apple, margin=0):
    """Test exact : suivre le cycle depuis la tête ne mord jamais le corps actuel.
    La tête atteint la cellule du cycle à l'offset k au pas k ; le segment i est libre au pas
    L - i + croissance (pending + 1 si la pomme connue est mangée avant). Marche aussi avec un
    corps "dans le désordre" (switch en cours de partie) ; un corps rangé dans le cycle passe
    toujours. margin = marge supplémentaire exigée (raccourcis, pommes futures inconnues)."""
    head = body[0]
    L = len(body)
    a = cyc.dist(head, apple) if apple is not None else N
    win_on_eat = L + pending >= N  # la pomme connue est la dernière case : manger = victoire
    for i in range(1, L):
        k = cyc.dist(head, body[i])
        if win_on_eat and k > a:
            continue
        if k < L - i + pending + (1 if a < k else 0) + margin:
            return False
    return True


def unfragmented(body):
    """Espace libre « d'un seul tenant » : toutes les cases libres forment un seul îlot,
    touché par la tête et par la queue, et aucune case libre n'est une impasse (moins de
    2 issues parmi cases libres + tête + queue). C'est la condition nécessaire pour qu'un
    chemin tête -> toutes les cases libres -> queue existe (cycle d'A). Coût O(N)."""
    head, tail = body[0], body[-1]
    occ = [False] * N
    for c in body:
        occ[c] = True
    free = [c for c in range(N) if not occ[c]]
    if not free:
        return True
    for c in free:
        if sum(1 for w in NB[c] if not occ[w] or w == head or w == tail) < 2:
            return False
    seen = [False] * N
    seen[free[0]] = True
    stack = [free[0]]
    count = 1
    touch_head = touch_tail = False
    while stack:
        u = stack.pop()
        for w in NB[u]:
            if w == head:
                touch_head = True
            elif w == tail:
                touch_tail = True
            elif not occ[w] and not seen[w]:
                seen[w] = True
                count += 1
                stack.append(w)
    return count == len(free) and touch_head and touch_tail


def to_action(game, target_cell):
    head = idx(*game.head)
    d = NB[head].index(target_cell)
    cur = CLOCKWISE.index(game.direction)
    if d == cur:
        return [1, 0, 0]
    if d == (cur + 1) % 4:
        return [0, 1, 0]
    if d == (cur - 1) % 4:
        return [0, 0, 1]
    return [1, 0, 0]  # aucun coup légal : la partie est perdue de toute façon


# --- le joueur -------------------------------------------------------------------

CYCLE_MODES = ("helice", "reconstruit", "temporel", "ilots", "prudent")
TRAP_FROM = 0.25  # en dessous, jamais coincé en pratique : on n'y paie pas la vérification
RETRY_EVERY = 10  # seuil atteint mais pas de cycle B : nouvel essai tous les N pas


class AlgoPlayer:
    """switch_fill : taux de remplissage (0..1) qui déclenche Hamilton (NEVER = jamais).
    C'est le seul paramètre benchmarké (sweep.py).
    cycle : "helice"      = cycle fixe en hélice + raccourcis (référence 20:14) ;
            "reconstruit" = entrée par l'hélice, puis cycle reconstruit à partir du corps
                            après chaque pomme, sans raccourcis (option A, 20:25) ;
            "temporel"    = cycle B (tient compte de la libération du corps) dès le switch,
                            reconstruit après chaque pomme, sans raccourcis (20:40).
            "ilots"       = switch juste avant que l'espace libre ne se fragmente (îlots,
                            impasses), avec le cycle d'A ; switch_fill ne sert que de
                            plafond de secours (20:55) ;
            "prudent"     = Dijkstra qui évite de fragmenter l'espace libre, switch au seuil
                            (cycle d'A, sinon B), puis reconstructions d'A (21:08).
    trap  : (temporel) switch aussi « juste avant d'être coincé » : dès que le coup Dijkstra
            ne laisserait plus aucun cycle B, on switche sur celui de l'état actuel.
    avoid : (ilots) Dijkstra écarte les coups qui fragmentent l'espace libre, pour faire
            reculer le switch.
    tol   : (ilots) tolérance en pas : un îlot qui se rouvre en <= tol pas (bordé par les tol
            derniers segments de la queue) ne compte pas comme fragmentation."""

    def __init__(self, switch_fill=0.5, cycle="helice", trap=False, avoid=False, tol=0,
                 width_k=1, width_slack=0):
        assert cycle in CYCLE_MODES
        self.switch_fill = switch_fill
        self.cycle_mode = cycle
        self.trap = trap
        self.avoid = avoid
        self.tol = tol
        self.width_k = width_k          # (A) nombre de cycles candidats ; 1 = pas de choix
        self.width_slack = width_slack  # (A) pas de plus tolérés vers la pomme pour un cycle plus large
        self.reset()

    def reset(self):
        self.hamilton = False
        self.cycle = HELIX
        self.tried_apple = None  # dernière pomme pour laquelle une reconstruction a été tentée
        self.mode = "dijkstra"   # affiché : dijkstra / queue / hamilton / raccourci / transition
        self.path = []           # chemin prévu (affichage)
        self.plan = None         # chemin d'entrée dans le cycle (transition)
        self.witness = None      # (temporel) cycle B valable pour l'état actuel si witness_ok
        self.witness_ok = False
        self.next_try = 0
        self.next_rescue = 0
        self.stats = {"switch_step": None, "switch_fill": None, "switch_by": None,
                      "ordered_step": None, "secours": 0, "avoided": 0,
                      "transition_steps": 0, "shortcuts": 0, "rebuilds": 0, "rebuild_kept": 0,
                      "rebuild_fails": 0}

    # -- phase 1 : Dijkstra torique + flood-fill --
    def _safe_after(self, body, pending):
        """Rejet si la queue n'est plus atteignable ou si l'espace atteignable < longueur."""
        ft = free_times(body, pending)
        space, tdist = flood(body[0], ft, body[-1])
        return tdist >= 0 and space >= len(body)

    def _greedy(self, body, pending, apple):
        head = body[0]
        ft = free_times(body, pending)
        path = None
        if apple is not None:
            _, prev, _ = dijkstra(head, ft, apple)
            path = path_to(prev, head, apple)
        if path:
            vb, vp = advance(body, pending, path, apple)
            if self._safe_after(vb, vp):
                self.mode, self.path = "dijkstra", path
                return path[0]
        # filet : suivre sa queue — coup qui garde la queue atteignable, max d'espace,
        # puis le plus long détour vers la queue (on laisse le temps au corps de se libérer)
        best, best_key = None, None
        for v in NB[head]:
            if ft[v] > 1:
                continue
            vb, vp = advance(body, pending, [v], apple)
            space, tdist = flood(v, free_times(vb, vp), vb[-1])
            key = (tdist >= 0, space, tdist)
            if best_key is None or key > best_key:
                best, best_key = v, key
        self.mode, self.path = "queue", []
        return best if best is not None else NB[head][0]

    # -- phase 2 : Hamilton + raccourcis --
    def _hamilton(self, body, pending, apple, retry=False):
        head = body[0]
        ft = free_times(body, pending)
        cyc = self.cycle
        nxt = cyc.succ(head)
        best, best_key = None, None
        for v in NB[head]:
            if ft[v] > 1:
                continue
            if v != nxt and self.cycle_mode != "helice":
                continue  # pas de raccourci : le corps doit rester collé le long du cycle
            vb, vp = advance(body, pending, [v], apple)
            eaten = v == apple
            m = 0 if v == nxt else SHORTCUT_MARGIN
            if not cycle_safe(cyc, vb, vp, None if eaten else apple, m):
                continue
            key = (0 if eaten else cyc.dist(v, apple) if apple is not None else 0, v != nxt)
            if best_key is None or key < best_key:
                best, best_key = v, key
        if best is None and self.cycle_mode in ("temporel", "ilots", "prudent"):
            # le cycle n'est plus sûr (pomme imprévue) : nouveau cycle B depuis l'état actuel,
            # sinon Dijkstra + flood-fill en secours
            # recherche coûteuse : au plus tous les RETRY_EVERY pas en secours
            if not retry and self._steps >= self.next_rescue:
                cyc = timed_cycle(body, pending, apple)
                if cyc is not None:
                    self.cycle = cyc
                    self.stats["rebuilds"] += 1
                    return self._hamilton(body, pending, apple, retry=True)
                self.next_rescue = self._steps + RETRY_EVERY
            self.stats["secours"] += 1
            v = self._greedy(body, pending, apple)
            self.mode = "secours"
            return v
        if best is None:
            # Transition : le corps pris "dans le désordre" n'est pas compatible avec le cycle
            # depuis la tête actuelle. Toutes les cases sont sur le cycle : la question est OÙ
            # y entrer. Dijkstra vers toutes les cases, on retient la plus proche telle qu'en
            # y arrivant (serpent virtuel), suivre le cycle soit sûr ; on s'engage sur ce chemin.
            self.stats["transition_steps"] += 1
            plan = self.plan if self.plan and self._entry_ok(body, pending, apple, self.plan) else None
            if plan is None:
                plan = self._find_entry(body, pending, apple, ft)
            self.plan = plan
            if plan:
                self.mode, self.path = "transition", plan
                v = plan.pop(0)
                return v
            v = self._greedy(body, pending, apple)  # pas d'entrée sûre : Dijkstra + flood-fill
            self.mode = "transition"
            return v
        self.plan = None
        if self.stats["ordered_step"] is None:
            self.stats["ordered_step"] = self._steps
        if best != nxt:
            self.stats["shortcuts"] += 1
        self.mode = "raccourci" if best != nxt else "hamilton"
        self.path = [best]
        return best

    def _entry_ok(self, body, pending, apple, path):
        vb, vp = advance(body, pending, path, apple)
        return cycle_safe(self.cycle, vb, vp, None if apple in path else apple)

    def _find_entry(self, body, pending, apple, ft):
        head = body[0]
        _, prev, order = dijkstra(head, ft)
        for c in order[1:]:  # ordre croissant de distance
            path = path_to(prev, head, c)
            if apple in path[:-1]:
                continue  # on n'avale pas la pomme en route (advance ne gère qu'une pomme finale)
            if self._entry_ok(body, pending, apple, path):
                return path
        return None

    def choose(self, game):
        """Cellule cible absolue (index plat) pour le prochain pas."""
        body = [idx(x, y) for x, y in game.body]
        apple = idx(*game.apple) if game.apple is not None else None
        pending = 1 if game.grow_pending else 0
        self._steps = game.steps
        if not self.hamilton:
            fill = len(body) / N
            if self.cycle_mode == "temporel":
                return self._phase1_timed(body, pending, apple, fill)
            if self.cycle_mode == "ilots":
                return self._phase1_islands(body, pending, apple, fill)
            if self.cycle_mode == "prudent":
                return self._phase1_prudent(body, pending, apple, fill)
            if fill >= self.switch_fill:
                self._switch(self.cycle, fill, "seuil")
        elif (self.cycle_mode != "helice" and apple is not None and apple != self.tried_apple
              and (self.cycle_mode in ("temporel", "ilots", "prudent") or self._on_cycle(body))):
            self._rebuild(body, pending, apple)  # nouvelle pomme : cycle qui y passe au plus tôt
        if self.hamilton:
            return self._hamilton(body, pending, apple)
        return self._greedy(body, pending, apple)

    def _switch(self, cyc, fill, why):
        self.hamilton = True
        self.cycle = cyc
        self.stats["switch_step"] = self._steps
        self.stats["switch_fill"] = fill
        self.stats["switch_by"] = why

    def _phase1_timed(self, body, pending, apple, fill):
        """Phase 1 du mode temporel : Dijkstra + flood-fill, et les deux déclenchements."""
        # 1) seuil atteint : on switche dès qu'un cycle B existe depuis l'état actuel
        if fill >= self.switch_fill and self._steps >= self.next_try:
            cyc = self.witness if self.witness_ok else timed_cycle(body, pending, apple)
            if cyc is not None:
                self._switch(cyc, fill, "seuil")
                return self._hamilton(body, pending, apple)
            self.next_try = self._steps + RETRY_EVERY
        v = self._greedy(body, pending, apple)
        if not self.trap or fill < TRAP_FROM:
            self.witness_ok = False
            return v
        # 2) piège : après le coup Dijkstra, existe-t-il encore un cycle B ? On réessaie
        #    d'abord le témoin actuel (test exact, O(L)), sinon on en cherche un nouveau.
        vb, vp = advance(body, pending, [v], apple)
        if self.witness is not None and cycle_safe(self.witness, vb, vp, None, margin=1):
            self.witness_ok = True
            return v
        known = None if v == apple else apple
        w = timed_cycle(vb, vp, known)
        if w is None:  # échec rapide : souvent une fausse alerte, on insiste avant de conclure
            w = timed_cycle(vb, vp, known, 0, TRAP_BUDGET, TRAP_TRIES, seed=self._steps)
        if w is not None:
            self.witness, self.witness_ok = w, True
            return v
        if self.witness_ok:
            # ce coup nous coincerait : on switche maintenant, sur le cycle de l'état actuel
            self._switch(self.witness, fill, "piege")
            return self._hamilton(body, pending, apple)
        self.witness_ok = False  # déjà coincé (aucun témoin) : on continue Dijkstra
        return v

    def _phase1_islands(self, body, pending, apple, fill):
        """Phase 1 du mode îlots. Le coup Dijkstra ne fragmente pas l'espace libre : on le
        joue. Sinon (option avoid) on cherche un autre coup sûr qui ne fragmente pas, le plus
        proche de la pomme. S'il n'y en a pas : on switche maintenant, tant que l'espace
        libre est encore d'un seul tenant (cycle d'A)."""
        if fill >= self.switch_fill:  # plafond de secours
            cyc = timed_cycle(body, pending, apple)
            if cyc is not None:
                self._switch(cyc, fill, "plafond")
                return self._hamilton(body, pending, apple)
        v = self._greedy(body, pending, apple)
        vb, _ = advance(body, pending, [v], apple)
        if self._clean(vb):
            return v
        clean_now = self._clean(body)
        if not clean_now:
            return v  # déjà fragmenté : on attend que les îlots se rouvrent
        if self.avoid:
            best = self._avoid_move(body, pending, apple, v)
            if best is not None:
                return best
        # aucun coup ne garde l'espace d'un seul tenant : switch maintenant
        cyc = complete_cycle(body, apple) or timed_cycle(body, pending, apple)
        if cyc is not None:
            self.tried_apple = apple
            self._switch(cyc, fill, "ilots")
            return self._hamilton(body, pending, apple)
        return v

    def _avoid_move(self, body, pending, apple, v):
        """Autre coup sûr que v qui garde l'espace libre d'un seul tenant, le plus proche de
        la pomme (Dijkstra torique) ; None s'il n'y en a pas."""
        ft = free_times(body, pending)
        best, best_d = None, None
        for u in NB[body[0]]:
            if u == v or ft[u] > 1:
                continue
            ub, up = advance(body, pending, [u], apple)
            if not self._clean(ub) or not self._safe_after(ub, up):
                continue
            d = 0
            if apple is not None and u != apple:
                dist, _, _ = dijkstra(u, free_times(ub, up), apple)
                d = dist[apple]
            if best_d is None or d < best_d:
                best, best_d = u, d
        if best is not None:
            self.stats["avoided"] += 1
            self.mode, self.path = "evite", [best]
        return best

    def _phase1_prudent(self, body, pending, apple, fill):
        """Dijkstra prudent jusqu'au seuil, puis switch : cycle d'A s'il existe, sinon B,
        sinon nouvel essai RETRY_EVERY pas plus tard."""
        if fill >= self.switch_fill and self._steps >= self.next_try:
            cyc = complete_cycle(body, apple) if self._clean(body) else None
            why = "seuil_A"
            if cyc is None:
                cyc, why = timed_cycle(body, pending, apple), "seuil_B"
            if cyc is not None:
                self.tried_apple = apple
                self._switch(cyc, fill, why)
                return self._hamilton(body, pending, apple)
            self.next_try = self._steps + RETRY_EVERY
        v = self._greedy(body, pending, apple)
        vb, _ = advance(body, pending, [v], apple)
        if self._clean(vb) or not self._clean(body):
            return v  # coup propre, ou espace déjà fragmenté : Dijkstra normal
        best = self._avoid_move(body, pending, apple, v)
        return best if best is not None else v

    def _clean(self, body):
        """Pas d'îlot durable : test d'îlots sur le corps privé de ses tol derniers segments
        (les cases qu'ils occupent seront libres dans <= tol pas)."""
        return unfragmented(body[:max(3, len(body) - self.tol)])

    def _widest(self, body, pending, apple):
        """Parmi width_k cycles d'A, ceux qui atteignent la pomme au plus tard width_slack pas
        après le plus direct ; on garde celui qui laisse l'espace libre le plus large une fois
        la pomme mangée (serpent virtuel)."""
        cands = complete_candidates(body, apple, self.width_k)
        if not cands:
            return None
        head = body[0]
        best_d = min(c.dist(head, apple) for c in cands)
        best, best_w = None, -1.0
        for c in cands:
            d = c.dist(head, apple)
            if d > best_d + self.width_slack:
                continue
            p = c.pos[head]
            path = [c.order[(p + j) % N] for j in range(1, d + 1)]
            vb, _ = advance(body, pending, path, apple)
            w = free_width(vb)
            if w > best_w:
                best, best_w = c, w
        return best

    def _on_cycle(self, body):
        """Corps collé le long du cycle (queue ... tête consécutifs) : le cycle en cours est
        alors lui-même une solution tête -> cases libres -> queue."""
        cyc = self.cycle
        p = cyc.pos[body[0]]
        return all(cyc.order[(p - i) % N] == c for i, c in enumerate(body))

    def _rebuild(self, body, pending, apple):
        self.tried_apple = apple
        if self.cycle_mode == "temporel" or (self.cycle_mode in ("ilots", "prudent")
                                             and not self._on_cycle(body)):
            cyc = timed_cycle(body, pending, apple)
        elif self.width_k > 1:
            cyc = self._widest(body, pending, apple)
        else:
            cyc = complete_cycle(body, apple)  # corps collé au cycle : cycle d'A, le plus direct
        if cyc is None:
            self.stats["rebuild_fails"] += 1
            return
        head = body[0]
        if (cyc.dist(head, apple) < self.cycle.dist(head, apple)
                or not cycle_safe(self.cycle, body, pending, apple)):
            self.cycle = cyc  # garde le nouveau cycle seulement s'il amène plus vite à la pomme
            self.stats["rebuilds"] += 1
        else:
            self.stats["rebuild_kept"] += 1

    def get_action(self, game):
        return to_action(game, self.choose(game))


# --- partie headless (sweep / tests) ----------------------------------------------

LOOP_LIMIT = 3 * N  # pas sans pomme avant de déclarer une boucle ; le cycle en demande <= N
                    # (écart max mesuré dans des parties gagnées : 214 pas)


def new_game(seed):
    # Le jeu de base (serpent-algo.py) n'a pas de faim : on la neutralise (hunger_k énorme)
    # et on détecte nous-mêmes les boucles infinies.
    return SnakeGame(seed=seed, hunger_mode="truncated", hunger_k=1e12)


def play_one(seed, switch_fill=0.5, cycle="helice", trap=False, avoid=False, tol=0):
    game = new_game(seed)
    player = AlgoPlayer(switch_fill, cycle, trap, avoid, tol)
    info = {}
    while not game.done:
        if game.steps_since_apple > LOOP_LIMIT:
            info = {"cause": "boucle"}
            break
        _, _, _, info = game.play_step(player.get_action(game))
    out = {"seed": seed, "score": game.score, "steps": game.steps, "victory": game.victory,
           "cause": info.get("cause", "?"), "fill": len(game.body) / N}
    out.update(player.stats)
    return out


if __name__ == "__main__":
    import time
    for args in ((0.4, "reconstruit"), (NEVER, "ilots"), (NEVER, "ilots", False, True)):
        t = time.time()
        r = play_one(1000, *args)
        print(args, {k: r[k] for k in ("score", "steps", "cause", "switch_fill", "switch_by",
                                       "secours", "avoided", "rebuilds", "rebuild_fails")},
              f"{time.time() - t:.1f}s", flush=True)
