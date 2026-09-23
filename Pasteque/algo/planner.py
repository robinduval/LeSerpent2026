"""Bot de décision pour le Snake torique 15x15 (groupe Pastèque).

Indépendant de Pygame. Le moteur (serpent-algo.py) ne fait que :

    planner = SnakePlanner(width=15, height=15)
    direction = planner.choose_action(snake.body, snake.direction,
                                      apple.position, snake.score,
                                      snake.grow_pending)

Stratégie (Model Predictive Control : on planifie loin, on joue 1 coup,
on replanifie) :

1. simulation exacte des règles du moteur (croissance différée d'un tick,
   collision ``head in body[1:]`` après retrait de la queue, demi-tour ignoré) ;
2. filtre de survie : un coup n'est candidat que s'il laisse un
   « certificat de queue » (BFS temporel : la tête peut atteindre une case
   du corps au moment où elle se libère => on peut suivre son corps à l'infini) ;
3. Beam Search sur les coups certifiés, évaluation rapide sur tous les nœuds,
   évaluation complète (flood fill + certificat) sur le top K et sur chaque
   état qui vient de manger ;
4. A* (temporel) vers la pomme, chemin simulé puis certifié, comme filet si le
   beam n'a pas eu le temps de trouver la pomme ;
5. poursuite de la queue quand la pomme n'est atteignable que dangereusement ;
6. cycle Hamiltonien avec raccourcis prouvés sûrs à forte densité ou en cas de
   stagnation.

Objectif (lexicographique) : maximiser P(score = 100), puis minimiser
ticks_to_100. La partie s'arrête à 100 : la 100e pomme est terminale, donc à
99 on ne demande plus de certificat après le repas (mode « endgame ») ; un
chemin simulé qui l'atteint vivant est une victoire garantie. Le poids de la
sécurité croît avec la progression : risk = 1 + 10 * (score / 100) ** 3.

Priorité : SURVIE > PROGRESSION VERS 100 > RAPIDITÉ > ESPACE / OPTIONS.
"""

from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass, field, fields
from typing import Optional

# --------------------------------------------------------------------------
# Plateau et tables pré-calculées
# --------------------------------------------------------------------------

W = H = 15
N = W * H

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
# Vecteurs identiques à ceux du moteur (UP = (0, -1), ...).
VECTORS = ((0, -1), (0, 1), (-1, 0), (1, 0))
VECTOR_TO_ACTION = {v: a for a, v in enumerate(VECTORS)}
OPPOSITE = (DOWN, UP, RIGHT, LEFT)
# Coups autorisés depuis une direction : tout sauf le demi-tour (règle du moteur).
LEGAL = tuple(tuple(a for a in ACTIONS if a != OPPOSITE[d]) for d in ACTIONS)

FULL = (1 << N) - 1
BIT = tuple(1 << c for c in range(N))
COL_FIRST = sum(BIT[y * W] for y in range(H))
COL_LAST = sum(BIT[y * W + W - 1] for y in range(H))
ROW_FIRST = sum(BIT[x] for x in range(W))
NOT_COL_FIRST = FULL & ~COL_FIRST
NOT_COL_LAST = FULL & ~COL_LAST


def cell_of(x: int, y: int) -> int:
    return y * W + x


def xy_of(cell: int) -> tuple[int, int]:
    return cell % W, cell // W


def _build_neighbors():
    table = []
    for c in range(N):
        x, y = xy_of(c)
        table.append(tuple(cell_of((x + dx) % W, (y + dy) % H) for dx, dy in VECTORS))
    return tuple(table)


def _build_distance():
    table = []
    for a in range(N):
        ax, ay = xy_of(a)
        row = []
        for b in range(N):
            bx, by = xy_of(b)
            dx = abs(ax - bx)
            dy = abs(ay - by)
            row.append(min(dx, W - dx) + min(dy, H - dy))
        table.append(tuple(row))
    return tuple(table)


NEIGHBORS = _build_neighbors()   # NEIGHBORS[cell][action] -> cell (wrap inclus)
DISTANCE = _build_distance()     # DISTANCE[a][b] -> distance torique


def toroidal_distance(a: int, b: int) -> int:
    return DISTANCE[a][b]


def expand(m: int) -> int:
    """Voisinage (4-connexe, torique) d'un ensemble de cases en bitset."""
    return (((m & NOT_COL_LAST) << 1) | ((m & COL_LAST) >> (W - 1))
            | ((m & NOT_COL_FIRST) >> 1) | ((m & COL_FIRST) << (W - 1))
            | ((m << W) & FULL) | (m >> (N - W))
            | (m >> W) | ((m & ROW_FIRST) << (N - W)))


# --------------------------------------------------------------------------
# Cycle Hamiltonien sur le tore
# --------------------------------------------------------------------------

def build_hamiltonian_cycle() -> tuple[int, ...]:
    """Colonne 0 de haut en bas, puis serpentin sur les colonnes 1..W-1 en
    remontant ; H impair => la dernière ligne (0) finit en (W-1, 0) qui est
    voisine de (0, 0) grâce au wrap."""
    order = [cell_of(0, y) for y in range(H)]
    for row in range(H - 1, -1, -1):
        xs = range(1, W) if (H - 1 - row) % 2 == 0 else range(W - 1, 0, -1)
        order.extend(cell_of(x, row) for x in xs)
    return tuple(order)


def build_helical_cycle() -> tuple[int, ...]:
    """Cycle hélicoïdal propre au tore : 14 pas à droite puis 1 vers le bas,
    15 fois. Chaque ligne commence une colonne plus à gauche que la
    précédente ; après 15 lignes, le dernier pas (wrap vertical) ferme le
    cycle. Forme très différente du serpentin : plus de variantes à essayer
    pour rejoindre un cycle avec un corps long."""
    order = []
    x = y = 0
    for _ in range(H):
        for _ in range(W):
            order.append(cell_of(x, y))
            x = (x + 1) % W
        x = (x - 1) % W
        y = (y + 1) % H
    return tuple(order)


def validate_cycle(cycle) -> None:
    assert len(cycle) == N, "le cycle doit couvrir les 225 cases"
    assert len(set(cycle)) == N, "chaque case doit apparaître une seule fois"
    for i, c in enumerate(cycle):
        nxt = cycle[(i + 1) % N]  # inclut dernière -> première
        assert nxt in NEIGHBORS[c], f"cases {c} et {nxt} non adjacentes"


def _index_of(cycle) -> tuple[int, ...]:
    idx = [0] * N
    for i, c in enumerate(cycle):
        idx[c] = i
    return tuple(idx)


def _dihedral(c: int, swap: bool, fx: bool, fy: bool) -> int:
    x, y = xy_of(c)
    if swap:
        x, y = y, x
    if fx:
        x = W - 1 - x
    if fy:
        y = H - 1 - y
    return cell_of(x, y)


CYCLE = build_hamiltonian_cycle()
validate_cycle(CYCLE)
CYCLE_INDEX = _index_of(CYCLE)

# Variantes du cycle : 8 symétries du carré x 2 sens de parcours (bases),
# puis 225 translations appliquées à la volée => 3600 cycles candidats.
# Toutes préservent l'adjacence torique ; chaque base est revalidée.
CYCLE_BASES = []
_seen_bases = set()
for _family in (CYCLE, build_helical_cycle()):
    for _swap in (False, True):
        for _fx in (False, True):
            for _fy in (False, True):
                _cyc = tuple(_dihedral(c, _swap, _fx, _fy) for c in _family)
                for _v in (_cyc, _cyc[::-1]):
                    validate_cycle(_v)
                    # Dédoublonnage à rotation près (même cycle, autre départ).
                    _i = _v.index(0)
                    _key = _v[_i:] + _v[:_i]
                    if _key in _seen_bases:
                        continue
                    _seen_bases.add(_key)
                    CYCLE_BASES.append((_v, _index_of(_v)))
# TRANSLATE[t][c] : case c décalée de (tx, ty) = xy_of(t).
TRANSLATE = tuple(
    tuple(cell_of((x + t % W) % W, (y + t // W) % H) for x, y in (xy_of(c) for c in range(N)))
    for t in range(N))
TRANSLATE_INV = tuple(cell_of((-(t % W)) % W, (-(t // W)) % H) for t in range(N))


def cycle_variant(base: int, t: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    cyc = tuple(TRANSLATE[t][c] for c in CYCLE_BASES[base][0])
    return cyc, _index_of(cyc)


def cycle_ordered(idx, body) -> bool:
    """Invariant : de la queue à la tête, les rangs (mesurés depuis la queue)
    croissent strictement => tout le corps tient dans un seul tour de cycle."""
    ti = idx[body[-1]]
    prev = N
    for c in body:
        p = (idx[c] - ti) % N
        if p >= prev:
            return False
        prev = p
    return True


# --------------------------------------------------------------------------
# État et simulation exacte
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SnakeState:
    body: tuple[int, ...]   # tête en premier
    direction: int
    food: int               # -1 = inconnue (réapparue au hasard après un repas simulé)
    score: int
    ticks: int
    grow: int = 0           # croissance en attente (Snake.grow_pending du moteur)
    mask: int = 0           # bitset des cases du corps

    @staticmethod
    def make(body, direction, food, score=0, ticks=0, grow=0) -> "SnakeState":
        body = tuple(body)
        mask = 0
        for c in body:
            mask |= BIT[c]
        return SnakeState(body, direction, food, score, ticks, int(grow), mask)

    @property
    def head(self) -> int:
        return self.body[0]


DEAD = None


def simulate(state: SnakeState, action: int) -> Optional[SnakeState]:
    """Reproduit Snake.set_direction + Snake.move + is_game_over + la pomme.

    - un demi-tour est ignoré par le moteur (la direction ne change pas) ;
    - la tête avance ; la queue est retirée sauf si grow_pending ;
    - mort si la nouvelle tête est dans body[1:] (après retrait de la queue) ;
    - si la tête est sur la pomme : score+1 et grow_pending (croissance au
      tick SUIVANT). La nouvelle pomme est aléatoire -> food = -1.
    """
    if action == OPPOSITE[state.direction]:
        action = state.direction
    body = state.body
    nh = NEIGHBORS[body[0]][action]
    if state.grow:
        if state.mask & BIT[nh]:
            return DEAD
        nbody = (nh,) + body
        nmask = state.mask | BIT[nh]
    else:
        m = state.mask ^ BIT[body[-1]]
        if m & BIT[nh]:
            return DEAD
        nbody = (nh,) + body[:-1]
        nmask = m | BIT[nh]
    if nh == state.food:
        return SnakeState(nbody, action, -1, state.score + 1, state.ticks + 1, 1, nmask)
    return SnakeState(nbody, action, state.food, state.score, state.ticks + 1, 0, nmask)


def legal_actions(state: SnakeState) -> tuple[int, ...]:
    return LEGAL[state.direction]


def _mobility(body, mask, grow, direction) -> int:
    """Nombre de coups immédiatement non mortels."""
    blocked = mask if grow else mask ^ BIT[body[-1]]
    nb = NEIGHBORS[body[0]]
    n = 0
    for a in LEGAL[direction]:
        if not blocked & BIT[nb[a]]:
            n += 1
    return n


def count_legal_moves(state: SnakeState) -> int:
    return _mobility(state.body, state.mask, state.grow, state.direction)


# --------------------------------------------------------------------------
# Flood fill, certificat de queue, A*
# --------------------------------------------------------------------------

def _flood(head: int, mask: int) -> int:
    free = FULL & ~mask
    seen = BIT[head]
    frontier = seen
    while frontier:
        frontier = expand(frontier) & free & ~seen
        seen |= frontier
    return seen.bit_count() - 1


def reachable_space(state: SnakeState) -> int:
    """Cases libres atteignables depuis la tête (tore pris en compte)."""
    return _flood(state.body[0], state.mask)


def _tail_certificate(body, mask, grow, food) -> bool:
    """Vrai si la tête peut atteindre une case du corps au moment (ou après)
    où celle-ci se libère, par un chemin simple sur les cases libres.

    Une fois sur une case libérée du corps, il suffit de suivre les segments
    suivants dans l'ordre (chacun se libère juste à temps) : survie infinie.
    La case body[i] se libère au tick L - i + grow (croissance différée).
    Deux canaux : sans passer par la pomme (0) et après l'avoir mangée (1),
    où chaque libération est retardée d'un tick. Le canal 1 n'emprunte pas
    les cases déjà visitées par le canal 0 (évite de repasser sur sa propre
    trace) : c'est conservateur, donc sûr.
    """
    L = len(body)
    free = FULL & ~mask
    foodbit = BIT[food] if food >= 0 else 0
    f0 = v0 = BIT[body[0]]
    f1 = v1 = 0
    rel0 = rel1 = 0
    for t in range(1, L + grow + 3):
        i = L - t + grow
        if 1 <= i < L:
            rel0 |= BIT[body[i]]
        if 1 <= i + 1 < L:
            rel1 |= BIT[body[i + 1]]
        e0 = expand(f0) if f0 else 0
        e1 = expand(f1) if f1 else 0
        if (e0 & rel0) or (e1 & rel1):
            return True
        n0 = e0 & free & ~v0
        ate = n0 & foodbit
        n0 ^= ate
        n1 = ((e1 & free) | ate) & ~v1 & ~v0
        v0 |= n0
        v1 |= n1
        f0, f1 = n0, n1
        if not (f0 or f1):
            return False
    return False


def _expected_next_distance(body, mask, grow) -> float:
    """Espérance du nombre de ticks pour atteindre la prochaine pomme, qui
    apparaît uniformément sur une case hors du corps.

    BFS temporel par couches en bitset depuis la tête : body[i] devient
    franchissable au tick L - i + grow. On étend depuis tout l'ensemble déjà
    atteint (approximation : le serpent peut « patienter » en tournant). Une
    case jamais atteinte compte pour 2 * W ticks. C'est ce qui rend la forme du
    corps coûteuse : un corps étalé rallonge tous les trajets suivants."""
    L = len(body)
    targets = FULL & ~mask
    total = targets.bit_count()
    if not total:
        return 0.0
    seen = BIT[body[0]]
    passable = targets
    acc = reached = 0
    t = 0
    while reached < total and t < N:
        t += 1
        i = L - t + grow
        if 1 <= i < L:
            passable |= BIT[body[i]]
        new = expand(seen) & passable & ~seen
        if not new and i < 1:
            break
        seen |= new
        k = (new & targets).bit_count()
        acc += k * t
        reached += k
    return (acc + (total - reached) * 2 * W) / total


def can_reach_tail(state: SnakeState) -> bool:
    return _tail_certificate(state.body, state.mask, state.grow, state.food)


def _release_times(body, grow):
    rel = {}
    L = len(body)
    for i in range(1, L):
        rel[body[i]] = L - i + grow
    return rel


def astar(state: SnakeState, target: int) -> Optional[list[int]]:
    """A* temporel : le corps est un obstacle dynamique (body[i] libre à
    partir du tick L - i + grow). Coût 1 par coup, heuristique torique."""
    body = state.body
    start = body[0]
    if start == target:
        return []
    rel = _release_times(body, state.grow)
    dist_t = DISTANCE[target]
    best = {start: 0}
    heap = [(dist_t[start], 0, start, state.direction)]
    parent = {start: None}
    while heap:
        _, g, cell, d = heapq.heappop(heap)
        if g > best.get(cell, 1 << 30):
            continue
        if cell == target:
            path = []
            while parent[cell] is not None:
                cell, a = parent[cell]
                path.append(a)
            path.reverse()
            return path
        nb = NEIGHBORS[cell]
        ng = g + 1
        for a in LEGAL[d]:
            c = nb[a]
            r = rel.get(c)
            if r is not None and r > ng:
                continue
            if ng < best.get(c, 1 << 30):
                best[c] = ng
                parent[c] = (cell, a)
                heapq.heappush(heap, (ng + dist_t[c], ng, c, a))
    return None


def astar_to_food(state: SnakeState) -> Optional[list[int]]:
    if state.food < 0:
        return None
    return astar(state, state.food)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    # Beam Search
    beam_width: int = 256
    search_depth: int = 48
    adaptive: bool = True            # profondeur/largeur selon la longueur
    max_planning_time_ms: float = 10.0
    max_nodes: int = 0               # >0 : budget déterministe (benchmarks)
    full_eval_top_k: int = 8
    extra_depth_after_food: int = 4  # on cherche encore un peu mieux après la 1re pomme

    # Heuristique rapide
    fast_food_base: float = 1000.0
    fast_food_tick: float = 5.0      # food_reward = 1000 - 5 t
    fast_distance: float = 3.0
    fast_mobility: float = 8.0
    fast_tick: float = 2.0
    fast_risk: float = 8.0           # risk_multiplier = 1 + 8 * fill^2
    fast_trapped: float = 60.0       # pénalité par coup légal manquant (x risque)

    # Évaluation complète
    full_food: float = 10000.0
    full_tick: float = 20.0
    full_distance: float = 25.0
    full_space: float = 15.0
    full_tail_ok: float = 5000.0
    full_tail_ko: float = 8000.0
    full_mobility: float = 200.0
    full_risk: float = 5000.0
    full_trapped: float = 50000.0    # espace atteignable < longueur (x risk_multiplier)
    full_next: float = 50.0         # par tick espéré jusqu'à la pomme suivante

    # Objectif : la partie s'arrête à target_score (0 = jamais).
    target_score: int = 100
    progress_risk_factor: float = 10.0   # risk = 1 + f * (score / target) ** 3
    endgame_beam_width: int = 512
    endgame_depth: int = 70
    endgame_time_ms: float = 50.0
    # En dessous de ce score, le plus court chemin A* certifié est pris avant
    # le beam (phase agressive : la pomme la plus rapide d'abord).
    astar_first_below: int = 40

    # Phases selon le score (bornes basses 40 / 80 / 95) : multiplicateurs
    # du poids de l'espace et du poids du temps.
    phase_space: tuple = (0.5, 1.0, 2.0, 3.0)
    phase_tick: tuple = (1.0, 1.0, 0.5, 0.25)

    # Densité à partir de laquelle on poursuit la queue si le beam ne mange pas.
    safe_above: float = 0.65

    # Cycle Hamiltonien
    use_cycle: bool = True
    # Occupation à partir de laquelle on bascule : 0.46 = juste après 100
    # (longueur 103 / 225), donc sans effet sur la course à 100. Plus tard,
    # le corps est trop emmêlé pour rejoindre le cycle de façon fiable.
    cycle_above: float = 0.46
    cycle_stall_ticks: int = 400     # ou si aucune pomme depuis N ticks
    cycle_shortcut_buffer: int = 3   # marge de cases gardée devant la queue
    cycle_join_margin: int = 0      # ticks de marge (pommes mangées pendant la jonction)
    cycle_join_retry: int = 1        # ticks entre deux recherches de variante

    @classmethod
    def from_overrides(cls, overrides: dict) -> "PlannerConfig":
        cfg = cls()
        types = {f.name: f.type for f in fields(cls)}
        for k, v in overrides.items():
            if k not in types:
                raise KeyError(f"paramètre inconnu : {k}")
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, tuple):
                v = tuple(float(x) for x in (v.split("/") if isinstance(v, str) else v))
            else:
                v = type(cur)(v)
            setattr(cfg, k, v)
        return cfg


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

# Classes de sécurité (tri lexicographique) : safe food > safe move > unsafe food.
SAFE_FOOD, SAFE_MOVE, UNSAFE_FOOD = 2, 1, 0


class SnakePlanner:
    def __init__(self, width: int = 15, height: int = 15, config: PlannerConfig | None = None):
        if (width, height) != (W, H):
            raise ValueError("planner compilé pour une grille 15x15")
        self.cfg = config or PlannerConfig()
        self.reset()

    def reset(self) -> None:
        self._last_score = None
        self._ticks = 0
        self._last_eat_tick = 0
        self.cycle_locked = False
        self.variant = None          # (cycle, index) suivi
        self._next_join_search = 0
        self._rng = random.Random(0)
        self.last_info: dict = {}

    # ---- API moteur ---------------------------------------------------

    def choose_action(self, snake_body, direction, food, score, grow_pending=False):
        """Entrée au format du moteur ([x, y], vecteurs), sortie : vecteur UP/DOWN/LEFT/RIGHT."""
        body = [cell_of(x, y) for x, y in snake_body]
        d = direction if isinstance(direction, int) else VECTOR_TO_ACTION[tuple(direction)]
        f = cell_of(*food) if food is not None else -1
        state = SnakeState.make(body, d, f, score, self._ticks, int(bool(grow_pending)))
        return VECTORS[self.choose(state)]

    # ---- Décision -----------------------------------------------------

    def choose(self, state: SnakeState) -> int:
        t0 = time.perf_counter()
        cfg = self.cfg
        if self._last_score is None or state.score != self._last_score:
            self._last_eat_tick = self._ticks
            self._last_score = state.score
        self._ticks += 1
        info = self.last_info = {"mode": "", "depth": 0, "nodes": 0}

        L = len(state.body)
        occupancy = L / N
        children = []
        for a in LEGAL[state.direction]:
            c = simulate(state, a)
            if c is not DEAD:
                children.append((a, c))
        if not children:
            info["mode"] = "doomed"
            return state.direction

        certified = [(a, c) for a, c in children
                     if _tail_certificate(c.body, c.mask, c.grow, c.food)]

        # 0. Dernière pomme : elle termine la partie, seule compte l'atteindre vivant.
        if self._is_final_apple(state):
            return self._endgame_planner(state, children, certified, t0)

        # 1. Aucune sortie certifiée : survivre le plus longtemps possible.
        if not certified:
            info["mode"] = "survival"
            return max(children, key=lambda ac: (reachable_space(ac[1]), count_legal_moves(ac[1])))[0]

        # 2. Cycle Hamiltonien à forte densité ou si on stagne. Testé avant le
        #    coup forcé : la recherche de jonction doit tourner à chaque tick
        #    (sinon, corps long => coups forcés en boucle, jamais de jonction).
        stalled = self._ticks - self._last_eat_tick
        if cfg.use_cycle and (self.cycle_locked or occupancy >= cfg.cycle_above
                              or stalled >= cfg.cycle_stall_ticks):
            self.cycle_locked = True
            move = self._cycle_step(state, children, certified)
            if move is not None:
                return move
            if stalled >= cfg.cycle_stall_ticks and len(certified) > 1:
                # Boucle derrière la queue sans jonction possible : un coup
                # certifié au hasard change la forme du corps (reste sûr).
                info["mode"] = "unstick"
                return self._rng.choice(certified)[0]

        if len(certified) == 1:
            info["mode"] = "forced"
            return certified[0][0]

        # 3. Phase agressive : le plus court chemin, s'il est certifié après le repas.
        if state.score < cfg.astar_first_below:
            pick = self._validated_food_route(state, certified)
            if pick is not None:
                info["mode"] = "astar-first"
                return pick

        # 4. Beam Search sur les coups certifiés.
        deadline = t0 + cfg.max_planning_time_ms / 1000.0 if cfg.max_planning_time_ms > 0 else None
        best_cls, best_val, best_action, unsafe_food_seen = self._beam(state, certified, deadline)

        # 4. Filet A* : chemin sûr vers la pomme trouvé plus vite que le beam.
        astar_pick = None
        if best_cls != SAFE_FOOD:
            astar_pick = self._validated_food_route(state, certified)
        if best_cls == SAFE_FOOD:
            info["mode"] = "beam-food"
            return best_action
        if astar_pick is not None:
            info["mode"] = "astar-food"
            return astar_pick

        # 5. Pomme dangereuse : poursuite de la queue / maximiser l'espace.
        if unsafe_food_seen or occupancy >= cfg.safe_above:
            info["mode"] = "tail-chase"
            return self._tail_chase(state, certified)

        info["mode"] = "beam-approach"
        return best_action

    # ---- Beam Search --------------------------------------------------

    def _params(self, L: int, endgame: bool = False) -> tuple[int, int]:
        cfg = self.cfg
        if endgame:
            return cfg.endgame_depth, cfg.endgame_beam_width
        if not cfg.adaptive:
            return cfg.search_depth, cfg.beam_width
        if L < 50:
            return 40, 192
        if L < 100:
            return 50, 256
        if L < 160:
            return 60, 384
        return 70, 512

    @staticmethod
    def _phase(score: int) -> int:
        """0 : 0-39 agressif, 1 : 40-79, 2 : 80-94 survie d'abord, 3 : 95-99."""
        return 0 if score < 40 else 1 if score < 80 else 2 if score < 95 else 3

    def _risk_multiplier(self, score: int) -> float:
        """Le prix d'une mort croît avec ce qu'on perdrait : 1 à 0, 11 à 100."""
        target = self.cfg.target_score or 100
        progress = min(score, target) / target
        return 1.0 + self.cfg.progress_risk_factor * progress ** 3

    def _full_value(self, body, mask, grow, direction, food, t, eaten: bool,
                    score: int) -> tuple[int, float]:
        """Évaluation complète (flood fill + certificat). Retourne (classe, valeur).

        La classe (sécurité) domine toujours la valeur. Dans la valeur, la pomme
        (+full_food) domine le temps, qui domine l'espace et la mobilité ; le
        poids du temps baisse et celui de l'espace monte avec la phase."""
        cfg = self.cfg
        L = len(body)
        space = _flood(body[0], mask)
        tail_ok = _tail_certificate(body, mask, grow, food)
        mob = _mobility(body, mask, grow, direction)
        fill = L / N
        phase = self._phase(score)
        risk = self._risk_multiplier(score)
        v = -cfg.full_tick * cfg.phase_tick[phase] * t
        if eaten:
            v += cfg.full_food
            if cfg.full_next:
                # Anticipation : ticks espérés pour la pomme suivante.
                v -= cfg.full_next * cfg.phase_tick[phase] * _expected_next_distance(body, mask, grow)
        elif food >= 0:
            v -= cfg.full_distance * DISTANCE[body[0]][food]
        v += cfg.full_space * cfg.phase_space[phase] * space
        v += cfg.full_tail_ok if tail_ok else -cfg.full_tail_ko
        v += cfg.full_mobility * mob
        v -= cfg.full_risk * fill * fill * max(0, L - space)
        if space < L:
            v -= cfg.full_trapped * risk
        if eaten:
            return (SAFE_FOOD if tail_ok else UNSAFE_FOOD), v
        return SAFE_MOVE, v

    def _beam(self, root: SnakeState, certified, deadline, endgame: bool = False):
        """Beam Search depuis les coups `certified`. En endgame, manger la pomme
        est terminal (victoire) : valeur 1e12 - 1000 * profondeur, sans
        certificat, et la recherche s'arrête à la première profondeur trouvée."""
        cfg = self.cfg
        L0 = len(root.body)
        depth_max, width = self._params(L0, endgame)
        s_eat = root.score + 1
        food = root.food
        fill = L0 / N
        risk_mult = 1.0 + cfg.fast_risk * fill * fill
        dist_food = DISTANCE[food] if food >= 0 else None
        info = self.last_info
        perf = time.perf_counter
        max_nodes = cfg.max_nodes

        # nœud = (classe, valeur, body, mask, grow, direction, root_action)
        beam = []
        best = None          # meilleur nœud « a mangé » (terminal)
        eaten_at = None
        unsafe_food_seen = False
        for a, c in certified:
            if c.score > root.score and endgame:
                return SAFE_FOOD, 1e12 - 1000.0, a, False
            if c.score > root.score:
                cls, v = self._full_value(c.body, c.mask, c.grow, c.direction, -1, 1, True, s_eat)
                v += cfg.fast_food_base - cfg.fast_food_tick
                node = (cls, v, c.body, c.mask, c.grow, c.direction, a)
                if cls == SAFE_FOOD:
                    if best is None or node[:2] > best[:2]:
                        best = node
                    eaten_at = 1
                else:
                    unsafe_food_seen = True
            else:
                cls, v = self._full_value(c.body, c.mask, c.grow, c.direction, food, 1, False, root.score)
                beam.append((cls, v, c.body, c.mask, c.grow, c.direction, a))
        beam.sort(key=lambda n: (n[0], n[1]), reverse=True)
        best_nonfood = beam[0] if beam else None

        nodes = 0
        seen = set()
        depth = 1
        f_base, f_tick = cfg.fast_food_base, cfg.fast_food_tick
        w_dist, w_mob, w_tick, w_trap = cfg.fast_distance, cfg.fast_mobility, cfg.fast_tick, cfg.fast_trapped
        timed_out = False
        while beam and depth < depth_max:
            depth += 1
            if eaten_at is not None and depth > eaten_at + cfg.extra_depth_after_food:
                break
            cands = []
            for node in beam:
                if deadline is not None and perf() > deadline:
                    timed_out = True
                    break
                if max_nodes and nodes >= max_nodes:
                    timed_out = True
                    break
                _, _, body, mask, grow, d, ra = node
                head = body[0]
                nb = NEIGHBORS[head]
                if grow:
                    blocked = mask
                else:
                    tailbit = BIT[body[-1]]
                    blocked = mask ^ tailbit
                for a in LEGAL[d]:
                    nh = nb[a]
                    hb = BIT[nh]
                    if blocked & hb:
                        continue
                    nodes += 1
                    if grow:
                        nbody = (nh,) + body
                    else:
                        nbody = (nh,) + body[:-1]
                    if nbody in seen:
                        continue
                    seen.add(nbody)
                    nmask = blocked | hb
                    if nh == food and endgame:
                        # Dernière pomme : victoire, la plus précoce est la meilleure.
                        info["nodes"] = nodes
                        return SAFE_FOOD, 1e12 - 1000.0 * depth, ra, False
                    if nh == food:
                        # Analyse complète immédiate après un repas.
                        cls, v = self._full_value(nbody, nmask, 1, a, -1, depth, True, s_eat)
                        if cls == SAFE_FOOD:
                            v += f_base - f_tick * depth
                            n2 = (cls, v, nbody, nmask, 1, a, ra)
                            if best is None or n2[:2] > best[:2]:
                                best = n2
                            if eaten_at is None:
                                eaten_at = depth
                        else:
                            unsafe_food_seen = True
                        continue
                    # Heuristique rapide.
                    mob = _mobility(nbody, nmask, 0, a)
                    if mob == 0:
                        continue  # mort au coup suivant, inutile de le garder
                    v = (-w_dist * dist_food[nh] if dist_food is not None else 0.0) \
                        - w_tick * depth + w_mob * mob - w_trap * risk_mult * (3 - mob)
                    cands.append((SAFE_MOVE, v, nbody, nmask, 0, a, ra))
                if timed_out:
                    break
            if not cands:
                break
            cands.sort(key=lambda n: n[1], reverse=True)
            beam = cands[:width]
            # Réévaluation complète du top K.
            k = min(cfg.full_eval_top_k, len(beam))
            for i in range(k):
                _, _, b, m, g, d, ra = beam[i]
                cls, v = self._full_value(b, m, g, d, food, depth, False, root.score)
                beam[i] = (cls, v / 25.0, b, m, g, d, ra)  # remis à l'échelle du score rapide
            beam.sort(key=lambda n: n[1], reverse=True)
            best_nonfood = beam[0]
            info["depth"] = depth
            if timed_out:
                break

        info["nodes"] = nodes
        info["timed_out"] = timed_out
        if best is not None:
            return SAFE_FOOD, best[1], best[6], unsafe_food_seen
        return SAFE_MOVE, best_nonfood[1], best_nonfood[6], unsafe_food_seen

    # ---- A*, queue, cycle ---------------------------------------------

    @staticmethod
    def _simulate_route(state: SnakeState, path) -> Optional[SnakeState]:
        """Joue `path` coup par coup ; None si le serpent meurt ou ne mange pas."""
        s = state
        for a in path:
            s = simulate(s, a)
            if s is DEAD:
                return None
        return s if s.score > state.score else None

    def _validated_food_route(self, state: SnakeState, certified) -> Optional[int]:
        """Un A* par premier coup certifié -> simulation complète du chemin ->
        évaluation complète de l'état après repas (certificat obligatoire).
        Parmi les routes sûres, la meilleure valeur : la plus rapide, en tenant
        compte de la forme laissée pour la pomme suivante."""
        if state.food < 0:
            return None
        best = None
        for a, c in certified:
            if c.score > state.score:
                s, t = c, 1
            else:
                path = astar(c, state.food)
                if path is None:
                    continue
                s = self._simulate_route(c, path)
                if s is None:
                    continue
                t = len(path) + 1
            cls, v = self._full_value(s.body, s.mask, s.grow, s.direction, -1, t, True, s.score)
            if cls != SAFE_FOOD or reachable_space(s) < 1:
                continue
            v -= self.cfg.fast_food_tick * t
            if best is None or v > best[0]:
                best = (v, a)
        return best[1] if best else None

    # ---- Endgame : la dernière pomme ----------------------------------

    def _is_final_apple(self, state: SnakeState) -> bool:
        target = self.cfg.target_score
        return target > 0 and state.food >= 0 and state.score + 1 >= target

    def _endgame_planner(self, state: SnakeState, children, certified, t0) -> int:
        """Score 99 : manger termine la partie. On ne cherche que (1) survivre
        et (2) atteindre la pomme ; le temps ne départage qu'en dernier.

        Un chemin simulé exactement qui atteint la pomme vivant est une
        victoire certaine (la pomme ne bouge pas avant d'être mangée) : aucun
        certificat n'est exigé après le repas. Sinon, on attend en restant
        certifié que la queue ouvre un passage."""
        cfg = self.cfg
        info = self.last_info
        for a, c in children:
            if c.score > state.score:
                info["mode"] = "endgame-eat"
                return a
        path = astar_to_food(state)
        if path and self._simulate_route(state, path) is not None:
            info["mode"] = "endgame-astar"
            return path[0]
        deadline = t0 + cfg.endgame_time_ms / 1000.0 if cfg.endgame_time_ms > 0 else None
        cls, _, action, _ = self._beam(state, children, deadline, endgame=True)
        if cls == SAFE_FOOD:
            info["mode"] = "endgame-beam"
            return action
        if certified:
            info["mode"] = "endgame-wait"
            return self._tail_chase(state, certified)
        info["mode"] = "survival"
        return max(children, key=lambda ac: (reachable_space(ac[1]), count_legal_moves(ac[1])))[0]

    def _tail_chase(self, state: SnakeState, certified) -> int:
        """Aller vers la queue pour la laisser libérer de l'espace, en gardant
        un maximum d'espace et d'options."""
        scored = [(reachable_space(c), count_legal_moves(c), a) for a, c in certified]
        best_space = max(s for s, _, _ in scored)
        path = astar(state, state.body[-1])
        if path:
            for s, _, a in scored:
                if a == path[0] and s >= 0.9 * best_space:
                    return a
        return max(scored)[2]

    def _cycle_move(self, state: SnakeState, cyc, idx) -> int:
        """Coup sur le cycle `cyc` quand le corps respecte l'invariant
        (cycle_ordered). Raccourcis autorisés s'ils ne dépassent ni la pomme
        ni la queue (avec marge) : l'invariant reste vrai, c'est prouvé sûr."""
        body = state.body
        head, tail = body[0], body[-1]
        hi = idx[head]
        next_action = NEIGHBORS[head].index(cyc[(hi + 1) % N])
        ti = idx[tail]

        gap = (ti - hi) % N  # cases à parcourir avant d'atteindre la queue
        if gap == 0:
            gap = N
        food = state.food
        if food < 0:
            return next_action
        food_d = (idx[food] - hi) % N
        fill = len(body) / N
        buffer = self.cfg.cycle_shortcut_buffer + (int(fill * 10) if fill > 0.5 else 0)
        # Toute case de rang relatif d (0 < d < gap) est libre sous l'invariant.
        # Raccourci glouton : le voisin qui avance le plus sans dépasser la
        # pomme ni s'approcher de la queue à moins de `buffer` (+ croissance).
        # NB : le plus court chemin exact vers la pomme (BFS sur les rangs
        # croissants) a été mesuré plus lent : il saute plus de cases, qui
        # deviennent des trous où la pomme suivante coûte un tour complet.
        best_a, best_rem = next_action, food_d - 1
        blocked = state.mask if state.grow else state.mask ^ BIT[tail]
        for a in LEGAL[state.direction]:
            c = NEIGHBORS[head][a]
            if blocked & BIT[c]:
                continue
            d = (idx[c] - hi) % N
            if d <= 1 or d > food_d:
                continue
            growth = state.grow + (1 if c == food else 0)
            if gap - d <= growth + buffer:
                continue
            rem = food_d - d
            if rem < best_rem:
                best_a, best_rem = a, rem
        return best_a

    def _cycle_step(self, state: SnakeState, children, certified) -> Optional[int]:
        info = self.last_info
        v = self.variant
        if v is not None and cycle_ordered(v[1], state.body):
            move = self._cycle_move(state, *v)
            if any(a == move for a, _ in children):
                info["mode"] = "cycle"
                return move
        if v is None or not self._join_still_ok(state, *v, self.cfg.cycle_join_margin):
            v = None
            if self._ticks >= self._next_join_search:
                v = self._find_join(state)
                if v is None:
                    self._next_join_search = self._ticks + self.cfg.cycle_join_retry
            self.variant = v
        if v is not None:
            move = NEIGHBORS[state.body[0]].index(v[0][(v[1][state.body[0]] + 1) % N])
            if any(a == move for a, _ in certified):
                info["mode"] = "cycle-join"
                return move
        return None

    def _find_join(self, state: SnakeState):
        """Cherche une variante du cycle qu'on peut rejoindre en la suivant
        simplement : chaque case du corps rencontrée sur les L prochaines
        positions doit s'être libérée avant l'arrivée de la tête (avec une
        marge pour les pommes mangées en route). Après ces L coups, le corps
        entier est rangé dans l'ordre du cycle. Préfère la variante qui
        atteint la pomme le plus tôt. Retourne (cyc, idx) ou None."""
        body = state.body
        L = len(body)
        g = state.grow
        margin = self.cfg.cycle_join_margin
        horizon = min(L + g + margin, N - 1)
        head = body[0]
        food = state.food
        # due[c] : premier rang j auquel la case c est franchissable.
        due = {}
        for i in range(1, L):
            due[body[i]] = L - i + g + margin
        best = None
        best_key = N + 1
        for b, (base, bidx) in enumerate(CYCLE_BASES):
            for t in range(N):
                tr = TRANSLATE[t]
                k0 = bidx[TRANSLATE[TRANSLATE_INV[t]][head]]
                ok = True
                for j in range(1, horizon + 1):
                    r = due.get(tr[base[(k0 + j) % N]])
                    if r is not None and r > j:
                        ok = False
                        break
                if not ok:
                    continue
                key = (bidx[TRANSLATE[TRANSLATE_INV[t]][food]] - k0) % N if food >= 0 else 0
                if key < best_key:
                    best, best_key = (b, t), key
        if best is None:
            return None
        return cycle_variant(*best)

    @staticmethod
    def _join_still_ok(state: SnakeState, cyc, idx, margin: int) -> bool:
        body = state.body
        L = len(body)
        g = state.grow
        due = {body[i]: L - i + g + margin for i in range(1, L)}
        k0 = idx[body[0]]
        for j in range(1, min(L + g + margin, N - 1) + 1):
            r = due.get(cyc[(k0 + j) % N])
            if r is not None and r > j:
                return False
        return True
