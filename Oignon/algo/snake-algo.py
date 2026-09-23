"""
Snake algo - équipe Oignon
==========================

DHCR : cycle hamiltonien dynamique (Dynamic Hamiltonian Cycle Repair) sur la grille torique.

  Jeu        : règles identiques à serpent-algo.py (grille 15x15 torique, clock 5, +1 par pomme).
  Cycle      : un chemin qui passe une fois par chaque case et revient au départ.
  Invariant  : le corps occupe un segment continu du cycle (queue -> tête). Tant que c'est
               vrai, suivre le cycle ne peut pas tuer le serpent : la victoire est garantie.
  Réparation : on peut changer de cycle à tout moment si le nouveau garde le corps au même
               endroit. À chaque pomme, BFS trace le plus court chemin tête -> pomme, puis on
               complète un chemin pomme -> queue qui couvre toutes les cases libres (détours
               par carrés 2x2). Si ça échoue ou si ce n'est pas plus court, on garde l'ancien cycle.
  Échanges   : tant que le trajet vers la pomme n'est pas le plus court, on améliore le cycle
               actuel par échanges 2x2 : une boucle entre la tête et la pomme est détachée,
               puis recollée après la pomme (jamais d'arête du corps retirée).

Usage (depuis LeSerpent2026/) :
  python Oignon/algo/snake-algo.py                       # partie officielle (= play)
  python Oignon/algo/snake-algo.py play [--algo cycle] [--debug-fps 60] [--trace]
  python Oignon/algo/snake-algo.py bench --games 30 [--check]
"""

import argparse
import os
import random
import sys
import time
from bisect import bisect_right
from collections import Counter, deque

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

# --- CONSTANTES DE JEU (inchangées) ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 5

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)
BLEU_TRACE = (70, 130, 220)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


# =============================================================================
# 1. JEU PYGAME (copie de serpent-algo.py, utilisée par le mode "play")
# =============================================================================

class Snake:
    """Représente le serpent, sa position, sa direction et son corps."""
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos,
                     [self.head_pos[0] - 1, self.head_pos[1]],
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        new_head_x = (self.head_pos[0] + self.direction[0]) % GRID_SIZE
        new_head_y = (self.head_pos[1] + self.direction[1]) % GRID_SIZE
        new_head_pos = [new_head_x, new_head_y]
        self.body.insert(0, new_head_pos)
        self.head_pos = new_head_pos
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False

    def grow(self):
        self.grow_pending = True
        self.score += 1

    def check_wall_collision(self):
        """Toujours False : move() applique un modulo, la grille est torique."""
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        return self.check_wall_collision() or self.check_self_collision()

    def draw(self, surface):
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)


class Apple:
    """Représente la pomme (nourriture) et sa position."""
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        if not available_positions:
            return None
        return random.choice(available_positions)

    def relocate(self, snake_body):
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)


def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font, snake, start_time):
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 20))


def display_message(surface, font, message, color=BLANC, y_offset=0):
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)


# =============================================================================
# 2. GÉOMÉTRIE DE LA GRILLE TORIQUE
# =============================================================================

N = GRID_SIZE
M = N * N                              # 225 cases
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]   # sens horaire : +1 = tourner à droite
TURN = (0, 1, 3)                       # actions relatives : tout droit, droite, gauche


def cell(x, y):
    return (y % N) * N + (x % N)


CELL_XY = [(c % N, c // N) for c in range(M)]
NEIGHBOR = [[cell(x + dx, y + dy) for dx, dy in DIRECTIONS] for (x, y) in CELL_XY]


def build_cycle():
    """Cycle hamiltonien torique : chaque ligne est parcourue vers la droite en
    commençant une colonne plus à gauche que la précédente ; la fin d'une ligne
    est juste au-dessus du début de la suivante, et la ligne 14 reboucle sur (0, 0).
    Sur une grille 15x15 à murs, aucun cycle hamiltonien n'existe (225 cases, damier) :
    ce cycle n'existe que grâce au passage d'un bord à l'autre."""
    order = []
    for y in range(N):
        start = (-y) % N
        order.extend(cell(start + k, y) for k in range(N))
    return order


CYCLE = build_cycle()
CYCLE_NEXT = [0] * M
for _i, _c in enumerate(CYCLE):
    CYCLE_NEXT[_c] = CYCLE[(_i + 1) % M]


# =============================================================================
# 3. MOTEUR SANS AFFICHAGE (mêmes règles que Snake/Apple, en entiers)
# =============================================================================

MOVE, APPLE, DEATH, WIN = "move", "apple", "death", "win"


class SnakeEnv:
    """Moteur rapide pour le bench. body[0] = tête, body[-1] = queue."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        head = cell(GRID_SIZE // 4, GRID_SIZE // 2)
        self.body = deque([head, cell(GRID_SIZE // 4 - 1, GRID_SIZE // 2),
                           cell(GRID_SIZE // 4 - 2, GRID_SIZE // 2)])
        self.dir = DIRECTIONS.index(RIGHT)
        self.occ = bytearray(M)
        for c in self.body:
            self.occ[c] = 1
        self.grow_pending = False
        self.score = 0
        self.steps = 0
        self.done = False
        self.won = False
        self.apple = self._random_free()

    def load(self, snake, apple):
        """Recopie l'état d'une partie pygame (mode play)."""
        self.body = deque(cell(x, y) for x, y in snake.body)
        self.occ = bytearray(M)
        for c in self.body:
            self.occ[c] = 1
        self.dir = DIRECTIONS.index(snake.direction)
        self.grow_pending = snake.grow_pending
        self.apple = cell(*apple.position)

    def _random_free(self):
        free = [c for c in range(M) if not self.occ[c]]
        return self.rng.choice(free) if free else None

    def target_cell(self, action):
        return NEIGHBOR[self.body[0]][(self.dir + TURN[action]) % 4]

    def step(self, action):
        d = (self.dir + TURN[action]) % 4
        c = NEIGHBOR[self.body[0]][d]
        self.dir = d
        self.steps += 1
        if self.grow_pending:
            self.grow_pending = False
        else:
            self.occ[self.body.pop()] = 0
        if self.occ[c]:
            self.done = True
            return DEATH
        self.body.appendleft(c)
        self.occ[c] = 1
        if c == self.apple:
            self.score += 1
            self.grow_pending = True
            self.apple = self._random_free()
            if self.apple is None:
                self.done = self.won = True
                return WIN
            return APPLE
        return MOVE


def action_towards(env, c):
    """Action relative (tout droit / droite / gauche) qui mène la tête sur la case c."""
    for a in range(3):
        if env.target_cell(a) == c:
            return a
    raise AssertionError(f"case {CELL_XY[c]} non voisine de la tête {CELL_XY[env.body[0]]}")


# =============================================================================
# 4. POLITIQUES : CYCLE FIXE ET DHCR
# =============================================================================

class CyclePolicy:
    """Suit le cycle hamiltonien fixe (référence : victoire garantie, mais lente)."""
    name = "cycle pur"

    def __init__(self):
        self.reset()

    def reset(self):
        self.nxt = list(CYCLE_NEXT)
        self.stats = Counter()

    def choose(self, env):
        return action_towards(env, self.nxt[env.body[0]])

    def path_to_apple(self, env):
        """Cases que la tête va parcourir jusqu'à la pomme (pour l'affichage --trace)."""
        out, c = [], env.body[0]
        while c != env.apple and len(out) < M:
            c = self.nxt[c]
            out.append(c)
        return out


def bfs_path(occ, src, dst):
    """Plus court chemin src -> dst sur les cases libres. Renvoie les cases après src
    (dst inclus), ou None. Toutes les arêtes valent 1 : BFS = A* = Dijkstra ici."""
    parent = {src: None}
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        for nb in NEIGHBOR[cur]:
            if nb in parent or occ[nb]:
                continue
            parent[nb] = cur
            if nb == dst:
                path = [nb]
                while parent[path[-1]] != src:
                    path.append(parent[path[-1]])
                return path[::-1]
            queue.append(nb)
    return None


def bfs_path_parity(used, src, dst, parity):
    """Plus court chemin src -> dst dont le nombre d'arêtes a la parité demandée.
    Renvoie les cases strictement entre src et dst, ou None. Sur le tore impair, les
    arêtes qui traversent un bord relient deux cases de même couleur : c'est ce qui
    permet de changer la parité d'un chemin."""
    start = (src, 0)
    parent = {start: None}
    queue = deque([start])
    while queue:
        cur, p = queue.popleft()
        q = p ^ 1
        for nb in NEIGHBOR[cur]:
            if nb == dst:
                if q != parity:
                    continue
                cells, s = [], (cur, p)
                while s != start:
                    cells.append(s[0])
                    s = parent[s]
                cells.reverse()
                # Un chemin de parité imposée peut repasser par une case : on le rejette.
                return cells if len(set(cells)) == len(cells) else None
            if used[nb] or (nb, q) in parent:
                continue
            parent[(nb, q)] = (cur, p)
            queue.append((nb, q))
    return None


def extend_path(path, used, start):
    """Allonge path par détours 2x2 (u->v devient u->c->d->v, c et d libres),
    uniquement sur les arêtes à partir de l'indice start. Renvoie le nombre de cases ajoutées."""
    added = 0
    changed = True
    while changed:
        changed = False
        i = start
        while i < len(path) - 1:
            u, v = path[i], path[i + 1]
            k = NEIGHBOR[u].index(v)
            for p in ((k + 1) % 4, (k + 3) % 4):
                c, d = NEIGHBOR[u][p], NEIGHBOR[v][p]
                if not used[c] and not used[d]:
                    used[c] = used[d] = 1
                    path[i + 1:i + 1] = [c, d]
                    added += 2
                    changed = True
                    break
            else:
                i += 1
    return added


def build_side_pairs():
    """Pour chaque carré 2x2 du tore, ses deux paires de côtés opposés (p1-p2, q1-q2),
    avec p1 voisin de q1 et p2 voisin de q2."""
    pairs = []
    for y in range(N):
        for x in range(N):
            a, b = cell(x, y), cell(x + 1, y)
            c, d = cell(x, y + 1), cell(x + 1, y + 1)
            pairs.append((a, b, c, d))   # côtés haut et bas
            pairs.append((a, c, b, d))   # côtés gauche et droit
    return pairs


SIDE_PAIRS = build_side_pairs()


def swap_edges(nxt, e):
    """u1 -> v1 et u2 -> v2 deviennent u1 -> v2 et u2 -> v1 (coupe ou recolle des boucles)."""
    u1, v1, u2, v2 = e
    nxt[u1] = v2
    nxt[u2] = v1


def improve_by_swaps(nxt, env, stats):
    """Rapproche la pomme par échanges 2x2, sans jamais retirer une arête du corps.

    Si le cycle emprunte deux côtés opposés d'un carré 2x2 en sens contraires, on peut
    les remplacer par les deux autres côtés : sur un même cycle, ça le coupe en deux
    boucles ; sur deux boucles, ça les recolle. Un coup = détacher une boucle située
    entre la tête et la pomme, puis la recoller après la pomme."""
    head, apple = env.body[0], env.apple
    free = M - len(env.body)       # rangs 1..free = cases libres, puis le corps
    while True:
        rank = [0] * M
        c, r = head, 0
        while r < M - 1:
            c = nxt[c]
            r += 1
            rank[c] = r
        dist = rank[apple]
        if dist <= 1:
            return

        # Une arête u -> nxt[u] est retirable si u est la tête ou une case libre (rang <= free).
        splits, merges = [], []
        for p1, p2, q1, q2 in SIDE_PAIRS:
            if nxt[p1] == p2 and nxt[q2] == q1:
                e = (p1, p2, q2, q1)
            elif nxt[p2] == p1 and nxt[q1] == q2:
                e = (p2, p1, q1, q2)
            else:
                continue
            r1, r2 = rank[e[0]], rank[e[2]]
            if r1 > free or r2 > free:
                continue
            lo, hi = min(r1, r2), max(r1, r2)
            if hi < dist:
                splits.append((hi - lo, lo, hi, e))    # détache les rangs lo+1..hi
            elif lo < dist:
                merges.append((lo, e))                 # une arête avant, une après la pomme
        if not splits or not merges:
            return

        # Coupure la plus longue qui peut être recollée après la pomme : il faut une
        # paire dont l'arête « avant » est à l'intérieur de la boucle (rang lo+1..hi-1).
        merges.sort()
        merge_ranks = [m[0] for m in merges]
        splits.sort(reverse=True)
        for gain, lo, hi, e in splits:
            k = bisect_right(merge_ranks, lo)
            if k < len(merges) and merges[k][0] < hi:
                swap_edges(nxt, e)
                swap_edges(nxt, merges[k][1])
                stats["echanges"] += 1
                stats["coups gagnes (echanges)"] += gain
                break
        else:
            return


class DHCRPolicy(CyclePolicy):
    """Cycle hamiltonien réparé dynamiquement pour rapprocher la pomme de la tête :
    reconstruction complète (BFS + détours) à chaque pomme, puis, tant que le trajet
    n'est pas le plus court, échanges 2x2 locaux à chaque pas."""
    name = "DHCR"

    def __init__(self, swaps=True):
        self.swaps = swaps
        if not swaps:
            self.name = "DHCR-A"
        super().__init__()

    def reset(self):
        super().reset()
        self.last_apple = None
        self.pending = False

    def cycle_dist(self, a, b):
        n, c, nxt = 0, a, self.nxt
        while c != b:
            c = nxt[c]
            n += 1
        return n

    def choose(self, env):
        if env.apple is not None and (env.apple != self.last_apple or self.pending):
            self.last_apple = env.apple
            self.repair(env)
        # La reconstruction complète a échoué ou n'est pas optimale : on grappille par échanges.
        if self.swaps and self.pending and env.apple is not None:
            improve_by_swaps(self.nxt, env, self.stats)
        return action_towards(env, self.nxt[env.body[0]])

    def repair(self, env):
        st = self.stats
        st["tentatives"] += 1
        head, tail, apple = env.body[0], env.body[-1], env.apple
        current = self.cycle_dist(head, apple)
        q = bfs_path(env.occ, head, apple)
        if q is None:
            # Pomme momentanément inaccessible hors du cycle : on retente au coup suivant.
            st["inaccessible"] += 1
            self.pending = True
            return
        if len(q) >= current:
            st["deja optimal"] += 1
            self.pending = False
            return
        path = self.build(env, q)
        if path is None:
            st["echec"] += 1
            self.pending = True
            return
        new = path.index(apple) + 1
        if new >= current:
            st["sans gain"] += 1
            self.pending = True
            return
        # Le corps garde ses arêtes (queue -> tête) ; seules les cases libres sont recâblées.
        nxt = self.nxt
        nxt[head] = path[0]
        for a, b in zip(path, path[1:]):
            nxt[a] = b
        nxt[path[-1]] = tail
        st["reparations"] += 1
        st["coups gagnes"] += current - new
        # Si des détours ont allongé le trajet vers la pomme, on retentera au coup suivant.
        self.pending = new > len(q)

    def build(self, env, q):
        """Chemin hamiltonien sur les cases libres : tête -> q -> pomme -> ... -> voisin de la queue."""
        body = env.body
        head, tail = body[0], body[-1]
        free = M - len(body)
        used = bytearray(env.occ)
        for c in q:
            used[c] = 1
        # Chaque détour ajoute 2 cases : la parité du chemin initial doit déjà être la bonne.
        r = bfs_path_parity(used, q[-1], tail, (free - len(q) + 1) % 2)
        if r is None:
            self.stats["echec parite"] += 1
            return None
        for c in r:
            used[c] = 1
        path = [head] + q + r + [tail]
        missing = free - len(q) - len(r)
        # D'abord des détours après la pomme (le trajet vers la pomme reste le plus court),
        # puis, s'il reste des cases, partout.
        missing -= extend_path(path, used, len(q))
        if missing:
            missing -= extend_path(path, used, 0)
        if missing:
            return None
        return path[1:-1]


def check_invariant(env, nxt):
    """Le cycle passe par les 225 cases et le corps y occupe un segment continu."""
    seen, c = 0, env.body[0]
    for _ in range(M):
        c = nxt[c]
        seen += 1
        if c == env.body[0]:
            break
    assert seen == M and c == env.body[0], "le cycle ne couvre pas les 225 cases"
    body = env.body
    for i in range(len(body) - 1):
        assert nxt[body[i + 1]] == body[i], "le corps n'est plus continu dans le cycle"


POLICIES = {
    "dhcr": DHCRPolicy,                         # A + B (par défaut)
    "dhcr-a": lambda: DHCRPolicy(swaps=False),  # reconstruction seule
    "cycle": CyclePolicy,
}


# =============================================================================
# 5. BENCH (sans affichage)
# =============================================================================

def run_game(policy, seed, max_steps, check=False):
    env = SnakeEnv(seed)
    policy.reset()
    while not env.done and env.steps < max_steps:
        env.step(policy.choose(env))
        if check and not env.done:
            check_invariant(env, policy.nxt)
    return {"score": env.score, "won": env.won, "steps": env.steps}


def minutes(steps):
    return steps / GAME_SPEED / 60


def bench(args):
    algos = args.algo.split(",")
    print(f"{args.games} parties par algo, graines {args.seed} à {args.seed + args.games - 1}"
          + (" (invariant vérifié à chaque coup)" if args.check else ""))
    for name in algos:
        policy = POLICIES[name]()
        results, stats = [], Counter()
        t0 = time.time()
        for g in range(args.games):
            r = run_game(policy, args.seed + g, args.max_steps, args.check)
            results.append(r)
            stats.update(policy.stats)
            if args.verbose:
                print(f"  {name} graine {args.seed + g} : score {r['score']} | victoire {r['won']} | "
                      f"{r['steps']} coups ≈ {minutes(r['steps']):.1f} min")
        n = len(results)
        steps = [r["steps"] for r in results]
        wins = sum(r["won"] for r in results)
        print(f"{policy.name:10s} | victoires {wins}/{n} | score moyen {sum(r['score'] for r in results) / n:6.1f} | "
              f"{sum(steps) / n:7.0f} coups ≈ {minutes(sum(steps) / n):5.1f} min "
              f"(min {minutes(min(steps)):.1f}, max {minutes(max(steps)):.1f}) | "
              f"calcul {(time.time() - t0) / n:.2f} s/partie")
        if stats:
            print("           " + ", ".join(f"{k} {v / n:.0f}" for k, v in stats.items()) + "  (moyennes par partie)")


# =============================================================================
# 6. EXÉCUTION (partie pygame, clock d'origine)
# =============================================================================

def draw_trace(surface, cells):
    for c in cells:
        x, y = CELL_XY[c]
        center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + SCORE_PANEL_HEIGHT + CELL_SIZE // 2)
        pygame.draw.circle(surface, BLEU_TRACE, center, CELL_SIZE // 6)


def play(args):
    policy = POLICIES[args.algo]()
    env = SnakeEnv()
    fps = args.debug_fps or GAME_SPEED
    print(f"Algo : {policy.name} | FPS : {fps}"
          + ("" if fps == GAME_SPEED else "  /!\\ vitesse de debug, partie NON officielle"))

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Snake Algo - Oignon ({policy.name})")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    snake = Snake()
    apple = Apple(snake.body)
    running = True
    game_over = False
    victory = False
    start_time = time.time()
    final_elapsed = None
    moves = 0
    move_counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and game_over and event.key == pygame.K_SPACE:
                snake = Snake()
                apple = Apple(snake.body)
                policy.reset()
                game_over = victory = False
                start_time, final_elapsed, moves = time.time(), None, 0

        if not game_over and not victory:
            move_counter += 1
            if move_counter >= GAME_SPEED // 10:
                # L'algorithme choisit la direction à la place du clavier.
                env.load(snake, apple)
                action = policy.choose(env)
                snake.set_direction(DIRECTIONS[(env.dir + TURN[action]) % 4])

                snake.move()
                moves += 1
                move_counter = 0

                if snake.is_game_over():
                    game_over = True
                elif snake.head_pos == list(apple.position):
                    snake.grow()
                    if not apple.relocate(snake.body):
                        victory = True
                        game_over = True
                if game_over:
                    final_elapsed = time.time() - start_time
                    print(f"{'VICTOIRE' if victory else 'GAME OVER'} | score {snake.score} | "
                          f"{moves} coups | temps {int(final_elapsed // 60):02d}:{int(final_elapsed % 60):02d}")

        screen.fill(GRIS_FOND)
        pygame.draw.rect(screen, NOIR, pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH))
        draw_grid(screen)
        if args.trace and not game_over:
            env.load(snake, apple)
            draw_trace(screen, policy.path_to_apple(env))
        apple.draw(screen)
        snake.draw(screen)
        # Le chrono s'arrête à la fin de la partie.
        shown_start = start_time if final_elapsed is None else time.time() - final_elapsed
        display_info(screen, font_main, snake, shown_start)
        if game_over:
            if victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
            display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, y_offset=100)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


# =============================================================================
# 7. LIGNE DE COMMANDE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Snake algo - équipe Oignon (cycle hamiltonien dynamique)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="partie pygame pilotée par l'algorithme")
    p.add_argument("--algo", choices=sorted(POLICIES), default="dhcr")
    p.add_argument("--debug-fps", type=int, default=None, help="accélère l'affichage (NON officiel)")
    p.add_argument("--trace", action="store_true", help="affiche le trajet prévu jusqu'à la pomme")

    b = sub.add_parser("bench", help="parties sans affichage, mêmes graines pour chaque algo")
    b.add_argument("--algo", default="cycle,dhcr-a,dhcr", help="liste séparée par des virgules")
    b.add_argument("--games", type=int, default=30)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--max-steps", type=int, default=30_000)
    b.add_argument("--check", action="store_true", help="vérifie l'invariant à chaque coup (plus lent)")
    b.add_argument("--verbose", action="store_true")

    # Sans argument (`python snake-algo.py`) : partie officielle.
    args = parser.parse_args(sys.argv[1:] or ["play"])
    {"play": play, "bench": bench}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
