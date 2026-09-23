"""
Snake IA - équipe Oignon
========================

Double DQN protégé par un garde-fou.

  Game   : règles identiques à serpent-algo.py (grille 15x15 torique, clock 5, +1 par pomme).
  Model  : Linear_QNet (MLP) + target network -> Double DQN.
  Agent  : encodeur d'état, replay buffer, epsilon-greedy parmi les actions autorisées.
  Shield : masque les actions dangereuses AVANT le choix du réseau.
           - "cycle" : raccourcis sur un cycle hamiltonien (victoire garantie par construction)
           - "tail"  : la queue doit rester accessible (flood-fill)
           - "none"  : seulement les collisions immédiates

Usage (depuis LeSerpent2026/) :
  python Oignon/snake-ia.py train [--episodes 800] [--shield cycle]
  python Oignon/snake-ia.py play                 # partie officielle, clock d'origine
  python Oignon/snake-ia.py eval --games 5       # parties sans affichage, modèle figé
  python Oignon/snake-ia.py bench --games 20     # politiques de référence sans réseau
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import deque

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

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

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
RUN_DIR = os.path.join(HERE, "runs")


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
M = N * N                       # 225 cases
MAX_TORUS_DIST = 2 * (N // 2)   # 14
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]   # sens horaire : +1 = tourner à droite
TURN = (0, 1, 3)                       # actions relatives : tout droit, droite, gauche
ACTION_NAMES = ("tout droit", "droite", "gauche")


def cell(x, y):
    return (y % N) * N + (x % N)


CELL_XY = [(c % N, c // N) for c in range(M)]
NEIGHBOR = [[cell(x + dx, y + dy) for dx, dy in DIRECTIONS] for (x, y) in CELL_XY]


def torus_delta(a, b):
    """Déplacement signé le plus court de a vers b (chaque axe dans [-7, 7])."""
    ax, ay = CELL_XY[a]
    bx, by = CELL_XY[b]
    dx = (bx - ax) % N
    dy = (by - ay) % N
    if dx > N // 2:
        dx -= N
    if dy > N // 2:
        dy -= N
    return dx, dy


def torus_dist(a, b):
    dx, dy = torus_delta(a, b)
    return abs(dx) + abs(dy)


def build_cycle():
    """Cycle hamiltonien torique : chaque ligne est parcourue vers la droite en
    commençant une colonne plus à gauche que la précédente ; la fin d'une ligne
    est juste au-dessus du début de la suivante, et la ligne 14 reboucle sur (0, 0)."""
    order = []
    for y in range(N):
        start = (-y) % N
        order.extend(cell(start + k, y) for k in range(N))
    return order


CYCLE = build_cycle()
CYCLE_POS = [0] * M
for _i, _c in enumerate(CYCLE):
    CYCLE_POS[_c] = _i
CYCLE_NEXT = [CYCLE[(CYCLE_POS[c] + 1) % M] for c in range(M)]


def cycle_dist(a, b):
    """Nombre de pas pour aller de a à b en suivant le cycle."""
    return (CYCLE_POS[b] - CYCLE_POS[a]) % M


# =============================================================================
# 3. MOTEUR SANS AFFICHAGE (mêmes règles que Snake/Apple, en entiers)
# =============================================================================

MOVE, APPLE, DEATH, WIN = "move", "apple", "death", "win"


class SnakeEnv:
    """Moteur rapide pour l'entraînement. body[0] = tête, body[-1] = queue."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.occ = bytearray(M)
        self.body = deque()
        self._seen = [0] * M
        self._stamp = 0

    def reset(self, length=3):
        if length <= 3:
            head = cell(GRID_SIZE // 4, GRID_SIZE // 2)
            self.body = deque([head, cell(GRID_SIZE // 4 - 1, GRID_SIZE // 2),
                               cell(GRID_SIZE // 4 - 2, GRID_SIZE // 2)])
            self.dir = DIRECTIONS.index(RIGHT)
        else:
            # Curriculum : un long serpent posé sur le cycle (état valide et sûr).
            k = self.rng.randrange(M)
            self.body = deque(CYCLE[(k - i) % M] for i in range(length))
            self.dir = NEIGHBOR[self.body[1]].index(self.body[0])
        self.occ = bytearray(M)
        for c in self.body:
            self.occ[c] = 1
        self.grow_pending = False
        self.score = 0
        self.steps = 0
        self.steps_since_apple = 0
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

    def collides(self, c):
        # La queue libère sa case pendant le déplacement, sauf si le serpent grandit.
        return bool(self.occ[c]) and not (c == self.body[-1] and not self.grow_pending)

    def step(self, action):
        d = (self.dir + TURN[action]) % 4
        c = NEIGHBOR[self.body[0]][d]
        self.dir = d
        self.steps += 1
        self.steps_since_apple += 1
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
            self.steps_since_apple = 0
            self.apple = self._random_free()
            if self.apple is None:
                self.done = self.won = True
                return WIN
            return APPLE
        return MOVE

    def flood(self, start, target):
        """Cases libres accessibles depuis start, et si target (la queue) est atteignable."""
        occ, seen = self.occ, self._seen
        self._stamp += 1
        stamp = self._stamp
        seen[start] = stamp
        stack = [start]
        area = 0
        tail_ok = False
        while stack:
            cur = stack.pop()
            for nb in NEIGHBOR[cur]:
                if nb == target:
                    tail_ok = True
                if not occ[nb] and seen[nb] != stamp:
                    seen[nb] = stamp
                    area += 1
                    stack.append(nb)
        return area, tail_ok

    def probe(self, action):
        """Simule une action sans la jouer -> (collision, case, aire, queue accessible)."""
        c = self.target_cell(action)
        occ, pending, tail = self.occ, self.grow_pending, self.body[-1]
        if not pending:
            occ[tail] = 0
        if occ[c]:
            if not pending:
                occ[tail] = 1
            return True, c, 0, False
        occ[c] = 1
        new_tail = tail if pending else self.body[-2]
        area, tail_ok = self.flood(c, new_tail)
        occ[c] = 0
        if not pending:
            occ[tail] = 1
        return False, c, area, tail_ok

    def free_arc(self):
        """Cases libres entre la tête et la queue en avançant sur le cycle."""
        return cycle_dist(self.body[0], self.body[-1]) - 1

    def cycle_ok(self, c, hole_factor, margin):
        """Règle des raccourcis : le corps reste rangé dans l'ordre du cycle, donc
        suivre le cycle reste toujours possible. On ne saute des cases que s'il
        reste assez de place devant la tête."""
        head = self.body[0]
        if c == CYCLE_NEXT[head]:
            return True
        pending, tail = self.grow_pending, self.body[-1]
        dc = cycle_dist(head, c)
        dt = cycle_dist(head, tail)
        if dc > dt or (pending and dc == dt):
            return False
        new_tail = tail if pending else self.body[-2]
        free_after = cycle_dist(c, new_tail) - 1
        grow_after = 1 if c == self.apple else 0
        length_after = len(self.body) + (1 if pending else 0)
        holes_after = M - length_after - free_after
        return free_after - grow_after >= hole_factor * holes_after + margin


# =============================================================================
# 4. GARDE-FOU ET ENCODEUR D'ÉTAT
# =============================================================================

STATE_SIZE = 3 * 7 + 4 + 4 + 4


class Shield:
    def __init__(self, mode="cycle", hole_factor=1.0, margin=2):
        self.mode = mode
        self.hole_factor = hole_factor
        self.margin = margin

    def mask(self, env, infos):
        """Renvoie (mask, fallback). fallback=True quand aucune action n'était sûre."""
        free = [not info[0] for info in infos]
        if self.mode == "cycle":
            allowed = [free[a] and env.cycle_ok(infos[a][1], self.hole_factor, self.margin) for a in range(3)]
        elif self.mode == "tail":
            allowed = [free[a] and infos[a][3] for a in range(3)]
        else:
            allowed = free
        if any(allowed):
            return allowed, False
        # Secours : la case suivante du cycle, sinon la plus grande zone libre.
        candidates = [a for a in range(3) if free[a]] or [0, 1, 2]
        succ = CYCLE_NEXT[env.body[0]]
        best = next((a for a in candidates if infos[a][1] == succ and free[a]), None)
        if best is None:
            best = max(candidates, key=lambda a: infos[a][2])
        return [a == best for a in range(3)], True


def cycle_probe(env, action):
    """Version sans flood-fill pour le garde-fou cycle : (collision, case, place libre
    devant la tête après le coup, le coup saute-t-il la pomme ?)."""
    c = env.target_cell(action)
    if env.collides(c):
        return True, c, 0, False
    head, tail = env.body[0], env.body[-1]
    new_tail = tail if env.grow_pending else env.body[-2]
    space = max(cycle_dist(c, new_tail) - 1, 0) if cycle_dist(head, c) <= cycle_dist(head, tail) else 0
    skips_apple = c != env.apple and cycle_dist(head, env.apple) < cycle_dist(head, c)
    return False, c, space, skips_apple


def observe(env, shield):
    probe = cycle_probe if shield.mode == "cycle" else SnakeEnv.probe
    infos = [probe(env, a) for a in range(3)]
    mask, fallback = shield.mask(env, infos)
    return encode(env, infos, mask), mask, infos, fallback


def encode(env, infos, mask):
    head, apple = env.body[0], env.apple
    f = []
    for a in range(3):
        # space/flag = aire accessible/queue accessible (flood-fill)
        #           ou place devant/saute la pomme (garde-fou cycle)
        col, c, space, flag = infos[a]
        f += [col, mask[a], c == apple, torus_dist(c, apple) / MAX_TORUS_DIST,
              cycle_dist(c, apple) / M, space / M, flag]
    f += [env.dir == d for d in range(4)]
    dx, dy = torus_delta(head, apple)
    vx, vy = DIRECTIONS[env.dir]
    fwd = dx * vx + dy * vy
    right = -dx * vy + dy * vx
    f += [fwd > 0, fwd < 0, right > 0, right < 0]
    length = len(env.body)
    free_arc = env.free_arc()
    f += [length / M, free_arc / M, (M - length - free_arc) / M, env.grow_pending]
    return np.array(f, dtype=np.float32)


# =============================================================================
# 5. MODÈLE (Linear_QNet) ET DOUBLE DQN
# =============================================================================

class Linear_QNet(nn.Module):
    def __init__(self, input_size=STATE_SIZE, hidden1=256, hidden2=128, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, state, mask):
        """Meilleure action autorisée (exécution : pas d'exploration, pas de gradient)."""
        with torch.inference_mode():
            q = self(torch.from_numpy(state)).tolist()
        return max((a for a in range(3) if mask[a]), key=q.__getitem__)

    def save(self, path, meta):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt.get("meta", {})


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.s = np.zeros((capacity, STATE_SIZE), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, STATE_SIZE), dtype=np.float32)
        self.m2 = np.zeros((capacity, 3), dtype=bool)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, s, a, r, s2, m2, done):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.m2[i], self.d[i] = s, a, r, s2, m2, done
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        j = np.random.randint(0, self.size, size=batch_size)
        return tuple(torch.from_numpy(x[j]) for x in (self.s, self.a, self.r, self.s2, self.m2, self.d))


class DoubleDQNTrainer:
    def __init__(self, online, target, lr, gamma):
        self.online, self.target, self.gamma = online, target, gamma
        self.optimizer = torch.optim.Adam(online.parameters(), lr=lr)

    def train_step(self, batch):
        s, a, r, s2, m2, d = batch
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN : l'online choisit l'action suivante (parmi les autorisées),
            # le target network l'évalue.
            a2 = self.online(s2).masked_fill(~m2, -1e9).argmax(1, keepdim=True)
            q2 = self.target(s2).gather(1, a2).squeeze(1)
            y = r + self.gamma * q2 * (1.0 - d)
        loss = F.smooth_l1_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()


# =============================================================================
# 6. RÉCOMPENSES (invisibles : le score affiché reste +1 par pomme)
# =============================================================================

REWARD_EVENT = {MOVE: 0.0, APPLE: 10.0, DEATH: -30.0, WIN: 100.0}
REWARD_STEP = -0.01
REWARD_NO_SAFE_MOVE = -20.0
REWARD_TIMEOUT = -10.0


def potential(mode, head, apple, area, tail_ok):
    """Phi(s). Garde-fou cycle : la sécurité est garantie, seule compte la distance
    à la pomme le long du cycle. Sinon : proche de la pomme, espace, queue accessible."""
    if mode == "cycle":
        return -5.0 * cycle_dist(head, apple) / M
    return -torus_dist(head, apple) / MAX_TORUS_DIST + 0.5 * area / M + 0.5 * tail_ok


# =============================================================================
# 7. ENTRAÎNEMENT
# =============================================================================

def run_game(env, choose, max_steps):
    """Joue une partie complète depuis le départ standard."""
    env.reset()
    while not env.done and env.steps < max_steps:
        env.step(choose(env))
    return {"score": env.score, "won": env.won, "steps": env.steps}


def evaluate(model, shield, games, seed, max_steps):
    def choose(env):
        state, mask, _, _ = observe(env, shield)
        return model.predict(state, mask)
    model.eval()
    results = [run_game(SnakeEnv(seed + g), choose, max_steps) for g in range(games)]
    model.train()
    return results


def summarize(results):
    n = len(results)
    return {
        "score": sum(r["score"] for r in results) / n,
        "wins": sum(r["won"] for r in results) / n,
        "steps": sum(r["steps"] for r in results) / n,
    }


def minutes(steps):
    return steps / GAME_SPEED / 60


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(1)

    shield = Shield(args.shield, args.hole_factor, args.margin)
    warmup_shield = Shield("none")
    online = Linear_QNet()
    target = Linear_QNet()
    target.load_state_dict(online.state_dict())
    trainer = DoubleDQNTrainer(online, target, args.lr, args.gamma)
    buffer = ReplayBuffer(args.buffer)
    env = SnakeEnv(args.seed)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUN_DIR, stamp)
    os.makedirs(run_dir, exist_ok=True)
    ep_file = open(os.path.join(run_dir, "episodes.csv"), "w", newline="")
    ep_log = csv.writer(ep_file)
    ep_log.writerow(["episode", "start_len", "apples", "steps", "end", "epsilon", "loss", "total_steps", "elapsed_s"])
    ev_file = open(os.path.join(run_dir, "eval.csv"), "w", newline="")
    ev_log = csv.writer(ev_file)
    ev_log.writerow(["episode", "total_steps", "score", "wins", "steps", "minutes_at_5fps"])
    meta = {"shield": args.shield, "hole_factor": args.hole_factor, "margin": args.margin}
    best_key = None
    history = {"episode": [], "apples_per_1000": [], "eval_episode": [], "eval_score": [], "eval_minutes": []}
    baselines = reference_baselines(args) if args.shield == "cycle" else {}

    print(f"Run : {run_dir}")
    print(f"Garde-fou : {args.shield} | épisodes : {args.episodes} | état : {STATE_SIZE} entrées")
    t0 = time.time()
    eval_time = 0.0
    total_steps = 0
    try:
        for ep in range(1, args.episodes + 1):
            active = warmup_shield if ep <= args.shield_warmup else shield
            start_len = 3 if rng.random() >= args.curriculum else rng.randint(4, M - 10)
            env.reset(start_len)
            state, mask, infos, _ = observe(env, active)
            head_area, head_tail = env.flood(env.body[0], env.body[-1])
            phi = potential(active.mode, env.body[0], env.apple, head_area, head_tail)
            losses = []
            while True:
                eps = max(args.eps_end, args.eps_start - total_steps * (args.eps_start - args.eps_end) / args.eps_decay)
                allowed = [a for a in range(3) if mask[a]]
                if rng.random() < eps:
                    action = rng.choice(allowed)
                else:
                    action = online.predict(state, mask)
                _, c, area, tail_ok = infos[action]
                event = env.step(action)
                total_steps += 1

                reward = REWARD_STEP + REWARD_EVENT[event]
                terminal = event in (DEATH, WIN)
                if not terminal and env.steps_since_apple >= args.timeout:
                    reward += REWARD_TIMEOUT
                    terminal = True
                truncated = not terminal and env.steps >= args.max_episode_steps

                if terminal:
                    next_state = np.zeros(STATE_SIZE, dtype=np.float32)
                    next_mask = [False, False, False]
                    phi_next = 0.0
                else:
                    next_state, next_mask, next_infos, next_fallback = observe(env, active)
                    if next_fallback:
                        reward += REWARD_NO_SAFE_MOVE
                    phi_next = potential(active.mode, c, env.apple, area, tail_ok)
                reward += args.eta * (args.gamma * phi_next - phi)

                buffer.push(state, action, reward, next_state, next_mask, float(terminal))
                if buffer.size >= args.warmup and total_steps % args.train_every == 0:
                    losses.append(trainer.train_step(buffer.sample(args.batch)))
                if total_steps % args.target_sync == 0:
                    target.load_state_dict(online.state_dict())

                if terminal or truncated:
                    end = event if event in (DEATH, WIN) else ("timeout" if terminal else "tronqué")
                    break
                state, mask, infos, phi = next_state, next_mask, next_infos, phi_next

            loss = sum(losses) / len(losses) if losses else float("nan")
            elapsed = time.time() - t0
            train_time = elapsed - eval_time
            ep_log.writerow([ep, start_len, env.score, env.steps, end, f"{eps:.3f}", f"{loss:.4f}", total_steps, f"{elapsed:.0f}"])
            history["episode"].append(ep)
            history["apples_per_1000"].append(1000 * env.score / max(env.steps, 1))
            print(f"ép {ep:4d} | départ {start_len:3d} | pommes {env.score:3d} | coups {env.steps:5d} | "
                  f"{end:8s} | eps {eps:.2f} | loss {loss:.3f} | {total_steps / max(train_time, 1e-9):.0f} coups/s | {elapsed / 60:.0f} min")

            if ep % args.eval_every == 0 or ep == args.episodes:
                t_eval = time.time()
                stats = summarize(evaluate(online, shield, args.eval_games, 10_000, args.eval_max_steps))
                eval_time += time.time() - t_eval
                ev_log.writerow([ep, total_steps, f"{stats['score']:.1f}", f"{stats['wins']:.2f}",
                                 f"{stats['steps']:.0f}", f"{minutes(stats['steps']):.1f}"])
                ev_file.flush()
                ep_file.flush()
                history["eval_episode"].append(ep)
                history["eval_score"].append(stats["score"])
                history["eval_minutes"].append(minutes(stats["steps"]))
                key = (stats["score"], -stats["steps"])
                tag = ""
                if best_key is None or key > best_key:
                    best_key = key
                    online.save(os.path.join(MODEL_DIR, "best.pth"), {**meta, "episode": ep, **stats})
                    tag = "  <- nouveau meilleur modèle"
                print(f"  ÉVAL ép {ep} : score {stats['score']:.1f} | victoires {stats['wins']:.0%} | "
                      f"{stats['steps']:.0f} coups ≈ {minutes(stats['steps']):.1f} min à 5 FPS{tag}")
                plot_history(history, baselines, os.path.join(run_dir, "courbe.png"))
    except KeyboardInterrupt:
        print("\nInterruption : sauvegarde du dernier modèle.")
    finally:
        online.save(os.path.join(MODEL_DIR, "last.pth"), {**meta, "total_steps": total_steps})
        ep_file.close()
        ev_file.close()
        if history["episode"]:
            plot_history(history, baselines, os.path.join(run_dir, "courbe.png"))
        print(f"Modèles : {MODEL_DIR} | journaux et courbe : {run_dir}")


def plot_history(history, baselines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, muted, grid = "#2a78d6", "#1f1f1e", "#6b6a63", "#e4e3dc"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.patch.set_facecolor("white")

    def style(ax, title, ylabel):
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        ax.set_xlabel("épisode", color=muted)
        ax.set_ylabel(ylabel, color=muted)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)

    ep, apm = history["episode"], history["apples_per_1000"]
    window = 25
    smooth = [sum(apm[max(0, i - window + 1):i + 1]) / (i - max(0, i - window + 1) + 1) for i in range(len(apm))]
    axes[0].plot(ep, smooth, color=blue, linewidth=2)
    style(axes[0], "Entraînement : pommes / 1000 coups (moyenne glissante 25)", "pommes / 1000 coups")

    axes[1].plot(history["eval_episode"], history["eval_score"], color=blue, linewidth=2, marker="o", markersize=5)
    # La dernière pomme est mangée quand le serpent occupe déjà les 225 cases.
    axes[1].axhline(M - 2, color=muted, linewidth=1, linestyle="--")
    axes[1].annotate("score max (223)", (0, M - 2), xycoords=("axes fraction", "data"),
                     xytext=(4, -12), textcoords="offset points", color=muted, fontsize=9)
    style(axes[1], "Évaluation : score moyen", "score")

    axes[2].plot(history["eval_episode"], history["eval_minutes"], color=blue, linewidth=2, marker="o", markersize=5)
    for label, value in baselines.items():
        axes[2].axhline(value, color=muted, linewidth=1, linestyle="--")
        axes[2].annotate(label, (0, value), xycoords=("axes fraction", "data"),
                         xytext=(4, 3), textcoords="offset points", color=muted, fontsize=9)
    style(axes[2], "Évaluation : durée d'une partie à 5 FPS", "minutes")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =============================================================================
# 8. POLITIQUES DE RÉFÉRENCE (sans réseau) POUR COMPARER
# =============================================================================

GREEDY_MAX_LEN = 115   # au-delà, les trous laissés par les raccourcis coûtent plus qu'ils ne rapportent


def policy_cycle_pure(env):
    succ = CYCLE_NEXT[env.body[0]]
    return next(a for a in range(3) if env.target_cell(a) == succ)


def make_cycle_policy(shield, greedy, rng, max_len=M):
    def choose(env):
        if len(env.body) >= max_len:
            return policy_cycle_pure(env)
        allowed = [a for a in range(3)
                   if not env.collides(env.target_cell(a))
                   and env.cycle_ok(env.target_cell(a), shield.hole_factor, shield.margin)]
        if not allowed:
            return policy_cycle_pure(env)
        if greedy:
            return min(allowed, key=lambda a: cycle_dist(env.target_cell(a), env.apple))
        return rng.choice(allowed)
    return choose


def make_tail_policy():
    shield = Shield("tail")

    def choose(env):
        infos = [env.probe(a) for a in range(3)]
        mask, _ = shield.mask(env, infos)
        allowed = [a for a in range(3) if mask[a]]
        return min(allowed, key=lambda a: (torus_dist(infos[a][1], env.apple), -infos[a][2]))
    return choose


def reference_baselines(args):
    """Durées moyennes (min à 5 FPS) des politiques sans réseau, sur les graines d'évaluation."""
    shield = Shield("cycle", args.hole_factor, args.margin)
    out = {}
    for label, choose in (("cycle pur", policy_cycle_pure),
                          ("glouton (raccourcis si L<115)", make_cycle_policy(shield, True, None, GREEDY_MAX_LEN))):
        res = [run_game(SnakeEnv(10_000 + g), choose, args.eval_max_steps) for g in range(args.eval_games)]
        out[label] = minutes(summarize(res)["steps"])
    return out


def bench(args):
    shield = Shield("cycle", args.hole_factor, args.margin)
    rng = random.Random(args.seed)
    policies = [
        ("cycle pur", policy_cycle_pure),
        ("cycle + raccourcis gloutons", make_cycle_policy(shield, True, rng)),
        ("glouton, raccourcis si L<115", make_cycle_policy(shield, True, rng, GREEDY_MAX_LEN)),
        ("cycle + raccourcis aléatoires", make_cycle_policy(shield, False, rng)),
        ("queue accessible + glouton", make_tail_policy()),
    ]
    if args.only:
        policies = [p for p in policies if args.only in p[0]]
    print(f"{args.games} parties par politique (hole_factor={args.hole_factor}, margin={args.margin})")
    for label, choose in policies:
        t0 = time.time()
        res = [run_game(SnakeEnv(args.seed + g), choose, args.eval_max_steps) for g in range(args.games)]
        s = summarize(res)
        print(f"{label:32s} | victoires {s['wins']:6.0%} | score {s['score']:6.1f} | "
              f"{s['steps']:7.0f} coups ≈ {minutes(s['steps']):5.1f} min | "
              f"min/max score {min(r['score'] for r in res)}/{max(r['score'] for r in res)} | {time.time() - t0:.0f}s")


def eval_cmd(args):
    model, meta = Linear_QNet.load(args.model)
    shield = Shield(meta.get("shield", "cycle"), meta.get("hole_factor", 1.0), meta.get("margin", 2))
    res = evaluate(model, shield, args.games, args.seed, args.eval_max_steps)
    for i, r in enumerate(res):
        print(f"partie {i + 1} : score {r['score']} | victoire {r['won']} | {r['steps']} coups ≈ {minutes(r['steps']):.1f} min")
    s = summarize(res)
    print(f"moyenne : score {s['score']:.1f} | victoires {s['wins']:.0%} | {s['steps']:.0f} coups ≈ {minutes(s['steps']):.1f} min à 5 FPS")


# =============================================================================
# 9. EXÉCUTION (partie pygame, clock d'origine)
# =============================================================================

def play(args):
    model, meta = Linear_QNet.load(args.model)
    shield = Shield(meta.get("shield", "cycle"), meta.get("hole_factor", 1.0), meta.get("margin", 2))
    env = SnakeEnv()
    fps = args.debug_fps or GAME_SPEED
    print(f"Modèle : {args.model} ({meta}) | garde-fou : {shield.mode} | FPS : {fps}"
          + ("" if fps == GAME_SPEED else "  /!\\ vitesse de debug, partie NON officielle"))

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Oignon (Double DQN + garde-fou)")
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
                snake, apple = Snake(), Apple(snake.body)
                game_over = victory = False
                start_time, final_elapsed, moves = time.time(), None, 0

        if not game_over and not victory:
            move_counter += 1
            if move_counter >= GAME_SPEED // 10:
                # L'IA choisit la direction à la place du clavier.
                env.load(snake, apple)
                state, mask, _, _ = observe(env, shield)
                action = model.predict(state, mask)
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
# 10. LIGNE DE COMMANDE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Snake IA - équipe Oignon (Double DQN + garde-fou)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_model = os.path.join(MODEL_DIR, "best.pth")

    def common(p):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--hole-factor", type=float, default=1.0)
        p.add_argument("--margin", type=int, default=2)
        p.add_argument("--eval-max-steps", type=int, default=30_000)

    t = sub.add_parser("train", help="entraîner le Double DQN (sans affichage)")
    common(t)
    t.add_argument("--episodes", type=int, default=800)
    t.add_argument("--shield", choices=["cycle", "tail", "none"], default="cycle")
    t.add_argument("--shield-warmup", type=int, default=0, help="épisodes initiaux avec le garde-fou 'none'")
    t.add_argument("--curriculum", type=float, default=0.5, help="proportion de départs avec un long serpent")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--eta", type=float, default=0.5, help="coefficient du shaping potentiel")
    t.add_argument("--buffer", type=int, default=100_000)
    t.add_argument("--warmup", type=int, default=5_000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--train-every", type=int, default=4)
    t.add_argument("--target-sync", type=int, default=2_000)
    t.add_argument("--eps-start", type=float, default=1.0)
    t.add_argument("--eps-end", type=float, default=0.05)
    t.add_argument("--eps-decay", type=int, default=150_000, help="coups pour passer de eps-start à eps-end")
    t.add_argument("--timeout", type=int, default=450, help="coups sans pomme avant fin d'épisode")
    t.add_argument("--max-episode-steps", type=int, default=2_000)
    t.add_argument("--eval-every", type=int, default=25)
    t.add_argument("--eval-games", type=int, default=3)

    p = sub.add_parser("play", help="partie pygame avec le modèle entraîné")
    p.add_argument("--model", default=default_model)
    p.add_argument("--debug-fps", type=int, default=None, help="accélère l'affichage (NON officiel)")

    e = sub.add_parser("eval", help="parties sans affichage avec le modèle figé")
    common(e)
    e.add_argument("--model", default=default_model)
    e.add_argument("--games", type=int, default=5)

    b = sub.add_parser("bench", help="politiques de référence sans réseau")
    common(b)
    b.add_argument("--games", type=int, default=20)
    b.add_argument("--only", default=None, help="filtrer les politiques par nom")

    # Sans argument (`python snake-ia.py`) : partie officielle avec le modèle entraîné.
    args = parser.parse_args(sys.argv[1:] or ["play"])
    {"train": train, "play": play, "eval": eval_cmd, "bench": bench}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
