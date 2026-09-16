# Agent DQN baseline Loeber, état 11 bits adapté au tore (danger corps après wrap).
import random
from collections import deque
import numpy as np

from game import GRID_SIZE
try:
    from model import Linear_QNet, QTrainer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


import math

class Agent:
    def __init__(self, gamma=0.9, use_target=False, eps_mode="loeber", eps_min=0.05, eps_decay=400):
        self.n_games = 0
        self.gamma = gamma
        self.use_target = use_target  # Fix 1/4
        self.eps_mode = eps_mode  # "loeber" (80-n) ou "decay" (Fix 2/4)
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.epsilon = 80
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3) if HAS_TORCH else None
        if HAS_TORCH and use_target:
            from model import Linear_QNet as _Q
            self.target_model = _Q(11, 256, 3)
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.target_model = None
        self.trainer = QTrainer(self.model, lr=0.001, gamma=gamma, target_model=self.target_model) if HAS_TORCH else None

    def get_epsilon(self):
        if self.eps_mode == "loeber":
            return max(0, 80 - self.n_games)
        # Fix 2/4 : décroissance exponentielle avec plancher (0-200 scale comme Loeber)
        # eps_0=80 -> proba 80/200=0.4 au début, plancher eps_min*200 en échelle 0-200
        eps01 = self.eps_min + (1.0 - self.eps_min) * math.exp(-self.n_games / self.eps_decay)
        return eps01 * 200

    def get_epsilon_proba(self):
        return self.get_epsilon() / 200.0

    def get_state(self, game):
        head = game.head
        # points voisins en tenant compte du tore
        def wrapped(x, y):
            return [x % GRID_SIZE, y % GRID_SIZE]

        dir_l = game.direction == (-1, 0)
        dir_r = game.direction == (1, 0)
        dir_u = game.direction == (0, -1)
        dir_d = game.direction == (0, 1)

        # directions relatives : tout droit / droite / gauche
        clockwise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = clockwise.index(game.direction)
        straight_d = clockwise[idx]
        right_d = clockwise[(idx + 1) % 4]
        left_d = clockwise[(idx - 1) % 4]

        def danger(d):
            p = wrapped(head[0] + d[0], head[1] + d[1])
            return game.is_collision(p)

        danger_straight = danger(straight_d)
        danger_right = danger(right_d)
        danger_left = danger(left_d)

        apple = game.apple if game.apple is not None else head
        food_left = apple[0] < head[0]
        food_right = apple[0] > head[0]
        food_up = apple[1] < head[1]
        food_down = apple[1] > head[1]

        state = [
            danger_straight, danger_right, danger_left,
            dir_l, dir_r, dir_u, dir_d,
            food_left, food_right, food_up, food_down,
        ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        self.epsilon = self.get_epsilon()
        move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            m = random.randint(0, 2)
            move[m] = 1
        else:
            import torch
            with torch.no_grad():
                pred = self.model(torch.tensor(state, dtype=torch.float))
                m = torch.argmax(pred).item()
                move[m] = 1
        return move

    def remember(self, s, a, r, s2, done, truncated=False):
        self.memory.append((s, a, r, s2, done, truncated))

    def train_short(self, s, a, r, s2, done, truncated=False):
        return self.trainer.train_step(s, a, r, s2, done, truncated)

    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = list(self.memory)
        if not batch:
            return 0.0
        # compat anciens tuples 5-éléments (sans truncated)
        if len(batch[0]) == 5:
            s, a, r, s2, d = zip(*batch)
            t = [False] * len(d)
        else:
            s, a, r, s2, d, t = zip(*batch)
        return self.trainer.train_step(s, a, r, s2, d, t)
