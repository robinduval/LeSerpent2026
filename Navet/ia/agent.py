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
    def __init__(self, gamma=0.9, use_target=False, eps_mode="loeber", eps_min=0.05, eps_decay=400,
                 torus_food=False, rich_state=False):
        self.n_games = 0
        self.torus_food = torus_food  # Fix 5/5 : bits pomme en distance torique
        self.rich_state = rich_state  # Fix 6/6 : + espace libre (flood-fill), queue, longueur
        self.state_size = 16 if rich_state else 11
        self.gamma = gamma
        self.use_target = use_target  # Fix 1/4
        self.eps_mode = eps_mode  # "loeber" (80-n) ou "decay" (Fix 2/4)
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.epsilon = 80
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(self.state_size, 256, 3) if HAS_TORCH else None
        if HAS_TORCH and use_target:
            from model import Linear_QNet as _Q
            self.target_model = _Q(self.state_size, 256, 3)
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
        if self.torus_food:
            # Fix 5/5 : direction de la pomme par le chemin LE PLUS COURT en tore.
            # Avant, `apple[0] < head[0]` comparait des coordonnees brutes : tete en x=1,
            # pomme en x=13 -> l'etat disait "a droite" alors que le court chemin fait
            # 3 cases a gauche en traversant le mur. Mesure : 26.8% des pas concernes,
            # et l'etat pointait a l'oppose dans 100% de ces cas. Le reseau ne pouvait
            # pas apprendre le raccourci, son entree lui mentait.
            dx = (apple[0] - head[0]) % GRID_SIZE  # pas vers la droite (avec wrap)
            dy = (apple[1] - head[1]) % GRID_SIZE  # pas vers le bas (avec wrap)
            # dx < GRID_SIZE - dx  <=>  aller a droite est plus court qu'aller a gauche
            food_right = dx != 0 and dx < GRID_SIZE - dx
            food_left = dx != 0 and GRID_SIZE - dx < dx
            food_down = dy != 0 and dy < GRID_SIZE - dy
            food_up = dy != 0 and GRID_SIZE - dy < dy
        else:
            # etat historique (coordonnees brutes), garde pour rejouer les modeles <= fix4
            food_left = apple[0] < head[0]
            food_right = apple[0] > head[0]
            food_up = apple[1] < head[1]
            food_down = apple[1] > head[1]

        state = [
            danger_straight, danger_right, danger_left,
            dir_l, dir_r, dir_u, dir_d,
            food_left, food_right, food_up, food_down,
        ]
        if not self.rich_state:
            return np.array(state, dtype=int)

        # Fix 6/6 : les 3 bits de danger ne voient qu'UNE case devant. A 70 pommes le
        # serpent fait 73 cases : il entre dans des poches sans issue qu'il ne peut pas
        # voir. Mesure : 50 morts sur 50 en eval sont des morsures, zero famine.
        # On ajoute l'espace libre atteignable par direction (flood-fill borne), la
        # distance a la queue et la longueur : de quoi anticiper l'enfermement.
        occupied = {tuple(p) for p in game.body}
        max_cells = GRID_SIZE * GRID_SIZE

        def free_space(d):
            start = (wrapped(head[0] + d[0], head[1] + d[1])[0],
                     wrapped(head[0] + d[0], head[1] + d[1])[1])
            if start in occupied:
                return 0.0
            # BFS borne : on plafonne au nombre de cases du corps, au-dela peu importe
            cap = min(max_cells, max(16, 2 * len(game.body)))
            seen = {start}
            stack = [start]
            while stack and len(seen) < cap:
                cx, cy = stack.pop()
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = ((cx + ddx) % GRID_SIZE, (cy + ddy) % GRID_SIZE)
                    if nxt not in seen and nxt not in occupied:
                        seen.add(nxt)
                        stack.append(nxt)
                        if len(seen) >= cap:
                            break
            return len(seen) / cap  # normalise 0-1

        tail = game.body[-1]
        tdx = min((tail[0] - head[0]) % GRID_SIZE, (head[0] - tail[0]) % GRID_SIZE)
        tdy = min((tail[1] - head[1]) % GRID_SIZE, (head[1] - tail[1]) % GRID_SIZE)

        extra = [
            free_space(straight_d),
            free_space(right_d),
            free_space(left_d),
            (tdx + tdy) / GRID_SIZE,
            len(game.body) / max_cells,
        ]
        return np.array([float(v) for v in state] + extra, dtype=float)

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
