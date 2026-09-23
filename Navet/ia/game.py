# Navet — Baseline DQN façon Loeber (Étape 1, pur RL)
# Contraintes respectées : GRID_SIZE=15, score affiché +1/pomme, monde tore (% GRID_SIZE).
# Reward interne (barème cours) isolée ici, ne touche jamais au score affiché.
# Flaws volontaires reproduits pour courbe "avant" (à fixer Étape 2) :
#  - eps = 80 - n_games (meurt après 80)
#  - gamma = 0.9 (myope)
#  - pas de target network
#  - timeout 100*len traité comme mort -10 (confond truncation/terminal)
#  - step +0.1 (pousse au S / tourner en rond)
import random
from collections import deque

GRID_SIZE = 15  # intouchable (contrainte 2)
GAME_SPEED = 5  # référence, non utilisé en headless training (éval seule à clock intacte)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

REWARD_APPLE = 10.0    # reward interne, pas score
REWARD_DEATH = -10.0
REWARD_VICTORY = 100.0
REWARD_STEP = 0.1      # flaw volontaire, remplacé par -0.01 Étape 3


class SnakeGame:
    """Env headless torique, miroir de serpent-algo.py sans pygame."""

    def __init__(self, seed=None, step_reward=0.1, hunger_mode="loeber", hunger_k=2.0):
        # step_reward : reward interne par pas (+0.1 flaw baseline, -0.01 anti-S)
        # hunger_mode : "loeber" (timeout 100*len = mort -10) ou "truncated"
        #   (faim = truncation avec bootstrap, limite = hunger_k * cases_libres)
        self.step_reward = step_reward
        self.hunger_mode = hunger_mode
        self.hunger_k = hunger_k
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        hx, hy = GRID_SIZE // 4, GRID_SIZE // 2
        self.head = [hx, hy]
        self.body = [[hx, hy], [hx - 1, hy], [hx - 2, hy]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0  # score affiché : +1 / pomme, intouchable
        self.steps = 0
        self.steps_since_apple = 0
        self.apple = self._random_free_cell()
        self.done = False
        self.victory = False
        return self.get_state_simple()

    def _random_free_cell(self):
        occupied = {tuple(p) for p in self.body}
        free = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                if (x, y) not in occupied]
        if not free:
            return None
        return list(self.rng.choice(free))

    def _next_head(self, direction):
        # Tore volontaire : % GRID_SIZE, les murs ne tuent pas (prof taquin).
        return [(self.head[0] + direction[0]) % GRID_SIZE,
                (self.head[1] + direction[1]) % GRID_SIZE]

    def _body_danger(self, pos):
        # La queue va se libérer sauf si grow_pending : on l'exclut.
        if self.grow_pending:
            trunk = self.body[1:]
        else:
            trunk = self.body[1:-1] if len(self.body) > 2 else self.body[1:]
        return list(pos) in trunk

    def is_collision(self, pos):
        return self._body_danger(pos)

    def get_state_simple(self):
        # placeholder, le vrai état 11 bits est dans agent.py (besoin direction relative)
        return self.head[:]

    def play_step(self, action):
        """action: [straight, right, left] one-hot (format Loeber).
        Retour: reward, done, score, info avec terminated/truncated (Gym-like)."""
        if self.done:
            return 0.0, True, self.score, {"cause": "already_done", "terminated": True, "truncated": False}

        clock_wise = [RIGHT, DOWN, LEFT, UP]  # sens horaire
        idx = clock_wise.index(self.direction)
        if action == [1, 0, 0]:
            new_dir = clock_wise[idx]
        elif action == [0, 1, 0]:
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir

        self.head = self._next_head(self.direction)
        self.body.insert(0, self.head[:])
        self.steps += 1
        self.steps_since_apple += 1

        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False

        # 1. morsure (seule vraie mort en tore) -> terminated, pas de bootstrap
        if self.head in self.body[1:]:
            self.done = True
            return REWARD_DEATH, True, self.score, {"cause": "corps", "terminated": True, "truncated": False}

        # 2. pomme mangée
        if self.apple is not None and self.head == self.apple:
            self.score += 1  # score affiché intouchable
            self.grow_pending = True
            self.steps_since_apple = 0
            new_apple = self._random_free_cell()
            if new_apple is None:
                self.apple = None
                self.done = True
                self.victory = True
                return REWARD_VICTORY, True, self.score, {"cause": "victoire", "terminated": True, "truncated": False}
            self.apple = new_apple
            return REWARD_APPLE, False, self.score, {"cause": "pomme", "terminated": False, "truncated": False}

        # 3. faim : loeber (flaw) vs truncated (fix)
        if self.hunger_mode == "loeber":
            if self.steps_since_apple > 100 * len(self.body):
                self.done = True
                return REWARD_DEATH, True, self.score, {"cause": "famine_timeout_flaw", "terminated": True, "truncated": False}
        else:
            free = GRID_SIZE * GRID_SIZE - len(self.body)
            limit = max(50, int(self.hunger_k * free))
            if self.steps_since_apple > limit:
                self.done = True
                # truncation : petite pénalité temps, mais on bootstrappe (pas une mort)
                return self.step_reward, True, self.score, {"cause": "famine_truncated", "terminated": False, "truncated": True}

        return self.step_reward, False, self.score, {"cause": "step", "terminated": False, "truncated": False}
