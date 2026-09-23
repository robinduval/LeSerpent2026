"""
serpent-algo.py — Snake piloté par Apprentissage par Renforcement (Deep Q-Learning)
Groupe Melon — LAMECHE Nazim, DE-SOUSA Basile

Basé sur le socle serpent.py (visuel, classes Snake/Apple, panneau de score
conservés à l'identique) et sur l'architecture en 3 blocs du cours :
    - Game  : SnakeGameAI.play_step(action) -> reward, game_over, score
    - Model : Linear_QNet (DQN) + QTrainer (torch)
    - Agent : get_state / get_action / remember / train_short|long_memory

Utilisation :
    python serpent-algo.py train --games 400            # entraînement headless (clock accélérée)
    python serpent-algo.py train --games 400 --render   # idem avec affichage (plus lent)
    python serpent-algo.py play                         # joue avec model.pth à la clock d'origine
    python serpent-algo.py play --model autre.pth

Contraintes respectées : GRID_SIZE, GAME_SPEED (clock du mode play) et le
scoring (1 point par pomme) sont ceux du socle et ne sont pas modifiés.
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONSTANTES DE JEU (socle, inchangées) ---
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
BLEU = (30, 144, 255)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
# Ordre horaire : tourner à droite = index+1, à gauche = index-1
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]

# --- HYPERPARAMÈTRES RL ---
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.95
HIDDEN_SIZE = 256
EPSILON_GAMES = 150       # exploration décroissante sur les N premières parties
TIMEOUT_FACTOR = 100      # partie stoppée si > TIMEOUT_FACTOR * len(serpent) coups sans pomme

# --- RÉCOMPENSES (modifiables, cf. consigne) ---
R_APPLE = 10.0
R_DEATH = -10.0
R_TIMEOUT = -10.0
R_VICTORY = 100.0
R_CLOSER = 0.3            # se rapproche de la pomme (distance torique)
R_FARTHER = -0.4          # s'en éloigne : légèrement plus pénalisant pour éviter les boucles

MODEL_PATH = "model.pth"
LOG_PATH = "training_log.csv"

# pygame est importé partout (dessin) mais initialisé seulement si on affiche.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402


# ============================================================================
# CLASSES DU SOCLE (inchangées)
# ============================================================================

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
        # Le socle fait un modulo : la grille est torique (pas de mur).
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


# --- FONCTIONS D'AFFICHAGE (socle) ---

def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(surface, font, snake, start_time, extra=None):
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

    if extra:
        small = pygame.font.Font(None, 26)
        extra_text = small.render(extra, True, BLEU)
        surface.blit(extra_text, (SCREEN_WIDTH // 2 - extra_text.get_width() // 2, 52))


def display_message(surface, font, message, color=BLANC, y_offset=0):
    text_surface = font.render(message, True, color)
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)


# ============================================================================
# BLOC 1 — GAME : environnement RL au-dessus du socle
# ============================================================================

def torus_delta(a, b):
    """Plus petit déplacement signé de a vers b sur un axe torique."""
    d = (b - a) % GRID_SIZE
    if d > GRID_SIZE // 2:
        d -= GRID_SIZE
    return d


def torus_distance(p, q):
    return abs(torus_delta(p[0], q[0])) + abs(torus_delta(p[1], q[1]))


class SnakeGameAI:
    """
    Enveloppe le socle pour l'agent.
    action = [tout droit, tourner à droite, tourner à gauche] (one-hot)
    play_step(action) -> reward, game_over, score
    """
    def __init__(self, render=False, speed=0):
        self.render = render
        self.speed = speed          # 0 = aucune limite (clock accélérée)
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake IA - Deep Q-Learning (entraînement)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 40)
        self.reset()

    def reset(self):
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.frame_iteration = 0
        self.start_time = time.time()
        self.victory = False

    @property
    def score(self):
        return self.snake.score

    # --- géométrie utilitaire ---
    @staticmethod
    def wrap(pt):
        return [pt[0] % GRID_SIZE, pt[1] % GRID_SIZE]

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake.head_pos
        pt = self.wrap(pt)
        # sur une grille torique, seul le corps est dangereux
        return pt in self.snake.body[1:]

    def free_space_from(self, pt):
        """Nombre de cases atteignables depuis pt (flood-fill), normalisé."""
        pt = tuple(self.wrap(pt))
        blocked = {tuple(s) for s in self.snake.body[:-1]}  # la queue va bouger
        if pt in blocked:
            return 0.0
        seen = {pt}
        stack = [pt]
        limit = GRID_SIZE * GRID_SIZE
        while stack and len(seen) < limit:
            x, y = stack.pop()
            for dx, dy in CLOCKWISE:
                n = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
                if n not in seen and n not in blocked:
                    seen.add(n)
                    stack.append(n)
        return len(seen) / limit

    def ray_distance(self, direction):
        """Distance (normalisée) jusqu'au premier segment de corps dans une direction."""
        x, y = self.snake.head_pos
        body = {tuple(s) for s in self.snake.body[1:]}
        for step in range(1, GRID_SIZE + 1):
            x = (x + direction[0]) % GRID_SIZE
            y = (y + direction[1]) % GRID_SIZE
            if (x, y) in body:
                return step / GRID_SIZE
        return 1.0

    # --- pas de simulation ---
    def play_step(self, action):
        self.frame_iteration += 1

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # 1. action relative -> direction absolue
        idx = CLOCKWISE.index(self.snake.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = CLOCKWISE[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = CLOCKWISE[(idx + 1) % 4]
        else:
            new_dir = CLOCKWISE[(idx - 1) % 4]
        self.snake.set_direction(new_dir)

        dist_before = torus_distance(self.snake.head_pos, self.apple.position)
        self.snake.move()

        # 2. fin de partie ?
        reward = 0.0
        game_over = False
        if self.snake.is_game_over():
            return R_DEATH, True, self.snake.score
        if self.frame_iteration > TIMEOUT_FACTOR * len(self.snake.body):
            return R_TIMEOUT, True, self.snake.score

        # 3. pomme
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            reward = R_APPLE
            self.frame_iteration = 0
            if not self.apple.relocate(self.snake.body):
                self.victory = True
                return R_VICTORY, True, self.snake.score
        else:
            dist_after = torus_distance(self.snake.head_pos, self.apple.position)
            reward = R_CLOSER if dist_after < dist_before else R_FARTHER

        # 4. affichage optionnel
        if self.render:
            self._draw()
            if self.speed > 0:
                self.clock.tick(self.speed)

        return reward, game_over, self.snake.score

    def _draw(self, extra=None):
        self.screen.fill(GRIS_FOND)
        pygame.draw.rect(self.screen, NOIR, pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH))
        draw_grid(self.screen)
        self.apple.draw(self.screen)
        self.snake.draw(self.screen)
        display_info(self.screen, self.font, self.snake, self.start_time, extra)
        pygame.display.flip()


# ============================================================================
# BLOC 2 — MODEL : réseau Q (torch) + entraîneur
# ============================================================================

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

    def save(self, path=MODEL_PATH):
        torch.save(self.state_dict(), path)

    def load(self, path=MODEL_PATH):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:  # un seul échantillon -> batch de 1
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Q(s,a) prédit
        pred = self.model(state)
        target = pred.clone().detach()
        with torch.no_grad():
            next_q = self.model(next_state).max(1)[0]
        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * next_q[i]
            target[i][torch.argmax(action[i]).item()] = q_new

        # Bellman : loss = (Q_new - Q_pred)^2
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============================================================================
# BLOC 3 — AGENT
# ============================================================================

STATE_SIZE = 17  # 11 états du cours + 3 espaces libres + 3 distances de corps


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)

    def get_state(self, game):
        snake = game.snake
        head = snake.head_pos
        d = snake.direction
        idx = CLOCKWISE.index(d)
        dir_straight = d
        dir_right = CLOCKWISE[(idx + 1) % 4]
        dir_left = CLOCKWISE[(idx - 1) % 4]

        pt_s = [head[0] + dir_straight[0], head[1] + dir_straight[1]]
        pt_r = [head[0] + dir_right[0], head[1] + dir_right[1]]
        pt_l = [head[0] + dir_left[0], head[1] + dir_left[1]]

        ax, ay = game.apple.position
        dx = torus_delta(head[0], ax)   # <0 : pomme à gauche (via le tore)
        dy = torus_delta(head[1], ay)   # <0 : pomme en haut

        state = [
            # Danger (en face, à droite, à gauche)
            game.is_collision(pt_s),
            game.is_collision(pt_r),
            game.is_collision(pt_l),
            # Direction actuelle
            d == LEFT, d == RIGHT, d == UP, d == DOWN,
            # Position de la pomme (distance torique la plus courte)
            dx < 0, dx > 0, dy < 0, dy > 0,
            # Enrichissement : espace atteignable après chaque action
            game.free_space_from(pt_s),
            game.free_space_from(pt_r),
            game.free_space_from(pt_l),
            # Enrichissement : distance au corps dans chaque direction
            game.ray_distance(dir_straight),
            game.ray_distance(dir_right),
            game.ray_distance(dir_left),
        ]
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
        states, actions, rewards, next_states, dones = zip(*sample)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, explore=True):
        # epsilon-greedy : exploration décroissante
        self.epsilon = max(0, EPSILON_GAMES - self.n_games)
        move = [0, 0, 0]
        if explore and random.randint(0, 200) < self.epsilon:
            move[random.randint(0, 2)] = 1
        else:
            with torch.no_grad():
                pred = self.model(torch.tensor(state, dtype=torch.float))
            move[torch.argmax(pred).item()] = 1
        return move


# ============================================================================
# ENTRAÎNEMENT
# ============================================================================

def train(n_games, render, speed, model_out, resume):
    agent = Agent()
    if resume and os.path.exists(resume):
        agent.model.load(resume)
        agent.model.train()
        agent.n_games = EPSILON_GAMES // 2  # réintroduit un peu d'exploration à la reprise
        print(f"Reprise depuis {resume}")
    game = SnakeGameAI(render=render, speed=speed)

    record = 0
    total_score = 0
    scores, means = [], []
    t0 = time.time()

    with open(LOG_PATH, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["game", "score", "record", "mean_score", "steps", "elapsed_s"])

        steps = 0
        while agent.n_games < n_games:
            state_old = agent.get_state(game)
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)
            steps += 1

            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save(model_out)

                total_score += score
                mean = total_score / agent.n_games
                scores.append(score)
                means.append(mean)
                log.writerow([agent.n_games, score, record, f"{mean:.2f}", steps, f"{time.time() - t0:.1f}"])
                f.flush()
                if agent.n_games % 10 == 0 or score >= record:
                    print(f"Partie {agent.n_games:4d} | score {score:3d} | record {record:3d} | "
                          f"moyenne {mean:5.2f} | {time.time() - t0:6.0f}s")
                steps = 0

    print(f"\nTerminé : {n_games} parties, record {record}, modèle -> {model_out}, log -> {LOG_PATH}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(scores, label="score", alpha=0.5)
        plt.plot(means, label="moyenne", linewidth=2)
        plt.xlabel("Partie"); plt.ylabel("Score"); plt.title("Entraînement DQN - Snake (Melon)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("training_curve.png", dpi=120)
        print("Courbe -> training_curve.png")
    except ImportError:
        pass


# ============================================================================
# MODE PLAY : l'agent joue avec le visuel et la clock du socle
# ============================================================================

def play(model_path):
    if not os.path.exists(model_path):
        print(f"Modèle introuvable : {model_path}. Lance d'abord : python serpent-algo.py train")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Deep Q-Learning")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    agent = Agent()
    agent.model.load(model_path)
    game = SnakeGameAI(render=False)

    running = True
    game_over = False
    end_time = None
    start_time = time.time()
    game.start_time = start_time
    move_counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and game_over and event.key == pygame.K_SPACE:
                game.reset()
                start_time = time.time()
                game.start_time = start_time
                game_over = False
                end_time = None

        if not game_over:
            # Même rythme que le socle (clock inchangée)
            move_counter += 1
            if move_counter >= GAME_SPEED // 10:
                move_counter = 0
                state = agent.get_state(game)
                action = agent.get_action(state, explore=False)
                _, done, score = game.play_step(action)
                if done:
                    game_over = True
                    end_time = time.time()
                    elapsed = end_time - start_time
                    ratio = score / elapsed if elapsed > 0 else 0
                    print(f"[RESULTAT] score={score} temps={elapsed:.1f}s ratio={ratio:.3f} pomme/s "
                          f"{'VICTOIRE' if game.victory else ''}")

        # Dessin (identique au socle)
        screen.fill(GRIS_FOND)
        pygame.draw.rect(screen, NOIR, pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH))
        draw_grid(screen)
        game.apple.draw(screen)
        game.snake.draw(screen)
        shown_time = start_time if end_time is None else start_time  # le chrono s'affiche depuis le départ
        display_info(screen, font_main, game.snake, shown_time, extra="Mode: IA (DQN)")

        if game_over:
            elapsed = end_time - start_time
            if game.victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
            display_message(screen, font_main, f"Score {game.score} en {elapsed:.1f}s - ESPACE pour rejouer", BLANC, y_offset=100)

        pygame.display.flip()
        clock.tick(GAME_SPEED)

    pygame.quit()


# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Snake - Deep Q-Learning (groupe Melon)")
    sub = parser.add_subparsers(dest="mode")

    p_train = sub.add_parser("train", help="entraîner l'agent")
    p_train.add_argument("--games", type=int, default=400, help="nombre de parties")
    p_train.add_argument("--render", action="store_true", help="afficher le jeu pendant l'entraînement")
    p_train.add_argument("--speed", type=int, default=0, help="FPS en mode --render (0 = illimité)")
    p_train.add_argument("--out", default=MODEL_PATH, help="fichier modèle de sortie")
    p_train.add_argument("--resume", default=None, help="reprendre depuis un modèle existant")

    p_play = sub.add_parser("play", help="faire jouer l'agent entraîné")
    p_play.add_argument("--model", default=MODEL_PATH)

    args = parser.parse_args()
    if args.mode == "train":
        if not args.render:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        train(args.games, args.render, args.speed, args.out, args.resume)
    else:
        play(args.model if args.mode == "play" else MODEL_PATH)


if __name__ == "__main__":
    main()
