"""
Snake + Reinforcement Learning (Double Deep Q Learning).

Basé sur serpent-algo.py (le jeu original n'est PAS modifié).
La grille et le score du jeu sont repris à l'identique.

Organisation :
    1. Game  -> Snake, Apple, SnakeGameAI (play_step)
    2. Model -> Linear_QNet, QTrainer (target network + Double DQN)
    3. Agent -> Agent (get_state, get_move, mémoire de replay)
    + train() / evaluate() / play()

Lancement (depuis le dossier Pomme) :
    python snake-ia.py            -> démonstration avec le meilleur modèle (model/best_eval.pth)
    python snake-ia.py --play     -> idem
    python snake-ia.py --train    -> entraînement (reprend model/latest.pth s'il existe)
    python snake-ia.py --new      -> nouvel entraînement depuis zéro
    python snake-ia.py --headless -> entraînement sans fenêtre ni clock (rapide)
"""

import argparse
import os
import random
import shutil
import time
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import matplotlib
    matplotlib.use("Agg")  # aucune fenêtre : la courbe est seulement enregistrée en PNG
    import matplotlib.pyplot as plt
except ImportError:  # la courbe est optionnelle
    plt = None


# ==================================================
# CONSTANTES DU JEU (identiques à serpent-algo.py)
# ==================================================

GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 5  # ATTENTION : serpent-algo.py utilise 5 (clock imposée)

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


# ==================================================
# HYPERPARAMÈTRES DU REINFORCEMENT LEARNING
# ==================================================

# Les 4 sorties du réseau correspondent à ces 4 directions, dans cet ordre
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Réseau : 11 -> HIDDEN_1 -> HIDDEN_2 -> 4
HIDDEN_1 = 256
HIDDEN_2 = 128

LEARNING_RATE = 0.001  # Adam, valeur standard
GAMMA = 0.9            # récompenses futures (voir compte rendu pour le choix)
MAX_MEMORY = 100_000   # mémoire de replay (les plus anciennes expériences sont oubliées)
BATCH_SIZE = 256       # mini-batch tiré au hasard à CHAQUE pas
WARMUP_STEPS = 1000    # pas d'entraînement par batch tant que la mémoire est plus petite

# Epsilon-greedy : epsilon = max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** parties)
# Avec 0.97 : ~0.74 à la partie 10, ~0.22 à la partie 50, ~0.05 à la partie 100, 0.01 ensuite
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.97

TARGET_UPDATE = 500    # copie policy_net -> target_net tous les 500 pas
GRAD_CLIP = 10.0       # norme maximale du gradient

# Statistiques, sauvegardes et évaluation
MEAN_WINDOW = 100      # moyenne glissante sur 100 parties
EVAL_EVERY = 100       # évaluation sans exploration toutes les 100 parties
EVAL_GAMES = 10        # nombre de parties par évaluation
SAVE_PLOT_SECONDS = 120  # latest.pth + courbe PNG toutes les 2 minutes (pas à chaque partie : c'est lent)
PRINT_EVERY = 10         # bloc de statistiques détaillé toutes les 10 parties

# Rewards du cours (différents du score du jeu !) - NE PAS MODIFIER
# Le cours donne "déplacement : +0.1" et "autres actions : 0".
# Choix : chaque pas est un déplacement, donc un pas normal (ni pomme, ni mort)
# rapporte +0.1. Remarque : ce bonus récompense aussi le fait de survivre sans
# manger ; c'est pour ça qu'il faut la limite TIMEOUT_FACTOR ci-dessous.
REWARD_APPLE = 10
REWARD_GAME_OVER = -10
REWARD_MOVE = 0.1
REWARD_VICTORY = 100

# Sécurité d'entraînement : si l'IA ne mange pas pendant
# TIMEOUT_FACTOR * longueur du serpent pas, la partie est perdue (-10).
TIMEOUT_FACTOR = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def model_path(name):
    return os.path.join(MODEL_DIR, name)


# Demi-tour interdit. Entrées "direction" du state = [LEFT, RIGHT, UP, DOWN]
# LEFT -> RIGHT (index 3), RIGHT -> LEFT (2), UP -> DOWN (1), DOWN -> UP (0)
FORBIDDEN_FROM_DIRECTION = [3, 2, 1, 0]
FORBIDDEN_TENSOR = torch.tensor(FORBIDDEN_FROM_DIRECTION)


def forbidden_action(state):
    """Index (dans ACTIONS) du demi-tour interdit pour ce state."""
    return FORBIDDEN_FROM_DIRECTION[int(np.argmax(state[3:7]))]


def wrapped_delta(target, source, size):
    """Écart le plus court de source vers target sur un axe circulaire (murs traversables).
    Ex. GRID_SIZE = 15 : de 0 vers 14 -> -1 (une case à gauche), pas +14."""
    delta = target - source
    if delta > size // 2:
        delta -= size
    elif delta < -(size // 2):
        delta += size
    return delta


# Version de la signification des 11 entrées, enregistrée dans les checkpoints.
# 1 : pomme comparée sans wrap-around ; 2 : pomme comparée avec wrap-around.
STATE_VERSION = 2


# ==================================================
# 1. GAME
# ==================================================

class Snake:
    """Serpent repris de serpent-algo.py (seul move() change, voir commentaire)."""

    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos,
                     [self.head_pos[0] - 1, self.head_pos[1]],
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse immédiat."""
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case dans la direction actuelle."""
        # Comme dans serpent-algo.py : "% GRID_SIZE" fait traverser les murs,
        # le serpent réapparaît de l'autre côté de la grille.
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
        """Prépare le serpent à grandir au prochain mouvement (score +1, inchangé)."""
        self.grow_pending = True
        self.score += 1

    def check_wall_collision(self):
        """Vérifie si la tête sort de la grille.
        Avec le modulo de move(), c'est toujours False : les murs ne tuent pas."""
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps."""
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
    """Pomme reprise telle quelle de serpent-algo.py."""

    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]

        if not available_positions:
            return None  # Toutes les cases sont pleines (Victoire)

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
    """Affiche le score, le temps et le remplissage (repris de serpent-algo.py)."""
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


class SnakeGameAI:
    """Version du jeu pilotée par l'agent au lieu du clavier."""

    def __init__(self, headless=False):
        # headless : aucune fenêtre, aucun dessin et pas de clock -> entraînement rapide.
        # Les règles du jeu (grille, collisions, score) restent exactement les mêmes.
        self.headless = headless
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake IA - Deep Q Learning")
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 40)
        self.font_small = pygame.font.Font(None, 24)

        # Informations affichées sous le score
        self.n_games = 0
        self.record = 0
        self.mode_text = "Entraînement"

        self.reset()

    def reset(self):
        """Nouvelle partie : serpent, pomme et compteurs remis à zéro."""
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.frame_iteration = 0  # pas joués depuis la dernière pomme
        self.victory = False
        self.start_time = time.time()

    def is_collision(self, point):
        """True si la tête du serpent mourrait en arrivant sur 'point'."""
        # Les murs se traversent : la case réelle est de l'autre côté de la grille.
        # Seul le corps du serpent est donc un danger.
        x, y = point[0] % GRID_SIZE, point[1] % GRID_SIZE
        # Au prochain pas, la queue libère sa case (sauf si le serpent grandit)
        body = self.snake.body if self.snake.grow_pending else self.snake.body[:-1]
        return [x, y] in body

    def play_step(self, action):
        """Joue un pas avec l'action de l'agent. Retourne (reward, game_over, score)."""
        self.frame_iteration += 1

        # La fenêtre peut être fermée pour arrêter l'entraînement
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # 1. Action choisie : one-hot [UP, DOWN, LEFT, RIGHT] -> direction
        new_dir = ACTIONS[int(np.argmax(action))]
        # L'agent masque le demi-tour : si on en reçoit un, c'est un bug.
        # (Sinon le jeu jouerait autre chose que l'action apprise.)
        if (-new_dir[0], -new_dir[1]) == self.snake.direction:
            raise ValueError("Demi-tour reçu : l'action doit être masquée par l'agent")
        self.snake.set_direction(new_dir)

        # 2. Déplacement
        self.snake.move()

        # 3. Collisions (murs, corps) ou trop longtemps sans manger
        if self.snake.is_game_over() or self.frame_iteration > TIMEOUT_FACTOR * len(self.snake.body):
            return REWARD_GAME_OVER, True, self.snake.score

        # 4. Pomme mangée ?
        reward = REWARD_MOVE
        game_over = False
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()  # score du jeu : +1, comme l'original
            reward = REWARD_APPLE
            self.frame_iteration = 0
            if not self.apple.relocate(self.snake.body):
                # Plus aucune case libre : le Snake est terminé
                self.victory = True
                game_over = True
                reward = REWARD_VICTORY

        # 5. Affichage + clock
        if not self.headless:
            self._update_ui()
            self.clock.tick(GAME_SPEED)

        return reward, game_over, self.snake.score

    def _update_ui(self):
        self.screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.screen, NOIR, game_area_rect)

        draw_grid(self.screen)
        self.apple.draw(self.screen)
        self.snake.draw(self.screen)
        display_info(self.screen, self.font_main, self.snake, self.start_time)

        ia_text = self.font_small.render(
            f"{self.mode_text}   Partie: {self.n_games + 1}   Record: {self.record}", True, BLANC)
        self.screen.blit(ia_text, (SCREEN_WIDTH // 2 - ia_text.get_width() // 2, 52))

        pygame.display.flip()


# ==================================================
# 2. MODEL
# ==================================================

class Linear_QNet(nn.Module):
    """Réseau 11 -> 256 -> 128 -> 4 : une Q-value par direction."""

    def __init__(self, input_size, hidden_1, hidden_2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_1)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.linear3 = nn.Linear(hidden_2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class QTrainer:
    """Entraîne policy_net avec Double DQN et un target network."""

    def __init__(self, policy_net, target_net, lr, gamma):
        self.policy_net = policy_net
        self.target_net = target_net
        self.gamma = gamma
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        # Huber : quadratique pour les petites erreurs, linéaire pour les grandes.
        # Une erreur de 100 (victoire) ou de 20 ne produit pas un gradient énorme.
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.float)

        # Une seule expérience (entraînement court) -> on ajoute la dimension batch
        if state.dim() == 1:
            state = state.unsqueeze(0)            # (1, 11)
            next_state = next_state.unsqueeze(0)  # (1, 11)
            action = action.unsqueeze(0)          # (1, 4)
            reward = reward.unsqueeze(0)          # (1,)
            done = done.unsqueeze(0)              # (1,)

        batch = torch.arange(state.shape[0])
        action_index = action.argmax(dim=1)       # (N,) action réellement jouée

        # Q(s, a) prédit par policy_net pour l'action jouée : (N,)
        q_pred = self.policy_net(state)[batch, action_index]

        with torch.no_grad():
            # Double DQN, étape 1 : policy_net CHOISIT la meilleure action future
            # (le demi-tour, impossible dans l'état suivant, est masqué)
            next_q_policy = self.policy_net(next_state)
            forbidden = FORBIDDEN_TENSOR[next_state[:, 3:7].argmax(dim=1)]
            next_q_policy[batch, forbidden] = -float("inf")
            best_next_action = next_q_policy.argmax(dim=1)

            # Double DQN, étape 2 : target_net ÉVALUE cette action
            next_q = self.target_net(next_state)[batch, best_next_action]

            # Bellman : Q = reward + gamma * Q(futur), sans futur si la partie est finie
            q_target = reward + self.gamma * next_q * (1 - done)

        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()
        return loss.item()


# ==================================================
# 3. AGENT
# ==================================================

class Agent:

    def __init__(self):
        self.n_games = 0
        self.steps = 0
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MAX_MEMORY)

        # policy_net apprend en permanence ; target_net est une copie figée,
        # mise à jour tous les TARGET_UPDATE pas, qui sert à calculer la cible.
        self.policy_net = Linear_QNet(11, HIDDEN_1, HIDDEN_2, 4)
        self.target_net = Linear_QNet(11, HIDDEN_1, HIDDEN_2, 4)
        self.update_target()
        for param in self.target_net.parameters():
            param.requires_grad_(False)  # le target_net n'est jamais entraîné directement

        self.trainer = QTrainer(self.policy_net, self.target_net, LEARNING_RATE, GAMMA)

    def get_state(self, game):
        """Les 11 entrées du réseau (0 ou 1)."""
        snake = game.snake
        head_x, head_y = snake.head_pos
        dir_x, dir_y = snake.direction

        # Directions relatives au serpent. Pygame : x vers la droite, y vers le BAS.
        # Tourner à droite = rotation horaire à l'écran : (dx, dy) -> (-dy, dx)
        #   RIGHT (1,0) -> DOWN (0,1)    DOWN (0,1) -> LEFT (-1,0)
        #   LEFT (-1,0) -> UP (0,-1)     UP (0,-1)  -> RIGHT (1,0)
        # Tourner à gauche = l'opposé : (dx, dy) -> (dy, -dx)
        front = (dir_x, dir_y)
        right = (-dir_y, dir_x)
        left = (dir_y, -dir_x)

        apple_x, apple_y = game.apple.position
        # Direction la plus courte vers la pomme en tenant compte de la traversée des murs
        apple_dx = wrapped_delta(apple_x, head_x, GRID_SIZE)
        apple_dy = wrapped_delta(apple_y, head_y, GRID_SIZE)

        state = [
            # Dangers
            game.is_collision((head_x + front[0], head_y + front[1])),
            game.is_collision((head_x + right[0], head_y + right[1])),
            game.is_collision((head_x + left[0], head_y + left[1])),

            # Direction actuelle
            snake.direction == LEFT,
            snake.direction == RIGHT,
            snake.direction == UP,
            snake.direction == DOWN,

            # Position de la pomme par rapport à la tête
            apple_dx < 0,  # pomme à gauche
            apple_dx > 0,  # pomme à droite
            apple_dy < 0,  # pomme en haut
            apple_dy > 0,  # pomme en bas
        ]
        return np.array(state, dtype=int)

    def update_epsilon(self):
        """Décroissance exponentielle bornée, une fois par partie."""
        self.epsilon = max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** self.n_games)

    def get_move(self, state, explore=True):
        """Epsilon-greedy avec masque du demi-tour. explore=False -> epsilon = 0."""
        forbidden = forbidden_action(state)

        if explore and random.random() < self.epsilon:
            # Exploration : uniquement parmi les 3 actions possibles
            move = random.choice([i for i in range(4) if i != forbidden])
        else:
            # Exploitation : model.predict, le demi-tour ne peut pas être choisi
            with torch.no_grad():
                prediction = self.policy_net(torch.tensor(state, dtype=torch.float))  # (4,)
            prediction[forbidden] = -float("inf")
            move = torch.argmax(prediction).item()

        final_move = [0, 0, 0, 0]
        final_move[move] = 1
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """Entraînement immédiat sur la dernière transition."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """Entraînement sur un mini-batch aléatoire de la mémoire (après le warmup)."""
        if len(self.memory) < WARMUP_STEPS:
            return  # warmup : on remplit d'abord la mémoire
        mini_sample = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def play_one_step(agent, game):
    """Un pas d'entraînement complet. Retourne (done, score)."""
    state_old = agent.get_state(game)
    final_move = agent.get_move(state_old)
    reward, done, score = game.play_step(final_move)
    state_new = agent.get_state(game)

    agent.train_short_memory(state_old, final_move, reward, state_new, done)
    agent.remember(state_old, final_move, reward, state_new, done)  # action réellement jouée

    # Le jeu est limité par la clock : entraîner un mini-batch à chaque pas
    # ne ralentit pas le jeu et exploite beaucoup mieux chaque expérience.
    agent.train_long_memory()

    agent.steps += 1
    if agent.steps % TARGET_UPDATE == 0:
        agent.update_target()
    return done, score


# ==================================================
# SAUVEGARDE / CHARGEMENT
# ==================================================

def new_stats():
    return {
        "scores": [],          # score de chaque partie
        "mean_scores": [],     # moyenne glissante sur MEAN_WINDOW parties
        "total_score": 0,
        "record": 0,
        "best_mean": 0.0,
        "best_eval": 0.0,
        "victories": 0,
        "training_time": 0.0,  # secondes (évaluations comprises)
        "eval_history": [],    # [partie, pas, temps, moyenne, max]
    }


def save_checkpoint(name, agent, stats, with_memory=False):
    checkpoint = {
        "policy_state": agent.policy_net.state_dict(),
        "target_state": agent.target_net.state_dict(),
        "optimizer_state": agent.trainer.optimizer.state_dict(),
        "n_games": agent.n_games,
        "steps": agent.steps,
        "epsilon": agent.epsilon,
        "stats": stats,
        "state_version": STATE_VERSION,
    }
    if with_memory and agent.memory:
        # Mémoire compacte (uint8) : ~3 Mo pour 100 000 expériences
        states, actions, rewards, next_states, dones = zip(*agent.memory)
        checkpoint["memory"] = {
            "states": torch.tensor(np.array(states), dtype=torch.uint8),
            "actions": torch.tensor(np.array(actions), dtype=torch.uint8),
            "rewards": torch.tensor(rewards, dtype=torch.float),
            "next_states": torch.tensor(np.array(next_states), dtype=torch.uint8),
            "dones": torch.tensor(dones, dtype=torch.bool),
        }
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Écriture dans un fichier temporaire puis remplacement : un Ctrl+C pendant
    # la sauvegarde ne peut pas corrompre l'ancien fichier.
    tmp_path = model_path(name + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, model_path(name))


def load_checkpoint(name, agent):
    checkpoint = torch.load(model_path(name), map_location="cpu")
    agent.policy_net.load_state_dict(checkpoint["policy_state"])
    agent.target_net.load_state_dict(checkpoint["target_state"])
    agent.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
    agent.n_games = checkpoint["n_games"]
    agent.steps = checkpoint["steps"]
    agent.epsilon = checkpoint["epsilon"]

    memory = checkpoint.get("memory")
    if memory is not None:
        for i in range(len(memory["rewards"])):
            agent.memory.append((memory["states"][i].numpy().astype(int),
                                 memory["actions"][i].tolist(),
                                 memory["rewards"][i].item(),
                                 memory["next_states"][i].numpy().astype(int),
                                 bool(memory["dones"][i])))
    return checkpoint["stats"]


def backup_previous_run():
    """--new : l'ancien entraînement est déplacé dans model/backup_<date>/ (rien n'est supprimé)."""
    names = ["latest.pth", "best_record.pth", "best_mean.pth", "best_eval.pth", "stats.csv", "eval.csv"]
    existing = [n for n in names if os.path.exists(model_path(n))]
    if not existing:
        return
    backup_dir = model_path("backup_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(backup_dir)
    for n in existing:
        shutil.move(model_path(n), os.path.join(backup_dir, n))
    print(f"Ancien entraînement déplacé dans {backup_dir}")


# ==================================================
# STATISTIQUES
# ==================================================

def format_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def plot(stats):
    """Courbe enregistrée dans Pomme/training_progress.png (aucune fenêtre bloquante)."""
    if plt is None or not stats["scores"]:
        return
    games = range(1, len(stats["scores"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(games, stats["scores"], color="#2a78d6", alpha=0.35, linewidth=1, label="Score par partie")
    ax.plot(games, stats["mean_scores"], color="#eb6834", linewidth=2,
            label=f"Moyenne glissante ({MEAN_WINDOW} parties)")
    if stats["eval_history"]:
        eval_games = [e[0] for e in stats["eval_history"]]
        eval_means = [e[3] for e in stats["eval_history"]]
        ax.plot(eval_games, eval_means, color="#1baf7a", linewidth=2, marker="o", markersize=6,
                label=f"Évaluation sans exploration ({EVAL_GAMES} parties)")
    ax.set_title("Entraînement du Snake IA (Double DQN)")
    ax.set_xlabel("Nombre de parties")
    ax.set_ylabel("Score")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "training_progress.png"), dpi=100)
    plt.close(fig)


def print_stats(agent, stats):
    minutes = stats["training_time"] / 60
    print("-" * 44)
    print(f"Game {agent.n_games}")
    print(f"Score : {stats['scores'][-1]} | Record : {stats['record']}")
    print(f"Mean {MEAN_WINDOW} : {stats['mean_scores'][-1]:.2f} (meilleure : {stats['best_mean']:.2f})"
          f" | Mean totale : {stats['total_score'] / agent.n_games:.2f}")
    print(f"Epsilon : {agent.epsilon:.3f} | Steps : {agent.steps}")
    print(f"Training time : {format_time(stats['training_time'])}"
          f" | Pommes / minute : {stats['total_score'] / max(minutes, 1e-9):.2f}")
    print(f"Meilleure évaluation (epsilon = 0) : {stats['best_eval']:.2f}")
    print("-" * 44)


# ==================================================
# ÉVALUATION, ENTRAÎNEMENT, DÉMONSTRATION
# ==================================================

def evaluate(agent, game, n_games=EVAL_GAMES):
    """Parties avec epsilon = 0, sans mémoire et sans entraînement."""
    game.mode_text = "EVALUATION"
    scores = []
    for _ in range(n_games):
        game.reset()
        done = False
        while not done:
            state = agent.get_state(game)
            move = agent.get_move(state, explore=False)  # pas d'exploration
            _, done, score = game.play_step(move)        # ni remember, ni train
        scores.append(score)
    game.reset()
    game.mode_text = "Entraînement"
    return sum(scores) / len(scores), max(scores)


def train(new=False, headless=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    agent = Agent()
    stats = new_stats()

    if new:
        backup_previous_run()
    elif os.path.exists(model_path("latest.pth")):
        old_version = torch.load(model_path("latest.pth"), map_location="cpu").get("state_version", 1)
        if old_version != STATE_VERSION:
            # Ne pas mélanger un réseau entraîné avec l'ancien sens des entrées "pomme"
            print("model/latest.pth a été entraîné avec l'ancien état (pomme sans wrap-around).")
            print("Reprise refusée : lance python snake-ia.py --new (les anciens fichiers seront archivés, pas supprimés).")
            return
        stats = load_checkpoint("latest.pth", agent)
        print(f"Reprise de model/latest.pth : partie {agent.n_games}, {agent.steps} pas, "
              f"record {stats['record']}, mémoire {len(agent.memory)}")
    else:
        print("Aucun checkpoint : nouvel entraînement")

    game = SnakeGameAI(headless=headless)
    game.n_games = agent.n_games
    game.record = stats["record"]

    stats_path = model_path("stats.csv")
    if not os.path.exists(stats_path):
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("partie,score,record,mean100,epsilon,steps,temps_s,victoire\n")
    eval_path = model_path("eval.csv")
    if not os.path.exists(eval_path):
        with open(eval_path, "w", encoding="utf-8") as f:
            f.write("partie,steps,temps_s,eval_mean,eval_max\n")

    agent.update_epsilon()
    time_offset = stats["training_time"]
    session_start = time.perf_counter()
    last_save = session_start
    print("Entraînement lancé (fermer la fenêtre ou Ctrl+C pour arrêter et sauvegarder)")

    try:
        while True:
            done, score = play_one_step(agent, game)
            if not done:
                continue

            # ----- Fin de partie -----
            victory = game.victory
            game.reset()
            agent.n_games += 1
            agent.update_epsilon()
            agent.train_long_memory()
            stats["training_time"] = time_offset + time.perf_counter() - session_start

            scores = stats["scores"]
            scores.append(score)
            stats["total_score"] += score
            stats["victories"] += int(victory)
            mean_recent = sum(scores[-MEAN_WINDOW:]) / len(scores[-MEAN_WINDOW:])
            stats["mean_scores"].append(mean_recent)

            if score > stats["record"]:
                stats["record"] = score
                save_checkpoint("best_record.pth", agent, stats)
            # Moyenne sur 100 parties complètes seulement (pas une chance au début)
            if len(scores) >= MEAN_WINDOW and mean_recent > stats["best_mean"]:
                stats["best_mean"] = mean_recent
                save_checkpoint("best_mean.pth", agent, stats)

            game.n_games = agent.n_games
            game.record = stats["record"]

            # Sans interface, afficher chaque partie ralentirait le terminal :
            # seul le bloc détaillé (toutes les PRINT_EVERY parties) est affiché.
            if not headless:
                print(f"Game {agent.n_games} | Score: {score} | Record: {stats['record']} | "
                      f"Mean {MEAN_WINDOW}: {mean_recent:.2f} | Epsilon: {agent.epsilon:.3f}"
                      + (" | VICTOIRE !" if victory else ""))
            with open(stats_path, "a", encoding="utf-8") as f:
                f.write(f"{agent.n_games},{score},{stats['record']},{mean_recent:.3f},{agent.epsilon:.4f},"
                        f"{agent.steps},{stats['training_time']:.1f},{int(victory)}\n")

            if agent.n_games % PRINT_EVERY == 0:
                print_stats(agent, stats)
            # Sauvegarde périodique sans la mémoire (rapide) ; la mémoire complète
            # n'est enregistrée qu'à l'arrêt, car la convertir prend du temps.
            if time.perf_counter() - last_save >= SAVE_PLOT_SECONDS:
                save_checkpoint("latest.pth", agent, stats)
                plot(stats)
                last_save = time.perf_counter()

            if agent.n_games % EVAL_EVERY == 0:
                eval_mean, eval_max = evaluate(agent, game)
                stats["training_time"] = time_offset + time.perf_counter() - session_start
                stats["eval_history"].append([agent.n_games, agent.steps, stats["training_time"], eval_mean, eval_max])
                print(f"*** Évaluation (epsilon = 0, {EVAL_GAMES} parties) : "
                      f"moyenne {eval_mean:.2f} | max {eval_max} ***")
                with open(eval_path, "a", encoding="utf-8") as f:
                    f.write(f"{agent.n_games},{agent.steps},{stats['training_time']:.1f},{eval_mean:.2f},{eval_max}\n")
                if eval_mean > stats["best_eval"]:
                    stats["best_eval"] = eval_mean
                    save_checkpoint("best_eval.pth", agent, stats)
                    print("*** Nouveau meilleur modèle : model/best_eval.pth ***")

    except (KeyboardInterrupt, SystemExit):
        stats["training_time"] = time_offset + time.perf_counter() - session_start
        print("\nArrêt demandé : sauvegarde de model/latest.pth...")
        save_checkpoint("latest.pth", agent, stats, with_memory=True)
        plot(stats)

        n = max(agent.n_games, 1)
        minutes = max(stats["training_time"] / 60, 1e-9)
        print("\n===== Fin de l'entraînement =====")
        print(f"Parties jouées          : {agent.n_games}")
        print(f"Pas joués (steps)       : {agent.steps}")
        print(f"Temps d'entraînement    : {format_time(stats['training_time'])}")
        print(f"Meilleur score          : {stats['record']}")
        print(f"Score moyen (total)     : {stats['total_score'] / n:.2f}")
        if stats["mean_scores"]:
            print(f"Moyenne {MEAN_WINDOW} dernières   : {stats['mean_scores'][-1]:.2f}")
        print(f"Meilleure moyenne {MEAN_WINDOW}   : {stats['best_mean']:.2f}")
        print(f"Meilleure évaluation    : {stats['best_eval']:.2f}")
        print(f"Pommes / minute         : {stats['total_score'] / minutes:.2f}")
        print(f"Snake terminé           : {stats['victories']} fois")


def play():
    """Démonstration : meilleur modèle, epsilon = 0, aucun entraînement."""
    for name in ["best_eval.pth", "best_mean.pth", "latest.pth"]:
        if os.path.exists(model_path(name)):
            break
    else:
        print("Aucun modèle dans Pomme/model/ : lance d'abord python snake-ia.py --train (ou --headless)")
        return
    if name != "best_eval.pth":
        print(f"best_eval.pth introuvable, utilisation de {name}")

    agent = Agent()
    checkpoint = torch.load(model_path(name), map_location="cpu")
    agent.policy_net.load_state_dict(checkpoint["policy_state"])
    agent.policy_net.eval()
    print(f"Modèle chargé : model/{name} (entraîné sur {checkpoint['n_games']} parties)")
    if checkpoint.get("state_version", 1) != STATE_VERSION:
        print("ATTENTION : ce modèle a été entraîné avec l'ancien état (pomme sans wrap-around) ;"
              " il peut mal jouer. Relance un entraînement avec --new.")

    game = SnakeGameAI()
    game.mode_text = "DEMO"
    scores = []
    try:
        while True:
            state = agent.get_state(game)
            _, done, score = game.play_step(agent.get_move(state, explore=False))
            if done:
                scores.append(score)
                game.reset()
                game.n_games = len(scores)
                game.record = max(scores)
                print(f"Game {len(scores)} | Score: {score} | Record: {max(scores)} | "
                      f"Moyenne: {sum(scores) / len(scores):.2f}")
    except (KeyboardInterrupt, SystemExit):
        if scores:
            print(f"\n{len(scores)} parties | moyenne {sum(scores) / len(scores):.2f} | max {max(scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake + Double Deep Q Learning")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="entraîner (reprend latest.pth)")
    mode.add_argument("--play", action="store_true", help="faire jouer le meilleur modèle (mode par défaut)")
    parser.add_argument("--new", action="store_true", help="entraîner depuis zéro (l'ancien entraînement est archivé)")
    parser.add_argument("--headless", action="store_true",
                        help="entraîner sans fenêtre ni clock (rapide, Ctrl+C pour arrêter)")
    args = parser.parse_args()

    # --new et --headless concernent l'entraînement : ils l'activent automatiquement
    training = args.train or args.new or args.headless
    if args.play and training:
        parser.error("--play ne peut pas être combiné avec --new ou --headless")

    if training:
        train(new=args.new, headless=args.headless)
    else:
        play()  # sans argument : démonstration avec le meilleur modèle
