"""
snake-ia.py — Agent Reinforcement Learning (DQN) pour le jeu Serpent.

==================================================================
ARCHITECTURE
==================================================================

    Snake Game (SnakeGameAI, headless ou avec fenêtre pygame)
        -> State (11 features booléennes, cf. get_state)
        -> DQN Agent (petit réseau de neurones : 11 -> 256 -> 3)
        -> Action (tout droit / droite / gauche, relative à la tête)
        -> reward (officiel : +10 pomme / -10 mort / +100 victoire / 0 déplacement)
        -> Replay Memory (deque, tirage aléatoire de batches)
        -> DQN Training (Q-learning avec réseau cible)

Pendant l'entraînement seulement, un expert A* peut proposer l'action à la
place du DQN, avec une probabilité qui décroît linéairement de 1.0 à 0.0.
Le DQN apprend quand même de CES transitions (reward, next_state) exactement
comme des siennes : c'est ce qui lui permet d'apprendre plus vite au début.
Cet expert est totalement désactivé à l'évaluation (use_astar_expert=False).

Ce fichier est volontairement un seul fichier (le jeu de base + le DQN +
l'expert A* + l'entraînement + l'évaluation), pour pouvoir être lancé
simplement avec :

    python snake-ia.py

Voir le README.md du dossier pour toutes les commandes disponibles.
"""

import argparse
import csv
import os
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass, asdict

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# Utilise le GPU si disponible (bien plus rapide pour lancer beaucoup
# d'entraînements en parallèle avec compare()), sinon retombe sur le CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================================
# 1. CONSTANTES DE JEU (reprises telles quelles de serpent-algo.py)
# ==================================================================
# Règle du cours : on ne touche ni à la taille de la grille, ni à l'horloge,
# ni au scoring officiel (score = nombre de pommes mangées).

GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 5  # images par seconde (= déplacements par seconde, cf. move_counter)

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
# Ordre "trigonométrique" utilisé pour tourner à droite/gauche par rapport
# à la direction actuelle (cf. action relative dans SnakeGameAI.play_step).
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]

# ==================================================================
# 2. RÉCOMPENSES OFFICIELLES (cf. PDF du cours, à ne pas réinventer)
# ==================================================================
REWARD_APPLE = 10
REWARD_DEATH = -10
REWARD_WIN = 100
REWARD_MOVE = 0  # "Autres actions : 0" — un reward de mouvement positif
                 # inciterait le serpent à tourner en rond indéfiniment
                 # plutôt qu'à manger, donc on retient 0 et non 0.1.

# Terme de shaping (EN PLUS des rewards officiels ci-dessus, qui restent
# inchangés) : à chaque pas, on ajoute un petit bonus proportionnel au
# score courant. Il pousse le DQN à valoriser le fait d'être déjà à un
# score élevé (donc à y arriver plus vite/plus souvent), sans dominer les
# rewards officiels (+10/-10/+100) tant que le coefficient reste petit —
# c'est un réglage explicitement demandé pour pousser la métrique
# score/temps, pas une réécriture du barème officiel.
SCORE_REWARD_COEFF = 0.01

# Ajout technique nécessaire (absent du jeu original) : sans limite, un
# serpent qui n'a rien à apprendre peut tourner en rond à l'infini pendant
# l'entraînement. On considère la partie perdue si aucune pomme n'est
# mangée depuis trop longtemps. Cela ne change ni le score, ni les règles
# de victoire/défaite officielles (mur/morsure), seulement une sécurité.
#
# Ce facteur influe directement sur la métrique principale (score/temps) :
# une partie qui erre longtemps sans manger fait grossir le temps sans
# faire grossir le score, donc écrase le ratio. Le laisser trop grand
# revient à "trop attendre avant de mourir" quand il n'y a plus de progrès
# possible ; mieux vaut couper court plus tôt (30 au lieu de 100) pour que
# le ratio reflète des parties efficaces plutôt que de la survie inutile.
STALL_LIMIT_FACTOR = 30

# Score maximum atteignable (plateau rempli = victoire) : GRID_SIZE² cellules
# moins la longueur initiale du corps (3). Sert de référence pour le malus
# de temps "non compétitif" ci-dessous (cf. train()).
MAX_SCORE = GRID_SIZE * GRID_SIZE - 3


# ==================================================================
# 3. HYPERPARAMÈTRES (tout est configurable, rien n'est codé en dur
#    ailleurs dans le fichier)
# ==================================================================
@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.9                     # facteur d'actualisation
    batch_size: int = 256
    replay_memory_size: int = 100_000
    hidden_size: int = 256                 # petit réseau -> rapide sur CPU

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.98            # multiplicatif, par épisode

    target_update_frequency: int = 500     # en nombre de steps d'entraînement

    number_of_episodes: int = 400

    # Probabilité d'utiliser l'expert A* plutôt que le DQN pendant l'entraînement.
    expert_probability_start: float = 1.0
    expert_probability_end: float = 0.0
    expert_probability_decay_episodes: int = 150  # décroissance linéaire sur N épisodes

    stall_limit_factor: int = STALL_LIMIT_FACTOR
    score_reward_coeff: float = SCORE_REWARD_COEFF
    seed: int = 0


# ==================================================================
# 4. JEU : classes reprises de serpent-algo.py (mécaniques inchangées)
# ==================================================================
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
        """Change la direction, empêchant le mouvement inverse immédiat."""
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case. Les bords sont enroulés (torus) :
        c'est le comportement original du jeu (% GRID_SIZE), volontaire et
        conservé tel quel — seule la morsure du corps termine la partie."""
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
        """Ne déclenche jamais (bords enroulés) : conservé pour fidélité au jeu original."""
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


# ==================================================================
# 5. ENVIRONNEMENT RL : enrobe Snake/Apple pour exposer play_step(action)
# ==================================================================
class SnakeGameAI:
    """Environnement RL au-dessus du jeu de base.

    - render=False (par défaut) : aucune fenêtre pygame, boucle la plus
      rapide possible -> utilisé pour l'entraînement (des milliers de
      parties) et pour l'évaluation "rapide".
    - render=True : ouvre une fenêtre et respecte l'horloge officielle du
      jeu (clock.tick(GAME_SPEED)) -> utilisé pour regarder l'agent jouer
      et pour l'évaluation "temps réel" (le vrai chronomètre du jeu).
    """

    def __init__(self, render=False, stall_limit_factor=STALL_LIMIT_FACTOR, score_reward_coeff=SCORE_REWARD_COEFF):
        self.render_enabled = render
        self.stall_limit_factor = stall_limit_factor
        self.score_reward_coeff = score_reward_coeff
        # Seuil de temps "compétitif" (cf. train()) : au-delà, même finir la
        # partie ne donnerait plus un bon ratio score/temps -> malus. None =
        # pas encore de référence (tout début d'entraînement), donc pas de
        # malus. Recalculé et resserré par train() à chaque épisode, à
        # mesure que le modèle s'améliore (cf. best_recent_avg).
        self.competitive_time_threshold = None
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake IA - DQN")
            self.clock = pygame.time.Clock()
            self.font_main = pygame.font.Font(None, 40)
            self.font_game_over = pygame.font.Font(None, 80)
        # quit_requested (fermeture de la fenêtre par l'humain) ne doit PAS
        # être remis à False par reset() : une fois la fenêtre fermée, on ne
        # doit plus relancer de nouvelle partie dedans.
        self.quit_requested = False
        self.reset()

    def reset(self):
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.steps = 0
        self.steps_since_apple = 0
        self.start_time = time.time()
        self.game_over = False
        self.victory = False
        self.time_malus_applied = False
        return self

    def _handle_events(self):
        """Vide la file d'événements pygame pour éviter que la fenêtre freeze
        (nécessaire même sans clavier, sur certains systèmes), et permet à un
        testeur humain de fermer la fenêtre proprement (croix ou Echap)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.game_over = True
                self.quit_requested = True

    def pause_on_game_over(self, seconds=1.5):
        """Laisse l'écran 'GAME OVER'/'VICTOIRE' affiché quelques secondes
        (au lieu de disparaître au prochain tick) pour qu'un testeur humain
        ait le temps de lire le score avant la partie suivante. Continue de
        traiter les événements pendant la pause pour rester réactif à une
        fermeture de fenêtre. Retourne True si la fenêtre a été fermée."""
        end_time = time.time() + seconds
        while time.time() < end_time:
            self._handle_events()
            if self.quit_requested:
                return True
            self._render()
        return False

    def _render(self):
        self.screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.screen, NOIR, game_area_rect)
        draw_grid(self.screen)
        self.apple.draw(self.screen)
        self.snake.draw(self.screen)
        display_info(self.screen, self.font_main, self.snake, self.start_time)
        if self.game_over:
            if self.victory:
                display_message(self.screen, self.font_game_over, "VICTOIRE !", VERT)
            else:
                display_message(self.screen, self.font_game_over, "GAME OVER", ROUGE)
        pygame.display.flip()
        self.clock.tick(GAME_SPEED)

    def play_step(self, action):
        """Exécute une action relative (cf. relative_to_absolute) et retourne
        (reward, done, score) en suivant exactement l'ordre du jeu original :
        set_direction -> move -> collision -> pomme -> grow/relocate."""
        self.steps += 1
        self.steps_since_apple += 1

        if self.render_enabled:
            self._handle_events()

        new_direction = relative_to_absolute(self.snake.direction, action)
        self.snake.set_direction(new_direction)
        self.snake.move()

        reward = REWARD_MOVE
        done = False

        if self.snake.is_game_over():
            self.game_over = True
            done = True
            reward = REWARD_DEATH
        elif self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.steps_since_apple = 0
            reward = REWARD_APPLE
            if not self.apple.relocate(self.snake.body):
                self.victory = True
                self.game_over = True
                done = True
                reward = REWARD_WIN
        elif self.steps_since_apple > self.stall_limit_factor * len(self.snake.body):
            # Sécurité anti-boucle-infinie (ajout technique, cf. plus haut).
            # N'affecte pas les règles officielles de victoire/défaite.
            self.game_over = True
            done = True
            reward = REWARD_DEATH

        if self.render_enabled:
            self._render()

        # Shaping additionnel (cf. SCORE_REWARD_COEFF) : un petit bonus par
        # pas proportionnel au score courant, EN PLUS du reward officiel
        # ci-dessus (qui reste +10/-10/+100/0 tel quel). Ça encourage le DQN
        # à valoriser le fait d'être déjà à un score élevé, donc à y arriver
        # plus vite (pousse la métrique score/temps sans changer le score
        # officiel ni réécrire le barème officiel).
        reward += self.score_reward_coeff * self.snake.score

        # + le score FINAL de la partie ajouté une seule fois, au tout
        # dernier pas (done=True) : un épisode qui se termine à un score
        # élevé rapporte un gros crédit terminal en plus du -10/+100 déjà
        # présent, ce qui pousse le DQN à préférer les trajectoires qui
        # finissent haut, pas seulement celles qui grappillent des +10 au
        # fil de l'eau. Toujours un AJOUT, jamais un remplacement du reward
        # officiel.
        if done:
            reward += self.snake.score

        # Malus de temps "non compétitif" : au-delà de competitive_time_
        # threshold (cf. train(), recalculé et resserré à chaque épisode à
        # mesure que le modèle s'améliore), même finir la partie ne donnerait
        # plus un score/temps intéressant -> -20, une seule fois par partie.
        # Condition supplémentaire : uniquement si le score n'a même pas
        # encore atteint 10 (l'objectif minimum du projet). Une partie qui a
        # déjà dépassé 10 points est déjà "réussie" et n'est pas pénalisée
        # même si elle traîne en longueur ensuite.
        if (self.competitive_time_threshold is not None
                and not self.time_malus_applied
                and self.snake.score < 10
                and self.elapsed_time > self.competitive_time_threshold):
            reward -= 20
            self.time_malus_applied = True

        return reward, done, self.snake.score

    @property
    def elapsed_time(self):
        """Temps de jeu. En rendu réel : vrai chronomètre (time.time()),
        identique au jeu original. En mode rapide (headless) : steps / GAME_SPEED,
        car le jeu original avance d'une case par tick à GAME_SPEED ticks/seconde
        -> équivalent exact du temps réel, sans attendre l'horloge."""
        if self.render_enabled:
            return time.time() - self.start_time
        return self.steps / GAME_SPEED


def relative_to_absolute(current_direction, relative_action):
    """Convertit une action relative (0=tout droit, 1=droite, 2=gauche) en
    direction absolue (UP/DOWN/LEFT/RIGHT), à partir de la direction actuelle.

    On utilise des actions relatives plutôt que les 4 directions absolues
    car "faire demi-tour" est toujours suicidaire (cf. set_direction qui
    l'interdit déjà) : autant ne pas donner ce choix inutile au réseau, ce
    qui réduit l'espace d'actions de 4 à 3 et accélère l'apprentissage.
    """
    idx = CLOCKWISE.index(current_direction)
    if relative_action == 0:
        return CLOCKWISE[idx]
    elif relative_action == 1:  # tourner à droite
        return CLOCKWISE[(idx + 1) % 4]
    else:  # relative_action == 2, tourner à gauche
        return CLOCKWISE[(idx - 1) % 4]


# ==================================================================
# 6. REPRÉSENTATION D'ÉTAT (11 features booléennes, cf. PDF du cours)
# ==================================================================
def get_state(game):
    """Construit l'état à partir du jeu, sous forme de vecteur de 11 flottants.

    Choix de représentation (pourquoi ces 11 features et pas la grille 15x15
    entière, soit 225 cellules) :
    - Une grille complète (225 valeurs) demanderait un réseau bien plus gros
      pour capturer un pattern spatial, donc plus lent à entraîner sur CPU.
    - Ces 11 booléens (danger immédiat dans les 3 directions possibles,
      direction actuelle, direction de la pomme) donnent au DQN exactement
      ce qu'il faut pour décider du prochain mouvement sans se noyer dans
      des détails inutiles (le reste du plateau, loin de la tête, n'influe
      pas sur la décision immédiate). C'est la représentation utilisée par
      le cours (Loeber) et elle suffit largement pour dépasser 10 points.
    - Limite connue : sans vision globale du corps, le serpent peut finir
      par s'enfermer lui-même vers 30-60 points. Ce n'est pas un problème
      pour l'objectif du projet (>= 10 points).

    Les dangers sont calculés en tenant compte du fait que les bords sont
    enroulés (torus) : la case "dangereuse" est le corps du serpent, jamais
    le mur (qui n'existe pas dans ce jeu).
    """
    snake = game.snake
    head = snake.head_pos
    direction = snake.direction
    idx = CLOCKWISE.index(direction)
    dir_straight = CLOCKWISE[idx]
    dir_right = CLOCKWISE[(idx + 1) % 4]
    dir_left = CLOCKWISE[(idx - 1) % 4]

    def is_danger(move_dir):
        next_pos = [(head[0] + move_dir[0]) % GRID_SIZE, (head[1] + move_dir[1]) % GRID_SIZE]
        # Le dernier segment (queue) va disparaître au prochain mouvement
        # (sauf si le serpent grandit), donc il n'est pas un danger.
        body_without_tail = snake.body[:-1] if not snake.grow_pending else snake.body
        return next_pos in body_without_tail

    apple_x, apple_y = game.apple.position if game.apple.position else head

    def wrapped_delta(a, b):
        """Plus petit déplacement signé sur un axe torique (plateau enroulé)."""
        d = a - b
        if d > GRID_SIZE / 2:
            d -= GRID_SIZE
        elif d < -GRID_SIZE / 2:
            d += GRID_SIZE
        return d

    dx = wrapped_delta(apple_x, head[0])
    dy = wrapped_delta(apple_y, head[1])

    state = [
        # Danger dans les 3 directions relatives possibles
        is_danger(dir_straight),
        is_danger(dir_right),
        is_danger(dir_left),
        # Direction actuelle du serpent (une seule à True)
        direction == LEFT,
        direction == RIGHT,
        direction == UP,
        direction == DOWN,
        # Position de la pomme par rapport à la tête (distance torique la plus courte)
        dx < 0,   # pomme à gauche
        dx > 0,   # pomme à droite
        dy < 0,   # pomme en haut
        dy > 0,   # pomme en bas
    ]
    return np.array(state, dtype=np.float32)


STATE_SIZE = 11
ACTION_SIZE = 3  # tout droit / droite / gauche


# ==================================================================
# 7. RÉSEAU DE NEURONES (DQN) — volontairement petit et simple
# ==================================================================
class QNet(nn.Module):
    """Petit réseau pleinement connecté : 11 -> hidden -> 3.
    Un seul hidden layer suffit largement pour un état de 11 booléens ;
    inutile de complexifier (moins de paramètres = entraînement plus
    rapide sur CPU, et moins de risque de sur-apprentissage)."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_SIZE),
        )

    def forward(self, x):
        return self.net(x)


# ==================================================================
# 8. REPLAY MEMORY — mémoire d'expérience (experience replay)
# ==================================================================
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayMemory:
    """Mémoire tampon simple : une deque de taille fixe + tirage aléatoire.

    L'experience replay casse la corrélation entre transitions successives
    (sinon le réseau apprend sur une séquence d'états très similaires d'affilée,
    ce qui déstabilise l'apprentissage) et permet de réutiliser plusieurs fois
    une même transition (y compris celles générées par l'expert A*)."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ==================================================================
# 9. EXPERT A* — guide d'exploration, UNIQUEMENT pendant l'entraînement
# ==================================================================
def astar_first_action(game):
    """Calcule un chemin A* de la tête du serpent vers la pomme, sur le
    plateau torique (bords enroulés, comme le jeu), en évitant le corps
    du serpent (sauf la queue, qui aura bougé). Retourne la PREMIÈRE action
    relative (0/1/2) du chemin trouvé, ou None si aucun chemin n'existe.

    A* n'est utilisé que pour guider l'exploration au début de l'entraînement
    (cf. train()) ; il ne joue jamais pendant l'évaluation. Le chemin le plus
    court vers la pomme n'est pas forcément "sûr" à long terme (il peut
    enfermer le serpent), mais ce n'est pas grave ici : son seul rôle est de
    montrer au DQN, via les transitions générées, à quoi ressemble "manger
    des pommes", pas de jouer parfaitement.
    """
    import heapq

    snake = game.snake
    head = tuple(snake.head_pos)
    goal = tuple(game.apple.position) if game.apple.position else None
    if goal is None:
        return None

    body_without_tail = set(tuple(seg) for seg in (snake.body[:-1] if not snake.grow_pending else snake.body))

    def wrapped_dist(a, b):
        dx = min((a[0] - b[0]) % GRID_SIZE, (b[0] - a[0]) % GRID_SIZE)
        dy = min((a[1] - b[1]) % GRID_SIZE, (b[1] - a[1]) % GRID_SIZE)
        return dx + dy

    # File de priorité A* : (f_score, compteur_unique, position, chemin_des_directions)
    counter = 0
    open_heap = [(wrapped_dist(head, goal), counter, head, [])]
    best_cost = {head: 0}

    while open_heap:
        f, _, pos, path = heapq.heappop(open_heap)
        if pos == goal:
            if not path:
                return None
            first_dir = path[0]
            idx = CLOCKWISE.index(snake.direction)
            target_idx = CLOCKWISE.index(first_dir)
            if target_idx == idx:
                return 0
            elif target_idx == (idx + 1) % 4:
                return 1
            elif target_idx == (idx - 1) % 4:
                return 2
            else:
                # Demi-tour demandé par le chemin : ne devrait pas arriver
                # (case occupée par le corps juste après la tête), on ignore.
                return None

        for move_dir in (UP, DOWN, LEFT, RIGHT):
            next_pos = ((pos[0] + move_dir[0]) % GRID_SIZE, (pos[1] + move_dir[1]) % GRID_SIZE)
            if next_pos in body_without_tail and next_pos != goal:
                continue
            new_cost = best_cost[pos] + 1
            if next_pos not in best_cost or new_cost < best_cost[next_pos]:
                best_cost[next_pos] = new_cost
                counter += 1
                new_path = path + [move_dir] if len(path) == 0 else path  # on ne garde que la 1re direction
                if len(path) == 0:
                    new_path = [move_dir]
                else:
                    new_path = path
                heapq.heappush(open_heap, (new_cost + wrapped_dist(next_pos, goal), counter, next_pos, new_path))

    return None  # aucun chemin trouvé -> l'appelant doit se rabattre sur le DQN


# ==================================================================
# 10. AGENT DQN — policy net + target net + entraînement par batch
# ==================================================================
class DQNAgent:
    def __init__(self, cfg: DQNConfig, device=None):
        self.cfg = cfg
        # device=None -> utilise le GPU si disponible (DEVICE global) ;
        # compare() force explicitement "cpu" dans ses sous-processus (cf.
        # _run_one_training) car CUDA ne se partage pas bien entre plusieurs
        # process forkés, et le réseau est de toute façon minuscule.
        self.device = device if device is not None else DEVICE
        self.policy_net = QNet(cfg.hidden_size).to(self.device)
        self.target_net = QNet(cfg.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayMemory(cfg.replay_memory_size)
        self.train_steps = 0

    def act(self, state, epsilon):
        """Politique epsilon-greedy : exploration aléatoire avec proba epsilon,
        sinon action gloutonne selon le réseau (policy_net)."""
        if random.random() < epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Un pas d'entraînement sur un batch tiré aléatoirement de la
        replay memory. Retourne la loss (ou None si pas assez de données)."""
        if len(self.memory) < self.cfg.batch_size:
            return None

        batch = self.memory.sample(self.cfg.batch_size)
        states = torch.from_numpy(np.stack([t.state for t in batch])).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        # Q(s, a) prédit par le réseau actuel
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Cible de Bellman : r + gamma * max_a' Q_target(s', a') * (1 - done)
        # Le réseau cible (target_net) stabilise l'entraînement : sans lui,
        # la cible bougerait à chaque mise à jour du même réseau qui prédit,
        # ce qui rend l'apprentissage instable (on "vise une cible mouvante").
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.cfg.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({"model": self.policy_net.state_dict(), "cfg": asdict(self.cfg)}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["model"])
        self.target_net.load_state_dict(checkpoint["model"])


# ==================================================================
# 11. ENTRAÎNEMENT
# ==================================================================
def train(mode, cfg: DQNConfig, out_dir="results", seed=None, quiet=False, device=None):
    """Boucle d'entraînement standard RL, avec injection optionnelle de
    l'expert A* (mode="astar") ou purement DQN (mode="baseline").

    mode="astar"    : use_astar_expert=True, expert_probability décroît
                      linéairement de expert_probability_start à _end.
    mode="baseline" : use_astar_expert=False, expert_probability toujours 0
                      (DQN pur dès le début, pour comparaison).
    """
    assert mode in ("astar", "baseline")
    use_astar_expert = (mode == "astar")

    if seed is not None:
        cfg.seed = seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{mode}_seed{cfg.seed}.csv")
    # Noms de fichiers : technique (mode = astar/baseline) + variante de
    # reward utilisée (cf. SCORE_REWARD_COEFF/malus de temps, ajoutés en
    # plus du barème officiel) + nombre d'épisodes, pour distinguer d'un
    # coup d'œil des modèles entraînés avec des réglages différents.
    reward_tag = "official" if cfg.score_reward_coeff == 0 else "shaped"
    model_path = os.path.join(out_dir, f"{mode}-{reward_tag}_seed{cfg.seed}_ep{cfg.number_of_episodes}_final.pth")

    agent = DQNAgent(cfg, device=device)
    game = SnakeGameAI(render=False, stall_limit_factor=cfg.stall_limit_factor,
                        score_reward_coeff=cfg.score_reward_coeff)

    epsilon = cfg.epsilon_start
    expert_probability = cfg.expert_probability_start if use_astar_expert else 0.0
    first_episode_score_10 = None

    # Fenêtres glissantes des 1000 dernières parties : score et temps sont
    # suivis SÉPARÉMENT (pas seulement le ratio combiné), pour pouvoir les
    # rapporter/nommer indépendamment l'un de l'autre.
    window_size = 1000
    recent_score_per_time = deque(maxlen=window_size)
    recent_scores = deque(maxlen=window_size)
    recent_times = deque(maxlen=window_size)
    recent_sum = 0.0  # sommes maintenues à la main (O(1) par épisode) plutôt
    recent_score_sum = 0.0  # que recalculées avec sum(deque) à chaque pas,
    recent_time_sum = 0.0   # pour un coût quasi nul même sur des parties très rapides.
    best_recent_avg = float("-inf")
    # Instantané séparé pour le meilleur score JAMAIS atteint sur UNE SEULE
    # partie (indépendant de la moyenne glissante ci-dessus, qui peut rester
    # bloquée si la partie exceptionnelle est un coup de chance isolé).
    best_single_score = float("-inf")
    # Chemins des 2 derniers instantanés sauvegardés (pour supprimer l'ancien
    # quand un nouveau record est battu : le nom change à chaque fois pour
    # embarquer l'épisode, donc sans ça les vieux fichiers s'accumuleraient).
    best_model_path = None
    best_score_model_path = None
    last_print_time = time.time()
    print_every_seconds = 20  # cadence d'affichage, pas de surcoût par step :
                               # un seul time.time() par épisode (déjà calculé
                               # via game.elapsed_time) suffit à décider d'imprimer

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "steps", "time", "score_per_time",
                          "reward", "epsilon", "expert_probability",
                          "expert_action_ratio", "loss"])

        for episode in range(1, cfg.number_of_episodes + 1):
            game.reset()
            # Seuil de temps "compétitif" recalculé à chaque épisode à partir
            # du record de moyenne glissante déjà atteint (cf. plus bas) :
            # plus le modèle s'améliore, plus best_recent_avg monte, plus ce
            # seuil se resserre automatiquement (demande explicite : "keep
            # making this threshold tighter as the model gets better").
            # Pas de seuil tant qu'on n'a pas encore de référence valide.
            if best_recent_avg > 0:
                game.competitive_time_threshold = MAX_SCORE / best_recent_avg
            state = get_state(game)
            done = False
            episode_reward = 0.0
            losses = []
            expert_used = 0
            steps_this_episode = 0

            while not done:
                steps_this_episode += 1

                # --- Choix de l'action : expert A* ou DQN epsilon-greedy ---
                action = None
                if use_astar_expert and random.random() < expert_probability:
                    expert_action = astar_first_action(game)
                    if expert_action is not None:
                        action = expert_action
                        expert_used += 1
                if action is None:
                    action = agent.act(state, epsilon)

                # --- Un pas de jeu (les transitions A* comptent EXACTEMENT
                #     comme des transitions DQN : même reward, même stockage) ---
                reward, done, score = game.play_step(action)
                next_state = get_state(game)

                agent.remember(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                episode_reward += reward

            # --- Fin d'épisode : décroissances ---
            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
            if use_astar_expert:
                decay_step = (cfg.expert_probability_start - cfg.expert_probability_end) / cfg.expert_probability_decay_episodes
                expert_probability = max(cfg.expert_probability_end, expert_probability - decay_step)

            if score >= 10 and first_episode_score_10 is None:
                first_episode_score_10 = episode

            elapsed = game.elapsed_time
            score_per_time = score / elapsed if elapsed > 0 else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            expert_ratio = expert_used / steps_this_episode if steps_this_episode else 0.0

            # Nouveau record de score sur une seule partie (indépendant de la
            # moyenne glissante) : snapshot séparé, pour ne pas perdre le
            # modèle qui a produit LE meilleur score jamais vu. Nom du
            # fichier = technique + épisode + score + temps (séparément).
            if score > best_single_score:
                best_single_score = score
                new_path = os.path.join(
                    out_dir, f"{mode}-{reward_tag}_seed{cfg.seed}_ep{episode}_score{int(score)}_t{elapsed:.0f}s.pth")
                agent.save(new_path)
                if best_score_model_path and os.path.exists(best_score_model_path):
                    os.remove(best_score_model_path)
                best_score_model_path = new_path
                if not quiet:
                    print(f"[{mode}] Nouveau meilleur score sur une partie : "
                          f"score={score} temps={elapsed:.1f}s -> {best_score_model_path}")

            # Mise à jour O(1) des moyennes glissantes (cf. plus haut) : si
            # la fenêtre est pleine, on retire la valeur qui va être évincée
            # par le deque AVANT de l'écraser avec append().
            if len(recent_score_per_time) == window_size:
                recent_sum -= recent_score_per_time[0]
                recent_score_sum -= recent_scores[0]
                recent_time_sum -= recent_times[0]
            recent_score_per_time.append(score_per_time)
            recent_scores.append(score)
            recent_times.append(elapsed)
            recent_sum += score_per_time
            recent_score_sum += score
            recent_time_sum += elapsed
            current_recent_avg = recent_sum / len(recent_score_per_time)
            current_avg_score = recent_score_sum / len(recent_scores)
            current_avg_time = recent_time_sum / len(recent_times)

            # Nouveau record de la moyenne glissante (score/temps sur les
            # 1000 dernières parties, ou moins si l'entraînement est plus
            # court) : on sauvegarde un instantané du modèle à cet instant,
            # séparé du modèle final (qui peut être légèrement moins bon si
            # l'agent régresse un peu ensuite, ce qui arrive parfois en RL).
            # Nom du fichier = technique + épisode + score et temps moyens
            # (séparés, comme demandé, plutôt que le seul ratio combiné).
            if current_recent_avg > best_recent_avg:
                best_recent_avg = current_recent_avg
                new_path = os.path.join(
                    out_dir,
                    f"{mode}-{reward_tag}_seed{cfg.seed}_ep{episode}_avgscore{current_avg_score:.1f}_avgt{current_avg_time:.1f}s.pth")
                agent.save(new_path)
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = new_path
                if not quiet:
                    print(f"[{mode}] Nouveau record (sur {len(recent_score_per_time)} parties) : "
                          f"avg_score={current_avg_score:.2f} avg_temps={current_avg_time:.2f}s "
                          f"(ratio={best_recent_avg:.4f}) -> {best_model_path}")

            writer.writerow([episode, score, steps_this_episode, f"{elapsed:.3f}",
                              f"{score_per_time:.4f}", f"{episode_reward:.1f}",
                              f"{epsilon:.4f}", f"{expert_probability:.4f}",
                              f"{expert_ratio:.4f}", f"{avg_loss:.5f}"])
            f.flush()

            # Affichage cadencé par le temps réel (~toutes les 20s), pas par
            # nombre d'épisodes : un seul appel time.time() par épisode, un
            # print seulement quand la fenêtre de 20s est passée -> overhead
            # négligeable même sur des parties très courtes/rapides.
            if not quiet:
                now = time.time()
                if now - last_print_time >= print_every_seconds:
                    print(f"[{mode}] Episode {episode}/{cfg.number_of_episodes} "
                          f"(last {len(recent_score_per_time)} games) avg_score={current_avg_score:.2f} "
                          f"avg_time={current_avg_time:.2f}s avg_score/time={current_recent_avg:.4f} "
                          f"epsilon={epsilon:.3f} expert_p={expert_probability:.3f} loss={avg_loss:.4f}")
                    last_print_time = now

    agent.save(model_path)
    if not quiet:
        print(f"[{mode}] Modèle final sauvegardé -> {model_path}")
        print(f"[{mode}] Meilleur instantané (ratio score/temps={best_recent_avg:.4f}) -> {best_model_path}")
        print(f"[{mode}] Meilleur score sur une partie ({best_single_score:.0f}) -> {best_score_model_path}")
        if first_episode_score_10:
            print(f"[{mode}] Premier épisode >= 10 points : {first_episode_score_10}")
        else:
            print(f"[{mode}] Aucun épisode n'a atteint 10 points pendant l'entraînement.")

    return model_path, csv_path, first_episode_score_10, best_model_path, best_score_model_path


# ==================================================================
# 12. ÉVALUATION — DQN SEUL, aucune influence de l'expert A*
# ==================================================================
def evaluate(model_path, n_games=20, real_clock=False, out_dir="results", quiet=False, device=None):
    """Évalue un modèle entraîné, DQN seul (epsilon=0, A* désactivé).

    real_clock=True  : ouvre une fenêtre et respecte l'horloge officielle
                        du jeu (clock.tick(GAME_SPEED)) -> "vrai" temps.
    real_clock=False : mode rapide headless, pour tourner beaucoup de
                        parties rapidement (steps / GAME_SPEED == temps réel
                        car le jeu avance d'une case par tick).
    """
    cfg = DQNConfig()
    agent = DQNAgent(cfg, device=device)
    agent.load(model_path)
    agent.policy_net.eval()

    use_astar_expert = False  # IMPORTANT : jamais d'A* en évaluation

    scores, times, lengths, wins, stalled = [], [], [], 0, 0

    # Test humain (real_clock=True) : UNE seule fenêtre pygame réutilisée
    # pour toutes les parties (pas de réouverture/flicker entre 2 parties),
    # avec une vraie pause sur l'écran "GAME OVER" / "VICTOIRE" pour que le
    # score soit lisible, au lieu de disparaître au frame suivant.
    shared_game = SnakeGameAI(render=real_clock, stall_limit_factor=cfg.stall_limit_factor) if real_clock else None
    quit_requested = False

    for i in range(n_games):
        if quit_requested:
            break
        game = shared_game if real_clock else SnakeGameAI(render=False, stall_limit_factor=cfg.stall_limit_factor)
        if real_clock and i > 0:
            game.reset()
        state = get_state(game)
        done = False
        while not done:
            # Assertion de garde-fou : l'expert ne doit jamais être appelé ici.
            assert not use_astar_expert
            action = agent.act(state, epsilon=0.0)  # pas d'exploration aléatoire
            reward, done, score = game.play_step(action)
            state = get_state(game)
            if real_clock and game.quit_requested:
                quit_requested = True

        scores.append(score)
        times.append(game.elapsed_time)
        lengths.append(game.steps)
        if game.victory:
            wins += 1
        if game.steps_since_apple > cfg.stall_limit_factor * len(game.snake.body) - 1:
            stalled += 1

        if real_clock and not quit_requested:
            quit_requested = game.pause_on_game_over(seconds=1.5)

    if shared_game is not None:
        pygame.quit()

    scores = np.array(scores, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    lengths = np.array(lengths, dtype=np.float64)
    score_per_time = np.divide(scores, times, out=np.zeros_like(scores), where=times > 0)

    # best_score et best_score_time forment une PAIRE : le temps de LA partie
    # où le meilleur score a été atteint (et non la moyenne des temps de
    # toutes les parties, qui ne correspondrait à aucune partie précise).
    best_idx = int(scores.argmax())

    results = {
        "n_games": n_games,
        "avg_score": scores.mean(),
        "best_score": scores[best_idx],           # score (1 point / pomme)
        "best_score_time": times[best_idx],       # temps de CETTE partie-là
        "avg_time": times.mean(),
        "avg_score_per_time": score_per_time.mean(),
        "pct_games_reaching_10": 100.0 * (scores >= 10).sum() / n_games,
        "avg_game_length_steps": lengths.mean(),
        "wins": wins,
        "stalled_games": stalled,
    }

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(model_path))[0]
    csv_path = os.path.join(out_dir, f"eval_{name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(results.keys()))
        writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in results.values()])

    if not quiet:
        print(f"--- Évaluation de {model_path} sur {n_games} parties ---")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print(f"Résultats sauvegardés -> {csv_path}")

    return results


# ==================================================================
# 13. COMPARAISON — plusieurs simulations en parallèle (baseline vs astar)
# ==================================================================
def _run_one_training(args):
    mode, episodes, seed, out_dir = args
    cfg = DQNConfig(number_of_episodes=episodes, seed=seed)
    torch.set_num_threads(1)  # évite que chaque process ne sature tous les coeurs
    # CUDA ne supporte pas d'être ré-initialisé dans un process forké : chaque
    # worker de ce Pool reste sur CPU. Le réseau est minuscule (11->256->3),
    # donc plusieurs entraînements CPU en parallèle restent très rapides,
    # et c'est plus simple/robuste que de partager le GPU entre process.
    cpu = torch.device("cpu")
    model_path, csv_path, first10, best_model_path, best_score_model_path = train(
        mode, cfg, out_dir=out_dir, seed=seed, quiet=True, device=cpu)
    eval_results = evaluate(model_path, n_games=100, real_clock=False, out_dir=out_dir, quiet=True, device=cpu)
    return mode, seed, first10, eval_results


def compare(seeds, episodes, workers=4, out_dir="results"):
    """Lance {baseline, astar} x seeds en parallèle (multiprocessing), pour
    obtenir des statistiques (moyenne/écart-type) au lieu d'une seule
    partie, comme demandé (le but est de comparer les deux approches de
    façon fiable, pas de tirer une conclusion sur une seule simulation)."""
    import multiprocessing

    jobs = [(mode, episodes, seed, out_dir) for mode in ("baseline", "astar") for seed in seeds]
    print(f"Lancement de {len(jobs)} entraînements ({workers} en parallèle)...")

    # "spawn" (et non le "fork" par défaut sous Linux) : le process parent a
    # déjà initialisé un contexte CUDA (DEVICE), qu'un fork ne peut pas
    # réutiliser dans les enfants. "spawn" démarre des process propres, qui
    # restent de toute façon sur CPU dans _run_one_training.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        all_results = pool.map(_run_one_training, jobs)

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    per_mode = {"baseline": [], "astar": []}
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "seed", "first_episode_score_10", "avg_score",
                          "best_score", "best_score_time", "avg_time", "avg_score_per_time",
                          "pct_games_reaching_10", "avg_game_length_steps", "wins"])
        for mode, seed, first10, res in all_results:
            per_mode[mode].append(res)
            writer.writerow([mode, seed, first10, f"{res['avg_score']:.3f}",
                              res["best_score"], f"{res['best_score_time']:.2f}",
                              f"{res['avg_time']:.3f}", f"{res['avg_score_per_time']:.4f}",
                              f"{res['pct_games_reaching_10']:.1f}", f"{res['avg_game_length_steps']:.1f}",
                              res["wins"]])

    # Statistiques sur plusieurs seeds (moyenne ± écart-type), pas une seule
    # simulation : c'est ce qui permet de dire si le guidage A* aide "pour
    # de vrai" ou si l'écart observé n'est que du bruit d'un seed particulier.
    print(f"\n=== Résumé statistique (mean ± std sur {len(seeds)} seeds) ===")
    stats = {}
    for mode in ("baseline", "astar"):
        results_list = per_mode[mode]
        if not results_list:
            continue
        spt_values = np.array([r["avg_score_per_time"] for r in results_list])
        score_values = np.array([r["avg_score"] for r in results_list])
        pct10_values = np.array([r["pct_games_reaching_10"] for r in results_list])
        stats[mode] = {"spt": spt_values, "score": score_values}
        print(f"  {mode:10s} avg_score_per_time={spt_values.mean():.4f} ± {spt_values.std():.4f}  "
              f"avg_score={score_values.mean():.2f} ± {score_values.std():.2f}  "
              f"%games>=10pts={pct10_values.mean():.1f}%")

    if "baseline" in stats and "astar" in stats:
        diff = stats["astar"]["spt"].mean() - stats["baseline"]["spt"].mean()
        # Écart-type combiné (pooled), pour situer l'écart de moyenne par
        # rapport au bruit entre seeds -> un simple "effect size", pas un
        # test statistique formel (pas assez de seeds ici pour un p-value
        # fiable), mais suffisant pour juger si l'écart est net ou pas.
        pooled_std = np.sqrt((stats["astar"]["spt"].std() ** 2 + stats["baseline"]["spt"].std() ** 2) / 2)
        effect = diff / pooled_std if pooled_std > 0 else float("inf")
        print(f"\n  Écart (astar - baseline) sur avg_score_per_time : {diff:+.4f} "
              f"(effect size ≈ {effect:+.2f} écarts-types combinés)")

    print(f"Détails complets -> {summary_path}")
    return summary_path


# ==================================================================
# 14. CLI
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="Snake IA - DQN guidé par A* pendant l'entraînement.")
    subparsers = parser.add_subparsers(dest="command")

    p_train = subparsers.add_parser("train", help="Entraîne un agent DQN.")
    p_train.add_argument("--mode", choices=["baseline", "astar"], default="astar")
    p_train.add_argument("--episodes", type=int, default=DQNConfig().number_of_episodes)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--out-dir", type=str, default="results")

    p_eval = subparsers.add_parser("eval", help="Évalue un modèle entraîné (DQN seul, sans A*).")
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--games", type=int, default=20)
    p_eval.add_argument("--fast", action="store_true", help="Mode rapide sans fenêtre (sinon : horloge réelle).")

    p_compare = subparsers.add_parser("compare", help="Compare baseline vs A*-guided sur plusieurs seeds, en parallèle.")
    p_compare.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p_compare.add_argument("--episodes", type=int, default=DQNConfig().number_of_episodes)
    p_compare.add_argument("--workers", type=int, default=4)
    p_compare.add_argument("--out-dir", type=str, default="results")

    args = parser.parse_args()

    if args.command == "train":
        cfg = DQNConfig(number_of_episodes=args.episodes, seed=args.seed)
        train(args.mode, cfg, out_dir=args.out_dir, seed=args.seed)

    elif args.command == "eval":
        evaluate(args.model, n_games=args.games, real_clock=not args.fast)

    elif args.command == "compare":
        compare(args.seeds, args.episodes, workers=args.workers, out_dir=args.out_dir)

    else:
        # Lancement sans argument : python snake-ia.py
        # -> entraîne (si besoin) puis fait jouer l'agent en temps réel,
        #    fenêtre ouverte, horloge officielle du jeu, SANS A* (final DQN only).
        default_model = os.path.join("results", "astar_seed0.pth")
        if not os.path.exists(default_model):
            print("Aucun modèle trouvé, entraînement rapide d'un agent A*-guidé...")
            cfg = DQNConfig(number_of_episodes=DQNConfig().number_of_episodes, seed=0)
            train("astar", cfg, seed=0)
        print("Évaluation en temps réel (5 parties, fenêtre pygame, DQN seul)...")
        evaluate(default_model, n_games=5, real_clock=True)


if __name__ == "__main__":
    main()
