"""
Serpent joue tout seul : Deep Q-Learning (DQN) avec PyTorch + PyGame.

Ce fichier est autonome : il ne modifie ni n'importe serpent-algo.py.

REGLE IMPORTANTE : la logique du jeu de base n'est PAS modifiee.
Les classes Snake et Apple sont reprises a l'identique du socle serpent-algo.py,
y compris le deplacement modulo GRID_SIZE (les bords teleportent le serpent) et
check_wall_collision() qui, de ce fait, ne se declenche jamais.
Seule est AJOUTEE la couche necessaire au RL (reset, play_step, perception),
sans toucher aux methodes existantes.

Architecture en 3 blocs (cf. sujet) :
    - SnakeGameIA (PyGame)  : l'environnement, play_step(action)
    - Linear_QNet / QTrainer: le modele (Torch)
    - Agent                 : l'orchestration de la boucle d'entrainement

Usage :
    python snake-ia.py                      # entraine sans affichage (rapide), puis sauvegarde
    python snake-ia.py --render             # entraine avec affichage
    python snake-ia.py --games 400          # nombre de parties d'entrainement
    python snake-ia.py --play               # regarde le modele entraine jouer

Chaque partie est journalisee (score, duree, nombre de pas) dans parties.csv.
"""

import argparse
import csv
import math
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONSTANTES DE JEU (identiques au socle serpent-algo.py, sauf GAME_SPEED) ---
GRID_SIZE = 15
CELL_SIZE = 30
# GAME_SPEED vaut 5 dans le socle. Ici 40, sinon l'entrainement avec --render
# durerait des heures. C'est une cadence d'AFFICHAGE : le serpent avance d'une
# case par image dans les deux cas, donc la logique du jeu est inchangee.
# Surchargeable en ligne de commande avec --speed (ex. --speed 5 pour le socle).
GAME_SPEED = 40  # images/s quand l'affichage est actif

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)   # Tete du serpent
VERT = (0, 200, 0)       # Corps du serpent
ROUGE = (200, 0, 0)      # Pomme
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Ordre horaire : sert a traduire une action relative en direction absolue.
# RIGHT -> DOWN -> LEFT -> UP -> RIGHT ...
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]

# --- RECOMPENSES ---
# Attention : ne JAMAIS mettre une recompense positive sur un simple deplacement,
# sinon la politique optimale devient "tourner en rond sans jamais manger".
REWARD_POMME = 10
REWARD_MORT = -10
REWARD_PAS = 0
REWARD_VICTOIRE = 100
# Troncation par le garde-fou anti-boucle : ce n'est PAS une mort, l'etat reste
# parfaitement viable. On ne la penalise donc pas. Tourner en rond rapporte deja
# 0, alors que manger rapporte +10 actualise : la boucle n'est jamais optimale.
REWARD_TIMEOUT = 0

# --- HYPERPARAMETRES DQN ---
STATE_SIZE = 17         # 9 dangers (3 directions x 3 profondeurs) + 4 direction + 4 pomme
MAX_MEMORY = 100_000    # taille du replay buffer
BATCH_SIZE = 1000       # taille du mini-batch rejoue en fin de partie
LR = 0.001              # learning rate (Adam)
GAMMA = 0.9             # facteur d'actualisation
# epsilon-greedy : decroissance exponentielle indexee sur les COUPS joues (et non
# sur les parties, dont la duree est tres variable), avec un plancher.
EPS_START = 1.0         # probabilite de coup aleatoire au demarrage
EPS_END = 0.02          # plancher : on garde toujours un peu d'exploration
EPS_DECAY = 20_000      # constante de temps, en nombre de coups joues
TARGET_UPDATE = 10      # copie du reseau vers le target network (en parties)

ICI = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ICI, "model.pth")
LOG_PATH = os.path.join(ICI, "parties.csv")


# =====================================================================
#  BLOC 0 : LES CLASSES DU SOCLE (reprises a l'identique)
# =====================================================================

class Snake:
    """Represente le serpent, sa position, sa direction et son corps.

    Copie conforme du socle serpent-algo.py : aucune ligne de logique modifiee.
    """

    def __init__(self):
        # Position initiale au centre
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        # Le corps est une liste de positions (x, y), incluant la tete
        self.body = [self.head_pos,
                     [self.head_pos[0] - 1, self.head_pos[1]],
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empechant le mouvement inverse immediat."""
        # Verifie que la nouvelle direction n'est pas l'inverse de l'actuelle
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Deplace le serpent d'une case dans la direction actuelle."""
        # Calcul de la nouvelle position de la tete
        new_head_x = (self.head_pos[0] + self.direction[0]) % GRID_SIZE
        new_head_y = (self.head_pos[1] + self.direction[1]) % GRID_SIZE

        # Mettre a jour la tete (la nouvelle position devient la nouvelle tete)
        new_head_pos = [new_head_x, new_head_y]
        self.body.insert(0, new_head_pos)
        self.head_pos = new_head_pos

        # Si le serpent ne doit pas grandir, supprime la queue (mouvement normal)
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False  # Reinitialise le drapeau

    def grow(self):
        """Prepare le serpent a grandir au prochain mouvement."""
        self.grow_pending = True
        self.score += 1

    def check_wall_collision(self):
        """Verifie si la tete touche les bords (Game Over si hors grille)."""
        # Note du socle : ne se declenche jamais, puisque move() applique un
        # modulo GRID_SIZE. Comportement conserve tel quel.
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        """Verifie si la tete touche une partie du corps (Game Over si auto-morsure)."""
        # On verifie si la position de la tete est dans le reste du corps (body[1:])
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est termine (mur ou morsure)."""
        return self.check_wall_collision() or self.check_self_collision()

    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        # Dessine le corps
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE,
                               segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)  # Bordure

        # Dessine la tete (couleur differente)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE,
                                self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                                CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)  # Bordure plus epaisse


class Apple:
    """Represente la pomme (nourriture) et sa position.

    Copie conforme du socle serpent-algo.py.
    """

    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aleatoire non occupee par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions
                               if list(pos) not in occupied_positions]

        if not available_positions:
            return None  # Toutes les cases sont pleines (condition de Victoire)

        return random.choice(available_positions)

    def relocate(self, snake_body):
        """Deplace la pomme vers une nouvelle position aleatoire."""
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        """Dessine la pomme sur la surface de jeu."""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE,
                               self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            # Ajout d'un petit reflet pour un aspect "pomme"
            pygame.draw.circle(surface, BLANC,
                               (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3),
                               CELL_SIZE // 8)


def case_suivante(position, direction, pas=1):
    """Case atteinte depuis `position` en faisant `pas` cases dans `direction`.

    Applique EXACTEMENT le meme modulo que Snake.move() : la perception de
    l'agent doit coller a la physique reelle du jeu, bords teleportants inclus.
    """
    return [(position[0] + direction[0] * pas) % GRID_SIZE,
            (position[1] + direction[1] * pas) % GRID_SIZE]


def delta_torique(a, b, n=GRID_SIZE):
    """Ecart signe le plus court pour aller de `a` a `b` sur un anneau de taille n.

    Indispensable ici : la grille est un TORE (Snake.move() applique un modulo),
    donc comparer betement deux coordonnees donne la mauvaise direction des que
    le chemin le plus court passe par un bord. Exemple avec GRID_SIZE = 15 :
    a = 1, b = 13 -> le resultat est -3 (3 cases a gauche en passant par le
    bord), et non +12 (12 cases a droite).
    """
    return ((b - a + n // 2) % n) - n // 2


# =====================================================================
#  BLOC 1 : LE JEU (environnement RL)
# =====================================================================

class SnakeGameIA:
    """Enveloppe RL autour des classes du socle.

    N'ajoute que ce qui manque pour entrainer un agent :
      - reset()             : reinitialiser sans relancer main() recursivement
      - play_step(action)   -> (reward, game_over, score)
      - is_collision(case)  : lecture seule, pour la perception de l'agent
      - un mode headless    : pas de fenetre, pas de limite de FPS
    """

    def __init__(self, render=True, speed=GAME_SPEED):
        self.render = render
        self.speed = speed

        if self.render:
            # On n'initialise que l'affichage et les polices : pygame.init() tenterait
            # aussi d'ouvrir le mixer audio, ce qui inonde WSL de warnings ALSA.
            pygame.display.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Serpent IA - Deep Q-Learning")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 32)
        else:
            # Mode "headless" : pas de fenetre, pas de limite de FPS -> entrainement rapide.
            self.display = None
            self.clock = None
            self.font = None

        self.n_games_display = 0
        self.record_display = 0
        self.reset()

    # -- cycle de vie -------------------------------------------------

    def reset(self):
        """Remet une partie a zero et redemarre le chronometre de la partie."""
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.frame_iteration = 0
        self.victory = False
        self.truncated = False
        self.start_time = time.time()  # chrono de la partie en cours

    # -- proprietes de confort ---------------------------------------

    @property
    def score(self):
        return self.snake.score

    @property
    def head(self):
        return self.snake.head_pos

    @property
    def direction(self):
        return self.snake.direction

    @property
    def elapsed(self):
        """Duree de la partie en cours, en secondes."""
        return time.time() - self.start_time

    # -- perception (lecture seule, n'altere pas l'etat du jeu) -------

    def is_collision(self, case, pas=1):
        """True si `case` est occupee au moment ou la tete y arrivera.

        `pas` = dans combien de coups la tete atteindrait cette case. La queue
        avance elle aussi : au bout de `pas` coups, les `pas` derniers segments
        ont libere leur case. Sans ca la perception est pessimiste et l'agent
        refuse des cases parfaitement sures, a commencer par celle que la queue
        vient de quitter. Exception : si le serpent vient de manger il grandit
        et ne libere rien (grow_pending).

        Lecture seule : n'altere pas l'etat du jeu. Il n'y a volontairement pas
        de test de mur : dans la logique du socle, les bords teleportent
        (modulo), donc un mur ne peut pas tuer.
        """
        corps = self.snake.body[1:]
        if pas > 0 and not self.snake.grow_pending:
            libere = min(pas, len(corps))
            corps = corps[:len(corps) - libere]
        return list(case) in corps

    # -- boucle de jeu ------------------------------------------------

    def play_step(self, action):
        """Joue un pas de jeu.

        action : vecteur one-hot [tout_droit, tourner_droite, tourner_gauche]
        retour : (reward, game_over, truncated, score)
                 game_over arrete la partie ; truncated distingue une coupure
                 par garde-fou d'une vraie mort (cf. etape 2).
        """
        self.frame_iteration += 1

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        # 1. deplacement (logique du socle : set_direction puis move)
        self._apply_action(action)
        self.snake.move()

        # 2. fin de partie ? (logique du socle, + un garde-fou anti-boucle infinie)
        reward = REWARD_PAS
        game_over = False
        self.truncated = False

        # 2a. vraie mort : etat TERMINAL, il n'y a aucun avenir a estimer.
        if self.snake.is_game_over():
            game_over = True
            reward = REWARD_MORT
            return reward, game_over, self.truncated, self.score

        # 2b. garde-fou anti-boucle : TRONCATION, pas une mort. La partie est
        # coupee alors que la position reste parfaitement jouable. Le signaler
        # a part est indispensable : sinon l'agent apprend V(s) = REWARD_MORT
        # sur un etat sain, et comme mourir tout de suite coute alors aussi
        # cher que survivre longtemps, ca l'encourage a se suicider tot.
        if self.frame_iteration > 100 * len(self.snake.body):
            game_over = True
            self.truncated = True
            reward = REWARD_TIMEOUT
            return reward, game_over, self.truncated, self.score

        # 3. pomme mangee ? (logique du socle)
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            reward = REWARD_POMME
            if not self.apple.relocate(self.snake.body):
                # Plus aucune case libre : la grille est remplie, c'est la victoire.
                self.victory = True
                game_over = True
                reward = REWARD_VICTOIRE

        # 4. affichage
        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, game_over, self.truncated, self.score

    def _apply_action(self, action):
        """Traduit une action *relative* en direction absolue, puis la transmet
        a set_direction() du socle.

        On utilise 3 actions relatives plutot que 4 absolues pour deux raisons :
        - c'est coherent avec l'etat (danger en face / a droite / a gauche) ;
        - un demi-tour ne peut pas etre produit, donc set_direction ne rejette
          jamais l'action choisie par l'agent.
        """
        idx = CLOCKWISE.index(self.snake.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = CLOCKWISE[idx]                    # tout droit
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = CLOCKWISE[(idx + 1) % 4]          # tourner a droite
        else:
            new_dir = CLOCKWISE[(idx - 1) % 4]          # tourner a gauche

        self.snake.set_direction(new_dir)

    # -- affichage (repris du socle) ----------------------------------

    def _update_ui(self):
        self.display.fill(GRIS_FOND)
        pygame.draw.rect(self.display, NOIR,
                         pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH))

        # grille
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.display, GRIS_GRILLE,
                             (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
        for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.display, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

        self.apple.draw(self.display)
        self.snake.draw(self.display)
        self._draw_panel()
        pygame.display.flip()

    def _draw_panel(self):
        pygame.draw.rect(self.display, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
        pygame.draw.line(self.display, BLANC,
                         (0, SCORE_PANEL_HEIGHT - 2),
                         (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

        score_text = self.font.render(f"Score: {self.score}", True, BLANC)
        self.display.blit(score_text, (10, 10))

        games_text = self.font.render(f"Partie: {self.n_games_display}", True, BLANC)
        self.display.blit(games_text, (10, 45))

        # Chrono de la partie en cours (comme le socle)
        elapsed = self.elapsed
        temps_text = self.font.render(
            f"Temps: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}", True, BLANC)
        self.display.blit(temps_text, (SCREEN_WIDTH - temps_text.get_width() - 10, 45))

        record_text = self.font.render(f"Record: {self.record_display}", True, BLANC)
        self.display.blit(record_text, (SCREEN_WIDTH - record_text.get_width() - 10, 10))

        fill_rate = (len(self.snake.body) / (GRID_SIZE * GRID_SIZE)) * 100
        fill_text = self.font.render(f"Rempl.: {fill_rate:.1f}%", True, BLANC)
        self.display.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 45))


# =====================================================================
#  BLOC 2 : LE MODELE (Torch)
# =====================================================================

class Linear_QNet(nn.Module):
    """Reseau de neurones qui approxime Q(etat, action).

    STATE_SIZE entrees (l'etat) -> 256 neurones caches -> 3 sorties
    (une Q-value par action).
    """

    def __init__(self, input_size=STATE_SIZE, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, path=MODEL_PATH):
        torch.save(self.state_dict(), path)

    def load(self, path=MODEL_PATH):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()


class QTrainer:
    """Une etape de descente de gradient sur l'equation de Bellman :

        Q(s, a) <- r + gamma * max_a' Q_target(s', a')

    On utilise un *target network* (copie figee du reseau) pour calculer la cible.
    Sans ca, la cible bouge a chaque mise a jour et l'entrainement oscille :
    c'est l'apport central du papier DQN de Mnih et al. (Nature, 2015).
    """

    def __init__(self, model, target_model, lr=LR, gamma=GAMMA):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:  # un seul echantillon -> on ajoute la dimension batch
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        with torch.no_grad():
            next_q = self.target_model(next_state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next_q[idx])
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        """Copie les poids du reseau courant vers le target network."""
        self.target_model.load_state_dict(self.model.state_dict())


# =====================================================================
#  BLOC 3 : L'AGENT (orchestration)
# =====================================================================

class Agent:
    def __init__(self):
        self.n_games = 0
        self.steps = 0              # coups joues : pilote la decroissance d'epsilon
        self.epsilon = EPS_START
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(STATE_SIZE, 256, 3)
        self.target_model = Linear_QNet(STATE_SIZE, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.trainer = QTrainer(self.model, self.target_model)

    # -- perception ---------------------------------------------------

    def get_state(self, game):
        """Construit le vecteur d'etat a STATE_SIZE (17) valeurs binaires.

        Danger    (9) : collision tout droit / a droite / a gauche, a 1, 2 et 3 cases
        Direction (4) : direction courante, en one-hot
        Pomme     (4) : position relative de la pomme, en ecart TORIQUE

        L'etat est *relatif* au serpent : c'est ce qui permet au reseau de
        generaliser (une situation apprise en haut de la grille est reutilisable
        en bas), et c'est ce qui rend les 3 actions relatives coherentes.
        """
        head = game.head
        d = game.direction
        idx = CLOCKWISE.index(d)

        dir_droite = CLOCKWISE[(idx + 1) % 4]
        dir_gauche = CLOCKWISE[(idx - 1) % 4]

        # Danger sur 3 profondeurs. A 1 case l'agent n'evite que le coup fatal
        # immediat ; a 2 et 3 cases il commence a voir l'impasse qu'il se creuse.
        # Comme les murs ne tuent pas (bords teleportants), l'auto-enfermement est
        # la SEULE cause de mort : un horizon de 1 case ne peut pas l'anticiper.
        dangers = []
        for pas in (1, 2, 3):
            for direction in (d, dir_droite, dir_gauche):
                # case_suivante applique le meme modulo que Snake.move()
                case = case_suivante(head, direction, pas)
                dangers.append(game.is_collision(case, pas))

        pomme = game.apple.position

        # Ecart TORIQUE, et non comparaison de coordonnees : les bords
        # teleportent, donc "pomme[0] > head[0]" designe la mauvaise direction
        # des que le chemin le plus court passe par un bord (tete en x=1, pomme
        # en x=13 -> la pomme est a 3 cases a GAUCHE, pas a 12 cases a droite).
        dx = delta_torique(head[0], pomme[0])
        dy = delta_torique(head[1], pomme[1])

        state = dangers + [
            # Direction courante
            d == LEFT,
            d == RIGHT,
            d == UP,
            d == DOWN,

            # Position de la pomme (chemin le plus court sur le tore)
            dx < 0,  # pomme a gauche
            dx > 0,  # pomme a droite
            dy < 0,  # pomme en haut
            dy > 0,  # pomme en bas
        ]
        return np.array(state, dtype=int)

    # -- memoire ------------------------------------------------------

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Experience replay : on rejoue un mini-batch tire au hasard dans la memoire.
        Tirer au hasard casse la correlation entre transitions successives."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)
        if not mini_sample:
            return
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # -- decision -----------------------------------------------------

    def get_action(self, state, train=True):
        """Strategie epsilon-greedy : au debut on explore beaucoup (coups au hasard),
        puis on exploite de plus en plus ce que le reseau a appris.

        Decroissance EXPONENTIELLE indexee sur le nombre de COUPS joues, avec un
        plancher EPS_END. L'ancienne regle (epsilon = 80 - n_games, bornee a 0)
        tombait a zero des la partie 80 : au-dela, l'agent n'explorait plus du
        tout et ne pouvait plus decouvrir une strategie qu'il n'avait pas deja.
        """
        final_move = [0, 0, 0]

        if train:
            self.steps += 1
            self.epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps / EPS_DECAY)
        else:
            self.epsilon = 0.0  # en evaluation : que de l'exploitation

        if train and random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move


# =====================================================================
#  JOURNAL DES PARTIES
# =====================================================================

class JournalParties:
    """Enregistre une ligne par partie dans parties.csv.

    Ecrit et vide le tampon a chaque partie : si l'entrainement est interrompu
    (Ctrl-C, fermeture de la fenetre), rien n'est perdu.
    """

    COLONNES = ["horodatage", "mode", "partie", "score", "record",
                "duree_s", "pas", "pas_par_s", "moyenne", "moyenne_100", "epsilon"]

    def __init__(self, path=LOG_PATH):
        self.path = path
        nouveau = not os.path.exists(path) or os.path.getsize(path) == 0
        self.fichier = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fichier)
        if nouveau:
            self.writer.writerow(self.COLONNES)
            self.fichier.flush()

    def ajouter(self, mode, partie, score, record, duree, pas,
                moyenne=None, moyenne_100=None, epsilon=None):
        self.writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            mode,
            partie,
            score,
            record,
            f"{duree:.3f}",
            pas,
            f"{pas / duree:.1f}" if duree > 0 else "",
            f"{moyenne:.3f}" if moyenne is not None else "",
            f"{moyenne_100:.3f}" if moyenne_100 is not None else "",
            f"{epsilon:.4f}" if epsilon is not None else "",
        ])
        self.fichier.flush()  # ecriture immediate sur disque

    def fermer(self):
        self.fichier.close()


# =====================================================================
#  BOUCLES PRINCIPALES
# =====================================================================

def train(n_games=1000, render=False, speed=GAME_SPEED):
    """Boucle d'entrainement (cf. sujet) :
        state -> action -> play_step -> new_state -> remember -> train
    """
    agent = Agent()
    game = SnakeGameIA(render=render, speed=speed)
    journal = JournalParties()

    record = 0
    total_score = 0
    scores = []
    t0 = time.time()

    print(f"Entrainement sur {n_games} parties (affichage: {'oui' if render else 'non'})")
    print(f"Journal des parties : {LOG_PATH}\n")

    try:
        while agent.n_games < n_games:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old, train=True)

            reward, done, truncated, score = game.play_step(final_move)

            # `done` arrete la partie ; seul `terminated` coupe le bootstrapping.
            # Sur une troncation l'etat suivant est reel et parfaitement valide :
            # on le transmet pour que la cible reste r + gamma * max Q(s', .).
            # Sur une vraie mort il n'y a pas d'avenir : la cible se reduit a r,
            # et state_new (neutralise) est de toute facon ignore.
            terminated = done and not truncated
            state_new = (np.zeros(STATE_SIZE, dtype=int) if terminated
                         else agent.get_state(game))

            agent.train_short_memory(state_old, final_move, reward, state_new, terminated)
            agent.remember(state_old, final_move, reward, state_new, terminated)

            if done:
                # On releve la duree et le nombre de pas AVANT le reset.
                duree = game.elapsed
                pas = game.frame_iteration

                agent.n_games += 1
                total_score += score
                scores.append(score)
                if score > record:
                    record = score
                    agent.model.save()

                moyenne = total_score / agent.n_games
                moyenne_100 = sum(scores[-100:]) / len(scores[-100:])

                journal.ajouter("train", agent.n_games, score, record, duree, pas,
                                moyenne, moyenne_100, agent.epsilon)

                print(f"Partie {agent.n_games:4d} | Score {score:3d} | Record {record:3d} "
                      f"| Moyenne {moyenne:5.2f} | Moyenne(100) {moyenne_100:5.2f} "
                      f"| eps {agent.epsilon:.3f} | {duree:6.2f}s | {pas:5d} pas")

                game.reset()
                game.n_games_display = agent.n_games
                game.record_display = record
                agent.train_long_memory()

                if agent.n_games % TARGET_UPDATE == 0:
                    agent.trainer.sync_target()
    except KeyboardInterrupt:
        print("\nInterrompu : le modele et le journal sont sauvegardes.")
    finally:
        agent.model.save()
        journal.fermer()

    duree_totale = time.time() - t0
    print(f"\nTermine en {duree_totale:.1f}s. Record: {record}. "
          f"Moyenne globale: {total_score / max(1, agent.n_games):.2f}")
    print(f"Modele sauvegarde dans {MODEL_PATH}")
    print(f"Temps de chaque partie dans {LOG_PATH}")
    print("Pour le regarder jouer :  python snake-ia.py --play")
    return agent


def play(n_games=5, speed=15):
    """Charge le modele entraine et le regarde jouer (sans exploration)."""
    if not os.path.exists(MODEL_PATH):
        print(f"Aucun modele trouve ({MODEL_PATH}). Lance d'abord : python snake-ia.py")
        return

    agent = Agent()
    agent.model.load(MODEL_PATH)
    game = SnakeGameIA(render=True, speed=speed)
    journal = JournalParties()

    record = 0
    try:
        for partie in range(1, n_games + 1):
            game.reset()
            game.n_games_display = partie
            done = False
            score = 0
            while not done:
                state = agent.get_state(game)
                move = agent.get_action(state, train=False)
                _, done, _, score = game.play_step(move)

            duree = game.elapsed
            pas = game.frame_iteration
            record = max(record, score)
            game.record_display = record

            journal.ajouter("play", partie, score, record, duree, pas)
            print(f"Partie {partie} | Score {score} | {duree:.2f}s | {pas} pas")
    finally:
        journal.fermer()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Serpent joue tout seul (Deep Q-Learning).")
    parser.add_argument("--games", type=int, default=1000,
                        help="nombre de parties d'entrainement (defaut: 1000)")
    parser.add_argument("--render", action="store_true",
                        help="afficher la fenetre pendant l'entrainement (plus lent)")
    parser.add_argument("--speed", type=int, default=GAME_SPEED,
                        help="images par seconde quand l'affichage est actif")
    parser.add_argument("--play", action="store_true",
                        help="ne pas entrainer : regarder le modele sauvegarde jouer")
    args = parser.parse_args()

    if args.play:
        play(speed=args.speed)
    else:
        train(n_games=args.games, render=args.render, speed=args.speed)


if __name__ == "__main__":
    main()
