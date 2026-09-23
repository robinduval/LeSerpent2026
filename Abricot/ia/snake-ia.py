"""
snake-ia.py — Snake joué par un agent de Deep Q-Learning (PyTorch).

Groupe ABRICOT — Projet STP 2026 — rendu du 16/09.
Architecture imposée par le sujet (cf. AGENTS.md §4), inspirée de
Patrick Loeber : https://github.com/patrickloeber/snake-ai-pytorch

    ┌─ Agent ─────────────┐   ┌─ Game (pygame) ──────┐   ┌─ Model (torch) ────┐
    │ get_state(game)     │   │ play_step(action)    │   │ Linear_QNet        │
    │ get_action(state)   │──▶│  -> reward,          │   │ QTrainer.train_step│
    │ remember / train    │◀──│     game_over, score │   │                    │
    └─────────────────────┘   └──────────────────────┘   └────────────────────┘

RÈGLES DU JEU — le fichier de base fait foi, et il fait « % GRID_SIZE » dans
move() : le serpent TRAVERSE les murs (WRAP = True). Seule l'auto-morsure tue.
Le README du prof dit l'inverse, mais check_wall_collision() ne peut jamais se
déclencher et sa docstring le documente. Voir AGENTS.md §6bis.

CONTRAINTES RESPECTÉES — GRID_SIZE = 15, GAME_SPEED = 5 et le scoring (+1 par
pomme) sont ceux du jeu de base et ne sont jamais modifiés. Seules les
RÉCOMPENSES varient : c'est le signal d'apprentissage interne de l'agent, pas
le score de la partie — deux barèmes différents sur la même partie donnent le
même score. L'entraînement tourne sans fenêtre, donc sans clock ; la démo
rejoue à la vitesse d'origine.

TROIS LEVIERS, croisés en plan factoriel (résultats dans README.md) :
  --bareme sujet     valeurs du PDF (+10 pomme, -10 mort, +0,1 déplacement)
           efficace  le déplacement coûte, se rapprocher de la pomme rapporte
  --etat   simple    les 11 booléens du sujet (danger à UNE case)
           etendu    + 3 entrées de flood-fill : « est-ce que je m'enferme ? »
  --securite         filet déterministe à l'inférence : le réseau garde la
                     main, on oppose seulement un veto aux actions qui mènent
                     dans une poche trop petite pour le corps.

Meilleure configuration mesurée : --bareme efficace --etat etendu --securite
(score moyen 94,2 et record 131 sur 100 parties, contre 26,6 et 56 pour le
sujet appliqué à la lettre).

Usage :
    python snake-ia.py train --games 400 --bareme efficace --etat etendu
    python snake-ia.py bench --games 100 --bareme efficace --etat etendu --securite
    python snake-ia.py play  --games 5   --bareme efficace --etat etendu --securite
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import deque


def _assurer_environnement():
    """Relance le script avec le Python du venv si PyTorch manque.

    Le Python du système est en 3.14, pour lequel PyTorch n'a pas de wheel.
    Sans ça, « python3 snake-ia.py » échoue sur un ModuleNotFoundError et il
    faut savoir qu'il existe un .venv à côté. On le fait nous-même.
    """
    try:
        import torch  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    if os.environ.get("SNAKE_IA_RELANCE"):          # garde anti-boucle
        raise SystemExit("PyTorch est introuvable, même dans .venv/. "
                         "Voir la section Environnement du README.")
    venv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        ".venv", "bin", "python")
    if os.path.exists(venv):
        print("PyTorch absent de ce Python — relance via .venv/ ...")
        os.environ["SNAKE_IA_RELANCE"] = "1"
        os.execv(venv, [venv, os.path.abspath(__file__)] + sys.argv[1:])
    raise SystemExit("PyTorch est introuvable et il n'y a pas de .venv/ ici. "
                     "Voir la section Environnement du README.")


_assurer_environnement()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================================
# CONSTANTES — reprises telles quelles du jeu de base (serpent-algo.py)
# ==========================================================================

GRID_SIZE = 15          # imposé par le sujet, ne pas modifier
CELL_SIZE = 30
GAME_SPEED = 5          # imposé par le sujet, ne pas modifier

# Vitesse d'AFFICHAGE de la démo, en images/seconde. GAME_SPEED ci-dessus n'est
# pas touché : il reste la référence du sujet et c'est toujours lui qui sert à
# convertir les pas en secondes de jeu dans les mesures. DEMO_SPEED ne change
# que la cadence de rafraîchissement de la fenêtre — mêmes coups, même partie,
# même score, simplement regardés plus vite. 40 est la valeur de la référence
# Patrick Loeber dont le sujet s'inspire.
# Avec le risque borné, une partie fait ~8 200 pas : à 60 img/s cela donne
# environ 2 minutes de démonstration, et sans phase morte.
# GAME_SPEED reste à 5 et demeure la référence de toutes les mesures :
# seule la cadence de rafraîchissement de la fenêtre change.
DEMO_SPEED = 60

# Le jeu de base fait « % GRID_SIZE » dans move() : le serpent TRAVERSE les
# murs et ressort de l'autre côté. check_wall_collision() ne peut donc jamais
# se déclencher, ce que le prof documente lui-même par « ne fonctionne pas
# volontairement ». Le fichier de base fait foi : on joue sur un tore, et seule
# l'auto-morsure tue. Passer à False pour retrouver des murs mortels.
WRAP = True
SCORE_PANEL_HEIGHT = 80
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions, dans l'ordre horaire : indispensable pour les actions relatives.
RIGHT, DOWN, LEFT, UP = (1, 0), (0, 1), (-1, 0), (0, -1)
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]

# --- Barèmes de récompense ------------------------------------------------
# "sujet"    : valeurs exactes de la diapo « Système de récompenses ».
# "efficace" : optimise le SCORE PAR SECONDE DE JEU. Deux changements.
#   1. Se déplacer coûte (-0.02) au lieu de rapporter (+0.1). Le barème du
#      sujet paie l'agent pour rester en vie sans manger, ce qui est
#      exactement l'inverse d'un bon ratio score/temps.
#   2. Récompense de rapprochement : se rapprocher de la pomme rapporte un
#      peu, s'en éloigner coûte un peu plus. L'asymétrie est volontaire —
#      symétriques, un aller-retour serait gratuit et l'agent oscillerait.
REWARD_SCHEMES = {
    "sujet": dict(apple=10.0, death=-10.0, win=100.0,
                  move=0.1, closer=0.0, farther=0.0, faconnage=None),
    "efficace": dict(apple=10.0, death=-10.0, win=100.0,
                     move=-0.02, closer=0.3, farther=-0.4, faconnage="distance"),
    # "potentiel" : la récompense de façonnage n'est plus une prime arbitraire
    # mais dérive d'un POTENTIEL construit sur les grandeurs que l'agent perçoit
    # réellement — proximité de la pomme et espace libre :
    #     Phi(s) = a * proximité_pomme + b * espace_libre
    #     F      = gamma * Phi(s') - Phi(s)
    # Cette forme est prouvée invariante pour la politique optimale (Ng, Harada
    # & Russell, 1999) : elle accélère l'apprentissage sans déplacer la cible.
    "potentiel": dict(apple=10.0, death=-10.0, win=100.0,
                      move=-0.01, closer=0.0, farther=0.0,
                      faconnage="potentiel"),
}
PHI_POMME, PHI_ESPACE = 2.0, 1.0

# Coefficient du façonnage : F = GAMMA_FACONNAGE * Phi(s') - Phi(s).
#
# La forme théorique de Ng, Harada & Russell (1999) utilise gamma (0,9 ici), et
# c'est elle qui garantit l'invariance de la politique optimale. Mais avec
# gamma < 1, un potentiel de magnitude Phi produit une taxe CONSTANTE de
# (gamma - 1) * Phi à chaque pas, sans rapport avec le progrès accompli :
# mesurée ici à -0,18 par pas, soit -180 sur une partie de 1000 pas, quand une
# pomme vaut +10. L'agent apprend alors surtout que vivre coûte cher.
#
# On prend donc 1,0 : la récompense devient l'ACCROISSEMENT pur du potentiel,
# sans dérive. On perd la garantie formelle d'invariance, on gagne un signal
# qui récompense le progrès au lieu de punir l'existence. Compromis assumé et
# mesuré (voir README).
GAMMA_FACONNAGE = 1.0
DEFAULT_SCHEME = "sujet"

# --- Hyperparamètres du DQN ----------------------------------------------
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
GAMMA = 0.9             # facteur d'actualisation
HIDDEN_SIZE = 256
EPSILON_GAMES = 80      # nb de parties sur lesquelles l'exploration décroît

DEFAULT_ETAT = "simple"

# Meilleure configuration mesurée (cf. matrice du README). C'est elle que
# lance une invocation sans argument : le prof tape « python snake-ia.py »
# et voit tout de suite le serpent jouer son meilleur niveau.
BEST_SCHEME, BEST_ETAT, BEST_SECURITE = "potentiel", "conscient", True

# Risque borné : au-delà de ce nombre de pas sans manger, le filet lâche son
# invariant et prend le coup non létal le plus proche de la pomme.
# Mesuré sur 60 parties : le score est inchangé (181,3 contre 181,2 sans
# risque) mais la partie est 3 fois plus courte — 8 200 pas au lieu de 24 100.
# Le gain n'est donc pas un gain de points, c'est la suppression d'une phase
# morte : sans ce mécanisme, l'agent passe 72 à 78 % de la partie à tourner en
# rond, score déjà figé, en attendant que le timeout le tue.
# Seuil choisi haut à dessein : à 300 ou 600 le risque se déclenche pendant des
# attentes légitimes (99e centile des attentes productives : 257 pas) et coûte
# une dizaine de points.
BEST_RISQUE = 2000

# Copie figée du modèle de démonstration. Un entraînement en cours réécrit
# model/snake-ia-<config>.pth à chaque nouveau record ; charger ce fichier
# pendant une sauvegarde planterait la démo. On lit donc un instantané que
# rien ne réécrit.
DEMO_MODEL = "model/DEMO-snake-ia.pth"


def paths(scheme=DEFAULT_SCHEME, etat=DEFAULT_ETAT):
    """Chemins de sortie, un jeu par (barème, état), pour ne rien écraser."""
    tag = scheme if etat == "simple" else f"{scheme}-{etat}"
    return (f"model/snake-ia-{tag}.pth",
            f"model/training_log-{tag}.csv",
            f"model/training_plot-{tag}.png")


MODEL_PATH, LOG_PATH, PLOT_PATH = paths()


# ==========================================================================
# BLOC 1/3 — GAME : le jeu, piloté par play_step(action)
# ==========================================================================

class SnakeGameAI:
    """Jeu de Serpent piloté par un agent.

    Mêmes règles que serpent-algo.py : mourir contre un mur ou en se mordant,
    +1 au score par pomme. Les positions sont des tuples (hashables) pour
    pouvoir tester les collisions en O(1) via un set — l'entraînement fait
    des centaines de milliers de pas, la version liste serait trop lente.
    """

    def __init__(self, render=False, speed=GAME_SPEED, scheme=DEFAULT_SCHEME):
        self.render = render
        self.speed = speed
        self.scheme = scheme
        self.R = REWARD_SCHEMES[scheme]
        self.display = None
        if self.render:
            self._init_display()
        self.reset()

    def _init_display(self):
        global pygame
        import pygame
        pygame.init()
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake IA - Deep Q-Learning - groupe Abricot")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 40)

    def reset(self):
        """Remet une partie à zéro. Même position de départ que le jeu de base."""
        self.direction = RIGHT
        head = (GRID_SIZE // 4, GRID_SIZE // 2)
        self.head = head
        self.body = deque([head,
                           (head[0] - 1, head[1]),
                           (head[0] - 2, head[1])])
        self.occupied = set(self.body)
        self.score = 0
        self.victory = False
        self.frame_iteration = 0
        self.steps = 0          # pas cumulés : c'est la mesure du TEMPS de jeu
        # nb de pas au moment où chaque pomme a été mangée : permet de mesurer
        # « en combien de temps l'agent atteint le score N », seule comparaison
        # équitable entre deux agents de niveaux différents (le ratio brut,
        # lui, favorise celui qui meurt tôt, quand les pommes sont faciles).
        self.steps_at_score = []
        self._phi = None        # Phi(s') d'un pas est Phi(s) du suivant
        self._place_food()
        return self

    def _place_food(self):
        """Pose la pomme sur une case libre. False si la grille est pleine."""
        free = [(x, y)
                for x in range(GRID_SIZE)
                for y in range(GRID_SIZE)
                if (x, y) not in self.occupied]
        if not free:
            self.food = None
            return False
        self.food = random.choice(free)
        return True

    @staticmethod
    def wrap(point):
        """Ramène une case dans la grille par les bords (jeu torique)."""
        if not WRAP:
            return point
        return (point[0] % GRID_SIZE, point[1] % GRID_SIZE)

    def peek(self, action_idx):
        """Case où atterrirait l'action donnée, sans jouer le coup."""
        idx = CLOCKWISE.index(self.direction)
        new_dir = CLOCKWISE[(idx + [0, 1, -1][action_idx]) % 4]
        return self.wrap((self.head[0] + new_dir[0], self.head[1] + new_dir[1]))

    def _bfs(self, depart, cible=None, cap=None, depart_libre=False):
        """BFS sur les cases libres. Rend (nb atteignable, cible jointe ?).

        `depart_libre` sert quand on part de la TÊTE : elle occupe sa propre
        case, donc sans ce drapeau le BFS s'arrêterait immédiatement et
        rendrait toujours 0. C'est exactement le bug qui rendait muettes les
        features « espace total » et « queue joignable » du prototype.
        """
        depart = self.wrap(depart)
        if not depart_libre and self.is_collision(depart):
            return 0, False
        if cap is None:
            cap = GRID_SIZE * GRID_SIZE
        queue_cell = self.body[-1]
        vus = bytearray(GRID_SIZE * GRID_SIZE)
        vus[depart[1] * GRID_SIZE + depart[0]] = 1
        f, n, trouve = deque([depart]), 0, False
        while f and n < cap:
            x, y = f.popleft()
            n += 1
            if cible is not None and (x, y) == cible:
                trouve = True
            for dx, dy in CLOCKWISE:
                nx, ny = x + dx, y + dy
                if WRAP:
                    nx, ny = nx % GRID_SIZE, ny % GRID_SIZE
                elif not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                    continue
                i = ny * GRID_SIZE + nx
                if not vus[i] and ((nx, ny) not in self.occupied
                                   or (nx, ny) == queue_cell):
                    vus[i] = 1
                    f.append((nx, ny))
        return n, trouve

    def offset(self, p, q):
        """Déplacement signé le plus court de p vers q, axe par axe.

        Sur un tore, aller « à gauche » peut être le chemin court vers une case
        située à droite : l'offset signé le dit, 4 booléens non.
        """
        out = []
        for a, b in ((p[0], q[0]), (p[1], q[1])):
            d = (b - a) % GRID_SIZE if WRAP else (b - a)
            if WRAP and d > GRID_SIZE // 2:
                d -= GRID_SIZE
            out.append(d)
        return out[0], out[1]

    def potentiel(self):
        """Phi(s), bâti sur les mêmes grandeurs que celles que l'agent perçoit."""
        if not self.food:
            return 0.0
        proximite = 1.0 - self._food_distance(self.head) / (GRID_SIZE - 1)
        espace = self._bfs(self.head, depart_libre=True)[0]
        return (PHI_POMME * proximite
                + PHI_ESPACE * espace / (GRID_SIZE * GRID_SIZE))

    def reachable(self, start, cap=None):
        """Nombre de cases atteignables depuis `start`, plafonné à `cap`.

        C'est la réponse à « si je vais là, est-ce que je m'enferme ? ».
        L'agent à 11 booléens ne voit le danger qu'à UNE case : il évite
        parfaitement les murs mais entre tout droit dans une poche fermée par
        son propre corps, et c'est ce qui le tue passé une vingtaine de pommes.

        Le plafond est une optimisation qui ne change pas la décision : dès
        qu'on a trouvé autant de cases libres que le serpent est long, on sait
        qu'il y a la place de manœuvrer, inutile de compter le reste.
        """
        if cap is None:
            cap = len(self.body) + 1
        return self._bfs(start, cap=cap)[0]

    def _food_distance(self, point):
        """Distance à la pomme (0 si la grille est pleine).

        Sur un tore, passer par le bord est souvent plus court que traverser
        la grille : la distance sur chaque axe est le minimum entre le trajet
        direct et le trajet par le bord.
        """
        if not self.food:
            return 0
        dx = abs(point[0] - self.food[0])
        dy = abs(point[1] - self.food[1])
        if WRAP:
            dx = min(dx, GRID_SIZE - dx)
            dy = min(dy, GRID_SIZE - dy)
        return dx + dy

    def is_collision(self, point=None):
        """Mur ou corps. Sert aussi à l'agent pour lire les dangers.

        La case de la queue est considérée LIBRE : à ce pas de jeu elle va se
        libérer, et le jeu de base retire la queue avant de tester la collision.
        Sans cette règle l'agent ne peut pas suivre sa propre queue, ce qui est
        exactement la manœuvre qui permet de survivre quand le serpent est long.
        (La pomme n'apparaissant jamais sur le corps, entrer sur la queue
        n'est jamais un cas de croissance : la règle est donc toujours sûre.)
        """
        if point is None:
            point = self.head
        x, y = point
        if WRAP:
            point = (x % GRID_SIZE, y % GRID_SIZE)
        elif x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return True
        if point not in self.occupied:
            return False
        return point != self.body[-1]

    def play_step(self, action):
        """Joue un pas de jeu.

        action : vecteur one-hot [tout_droit, tourner_droite, tourner_gauche].
        Retourne (reward, game_over, score).
        """
        self.frame_iteration += 1
        self.steps += 1

        # distance à la pomme AVANT le déplacement (pour la récompense de
        # rapprochement du barème "efficace")
        dist_before = self._food_distance(self.head)
        if self.R["faconnage"] == "potentiel":
            phi_avant = self._phi if self._phi is not None else self.potentiel()

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        # 1. déplacement
        self._move(action)

        # 2. fin de partie ? (collision, ou l'agent tourne en rond)
        game_over = False
        timeout = self.frame_iteration > 100 * len(self.body)
        if self.is_collision(self.new_head) or timeout:
            game_over = True
            if self.render:
                self._update_ui(game_over=True)
                self.clock.tick(self.speed)
            return self.R["death"], game_over, self.score

        # 3. la tête avance vraiment.
        # La queue est retirée AVANT d'ajouter la tête : si la tête vient
        # occuper l'ancienne case de queue, l'ordre inverse effacerait la tête
        # du set des cases occupées.
        growing = (self.new_head == self.food)
        if not growing:
            tail = self.body.pop()
            self.occupied.discard(tail)

        self.head = self.new_head
        self.body.appendleft(self.head)
        self.occupied.add(self.head)

        # 4. pomme mangée ?
        if growing:
            self.score += 1
            self.steps_at_score.append(self.steps)
            reward = self.R["apple"]
            self.frame_iteration = 0          # on relance le compteur anti-boucle
            if not self._place_food():        # plus une case libre -> victoire
                self.victory = True
                game_over = True
                reward = self.R["win"]
        else:
            reward = self.R["move"]
            if self.R["faconnage"] == "distance":
                dist_after = self._food_distance(self.head)
                if dist_after < dist_before:
                    reward += self.R["closer"]
                else:
                    reward += self.R["farther"]

        if self.R["faconnage"] == "potentiel" and not game_over:
            self._phi = self.potentiel()
            reward += GAMMA_FACONNAGE * self._phi - phi_avant

        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, game_over, self.score

    def _move(self, action):
        """Applique une action RELATIVE et calcule la future tête.

        [1,0,0] tout droit · [0,1,0] tourner à droite · [0,0,1] tourner à gauche.

        Pourquoi relatif et non les 4 directions absolues du PDF : l'état
        contient « danger en face / à droite / à gauche », qui est déjà exprimé
        dans le repère du serpent. Garder le même repère pour les actions rend
        la correspondance état->action directe, et supprime par construction le
        demi-tour interdit. C'est aussi le choix de Patrick Loeber, dont le
        sujet s'inspire explicitement.
        """
        idx = CLOCKWISE.index(self.direction)
        if action[0] == 1:
            new_dir = CLOCKWISE[idx]                    # tout droit
        elif action[1] == 1:
            new_dir = CLOCKWISE[(idx + 1) % 4]          # droite (sens horaire)
        else:
            new_dir = CLOCKWISE[(idx - 1) % 4]          # gauche

        self.direction = new_dir
        self.new_head = self.wrap((self.head[0] + new_dir[0],
                                   self.head[1] + new_dir[1]))

    # ---------------- affichage (identique au jeu de base) ----------------

    def _update_ui(self, game_over=False):
        self.display.fill(GRIS_FOND)
        pygame.draw.rect(self.display, NOIR,
                         (0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH))

        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.display, GRIS_GRILLE,
                             (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
        for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.display, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

        if self.food:
            rect = pygame.Rect(self.food[0] * CELL_SIZE,
                               self.food[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.display, ROUGE, rect, border_radius=5)

        for segment in list(self.body)[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE,
                               segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.display, VERT, rect)
            pygame.draw.rect(self.display, NOIR, rect, 1)

        head_rect = pygame.Rect(self.head[0] * CELL_SIZE,
                                self.head[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                                CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.display, ORANGE, head_rect)
        pygame.draw.rect(self.display, NOIR, head_rect, 2)

        info = f"Score: {self.score}"
        self.display.blit(self.font.render(info, True, BLANC), (10, 20))
        fill = len(self.body) / (GRID_SIZE * GRID_SIZE) * 100
        fill_txt = self.font.render(f"Remplissage: {fill:.1f}%", True, BLANC)
        self.display.blit(fill_txt, (SCREEN_WIDTH // 2 - fill_txt.get_width() // 2, 20))

        if game_over:
            txt = self.font.render("GAME OVER", True, ROUGE)
            self.display.blit(txt, txt.get_rect(center=(SCREEN_WIDTH // 2,
                                                        SCREEN_HEIGHT // 2)))
        pygame.display.flip()


# ==========================================================================
# BLOC 2/3 — MODEL : le réseau (torch) et son entraîneur
# ==========================================================================

class Linear_QNet(nn.Module):
    """Réseau dense 11 -> 256 -> 3 : estime la Q-valeur de chaque action."""

    def __init__(self, input_size=11, hidden_size=HIDDEN_SIZE, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

    def predict(self, state):
        """Retourne l'action one-hot jugée la meilleure pour cet état."""
        with torch.no_grad():
            q_values = self(torch.tensor(state, dtype=torch.float))
        action = [0, 0, 0]
        action[int(torch.argmax(q_values).item())] = 1
        return action

    def q_order_hasard(self, state):
        """Classement aléatoire, pour mesurer ce que le filet fait TOUT SEUL.

        Si le filet avec une politique aléatoire atteint le même score qu'avec
        la politique apprise, c'est que l'apprentissage n'apporte rien et que
        le résultat est entièrement algorithmique. C'est le contrôle honnête.
        """
        return random.sample([0, 1, 2], 3)

    def q_order(self, state):
        """Indices des actions, de la meilleure à la pire selon le réseau."""
        with torch.no_grad():
            q = self(torch.tensor(state, dtype=torch.float))
        return torch.argsort(q, descending=True).tolist()

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path=MODEL_PATH):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()
        return self


class QTrainer:
    """Descente de gradient sur l'équation de Bellman.

        Q(s, a) <- r                          si l'état est terminal
        Q(s, a) <- r + gamma * max_a' Q(s', a')  sinon
    """

    def __init__(self, model, lr=LEARNING_RATE, gamma=GAMMA):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if state.dim() == 1:            # un seul échantillon -> on ajoute la dim batch
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)

        # Version vectorisée. La boucle Python d'origine appelait
        # self.model(next_state[i]) UNE FOIS PAR ÉCHANTILLON : sur un lot de
        # 1000, cela faisait 1000 passes avant séparées là qu'une seule passe
        # groupée suffit. Mesuré : 218 ms -> 4,7 ms, soit x47.
        #
        # Le calcul est strictement identique (écart max 4e-08 sur les poids
        # après 20 pas). En particulier on garde DEUX choix de l'original :
        #   - q_next est calculé AVEC gradient (pas de torch.no_grad),
        #   - target est un clone NON détaché de pred.
        # Le gradient remonte donc dans le terme de bootstrap. Ce n'est pas le
        # DQN canonique, qui fige la cible ; détacher changerait la dynamique
        # d'apprentissage, donc les résultats. Ce serait une autre expérience,
        # pas une optimisation — à mesurer séparément si on veut la tenter.
        q_next = self.model(next_state).max(1).values
        pas_fini = torch.tensor([not d for d in done], dtype=torch.float)
        target = pred.clone()
        lignes = torch.arange(len(target))
        target[lignes, action.argmax(1)] = reward + self.gamma * q_next * pas_fini

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ==========================================================================
# BLOC 3/3 — AGENT : fait le lien entre le jeu et le modèle
# ==========================================================================

class Agent:
    STATE_SIZES = {"simple": 11, "etendu": 14, "conscient": 16}

    def __init__(self, model=None, etat="simple", politique="reseau"):
        self.etat = etat
        self.politique = politique      # "reseau" ou "hasard" (contrôle)
        self.n_games = 0
        self.epsilon = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = (model if model is not None
                      else Linear_QNet(input_size=self.STATE_SIZES[etat]))
        self.trainer = QTrainer(self.model)

    def get_state(self, game):
        return self.build_state(game, self.etat)

    @staticmethod
    def build_state(game, etat="simple"):
        """Les 11 booléens de l'état, dans l'ordre de la diapo « Les états ».

        En mode « etendu », 3 réels supplémentaires : la proportion de cases
        encore atteignables après chaque action (tout droit / droite / gauche),
        rapportée à la longueur du serpent et plafonnée à 1. Lire 0 signifie
        « cette action me tue », lire 1 « j'ai de la place pour tout mon corps ».
        """
        head = game.head
        dir_r = game.direction == RIGHT
        dir_d = game.direction == DOWN
        dir_l = game.direction == LEFT
        dir_u = game.direction == UP

        # les 3 cases voisines dans le repère du serpent
        straight = (head[0] + game.direction[0], head[1] + game.direction[1])
        idx = CLOCKWISE.index(game.direction)
        d_right = CLOCKWISE[(idx + 1) % 4]
        d_left = CLOCKWISE[(idx - 1) % 4]
        straight = game.wrap(straight)
        right = game.wrap((head[0] + d_right[0], head[1] + d_right[1]))
        left = game.wrap((head[0] + d_left[0], head[1] + d_left[1]))

        food = game.food if game.food else head

        state = [
            # --- Danger (3) ---
            game.is_collision(straight),      # danger en face
            game.is_collision(right),         # danger à droite
            game.is_collision(left),          # danger à gauche
            # --- Direction (4) ---
            dir_l, dir_r, dir_u, dir_d,
            # --- Pomme (4) ---
            food[0] < head[0],                # pomme à gauche
            food[0] > head[0],                # pomme à droite
            food[1] < head[1],                # pomme en haut
            food[1] > head[1],                # pomme en bas
        ]
        state = [float(v) for v in state]

        if etat == "etendu":
            cap = len(game.body) + 1
            state += [min(1.0, game.reachable(straight, cap) / cap),
                      min(1.0, game.reachable(right, cap) / cap),
                      min(1.0, game.reachable(left, cap) / cap)]
            return np.array(state, dtype=float)

        if etat == "conscient":
            # Remplace les 4 booléens de position de pomme par un offset SIGNÉ
            # dans le repère du serpent, et ajoute la conscience de l'espace.
            cases = GRID_SIZE * GRID_SIZE
            L = len(game.body)
            cap = L + 1
            d_s = game.direction
            if game.food:
                ox, oy = game.offset(head, game.food)
                demi = GRID_SIZE // 2
                devant = (ox * d_s[0] + oy * d_s[1]) / demi
                cote = (ox * d_right[0] + oy * d_right[1]) / demi
                dnorm = game._food_distance(head) / (GRID_SIZE - 1)
            else:
                devant = cote = dnorm = 0.0
            # depart_libre=True : on part de la tête, qui occupe sa propre case
            libre, queue_ok = game._bfs(head, cible=game.body[-1],
                                        depart_libre=True)
            return np.array([
                float(game.is_collision(straight)),      # danger en face
                float(game.is_collision(right)),         # danger à droite
                float(game.is_collision(left)),          # danger à gauche
                min(1.0, game.reachable(straight, cap) / cap),
                min(1.0, game.reachable(right, cap) / cap),
                min(1.0, game.reachable(left, cap) / cap),
                float(dir_l), float(dir_r), float(dir_u), float(dir_d),
                devant, cote, dnorm,                     # où est vraiment la pomme
                L / cases,                               # place que j'occupe
                float(queue_ok),                         # puis-je rejoindre ma queue
                libre / cases,                           # espace total accessible
            ], dtype=float)

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Rejoue un lot d'expériences passées (experience replay).

        Sans ça, les transitions consécutives sont très corrélées et le réseau
        oublie ce qu'il a appris il y a 50 parties.
        """
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = list(self.memory)
        if not sample:
            return
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def safe_action(self, state, game, risque_apres=None):
        """FILET DE SÉCURITÉ DÉTERMINISTE (hybride, à déclarer).

        Le réseau garde la main : on suit son classement des actions et on
        retient la PREMIÈRE qui ne soit pas un suicide. Une action est rejetée
        si elle tue immédiatement, ou si elle mène dans une poche trop petite
        pour contenir le serpent (flood-fill) — c'est-à-dire un piège différé.

        Ce n'est pas un algorithme de recherche de chemin : il ne choisit
        jamais la direction, il ne fait qu'opposer un veto. Si toutes les
        actions sont condamnées, on rend la meilleure selon le réseau.
        """
        order = (self.model.q_order_hasard(state) if self.politique == "hasard"
                 else self.model.q_order(state))
        besoin = len(game.body)
        queue = game.body[-1]

        # RISQUE BORNÉ. Mesure faite sur trois parties : l'agent joue à ~30 pas
        # par pomme pendant le premier quart, puis atteint un état où plus aucun
        # coup ne préserve l'invariant ET ne progresse vers la pomme. Il tourne
        # alors en rond pendant EXACTEMENT la durée du timeout — 72 à 78 % de la
        # partie — et meurt sans avoir rien tenté. Le score est figé depuis
        # longtemps ; l'attente ne coûte rien mais ne rapporte rien non plus.
        #
        # Au-delà de `risque_apres` pas sans manger, on lâche donc l'invariant
        # et on prend le coup NON LÉTAL qui rapproche le plus de la pomme. Ce
        # n'est pas un suicide : la mort immédiate reste exclue. On accepte
        # seulement de pouvoir s'enfermer, ce qui au pire avance une fin déjà
        # certaine.
        if risque_apres is not None and game.frame_iteration > risque_apres:
            meilleur, d_min = None, float("inf")
            for idx in order:
                cible = game.peek(idx)
                if game.is_collision(cible):
                    continue
                d = game._food_distance(cible)
                if d < d_min:
                    meilleur, d_min = idx, d
            if meilleur is not None:
                return self._one_hot(meilleur)

        # Deux critères, du plus fort au plus faible :
        #
        #   1. QUEUE JOIGNABLE — après ce coup, existe-t-il encore un chemin de
        #      la tête jusqu'à sa propre queue ? C'est l'invariant de survie
        #      classique du Snake : tant qu'il tient, le serpent peut toujours
        #      suivre sa queue indéfiniment, donc il n'est jamais piégé.
        #   2. ASSEZ DE PLACE — la poche contient au moins la longueur du corps.
        #      Plus faible : une poche peut être assez grande et pourtant sans
        #      issue, parce que la queue n'y est pas.
        #
        # On applique (1) d'abord, puis (2) en repli, puis le plus d'espace.
        avec_queue, avec_place, repli, meilleure_place = None, None, None, -1
        for idx in order:
            cible = game.peek(idx)
            if game.is_collision(cible):
                continue
            place, queue_ok = game._bfs(cible, cible=queue, cap=None)
            if queue_ok and avec_queue is None:
                avec_queue = idx                 # critère fort satisfait
            if place >= besoin and avec_place is None:
                avec_place = idx                 # critère faible satisfait
            if place > meilleure_place:
                repli, meilleure_place = idx, place

        if avec_queue is not None:
            return self._one_hot(avec_queue)
        if avec_place is not None:
            return self._one_hot(avec_place)
        # Plus aucun coup ne garantit la survie : on prend celui qui laisse le
        # plus d'espace. Repousser l'échéance suffit souvent, la queue avançant
        # elle rouvre parfois le passage au coup suivant.
        return self._one_hot(repli if repli is not None else order[0])

    @staticmethod
    def _one_hot(idx):
        a = [0, 0, 0]
        a[idx] = 1
        return a

    def get_action(self, state, greedy=False):
        if self.politique == "hasard":
            return self._one_hot(random.randint(0, 2))

        """Compromis exploration / exploitation.

        L'exploration décroît avec le nombre de parties : au début l'agent joue
        beaucoup au hasard, après ~80 parties il suit presque toujours le réseau.
        """
        if greedy:
            return self.model.predict(state)

        self.epsilon = EPSILON_GAMES - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            action[random.randint(0, 2)] = 1
        else:
            action = self.model.predict(state)
        return action


# ==========================================================================
# ENTRAÎNEMENT / ÉVALUATION
# ==========================================================================

def save_plot(scores, mean_scores, path=None):
    if path is None:
        path = PLOT_PATH
    """Courbe d'apprentissage — c'est la « courbe » demandée par le sujet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Snake DQN — groupe Abricot")
    plt.xlabel("Partie")
    plt.ylabel("Score")
    plt.plot(scores, linewidth=0.8, alpha=0.5, label="Score par partie")
    plt.plot(mean_scores, linewidth=2, label="Moyenne cumulée")

    # Moyenne glissante sur 50 parties : la moyenne cumulée traîne derrière le
    # niveau réel de l'agent, puisqu'elle inclut encore les parties du début.
    window = 50
    if len(scores) >= window:
        rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(scores)), rolling, linewidth=2,
                 label=f"Moyenne glissante ({window} parties)")
    if scores:
        plt.axhline(max(scores), color="red", linestyle="--", linewidth=0.8,
                    label=f"Record : {max(scores)}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def train(n_games=300, render=False, resume=False, scheme=DEFAULT_SCHEME,
          etat=DEFAULT_ETAT):
    model_path, log_path, plot_path = paths(scheme, etat)
    scores, mean_scores, ratios = [], [], []
    total_score, record = 0, 0

    agent = Agent(etat=etat)
    if resume and os.path.exists(model_path):
        agent.model.load(model_path)
        agent.model.train()
        # Sans cette ligne, epsilon repartirait à son maximum et l'agent
        # rejouerait 80 parties au hasard, détruisant le modèle qu'on reprend.
        agent.n_games = EPSILON_GAMES
        print(f"Modèle repris depuis {model_path} "
              f"(exploration déjà éteinte)")
    cible_parties = agent.n_games + n_games

    game = SnakeGameAI(render=render, scheme=scheme)
    start = time.time()

    os.makedirs("model", exist_ok=True)
    # En reprise on complète le log au lieu de l'écraser, sinon la courbe des
    # parties précédentes est perdue.
    ajout = resume and os.path.exists(log_path)
    log_file = open(log_path, "a" if ajout else "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if not ajout:
        log.writerow(["partie", "score", "record", "moyenne",
                      "pas", "pas_par_pomme", "score_par_sec_jeu", "secondes"])

    print(f"Entraînement sur {n_games} parties | barème « {scheme} » "
          f"| état « {etat} » ({Agent.STATE_SIZES[etat]} entrées) "
          f"| grille {GRID_SIZE}x{GRID_SIZE}\n")

    while agent.n_games < cible_parties:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            victory = game.victory
            steps_played = game.steps
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(model_path)

            scores.append(score)
            total_score += score
            mean_scores.append(total_score / len(scores))

            # Le TEMPS DE JEU : le jeu tourne à GAME_SPEED pas/seconde, donc
            # une partie de N pas dure N / GAME_SPEED secondes à l'écran.
            game_seconds = steps_played / GAME_SPEED
            ratio = score / game_seconds if game_seconds else 0.0
            ratios.append(ratio)
            ppp = steps_played / score if score else float("nan")

            elapsed = time.time() - start
            log.writerow([agent.n_games, score, record,
                          round(mean_scores[-1], 3), steps_played,
                          round(ppp, 2), round(ratio, 4), round(elapsed, 2)])

            if victory:
                print(f"*** GRILLE REMPLIE partie {agent.n_games} "
                      f"en {elapsed:.1f}s ***")
            if agent.n_games % 10 == 0:
                log_file.flush()
                print(f"Partie {agent.n_games:4d} | score {score:3d} | "
                      f"record {record:3d} | moyenne {mean_scores[-1]:6.2f} | "
                      f"pas/pomme {ppp:5.1f} | {elapsed:6.1f}s")

    log_file.close()
    agent.model.save(model_path.replace(".pth", "_final.pth"))
    save_plot(scores, mean_scores, plot_path)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Terminé : {n_games} parties en {elapsed:.1f}s "
          f"({n_games/elapsed:.1f} parties/s)")
    print(f"Record        : {record}")
    print(f"Moyenne finale: {mean_scores[-1]:.2f}")
    print(f"Moyenne 50 der: {np.mean(scores[-50:]):.2f}")
    print(f"Ratio 50 der  : {np.mean(ratios[-50:]):.3f} pomme/s de jeu")
    print(f"Modèle  -> {model_path}")
    print(f"Courbe  -> {plot_path}")
    print(f"Log CSV -> {log_path}")
    print(f"{'='*60}")
    return scores


def play(n_games=5, speed=GAME_SPEED, model_path=None, scheme=DEFAULT_SCHEME,
         etat=DEFAULT_ETAT, securite=False, risque=None):
    """Rejoue le modèle entraîné, en mode glouton, à la vitesse du jeu de base."""
    if model_path is None:
        model_path = paths(scheme, etat)[0]
    if not os.path.exists(model_path):
        raise SystemExit(f"Aucun modèle à {model_path} — lance d'abord "
                         f"« python snake-ia.py train ».")
    net = Linear_QNet(input_size=Agent.STATE_SIZES[etat]).load(model_path)
    agent = Agent(model=net, etat=etat)
    game = SnakeGameAI(render=True, speed=speed, scheme=scheme)

    for i in range(1, n_games + 1):
        game.reset()
        done = False
        while not done:
            state = agent.get_state(game)
            action = (agent.safe_action(state, game, risque) if securite
                      else agent.get_action(state, greedy=True))
            _, done, score = game.play_step(action)
        print(f"Partie {i} : score {score} en {game.steps} pas "
              f"({game.steps/GAME_SPEED:.0f}s de jeu, "
              f"{score/(game.steps/GAME_SPEED):.2f} pomme/s)"
              + ("  *** GRILLE REMPLIE ***" if game.victory else ""))


def bench(n_games=100, model_path=None, scheme=DEFAULT_SCHEME,
          etat=DEFAULT_ETAT, securite=False, politique="reseau",
          risque=None):
    """Évaluation sans fenêtre : les chiffres à présenter au tour de table.

    Mesure le RATIO SCORE/TEMPS. Le temps est le temps de jeu, pas le temps
    de calcul : le jeu tourne à GAME_SPEED pas/seconde, donc une partie de N
    pas dure N / GAME_SPEED secondes à l'écran, que le bench tourne avec ou
    sans fenêtre.
    """
    if model_path is None:
        model_path = paths(scheme, etat)[0]
    if not os.path.exists(model_path):
        raise SystemExit(f"Aucun modèle à {model_path}.")
    net = Linear_QNet(input_size=Agent.STATE_SIZES[etat]).load(model_path)
    agent = Agent(model=net, etat=etat, politique=politique)
    game = SnakeGameAI(render=False, scheme=scheme)

    scores, steps, ratios, ppps, wins = [], [], [], [], 0
    jalons = [10, 20, 30, 40, 50]
    temps_jalon = {j: [] for j in jalons}
    start = time.time()
    for _ in range(n_games):
        game.reset()
        done = False
        while not done:
            state = agent.get_state(game)
            action = (agent.safe_action(state, game, risque) if securite
                      else agent.get_action(state, greedy=True))
            _, done, score = game.play_step(action)
        scores.append(score)
        steps.append(game.steps)
        game_seconds = game.steps / GAME_SPEED
        ratios.append(score / game_seconds if game_seconds else 0.0)
        if score:
            ppps.append(game.steps / score)
        wins += int(game.victory)
        for j in jalons:
            if len(game.steps_at_score) >= j:
                temps_jalon[j].append(game.steps_at_score[j - 1] / GAME_SPEED)

    tot_score, tot_steps = sum(scores), sum(steps)
    tot_seconds = tot_steps / GAME_SPEED

    print(f"\n=== {n_games} parties | modèle {os.path.basename(model_path)} "
          f"| état « {etat} »"
          f"{' | FILET DE SÉCURITÉ' if securite else ''}"
          f"{' | POLITIQUE ALÉATOIRE (contrôle)' if politique == 'hasard' else ''}"
          f" | calcul {time.time()-start:.1f}s ===")
    print(f"  SCORE")
    print(f"    moyen        : {np.mean(scores):.2f}  (écart-type {np.std(scores):.2f})")
    print(f"    médiane      : {np.median(scores):.0f}")
    print(f"    meilleur     : {max(scores)}    pire : {min(scores)}")
    print(f"    grilles remplies : {wins}/{n_games}")
    print(f"  TEMPS DE JEU (à {GAME_SPEED} pas/s)")
    print(f"    pas / partie : {np.mean(steps):.1f}")
    print(f"    durée moyenne: {np.mean(steps)/GAME_SPEED:.1f} s")
    print(f"  RATIO SCORE / TEMPS")
    print(f"    pas par pomme      : {np.mean(ppps):.2f}"
          f"   (plancher théorique ~{(GRID_SIZE)/2:.0f})")
    print(f"    pommes / s de jeu  : {np.mean(ratios):.3f}  (moyenne par partie)")
    print(f"    pommes / s cumulé  : {tot_score/tot_seconds:.3f}"
          f"   ({tot_score} pommes en {tot_seconds:.0f} s de jeu)")
    print(f"  TEMPS DE JEU POUR ATTEINDRE UN SCORE  (médiane, "
          f"% de parties qui y arrivent)")
    for j in jalons:
        v = temps_jalon[j]
        if v:
            print(f"    score {j:3d} : {np.median(v):6.1f} s"
                  f"   ({100*len(v)/n_games:3.0f} % des parties)")
        else:
            print(f"    score {j:3d} :      -    (  0 % des parties)")
    return dict(scores=scores, steps=steps, ratio=tot_score / tot_seconds,
                ppp=float(np.mean(ppps)), wins=wins,
                jalons={j: (float(np.median(v)) if v else None)
                        for j, v in temps_jalon.items()})


def main():
    parser = argparse.ArgumentParser(
        description="Snake joué par un agent DQN. Sans argument : démonstration "
                    "du meilleur modèle entraîné.")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="entraîner l'agent")
    p_train.add_argument("--games", type=int, default=300)
    p_train.add_argument("--render", action="store_true",
                         help="afficher la fenêtre (beaucoup plus lent)")
    p_train.add_argument("--resume", action="store_true",
                         help="repartir du modèle sauvegardé")
    p_train.add_argument("--bareme", choices=list(REWARD_SCHEMES),
                         default=DEFAULT_SCHEME,
                         help="sujet = valeurs du PDF ; efficace = score/temps")
    p_train.add_argument("--etat", choices=list(Agent.STATE_SIZES),
                         default=DEFAULT_ETAT,
                         help="simple = 11 booléens du sujet ; "
                              "etendu = + flood-fill (14 entrées)")

    p_play = sub.add_parser("play", help="regarder l'agent entraîné jouer")
    p_play.add_argument("--games", type=int, default=5)
    p_play.add_argument("--speed", type=int, default=DEMO_SPEED,
                        help=f"images/s à l'affichage (défaut {DEMO_SPEED} ; "
                             f"{GAME_SPEED} = vitesse du sujet)")
    p_play.add_argument("--model", default=None)
    p_play.add_argument("--bareme", choices=list(REWARD_SCHEMES),
                        default=BEST_SCHEME)
    p_play.add_argument("--etat", choices=list(Agent.STATE_SIZES),
                        default=BEST_ETAT)
    p_play.add_argument("--securite", action="store_true",
                        help="filet de sécurité déterministe (hybride)")
    p_play.add_argument("--risque", type=int, default=BEST_RISQUE,
                        help=f"risque borné après N pas sans manger "
                             f"(défaut {BEST_RISQUE})")

    p_bench = sub.add_parser("bench", help="évaluer sans fenêtre")
    p_bench.add_argument("--games", type=int, default=100)
    p_bench.add_argument("--model", default=None)
    p_bench.add_argument("--bareme", choices=list(REWARD_SCHEMES),
                         default=DEFAULT_SCHEME)
    p_bench.add_argument("--etat", choices=list(Agent.STATE_SIZES),
                         default=DEFAULT_ETAT)
    p_bench.add_argument("--securite", action="store_true",
                         help="filet de sécurité déterministe (hybride)")
    p_bench.add_argument("--risque", type=int, default=None,
                         help="prendre un risque après N pas sans manger "
                              "(défaut : jamais)")
    p_bench.add_argument("--politique", choices=["reseau", "hasard"],
                         default="reseau",
                         help="hasard = contrôle : mesure ce que le filet "
                              "accomplit sans politique apprise")

    args = parser.parse_args()

    if args.cmd is None:
        # Invocation nue : c'est le cas du jour de la démo.
        modele = (DEMO_MODEL if os.path.exists(DEMO_MODEL)
                  else paths(BEST_SCHEME, BEST_ETAT)[0])
        print("=" * 62)
        print("  SNAKE — agent Deep Q-Learning — groupe ABRICOT")
        print(f"  modèle : {modele}")
        print(f"  barème « {BEST_SCHEME} », état « {BEST_ETAT} », "
              f"filet de sécurité {'actif' if BEST_SECURITE else 'inactif'}")
        print(f"  jeu torique {GRID_SIZE}x{GRID_SIZE} | affichage à "
              f"{DEMO_SPEED} img/s")
        print(f"  (GAME_SPEED = {GAME_SPEED} reste la référence du sujet pour "
              f"les mesures ;")
        print("   accélérer l'affichage ne change ni les coups ni le score)")
        print("  fermer la fenêtre pour arrêter  |  vitesse : --speed N  |  "
              "plusieurs parties : play --games N")
        print("=" * 62)
        play(n_games=1, speed=DEMO_SPEED, model_path=modele,
             scheme=BEST_SCHEME, etat=BEST_ETAT, securite=BEST_SECURITE,
             risque=BEST_RISQUE)
        return

    if args.cmd == "train":
        train(args.games, args.render, args.resume, args.bareme, args.etat)
    elif args.cmd == "play":
        play(args.games, args.speed, args.model, args.bareme, args.etat,
             args.securite, args.risque)
    elif args.cmd == "bench":
        bench(args.games, args.model, args.bareme, args.etat, args.securite,
              args.politique, args.risque)


if __name__ == "__main__":
    main()
