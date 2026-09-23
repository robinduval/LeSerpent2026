#!/usr/bin/env python3
"""Snake torique piloté par un algorithme de recherche (groupe Poire).

Le socle de base est conservé tel quel (classes Snake / Apple, boucle pygame).
Le pilote automatique vit dans la section MOTEUR DE DECISION et n'utilise
aucun cycle hamiltonien : BFS temporel vers la pomme + filtres de sureté.

Usage :
    python serpent-algo.py                 # partie visuelle, pilote auto
    python serpent-algo.py --speed 20      # 20x plus rapide, chrono mis a l'echelle
    python serpent-algo.py --bench 50      # 50 parties sans affichage, statistiques
    python serpent-algo.py --selftest      # scenarios de verification cibles
"""

import argparse
import math
import random
import sys
import time
from collections import deque

try:
    import pygame
except ImportError:  # le mode --bench / --selftest ne depend pas de pygame
    pygame = None

# --- CONSTANTES DE JEU ---
# Taille de la grille (20x20)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse de jeu (images par seconde)
GAME_SPEED = 5

# Dimensions de l'écran (avec espace pour le score/timer)
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0) # Tête du serpent
VERT = (0, 200, 0)    # Corps du serpent
ROUGE = (200, 0, 0)   # Pomme
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)
BLEU = (80, 160, 255)
JAUNE = (240, 200, 60)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
NOM_DIRECTION = {UP: "haut", DOWN: "bas", LEFT: "gauche", RIGHT: "droite"}

# --- CONSTANTES DU PILOTE ---
# Plafond de calcul par coup (millisecondes). Le budget reel est le minimum entre
# cette valeur et 80 % du temps disponible entre deux coups (cf. budget_for_speed).
DEFAULT_TIME_BUDGET_MS = 100.0
# Nombre de coups simules en « chasse a la queue » pour valider un plan.
ROLLOUT_HORIZON = 48
# En dessous de ce nombre de cases libres, on considere etre en fin de partie :
# si aucun trajet vers la pomme n'existe, le serpent va se poster a cote d'elle
# au lieu de se derouler, pour saisir l'ouverture des que la queue libere le passage.
ENDGAME_CELLS = 14
# Plafond d'images par seconde en mode accelere.
MAX_RENDER_FPS = 120
# Profils de jeu : compromis entre score final et score par seconde.
#   late_area_ratio : fraction de l'espace libre que la tete doit encore atteindre
#                     apres avoir mange, une fois la grille plus pleine que longue.
#   relax_rollout / relax_area / relax_minimal : nombre de coups sans pomme au-dela
#                     duquel chaque critere de surete est abandonne (anti-blocage).
PROFILS = {
    # Mesure sur 12 parties : seul le critere « queue joignable » permet de finir
    # au-dessus de 220. Les controles d'aire font errer le serpent et le tuent avant.
    "ratio": {"late_area_ratio": 0.0, "relax_rollout": 0,
              "relax_area": 0, "relax_minimal": 300},
    "equilibre": {"late_area_ratio": 0.0, "relax_rollout": 600,
                  "relax_area": 0, "relax_minimal": 600},
    "prudent": {"late_area_ratio": 0.5, "relax_rollout": 600,
                "relax_area": 600, "relax_minimal": 600},
}
PROFIL_DEFAUT = "ratio"

# --- CLASSES DU JEU ---

class Snake:
    """Représente le serpent, sa position, sa direction et son corps."""
    def __init__(self):
        # Position initiale au centre
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        # Le corps est une liste de positions (x, y), incluant la tête
        self.body = [self.head_pos, 
                     [self.head_pos[0] - 1, self.head_pos[1]], 
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse immédiat."""
        # Vérifie que la nouvelle direction n'est pas l'inverse de l'actuelle
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case dans la direction actuelle."""
        # Calcul de la nouvelle position de la tête
        new_head_x = (self.head_pos[0] + self.direction[0]) % GRID_SIZE
        new_head_y = (self.head_pos[1] + self.direction[1]) % GRID_SIZE
        
        # Mettre à jour la tête (la nouvelle position devient la nouvelle tête)
        new_head_pos = [new_head_x, new_head_y]
        self.body.insert(0, new_head_pos)
        self.head_pos = new_head_pos

        # Si le serpent ne doit pas grandir, supprime la queue (mouvement normal)
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False # Réinitialise le drapeau

    def grow(self):
        """Prépare le serpent à grandir au prochain mouvement."""
        self.grow_pending = True
        self.score += 1

    def check_wall_collision(self):
        """Vérifie si la tête touche les bords (inactif : la grille est torique)."""
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps (Game Over si auto-morsure)."""
        # On vérifie si la position de la tête est dans le reste du corps (body[1:])
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé (mur ou morsure)."""
        return self.check_wall_collision() or self.check_self_collision()

    def state(self):
        """Etat compact et immuable consomme par le pilote automatique."""
        return tuple((c[0], c[1]) for c in self.body), bool(self.grow_pending)

    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        # Dessine le corps
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1) # Bordure

        # Dessine la tête (couleur différente)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2) # Bordure plus épaisse

class Apple:
    """Représente la pomme (nourriture) et sa position."""
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available_positions = [pos for pos in all_positions if list(pos) not in occupied_positions]
        
        if not available_positions:
            return None # Toutes les cases sont pleines (condition de Victoire)
            
        return random.choice(available_positions)

    def relocate(self, snake_body):
        """Déplace la pomme vers une nouvelle position aléatoire."""
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False

    def draw(self, surface):
        """Dessine la pomme sur la surface de jeu."""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            # Ajout d'un petit reflet pour un aspect "pomme"
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

# =====================================================================
# MOTEUR DE DECISION
# =====================================================================
#
# Modele : le corps est un tuple de cases, tete en tete de liste, plus un
# drapeau grow_pending. Toute la recherche raisonne sur ce modele, qui reproduit
# exactement move() puis grow() du socle.

def step_cell(cell, direction):
    """Case atteinte depuis `cell` en suivant `direction`, en geometrie torique."""
    return ((cell[0] + direction[0]) % GRID_SIZE,
            (cell[1] + direction[1]) % GRID_SIZE)


def neighbours(cell):
    return [step_cell(cell, d) for d in DIRECTIONS]


def opposite(direction):
    return (-direction[0], -direction[1])


def toric_distance(a, b):
    """Distance de Manhattan sur le tore : les bords sont des raccourcis."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return min(dx, GRID_SIZE - dx) + min(dy, GRID_SIZE - dy)


def release_times(body, grow_pending):
    """Instant (en coups) a partir duquel chaque case du corps redevient libre.

    La queue (indice len-1) se libere au coup 1, sauf si une croissance est en
    attente : dans ce cas le premier move() ne depile pas la queue, donc tout est
    decale d'un coup.
    """
    delay = 1 if grow_pending else 0
    length = len(body)
    released = {}
    for index, cell in enumerate(body):
        free_at = length - index + delay
        if free_at > released.get(cell, 0):
            released[cell] = free_at
    return released


def blocked_cells(body, grow_pending):
    """Cases fatales pour le prochain coup.

    Sans croissance la queue s'efface pendant le meme move() : y entrer est legal.
    Avec grow_pending elle reste en place : y entrer tue.
    """
    return frozenset(body) if grow_pending else frozenset(body[:-1])


def legal_directions(body, grow_pending, direction):
    """Coups autorises : pas de demi-tour, pas de collision immediate."""
    blocked = blocked_cells(body, grow_pending)
    banned = opposite(direction)
    return [d for d in DIRECTIONS
            if d != banned and step_cell(body[0], d) not in blocked]


def advance(body, grow_pending, direction, apple=None):
    """Applique un coup : move() puis grow() si la tete atterrit sur la pomme."""
    head = step_cell(body[0], direction)
    if grow_pending:
        new_body = (head,) + body
    else:
        new_body = (head,) + body[:-1]
    pending = apple is not None and head == apple
    return new_body, pending


def simulate_path(body, grow_pending, path, apple):
    for direction in path:
        body, grow_pending = advance(body, grow_pending, direction, apple)
    return body, grow_pending


def _rebuild(parents, cell):
    path = []
    while parents[cell] is not None:
        previous, direction = parents[cell]
        path.append(direction)
        cell = previous
    path.reverse()
    return path


def shortest_path(body, grow_pending, direction, targets, extra_blocked=frozenset(),
                  deadline=None):
    """BFS torique vers `targets`, en tenant compte de la liberation du corps.

    Une case du corps est franchissable si la tete y arrive apres son instant de
    liberation. Le parcours ne repasse jamais sur une case deja utilisee, donc le
    chemin rendu est toujours realisable ; en revanche il peut rater un chemin qui
    exigerait d'attendre devant une case encore occupee (recherche conservative).
    """
    head = body[0]
    if head in targets:
        return []
    released = release_times(body, grow_pending)
    banned = opposite(direction)
    parents = {head: None}
    frontier = deque([(head, 0)])
    while frontier:
        if deadline is not None and time.perf_counter() > deadline:
            return None
        cell, when = frontier.popleft()
        arrival = when + 1
        for move in DIRECTIONS:
            if cell == head and move == banned:
                continue
            nxt = step_cell(cell, move)
            if nxt in parents or nxt in extra_blocked:
                continue
            if released.get(nxt, 0) > arrival:
                continue
            parents[nxt] = (cell, move)
            if nxt in targets:
                return _rebuild(parents, nxt)
            frontier.append((nxt, arrival))
    return None


def reachable_area(body, grow_pending):
    """Nombre de cases libres atteignables depuis la tete (inondation torique)."""
    blocked = blocked_cells(body, grow_pending)
    seen = set()
    stack = []
    for cell in neighbours(body[0]):
        if cell not in blocked and cell not in seen:
            seen.add(cell)
            stack.append(cell)
    while stack:
        cell = stack.pop()
        for nxt in neighbours(cell):
            if nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen)


def survival_move(body, grow_pending, direction, apple=None, viser_pomme=False,
                  etats_vus=frozenset()):
    """Coup de repli : rester en vie, puis se rapprocher de la pomme.

    Filtres successifs : ne pas manger (un repas non valide par le mode chasse
    serait un pari), garder la queue joignable, conserver l'espace libre
    accessible. Parmi les coups restants on prend le plus proche de la pomme,
    puis celui qui deroule le plus le serpent.
    """
    scored = []
    avoid = frozenset() if apple is None else frozenset({apple})
    for move in legal_directions(body, grow_pending, direction):
        nbody, npending = advance(body, grow_pending, move, apple)
        tail_path = shortest_path(nbody, npending, move, {nbody[-1]}, extra_blocked=avoid)
        scored.append((
            tail_path is not None,
            reachable_area(nbody, npending),
            len(tail_path) if tail_path is not None else 0,
            toric_distance(nbody[0], apple) if apple is not None else 0,
            npending,
            move,
            (nbody, npending, move) in etats_vus,
        ))
    if not scored:
        return None

    without_apple = [c for c in scored if not c[4]]
    if without_apple:
        scored = without_apple

    # Ne pas repasser par un etat deja visite depuis la derniere pomme : c'est la
    # signature d'un cycle de survie dont le serpent ne sortirait jamais.
    inedits = [c for c in scored if not c[6]]
    if inedits:
        scored = inedits

    best_tail = max(c[0] for c in scored)
    scored = [c for c in scored if c[0] == best_tail]
    best_area = max(c[1] for c in scored)
    scored = [c for c in scored if c[1] == best_area]
    if viser_pomme:
        # Fin de partie sans trajet : se rapprocher pour saisir l'ouverture des
        # que la queue libere le verrou.
        scored.sort(key=lambda c: (c[3], -c[2]))
    else:
        # Se derouler au maximum rouvre un chemin vers la pomme plus vite que de la
        # suivre de pres : mesure a score egal, c'est moins de coups et un meilleur score.
        scored.sort(key=lambda c: (-c[2], c[3]))
    return scored[0][5]


def survives_rollout(body, grow_pending, direction, horizon, deadline=None):
    """Simule `horizon` coups de chasse a la queue sans manger : survie longue."""
    for _ in range(horizon):
        if deadline is not None and time.perf_counter() > deadline:
            return True
        move = survival_move(body, grow_pending, direction, None)
        if move is None:
            return False
        body, grow_pending = advance(body, grow_pending, move, None)
        direction = move
    return True


class SnakeBrain:
    """Pilote automatique : plus court chemin filtre par des criteres de surete."""

    def __init__(self, time_budget_ms=DEFAULT_TIME_BUDGET_MS,
                 rollout_horizon=ROLLOUT_HORIZON, profil=PROFIL_DEFAUT):
        reglages = PROFILS[profil]
        self.profil = profil
        self.late_area_ratio = reglages["late_area_ratio"]
        self.relax_rollout = reglages["relax_rollout"]
        self.relax_area = reglages["relax_area"]
        self.relax_minimal = reglages["relax_minimal"]
        self.time_budget_ms = time_budget_ms
        self.rollout_horizon = rollout_horizon
        self.endgame_cells = ENDGAME_CELLS
        self.mode = "chasse"
        self.plan_length = 0
        self.decision_ms = 0.0
        self.moves_since_apple = 0
        self._last_apple = None
        self._etats_vus = set()

    def reset(self):
        self.moves_since_apple = 0
        self._last_apple = None
        self._etats_vus = set()

    def decide(self, body, grow_pending, direction, apple):
        started = time.perf_counter()
        deadline = started + self.time_budget_ms / 1000.0

        if apple != self._last_apple:
            self._last_apple = apple
            self.moves_since_apple = 0
            self._etats_vus.clear()
        else:
            self.moves_since_apple += 1
            self._etats_vus.add((body, grow_pending, direction))

        move = self._choose(body, grow_pending, direction, apple, deadline)
        self.decision_ms = (time.perf_counter() - started) * 1000.0
        return move

    def _choose(self, body, grow_pending, direction, apple, deadline):
        legal = legal_directions(body, grow_pending, direction)
        if not legal:
            self.mode = "piege"
            self.plan_length = 0
            return direction

        plans = []
        if apple is not None:
            for first in legal:
                nbody, npending = advance(body, grow_pending, first, apple)
                if nbody[0] == apple:
                    plans.append([first])
                    continue
                rest = shortest_path(nbody, npending, first, {apple}, deadline=deadline)
                if rest is not None:
                    plans.append([first] + rest)
            plans.sort(key=lambda p: (len(p), 0 if p[0] == direction else 1))

        for plan in plans:
            final_body, final_pending = simulate_path(body, grow_pending, plan, apple)
            if self._is_viable(final_body, final_pending, plan[-1], deadline):
                self.mode = "chasse"
                self.plan_length = len(plan)
                return plan[0]

        self.mode = "survie"
        self.plan_length = 0
        libres = GRID_SIZE * GRID_SIZE - len(blocked_cells(body, grow_pending))
        viser = not plans and libres <= self.endgame_cells
        fallback = survival_move(body, grow_pending, direction, apple,
                                 viser_pomme=viser, etats_vus=self._etats_vus)
        return fallback if fallback is not None else legal[0]

    def _is_viable(self, body, grow_pending, direction, deadline):
        """Un plan n'est retenu que si l'etat d'arrivee reste vivable."""
        if not legal_directions(body, grow_pending, direction):
            return False

        if self.moves_since_apple >= self.relax_minimal:
            return True

        if shortest_path(body, grow_pending, direction, {body[-1]}) is None:
            return False

        if self.moves_since_apple < self.relax_area:
            free_total = GRID_SIZE * GRID_SIZE - len(blocked_cells(body, grow_pending))
            area = reachable_area(body, grow_pending)
            if free_total >= len(body):
                # Le serpent tient encore dans l'espace libre : il lui faut de quoi se deployer.
                if area < len(body):
                    return False
            elif area < free_total * self.late_area_ratio:
                # Fin de partie : ne pas isoler de poche que l'on devra remplir.
                return False

        if self.moves_since_apple < self.relax_rollout:
            horizon = min(self.rollout_horizon, len(body))
            if not survives_rollout(body, grow_pending, direction, horizon, deadline):
                return False

        return True


def choose_direction(snake, apple, brain):
    """Point d'entree appele a chaque tour de jeu."""
    body, grow_pending = snake.state()
    target = tuple(apple.position) if apple.position is not None else None
    return brain.decide(body, grow_pending, snake.direction, target)

# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, small_font, snake, hud):
    """Affiche le score, le chrono mis a l'echelle et l'etat du pilote."""
    
    # Dessiner le panneau de score
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Afficher le score
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 8))

    # Chrono virtuel : temps qu'aurait pris la partie jouee a la vitesse nominale.
    virtual = hud["virtual_seconds"]
    time_text = font.render(f"Temps: {int(virtual // 60):02d}:{int(virtual % 60):02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 8))
    
    # Afficher le taux de remplissage
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"{fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 8))

    couleur_mode = {"chasse": VERT, "survie": JAUNE, "piege": ROUGE, "manuel": BLEU}
    mode = hud["mode"]
    mode_text = small_font.render(f"[{mode}]", True, couleur_mode.get(mode, BLANC))
    surface.blit(mode_text, (10, 46))

    detail = f"plan {hud['plan_length']}  {hud['decision_ms']:.1f}ms"
    detail_text = small_font.render(detail, True, BLANC)
    surface.blit(detail_text, (SCREEN_WIDTH // 2 - detail_text.get_width() // 2, 46))

    ratio = snake.score / max(virtual, 1e-9)
    ratio_text = small_font.render(f"ratio {ratio:.3f}  x{hud['speed']:g}", True,
                                   VERT if ratio >= 0.15 else BLANC)
    surface.blit(ratio_text, (SCREEN_WIDTH - ratio_text.get_width() - 10, 46))

def display_message(surface, font, message, color=BLANC, y_offset=0):
    """
    Affiche un message central sur l'écran, avec un décalage vertical optionnel.
    y_offset permet de positionner plusieurs messages.
    """
    text_surface = font.render(message, True, color)
    # Applique le décalage vertical au centre
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    
    # Dessine un fond semi-transparent pour la lisibilité
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)

# --- BOUCLE PRINCIPALE DU JEU ---

def compute_pacing(speed):
    """Repartit la vitesse demandee entre images par seconde et coups par image."""
    moves_per_second = GAME_SPEED * speed
    moves_per_frame = max(1, math.ceil(moves_per_second / MAX_RENDER_FPS))
    render_fps = max(1, int(round(moves_per_second / moves_per_frame)))
    return moves_per_second, moves_per_frame, render_fps


def budget_for_speed(moves_per_second, requested_ms):
    """Budget de reflexion : 80 % du temps disponible entre deux coups.

    Plus le budget est large, plus loin va la simulation de survie ; l'algorithme
    est anytime, il rend simplement une decision moins bien verifiee s'il est coupe.
    """
    return min(requested_ms, 800.0 / max(moves_per_second, 1e-9))


def play_one_headless(brain, max_moves=200000):
    """Joue une partie complete sans affichage, avec la logique exacte du socle."""
    snake = Snake()
    apple = Apple(snake.body)
    moves = 0
    victory = False
    while moves < max_moves:
        snake.set_direction(choose_direction(snake, apple, brain))
        snake.move()
        moves += 1
        if snake.is_game_over():
            return {"score": snake.score, "moves": moves, "victory": False,
                    "length": len(snake.body)}
        if snake.head_pos == list(apple.position):
            snake.grow()
            if not apple.relocate(snake.body):
                victory = True
                break
    return {"score": snake.score, "moves": moves, "victory": victory,
            "length": len(snake.body)}


def run_bench(games, seed, budget_ms, profil=PROFIL_DEFAUT):
    random.seed(seed)
    brain = SnakeBrain(time_budget_ms=budget_ms, profil=profil)
    results = []
    started = time.perf_counter()
    for index in range(games):
        brain.reset()
        outcome = play_one_headless(brain)
        results.append(outcome)
        seconds = outcome["moves"] / GAME_SPEED
        print(f"  partie {index + 1:3d}/{games}  score={outcome['score']:3d}  "
              f"coups={outcome['moves']:6d}  duree={seconds:6.0f}s  "
              f"pommes/s={outcome['score'] / max(seconds, 1e-9):5.3f}  "
              f"{'VICTOIRE' if outcome['victory'] else 'mort'}")
    elapsed = time.perf_counter() - started
    scores = [r["score"] for r in results]
    moves = [r["moves"] for r in results]
    ratio = [r["moves"] / max(r["score"], 1) for r in results]
    rendement = [r["score"] / max(r["moves"] / GAME_SPEED, 1e-9) for r in results]
    print("\n--- Bilan ---")
    print(f"parties            : {games}")
    print(f"pommes par seconde : {sum(rendement) / games:.3f}  "
          f"(min {min(rendement):.3f}, max {max(rendement):.3f})")
    print(f"score moyen        : {sum(scores) / games:.1f}  (min {min(scores)}, max {max(scores)})")
    print(f"remplissage moyen  : {(sum(scores) / games + 3) / (GRID_SIZE ** 2) * 100:.1f} %")
    print(f"coups moyens       : {sum(moves) / games:.0f}  "
          f"({sum(moves) / games / GAME_SPEED:.0f} s de jeu a vitesse nominale)")
    print(f"coups par pomme    : {sum(ratio) / games:.2f}")
    print(f"victoires          : {sum(1 for r in results if r['victory'])}")
    print(f"temps de calcul    : {elapsed:.1f}s pour {sum(moves)} coups "
          f"({elapsed / max(sum(moves), 1) * 1000:.2f} ms/coup)")
    return 0


def run_game(args):
    """Partie visuelle pilotee par l'algorithme."""
    if pygame is None:
        print("pygame est introuvable. Utilise nix-shell Poire/shell.nix, "
              "ou lance --bench / --selftest.", file=sys.stderr)
        return 1

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Algo - Groupe Poire")
    clock = pygame.time.Clock()

    font_main = pygame.font.Font(None, 34)
    font_small = pygame.font.Font(None, 24)
    font_game_over = pygame.font.Font(None, 80)

    if args.seed:
        random.seed(args.seed)
    speed = args.speed
    auto = not args.manual
    running = True

    while running:
        snake = Snake()
        apple = Apple(snake.body)
        brain = SnakeBrain(time_budget_ms=args.budget, profil=args.profil)
        game_over = False
        victory = False
        total_moves = 0
        start_time = time.time()
        restart = False

        while running and not restart:
            moves_per_second, moves_per_frame, render_fps = compute_pacing(speed)
            brain.time_budget_ms = budget_for_speed(moves_per_second, args.budget)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_a:
                        auto = not auto
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        speed = min(speed * 2, 400)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed = max(speed / 2, 0.125)
                    elif game_over and event.key == pygame.K_SPACE:
                        restart = True
                    elif not auto and not game_over:
                        if event.key == pygame.K_UP:
                            snake.set_direction(UP)
                        elif event.key == pygame.K_DOWN:
                            snake.set_direction(DOWN)
                        elif event.key == pygame.K_LEFT:
                            snake.set_direction(LEFT)
                        elif event.key == pygame.K_RIGHT:
                            snake.set_direction(RIGHT)

            if not game_over and not victory:
                for _ in range(moves_per_frame):
                    if auto:
                        snake.set_direction(choose_direction(snake, apple, brain))
                    snake.move()
                    total_moves += 1

                    if snake.is_game_over():
                        game_over = True
                        break

                    if snake.head_pos == list(apple.position):
                        snake.grow()
                        if not apple.relocate(snake.body):
                            victory = True
                            game_over = True
                            break

            screen.fill(GRIS_FOND)
            game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
            pygame.draw.rect(screen, NOIR, game_area_rect)
            draw_grid(screen)
            apple.draw(screen)
            snake.draw(screen)

            display_info(screen, font_main, font_small, snake, {
                "mode": brain.mode if auto else "manuel",
                "plan_length": brain.plan_length,
                "decision_ms": brain.decision_ms,
                "speed": speed,
                "virtual_seconds": total_moves / GAME_SPEED,
                "real_seconds": time.time() - start_time,
            })

            if game_over:
                if victory:
                    display_message(screen, font_game_over, "VICTOIRE !", VERT)
                    display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, y_offset=100)
                else:
                    display_message(screen, font_game_over, "GAME OVER", ROUGE)
                    display_message(screen, font_main, "ESPACE pour rejouer.", BLANC, y_offset=100)

            pygame.display.flip()
            clock.tick(render_fps)

    pygame.quit()
    return 0

# =====================================================================
# VERIFICATIONS CIBLEES
# =====================================================================

def _chain_state():
    """Tete enfermee par son corps, seule issue : la case de la queue.

    Corps (tete en premier) : (5,5) (5,6) (4,6) (4,5) (4,4) (5,4) (6,4) (6,5).
    Direction UP, donc le demi-tour vers le cou (5,6) est interdit ; (5,4) et
    (4,5) sont du corps ; (6,5) est la queue.
    """
    body = ((5, 5), (5, 6), (4, 6), (4, 5), (4, 4), (5, 4), (6, 4), (6, 5))
    return body, UP


def _serpentine(count):
    cells = []
    for y in range(GRID_SIZE):
        xs = range(GRID_SIZE) if y % 2 == 0 else range(GRID_SIZE - 1, -1, -1)
        cells.extend((x, y) for x in xs)
    return cells[:count]


def _check(name, condition, detail=""):
    return (name, bool(condition), detail)


def run_selftest():
    brain = SnakeBrain()
    results = []

    # 1. Refus d'entrer dans le corps.
    body, direction = _chain_state()
    legal = legal_directions(body, False, direction)
    results.append(_check("refus d'entrer dans le corps",
                          UP not in legal and LEFT not in legal,
                          f"coups legaux = {[NOM_DIRECTION[d] for d in legal]}"))

    # 2. Interdiction du demi-tour.
    results.append(_check("demi-tour interdit",
                          DOWN not in legal,
                          "direction UP, cou en (5,6)"))

    # 3. Case de la queue SANS croissance : autorisee.
    move = brain.decide(body, False, direction, (0, 0))
    after, _ = advance(body, False, move, (0, 0))
    results.append(_check("case de la queue sans croissance : autorisee",
                          move == RIGHT and after[0] not in after[1:],
                          f"coup joue = {NOM_DIRECTION[move]}"))

    # 4. Case de la queue AVEC croissance : interdite.
    legal_grow = legal_directions(body, True, direction)
    move_grow = brain.decide(body, True, direction, (0, 0))
    results.append(_check("case de la queue avec croissance : interdite",
                          legal_grow == [] and brain.mode == "piege",
                          f"aucun coup legal, mode = {brain.mode}"))

    # 5. Absence de coup legal : pas de plantage, mode piege.
    results.append(_check("aucun coup legal : renvoie un coup sans planter",
                          move_grow in DIRECTIONS,
                          f"coup renvoye = {NOM_DIRECTION[move_grow]}"))

    # 6. Passage par les bords (tore).
    body_edge = ((0, 7), (1, 7), (2, 7))
    apple_edge = (14, 7)
    path = shortest_path(body_edge, False, LEFT, {apple_edge})
    results.append(_check("passage par le bord gauche",
                          path == [LEFT],
                          f"chemin = {[NOM_DIRECTION[d] for d in path] if path else None}"))

    # 7. Pomme proche : nombre de coups minimal.
    snake = Snake()
    apple = Apple(snake.body)
    apple.position = (snake.head_pos[0] + 2, snake.head_pos[1])
    expected = toric_distance(tuple(snake.head_pos), apple.position)
    played = 0
    fresh = SnakeBrain()
    while played < 20:
        snake.set_direction(choose_direction(snake, apple, fresh))
        snake.move()
        played += 1
        if snake.is_game_over():
            played = -1
            break
        if snake.head_pos == list(apple.position):
            break
    results.append(_check("pomme proche atteinte en un minimum de coups",
                          played == expected,
                          f"{played} coups pour une distance de {expected}"))

    # 8. Serpent presque plein : decision legale et non fatale.
    cells = _serpentine(220)
    body_full = tuple(reversed(cells))
    direction_full = (body_full[0][0] - body_full[1][0], body_full[0][1] - body_full[1][1])
    apple_full = (12, 14)
    dense_brain = SnakeBrain()
    move_full = dense_brain.decide(body_full, False, direction_full, apple_full)
    after_full, _ = advance(body_full, False, move_full, apple_full)
    results.append(_check("serpent presque plein (220/225) : coup sur",
                          move_full in legal_directions(body_full, False, direction_full)
                          and after_full[0] not in after_full[1:],
                          f"coup = {NOM_DIRECTION[move_full]}, "
                          f"{dense_brain.decision_ms:.1f} ms, mode = {dense_brain.mode}"))

    # 9. Le modele de simulation colle au moteur du socle (croissance differee).
    random.seed(7)
    snake = Snake()
    apple = Apple(snake.body)
    mirror_brain = SnakeBrain()
    body_sim, pending_sim = snake.state()
    drift = None
    for turn in range(600):
        target = tuple(apple.position)
        move = choose_direction(snake, apple, mirror_brain)
        body_sim, pending_sim = advance(body_sim, pending_sim, move, target)
        snake.set_direction(move)
        snake.move()
        if snake.is_game_over():
            drift = f"mort au coup {turn}"
            break
        if snake.head_pos == list(apple.position):
            snake.grow()
            if not apple.relocate(snake.body):
                break
        real_body, real_pending = snake.state()
        if real_body != body_sim or real_pending != pending_sim:
            drift = f"divergence au coup {turn}"
            break
    results.append(_check("simulation identique au moteur du jeu sur 600 coups",
                          drift is None, drift or "aucune divergence"))

    # 10. Budget de calcul respecte.
    random.seed(11)
    timed = SnakeBrain(time_budget_ms=15.0)
    snake = Snake()
    apple = Apple(snake.body)
    worst = 0.0
    for _ in range(400):
        snake.set_direction(choose_direction(snake, apple, timed))
        worst = max(worst, timed.decision_ms)
        snake.move()
        if snake.is_game_over():
            break
        if snake.head_pos == list(apple.position):
            snake.grow()
            if not apple.relocate(snake.body):
                break
    results.append(_check("budget par coup respecte",
                          worst < 60.0, f"pire decision = {worst:.1f} ms"))

    print("\n--- Verifications ciblees ---")
    failures = 0
    for name, ok, detail in results:
        status = "OK  " if ok else "ECHEC"
        failures += 0 if ok else 1
        print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    print(f"\n{len(results) - failures}/{len(results)} verifications passees.")
    return 0 if failures == 0 else 1


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Snake torique pilote par recherche.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="multiplicateur de vitesse ; le chrono est mis a l'echelle")
    parser.add_argument("--budget", type=float, default=DEFAULT_TIME_BUDGET_MS,
                        help="budget de calcul par coup en millisecondes")
    parser.add_argument("--profil", choices=sorted(PROFILS), default=PROFIL_DEFAUT,
                        help="compromis score / temps (defaut : %(default)s)")
    parser.add_argument("--manual", action="store_true",
                        help="demarrer en pilotage clavier (touche A pour basculer)")
    parser.add_argument("--bench", type=int, metavar="N",
                        help="jouer N parties sans affichage et afficher les statistiques")
    parser.add_argument("--seed", type=int, default=0, help="graine aleatoire du bench")
    parser.add_argument("--selftest", action="store_true",
                        help="executer les verifications ciblees")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()
    if args.bench:
        return run_bench(args.bench, args.seed, args.budget, args.profil)
    return run_game(args)


if __name__ == '__main__':
    sys.exit(main())
