import pygame
import random
import time
import argparse
import json
import math
import statistics
import hashlib
import sys
from pathlib import Path
from typing import NamedTuple
from concurrent.futures import ProcessPoolExecutor

# --- CONSTANTES DE JEU ---
# Taille de la grille (15x15)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse du serpent (déplacements par seconde)
GAME_SPEED = 5
DEFAULT_POLICY = "all128"

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

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = (UP, RIGHT, DOWN, LEFT)
BOARD_CELLS = GRID_SIZE * GRID_SIZE
MAX_SCORE = BOARD_CELLS - 3 + 1

# --- CLASSES DU JEU ---

class Snake:
    """Représente le serpent, sa position, sa direction et son corps."""
    def __init__(self):
        # Position initiale à gauche du centre
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        # Le corps est une liste de positions (x, y), incluant la tête
        self.body = [self.head_pos, 
                     [self.head_pos[0] - 1, self.head_pos[1]], 
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self._direction_changed = False
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Accepte un seul virage par déplacement, sans demi-tour."""
        if self._direction_changed or new_dir == self.direction:
            return
        if new_dir in (UP, DOWN, LEFT, RIGHT) and (-new_dir[0], -new_dir[1]) != self.direction:
            self.direction = new_dir
            self._direction_changed = True

    def next_head_position(self):
        """Calcule la prochaine case en traversant les bords de la grille."""
        return [(self.head_pos[0] + self.direction[0]) % GRID_SIZE,
                (self.head_pos[1] + self.direction[1]) % GRID_SIZE]

    def move(self):
        """Déplace le serpent d'une case dans la direction actuelle."""
        # Mettre à jour la tête (la nouvelle position devient la nouvelle tête)
        new_head_pos = self.next_head_position()
        self.body.insert(0, new_head_pos)
        self.head_pos = new_head_pos
        self._direction_changed = False

        # Si le serpent ne doit pas grandir, supprime la queue (mouvement normal)
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False # Réinitialise le drapeau

    def grow(self):
        """Prépare le serpent à grandir au prochain mouvement."""
        self.grow_pending = True
        self.score += 1

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps (Game Over si auto-morsure)."""
        # On vérifie si la position de la tête est dans le reste du corps (body[1:])
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Seule une auto-morsure termine la partie : les bords sont traversables."""
        return self.check_self_collision()

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
    def __init__(self, snake_body, rng=None):
        self._rng = rng if rng is not None else random
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        occupied = {tuple(pos) for pos in occupied_positions}
        available_positions = [pos for pos in all_positions if pos not in occupied]
        
        if not available_positions:
            return None # Toutes les cases sont pleines (condition de Victoire)
            
        return self._rng.choice(available_positions)

    def relocate(self, snake_body):
        """Déplace la pomme vers une nouvelle position aléatoire."""
        self.position = self.random_position(snake_body)
        return self.position is not None

    def draw(self, surface):
        """Dessine la pomme sur la surface de jeu."""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            # Ajout d'un petit reflet pour un aspect "pomme"
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

class GameState(NamedTuple):
    """État observable immuable ; aucune référence au générateur des pommes."""

    body: tuple
    direction: int
    grow_pending: bool
    score: int
    steps: int
    apple: tuple | None
    terminated: bool


class Game:
    """Transitions partagées par l'interface et les simulations accélérées."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self._rng.seed(seed)
        self.snake = Snake()
        self.apple = Apple(self.snake.body, rng=self._rng)
        self.steps = 0
        self.terminated = False
        self.completed = False
        return self

    def state(self):
        return GameState(
            tuple(tuple(cell) for cell in self.snake.body),
            DIRECTIONS.index(self.snake.direction),
            self.snake.grow_pending, self.snake.score, self.steps,
            self.apple.position, self.terminated,
        )

    def step(self, action):
        """Déplacement/croissance en attente, collision, consommation, nouvelle pomme.

        La pomme consommée programme la conservation de la queue au prochain
        déplacement. Il faut donc 223 pommes pour passer de 3 à 225 cases.
        """
        if self.terminated:
            raise RuntimeError("La partie est terminée ; appeler reset().")
        if not isinstance(action, int) or not 0 <= action < len(DIRECTIONS):
            raise ValueError("L'action doit être un entier entre 0 et 3.")
        self.snake.set_direction(DIRECTIONS[action])
        grew = self.snake.grow_pending
        self.snake.move()
        self.steps += 1
        collision = self.snake.is_game_over()
        ate_apple = False
        if collision:
            self.terminated = True
        elif self.apple.position is not None and self.snake.head_pos == list(self.apple.position):
            ate_apple = True
            self.snake.grow()
            if not self.apple.relocate(self.snake.body):
                self.completed = True
                self.terminated = True
        return {"ate_apple": ate_apple, "grew": grew,
                "collision": collision, "completed": self.completed}


# --- AGENT HAMILTONIEN ---

def cycle_rank(position):
    """Rang du cycle torique orienté droite/bas sur la grille impaire."""
    x, y = position
    return GRID_SIZE * y + (x + y) % GRID_SIZE


CYCLE = tuple(sorted(((x, y) for y in range(GRID_SIZE)
                      for x in range(GRID_SIZE)), key=cycle_rank))
INDEX = {cell: rank for rank, cell in enumerate(CYCLE)}
_CYCLE_NEIGHBORS = {
    cell: tuple(((cell[0] + dx) % GRID_SIZE, (cell[1] + dy) % GRID_SIZE)
                for dx, dy in DIRECTIONS)
    for cell in CYCLE
}


class RewiredCycleShield:
    """Cycle complet contenant le corps contigu, orienté queue vers tête.

    Une inversion de l'arc libre remplace seulement deux arêtes. Les deux
    nouveaux ponts doivent être voisins sur le tore. L'inversion conserve
    toutes les cases et toutes les arêtes du corps: elle donne encore un
    unique cycle. Son successeur reste donc sûr avec une croissance différée.
    Toute adoption est validée avant de modifier ce parcours de secours.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.cycle = list(CYCLE)
        self.index = dict(INDEX)
        self.last_exploration_attempts = 0

    @staticmethod
    def adjacent(first, second):
        return second in _CYCLE_NEIGHBORS.get(first, ())

    @staticmethod
    def _validate_cycle(cycle, index, body):
        if len(cycle) != BOARD_CELLS or set(cycle) != set(INDEX):
            raise ValueError("Le cycle doit couvrir exactement toutes les cases")
        if (len(index) != BOARD_CELLS
                or any(index.get(cell) != rank for rank, cell in enumerate(cycle))):
            raise ValueError("Les rangs du cycle sont incohérents")
        if any(cycle[(i + 1) % BOARD_CELLS] not in _CYCLE_NEIGHBORS[cell]
               for i, cell in enumerate(cycle)):
            raise ValueError("Une arête du cycle ne respecte pas le tore")
        if not body or len(set(body)) != len(body):
            raise ValueError("Le corps est vide ou contient une collision")
        head_rank = index.get(body[0])
        if head_rank is None or any(
                index.get(cell) != (head_rank - offset) % BOARD_CELLS
                for offset, cell in enumerate(body)):
            raise ValueError("Le corps doit être un segment contigu du cycle")

    def validate(self, state):
        """Vérifie complètement le cycle et son alignement avec l'état public."""
        self._validate_cycle(self.cycle, self.index, state.body)
        if not state.terminated and (state.apple not in self.index
                                     or state.apple in state.body):
            raise ValueError("Une partie active doit avoir une pomme libre")

    def is_aligned(self, state):
        if not state.body or len(set(state.body)) != len(state.body):
            return False
        head_rank = self.index.get(state.body[0])
        return head_rank is not None and all(
            self.index.get(cell) == (head_rank - offset) % BOARD_CELLS
            for offset, cell in enumerate(state.body))

    def distance(self, start, end):
        return (self.index[end] - self.index[start]) % BOARD_CELLS

    def _adopt(self, state, cycle):
        """Transaction: préparer et vérifier, puis seulement publier le cycle."""
        candidate = list(cycle)
        candidate_index = {cell: rank for rank, cell in enumerate(candidate)}
        self._validate_cycle(candidate, candidate_index, state.body)
        if state.apple is not None:
            new_distance = ((candidate_index[state.apple]
                             - candidate_index[state.body[0]]) % BOARD_CELLS)
            if new_distance > self.distance(state.body[0], state.apple):
                raise ValueError("Une reconfiguration ne doit pas éloigner la pomme")
        self.cycle, self.index = candidate, candidate_index

    def _options(self, state):
        if state.terminated:
            return []
        head = state.body[0]
        head_rank = self.index[head]
        apple_distance = self.distance(head, state.apple)
        successor = self.cycle[(head_rank + 1) % BOARD_CELLS]
        free_end = BOARD_CELLS - len(state.body)
        result = []
        for action, cell in enumerate(_CYCLE_NEIGHBORS[head]):
            if action == (state.direction + 2) % 4:
                continue
            rank = self.distance(head, cell)
            if rank == 1:
                result.append((action, 1, apple_distance - 1))
            elif 1 < rank <= free_end:
                after = self.cycle[(head_rank + rank + 1) % BOARD_CELLS]
                if not self.adjacent(successor, after):
                    continue
                # Inverser [1..rank] ramène la case choisie juste après la tête.
                remaining = (rank - apple_distance if apple_distance <= rank
                             else apple_distance - 1)
                if remaining < apple_distance:
                    result.append((action, rank, remaining))
        return result

    def allowed_actions(self, state):
        if state.terminated:
            return ()
        self.validate(state)
        return tuple(action for action, _, _ in self._options(state))

    def next_food_distance(self, state, action):
        for candidate, _, remaining in self._options(state):
            if candidate == action:
                return remaining
        raise ValueError("Action hors des mouvements hamiltoniens admissibles")

    def _prepare(self, state, action):
        self.validate(state)
        option = next((item for item in self._options(state)
                       if item[0] == action), None)
        if option is None:
            raise ValueError("Action hors des mouvements hamiltoniens admissibles")
        _, end, remaining = option
        head_rank = self.index[state.body[0]]
        cycle = self.cycle[head_rank:] + self.cycle[:head_rank]
        if end > 1:
            cycle[1:end + 1] = reversed(cycle[1:end + 1])
        index = {cell: rank for rank, cell in enumerate(cycle)}
        self._validate_cycle(cycle, index, state.body)
        next_head = _CYCLE_NEIGHBORS[state.body[0]][action]
        next_body = ((next_head,) + state.body if state.grow_pending
                     else (next_head,) + state.body[:-1])
        # Valider aussi le corps après le mouvement, AVANT de jouer celui-ci.
        # Une pomme déclenche seulement la croissance du mouvement suivant.
        if len(set(next_body)) != len(next_body) or any(
                index.get(cell) != (index[next_head] - offset) % BOARD_CELLS
                for offset, cell in enumerate(next_body)):
            raise ValueError("Le mouvement ne préserve pas le corps contigu")
        actual_remaining = (index[state.apple] - index[next_head]) % BOARD_CELLS
        if (actual_remaining != remaining
                or actual_remaining >= self.distance(state.body[0], state.apple)):
            raise ValueError("Le mouvement doit progresser strictement vers la pomme")
        return cycle, index

    def candidate_index(self, state, action):
        """Copie des rangs d'un candidat validé, sans modifier le parcours sûr."""
        _, index = self._prepare(state, action)
        return index

    def commit(self, state, action):
        """Valide le cycle et le déplacement prévu avant toute mutation du cycle."""
        cycle, index = self._prepare(state, action)
        self.cycle, self.index = cycle, index


def optimize_free_arc(shield, state):
    """Applique les 2-opt libres raccourcissant strictement la distance pomme.

    Avec la tête au rang zéro, inverser [i+1..j] envoie le rang a de la
    pomme vers i+j+1-a. On retient à chaque tour la plus forte réduction.
    Les extrémités satisfont 0 <= i < j <= nombre de cases libres.
    Les arêtes du corps ne sont donc jamais inversées ou supprimées.
    """
    if state.terminated:
        return 0
    free_end = BOARD_CELLS - len(state.body)
    changes = 0
    while True:
        head_rank = shield.index[state.body[0]]
        cycle = shield.cycle[head_rank:] + shield.cycle[:head_rank]
        index = {cell: rank for rank, cell in enumerate(cycle)}
        apple_rank = index[state.apple]
        best = None
        for first in range(min(apple_rank, max(0, free_end - 1))):
            for neighbor in _CYCLE_NEIGHBORS[cycle[first]]:
                last = index[neighbor]
                if not max(first + 2, apple_rank) <= last <= free_end:
                    continue
                reduction = 2 * apple_rank - first - last - 1
                if reduction <= 0 or (best is not None and reduction <= best[0]):
                    continue
                if shield.adjacent(cycle[first + 1], cycle[last + 1]):
                    best = (reduction, first, last)
        if best is None:
            return changes
        _, first, last = best
        candidate = list(cycle)
        candidate[first + 1:last + 1] = reversed(candidate[first + 1:last + 1])
        shield._adopt(state, candidate)
        changes += 1


def explore_free_arc(shield, state, attempts=64):
    """Explore au plus 64 inversions neutres, avec optimisation locale séparée.

    Le budget compte les propositions exploratoires, pas les améliorations
    déterministes de optimize_free_arc. La recherche travaille sur une copie
    et conserve sa meilleure configuration, départ inclus. Son générateur
    local dépend uniquement de l'état public; aucun accès au hasard du jeu.
    """
    if state.terminated:
        return 0
    work = RewiredCycleShield()
    work.cycle, work.index = list(shield.cycle), dict(shield.index)
    work.validate(state)
    best_cycle = list(work.cycle)
    best_distance = work.distance(state.body[0], state.apple)
    improvements = optimize_free_arc(work, state)
    distance = work.distance(state.body[0], state.apple)
    if distance < best_distance:
        best_distance, best_cycle = distance, list(work.cycle)
    rng = random.Random(state.steps * 101 + state.score * 31 + cycle_rank(state.apple))
    free_end = BOARD_CELLS - len(state.body)
    explored = 0
    for _ in range(max(0, min(64, attempts))):
        head_rank = work.index[state.body[0]]
        cycle = work.cycle[head_rank:] + work.cycle[:head_rank]
        index = {cell: rank for rank, cell in enumerate(cycle)}
        apple_rank = index[state.apple]
        neutral = []
        for first in range(max(0, free_end - 1)):
            for neighbor in _CYCLE_NEIGHBORS[cycle[first]]:
                last = index[neighbor]
                if not first + 2 <= last <= free_end:
                    continue
                if first < apple_rank <= last:
                    continue
                if work.adjacent(cycle[first + 1], cycle[last + 1]):
                    neutral.append((first, last))
        if not neutral:
            break
        first, last = rng.choice(neutral)
        candidate = list(cycle)
        candidate[first + 1:last + 1] = reversed(candidate[first + 1:last + 1])
        work._adopt(state, candidate)
        explored += 1
        improvements += optimize_free_arc(work, state)
        distance = work.distance(state.body[0], state.apple)
        if distance < best_distance:
            best_distance, best_cycle = distance, list(work.cycle)
    shield._adopt(state, best_cycle)
    shield.last_exploration_attempts = explored
    return improvements


class FixedCyclePolicy:
    """Référence algorithmique: suivre uniquement le cycle initial fixe."""
    model_id = "algorithmic_fixed_hamiltonian"
    learned = False

    def __init__(self):
        self.shield = RewiredCycleShield()

    def select_action(self, state):
        if state.terminated:
            raise ValueError("La partie est déjà terminée")
        if state.steps == 0:
            self.shield.reset()
        self.shield.validate(state)
        head_rank = self.shield.index[state.body[0]]
        successor = self.shield.cycle[(head_rank + 1) % BOARD_CELLS]
        action = _CYCLE_NEIGHBORS[state.body[0]].index(successor)
        self.shield.commit(state, action)
        return action


class ExploredRewiredGreedyPolicy:
    """Agent programmé: cycle reconfigurable, recherche64 et choix glouton."""
    model_id = "algorithmic_rewired_hamiltonian_explored_64"
    learned = False

    def __init__(self):
        self.shield = RewiredCycleShield()
        self._last_score = None

    def select_action(self, state):
        if state.terminated:
            raise ValueError("La partie est déjà terminée")
        if state.steps == 0:
            self.shield.reset()
            self._last_score = None
        self.shield.validate(state)
        if self._last_score is None or state.score != self._last_score:
            explore_free_arc(self.shield, state, attempts=64)
        else:
            optimize_free_arc(self.shield, state)
        self._last_score = state.score
        options = sorted(self.shield._options(state), key=lambda item: (item[2], item[0]))
        for action, _, _ in options:
            try:
                self.shield.commit(state, action)
            except ValueError:
                # Une proposition rejetée n'a pas modifié le cycle: le
                # successeur du parcours sûr reste disponible dans options.
                continue
            return action
        raise ValueError("Aucun mouvement sûr: invariant hamiltonien invalide")


def simulate_known_apple(state, action):
    """Avance une branche sans générer ni consulter la pomme suivante.

    La croissance en attente s'applique avant la collision et la consommation.
    La consommation termine uniquement cette simulation prospective : elle
    n'indique pas nécessairement la victoire dans le véritable moteur.
    """
    if state.terminated:
        raise ValueError("La branche prospective est déjà terminée")
    if not isinstance(action, int) or not 0 <= action < len(DIRECTIONS):
        raise ValueError("Action prospective invalide")
    # Même refus du demi-tour que Snake.set_direction, une fois par mouvement.
    if action == (state.direction + 2) % len(DIRECTIONS):
        action = state.direction
    head = _CYCLE_NEIGHBORS[state.body[0]][action]
    body = ((head,) + state.body if state.grow_pending
            else (head,) + state.body[:-1])
    if len(set(body)) != len(body):
        raise ValueError("Collision dans une branche prospective")
    consumed = head == state.apple
    return GameState(
        body, action, consumed, state.score + int(consumed), state.steps + 1,
        None if consumed else state.apple, consumed,
    ), consumed


def choose_lookahead(shield, state, depth=6, width=3, max_nodes=72,
                     optimizer=optimize_free_arc, deadline_ns=None,
                     initial_cycles=()):
    """Compare des continuations sûres sur 4 à 8 mouvements simulés.

    Chaque nœud possède son propre cycle complet et son corps simulé. Sa borne
    réalisable est mouvements simulés + distance restante sur ce cycle. Une
    reconfiguration future n'est jamais appliquée à l'état présent : seules
    la configuration et l'action certifiées du PREMIER mouvement reviennent.
    Le parcours d'entrée reste disponible et shield n'est jamais modifié.

    max_nodes borne le travail de façon reproductible. deadline_ns est une
    échéance facultative perf_counter_ns, partagée avec la recherche courante.
    L'optimizer(work_shield, public_state) doit conserver la distance ou la
    diminuer ; aucun état de générateur de pommes n'est transmis.
    initial_cycles permet de départager des cycles distincts du faisceau,
    certifiés pour cet état présent (jamais réutilisés après un mouvement).
    """
    if state.terminated:
        raise ValueError("Impossible d'anticiper une partie terminée")
    depth = max(0, min(8, int(depth)))
    width = max(1, min(8, int(width)))
    max_nodes = max(0, int(max_nodes))
    shield.validate(state)

    def clone(source):
        result = RewiredCycleShield()
        result.cycle, result.index = list(source.cycle), dict(source.index)
        return result

    def normalized(work, public_state):
        rank = work.index[public_state.body[0]]
        return tuple(work.cycle[rank:] + work.cycle[:rank])

    def expired():
        return deadline_ns is not None and time.perf_counter_ns() >= deadline_ns

    options = sorted(shield._options(state), key=lambda option: (option[2], option[0]))
    if not options:
        raise ValueError("Aucune continuation hamiltonienne sûre")
    initial_action = options[0][0]
    initial_cycle, _ = shield._prepare(state, initial_action)
    initial_estimate = options[0][2] + 1
    best_cycle, best_action = tuple(initial_cycle), initial_action
    best_estimate = initial_estimate
    stats = {"nodes": 0, "depth": 0, "initial_estimate": initial_estimate,
             "best_estimate": initial_estimate, "consumed_branches": 0,
             "expired": False, "distinct_states": 1}
    if not depth or not max_nodes or expired():
        stats["expired"] = expired()
        return best_cycle, best_action, stats

    root = clone(shield)
    if optimizer is not None:
        optimizer(root, state)
        # Vérifie aussi les callbacks alternatifs avant toute simulation.
        root.validate(state)
        if root.distance(state.body[0], state.apple) > shield.distance(
                state.body[0], state.apple):
            raise ValueError("L'optimisation prospective éloigne la pomme")
    roots = [root]
    root_keys = {normalized(root, state)}
    for candidate in initial_cycles:
        if len(roots) >= width or expired():
            break
        alternative = clone(shield)
        alternative._adopt(state, candidate)
        before_distance = alternative.distance(state.body[0], state.apple)
        if optimizer is not None:
            optimizer(alternative, state)
            alternative.validate(state)
            if alternative.distance(state.body[0], state.apple) > before_distance:
                raise ValueError("L'optimisation prospective éloigne la pomme")
        key = normalized(alternative, state)
        if key not in root_keys:
            root_keys.add(key)
            roots.append(alternative)
    # (cycle, état, premier cycle préparé, première action, profondeur)
    frontier = [(work, state, None, None, 0) for work in roots]
    seen = {(state.body, state.grow_pending, key) for key in root_keys}
    while frontier and stats["nodes"] < max_nodes and not expired():
        children = []
        for work, public_state, first_cycle, first_action, elapsed in frontier:
            if elapsed >= depth:
                continue
            options = sorted(work._options(public_state),
                             key=lambda option: (option[2], option[0]))
            for action, _, remaining in options:
                if stats["nodes"] >= max_nodes or expired():
                    break
                candidate, index = work._prepare(public_state, action)
                next_state, consumed = simulate_known_apple(public_state, action)
                child = clone(work)
                child.cycle, child.index = candidate, index
                child.validate(next_state)
                next_depth = elapsed + 1
                start_cycle = tuple(candidate) if first_cycle is None else first_cycle
                start_action = action if first_action is None else first_action
                stats["nodes"] += 1
                stats["depth"] = max(stats["depth"], next_depth)
                if consumed:
                    stats["consumed_branches"] += 1
                    estimate = next_depth
                else:
                    if optimizer is not None and not expired():
                        before_distance = child.distance(next_state.body[0], next_state.apple)
                        optimizer(child, next_state)
                        child.validate(next_state)
                        if child.distance(next_state.body[0], next_state.apple) > before_distance:
                            raise ValueError("L'optimisation prospective éloigne la pomme")
                    remaining = child.distance(next_state.body[0], next_state.apple)
                    estimate = next_depth + remaining
                if estimate < best_estimate:
                    best_cycle, best_action = start_cycle, start_action
                    best_estimate = estimate
                if not consumed and next_depth < depth:
                    key = (next_state.body, next_state.grow_pending,
                           normalized(child, next_state))
                    if key not in seen:
                        seen.add(key)
                        children.append((estimate, start_action, key[2], child,
                                         next_state, start_cycle, next_depth))
            if stats["nodes"] >= max_nodes or expired():
                break
        children.sort(key=lambda entry: entry[:3])
        frontier = [(work, public_state, first_cycle, first_action, elapsed)
                    for _, first_action, _, work, public_state, first_cycle, elapsed
                    in children[:width]]
    stats["best_estimate"] = best_estimate
    stats["distinct_states"] = len(seen)
    stats["expired"] = expired()
    # Transaction de retour : même si l'horizon a été interrompu, la décision
    # actuellement proposée a été certifiée depuis le véritable état initial.
    check = clone(shield)
    check._adopt(state, best_cycle)
    check._prepare(state, best_action)
    return best_cycle, best_action, stats


class AdvancedCyclePolicy:
    """Recherche bornée sur des cycles certifiés, sans information future.

    Chaque branche commence sur le même optimum local puis explore ses propres
    configurations neutres. Les empreintes sont des tuples normalisés sur la
    tête : aucune dépendance au hash aléatoire de Python ou au hasard du jeu.
    """

    model_id = "algorithmic_diverse_hamiltonian"
    learned = False

    def __init__(self, search_budget=64, beam_width=4, apple_neutral=False,
                 or_opt=False, adaptive=False, lookahead_depth=0,
                 time_limit_ms=None, or_lengths=(2,), lookahead_roots=False,
                 three_opt=False, stall_steps=4, search_distance=12,
                 retain_plateaus=False, four_opt=False, uphill_margin=0):
        self.shield = RewiredCycleShield()
        self.search_budget = max(0, int(search_budget))
        self.beam_width = max(1, int(beam_width))
        self.apple_neutral = bool(apple_neutral)
        self.or_opt = bool(or_opt)
        self.three_opt = bool(three_opt)
        self.four_opt = bool(four_opt)
        self.uphill_margin = max(0, int(uphill_margin))
        self.retain_plateaus = bool(retain_plateaus)
        self.stall_steps = max(1, int(stall_steps))
        self.search_distance = max(2, int(search_distance))
        self.adaptive = bool(adaptive)
        self.lookahead_depth = max(0, int(lookahead_depth))
        self.time_limit_ms = time_limit_ms
        self.lookahead_roots = bool(lookahead_roots)
        self.last_search_cycles = ()
        self._all_or_lengths = or_lengths is None
        self.or_lengths = (tuple(range(1, BOARD_CELLS)) if self._all_or_lengths else
                           tuple(sorted(set(max(1, int(n)) for n in or_lengths))))
        self._last_score = None
        self._stalled = 0
        self._last_search_step = -1000
        self.last_decision_kind = "ordinary"
        self.last_search_stats = {}

    def _normalized(self, state):
        start = self.shield.index[state.body[0]]
        return tuple(self.shield.cycle[start:] + self.shield.cycle[:start])

    @staticmethod
    def _expired(deadline):
        return deadline is not None and time.perf_counter() >= deadline

    def _two_opt(self, cycle, apple_rank, free_end, neutral=False):
        index = {cell: rank for rank, cell in enumerate(cycle)}
        stop = max(0, free_end - 1) if neutral is not False else min(apple_rank, max(0, free_end - 1))
        for first in range(stop):
            for neighbor in _CYCLE_NEIGHBORS[cycle[first]]:
                last = index[neighbor]
                if not first + 2 <= last <= free_end:
                    continue
                contains = first < apple_rank <= last
                new_rank = first + last + 1 - apple_rank if contains else apple_rank
                if neutral:
                    if new_rank != apple_rank or (contains and not self.apple_neutral):
                        continue
                elif neutral is False and new_rank >= apple_rank:
                    continue
                if cycle[last + 1] in _CYCLE_NEIGHBORS[cycle[first + 1]]:
                    yield new_rank, ("reverse", first, last)

    def _or_opt(self, cycle, apple_rank, free_end, neutral=False):
        """Déplace un bloc libre, éventuellement inversé, en vérifiant 3 ponts."""
        index = {cell: rank for rank, cell in enumerate(cycle)}
        # Un bloc [start..end] n'est retirable que si end+1 est voisin
        # de start-1. Quatre voisins suffisent donc pour toutes les longueurs;
        # les deux ponts d'insertion ci-dessous ferment ensuite un seul cycle.
        blocks = (((start, index[neighbor] - 1)
                   for start in range(1, free_end + 1)
                   for neighbor in _CYCLE_NEIGHBORS[cycle[start - 1]])
                  if self._all_or_lengths else
                  ((start, start + length - 1) for length in self.or_lengths
                   for start in range(1, free_end - length + 2)))
        for start, end in blocks:
            if not start <= end <= free_end or cycle[end + 1] not in _CYCLE_NEIGHBORS[cycle[start - 1]]:
                continue
            length = end - start + 1
            for reverse in (False, True) if length > 1 else (False,):
                first, last = (cycle[end], cycle[start]) if reverse else (cycle[start], cycle[end])
                for neighbor in _CYCLE_NEIGHBORS[first]:
                    after = index[neighbor]
                    if after > free_end or start - 1 <= after <= end:
                        continue
                    if cycle[after + 1] not in _CYCLE_NEIGHBORS[last]:
                        continue
                    if start <= apple_rank <= end:
                        offset = end - apple_rank if reverse else apple_rank - start
                        new_rank = after + 1 + offset - (length if after > end else 0)
                    elif after < apple_rank < start:
                        new_rank = apple_rank + length
                    elif end < apple_rank <= after:
                        new_rank = apple_rank - length
                    else:
                        new_rank = apple_rank
                    if neutral is None or (new_rank == apple_rank if neutral else new_rank < apple_rank):
                        yield new_rank, ("relocate", start, end, after, reverse)

    def _three_opt(self, cycle, apple_rank, free_end, neutral=False):
        """Trois coupures : A+B+C+D devient A+rev(B)+rev(C)+D.

        B et C sont deux blocs libres adjacents, chacun d'au moins deux
        cases (les blocs d'une case redonneraient un simple 2-opt).
        Les permutations gardent toutes les cases exactement une fois;
        les trois ponts vérifiés ferment une unique boucle et le corps
        situé dans A/D conserve son orientation et toutes ses arêtes.
        """
        index = {cell: rank for rank, cell in enumerate(cycle)}
        for first in range(max(0, free_end - 3)):
            for neighbor in _CYCLE_NEIGHBORS[cycle[first]]:
                middle = index[neighbor]
                if not first + 2 <= middle <= free_end - 2:
                    continue
                for other in _CYCLE_NEIGHBORS[cycle[first + 1]]:
                    last = index[other]
                    if not middle + 2 <= last <= free_end:
                        continue
                    if cycle[last + 1] not in _CYCLE_NEIGHBORS[cycle[middle + 1]]:
                        continue
                    if first < apple_rank <= middle:
                        new_rank = first + middle + 1 - apple_rank
                    elif middle < apple_rank <= last:
                        new_rank = middle + last + 1 - apple_rank
                    else:
                        new_rank = apple_rank
                    if (new_rank == apple_rank if neutral else new_rank < apple_rank):
                        yield new_rank, ("double_reverse", first, middle, last)

    def _bridges(self, cycle, apple_rank, free_end, neutral=False):
        """4-opt : A+B+C+D+E devient A+D+C+B+E sur trois blocs libres.

        Les quatre nouveaux raccordements sont vérifiés. Les blocs non
        vides gardent leur orientation, toutes les cases restent présentes
        une fois et le suffixe contenant le corps ne change jamais.
        """
        index = {cell: rank for rank, cell in enumerate(cycle)}
        neighbors = _CYCLE_NEIGHBORS
        for first in range(max(0, free_end - 2)):
            for cell in neighbors[cycle[first]]:
                third = index[cell] - 1
                if not first + 2 <= third <= free_end - 1:
                    continue
                if cycle[first + 1] not in neighbors[cycle[third]]:
                    continue
                for second in range(first + 1, third):
                    for cell in neighbors[cycle[second]]:
                        last = index[cell] - 1
                        if not third + 1 <= last <= free_end:
                            continue
                        if cycle[second + 1] not in neighbors[cycle[last]]:
                            continue
                        if first < apple_rank <= second:
                            new_rank = apple_rank + last - second
                        elif second < apple_rank <= third:
                            new_rank = apple_rank + last - third - second + first
                        elif third < apple_rank <= last:
                            new_rank = apple_rank - third + first
                        else:
                            new_rank = apple_rank
                        if neutral is None or (new_rank == apple_rank if neutral else new_rank < apple_rank):
                            yield new_rank, ("bridge", first, second, third, last)

    @staticmethod
    def _apply(cycle, descriptor):
        if descriptor[0] == "bridge":
            _, first, second, third, last = descriptor
            return (cycle[:first + 1] + cycle[third + 1:last + 1]
                    + cycle[second + 1:third + 1] + cycle[first + 1:second + 1]
                    + cycle[last + 1:])
        if descriptor[0] == "reverse":
            _, first, last = descriptor
            return cycle[:first + 1] + cycle[first + 1:last + 1][::-1] + cycle[last + 1:]
        if descriptor[0] == "double_reverse":
            _, first, middle, last = descriptor
            return (cycle[:first + 1] + cycle[first + 1:middle + 1][::-1]
                    + cycle[middle + 1:last + 1][::-1] + cycle[last + 1:])
        _, start, end, after, reverse = descriptor
        block = cycle[start:end + 1]
        if reverse:
            block = block[::-1]
        if after < start:
            return cycle[:after + 1] + block + cycle[after + 1:start] + cycle[end + 1:]
        return cycle[:start] + cycle[end + 1:after + 1] + block + cycle[after + 1:]

    def _moves(self, cycle, apple_rank, free_end, neutral=False):
        yield from self._two_opt(cycle, apple_rank, free_end, neutral)
        if self.or_opt:
            yield from self._or_opt(cycle, apple_rank, free_end, neutral)
        if self.three_opt:
            yield from self._three_opt(cycle, apple_rank, free_end, neutral)
        if self.four_opt:
            yield from self._bridges(cycle, apple_rank, free_end, neutral)

    @staticmethod
    def _certify(cycle, state):
        RewiredCycleShield._validate_cycle(
            cycle, {cell: rank for rank, cell in enumerate(cycle)}, state.body)

    def _improve(self, cycle, state, deadline=None):
        """Descente déterministe; conserve la dernière configuration certifiée."""
        cycle = tuple(cycle)
        free_end = BOARD_CELLS - len(state.body)
        apple_rank = cycle.index(state.apple)
        while apple_rank > 1 and not self._expired(deadline):
            best_rank, best_move = apple_rank, None
            for rank, move in self._moves(cycle, apple_rank, free_end):
                if rank < best_rank:
                    best_rank, best_move = rank, move
            if best_move is None:
                break
            candidate = self._apply(cycle, best_move)
            self._certify(candidate, state)
            cycle, apple_rank = candidate, best_rank
        return cycle

    def _search(self, cycle, state, deadline=None):
        if self.uphill_margin:
            return self._search_perturbed(cycle, state, deadline)
        original = tuple(cycle)
        cycle = self._improve(original, state, deadline)
        best, best_rank = cycle, cycle.index(state.apple)
        free_end = BOARD_CELLS - len(state.body)
        # Ces branches constituent plusieurs départs à budget TOTAL partagé.
        branches = [cycle for _ in range(self.beam_width)]
        seen = {original, cycle}
        rng = random.Random(state.steps * 101 + state.score * 31 + cycle_rank(state.apple))
        stats = {"candidates": 0, "unique": len(seen), "improvements": 0,
                 "timed_out": False, "branches": self.beam_width, "revisits": 0}
        for attempt in range(self.search_budget):
            if best_rank <= 1 or self._expired(deadline):
                stats["timed_out"] = self._expired(deadline)
                break
            lane = attempt % self.beam_width
            parent = branches[lane]
            apple_rank = parent.index(state.apple)
            moves = list(self._moves(parent, apple_rank, free_end, neutral=True))
            rng.shuffle(moves)
            candidate = None
            for _, move in moves:
                proposal = self._apply(parent, move)
                if proposal not in seen:
                    candidate = proposal
                    break
                stats["revisits"] += 1
            if candidate is None:
                # Une branche bloquée repart du meilleur parcours disponible.
                branches[lane] = best
                if parent == best and all(branch == best for branch in branches):
                    # Aucun départ distinct ni voisin non visité ne subsiste.
                    break
                continue
            self._certify(candidate, state)
            seen.add(candidate)
            stats["candidates"] += 1
            plateau = candidate
            candidate = self._improve(candidate, state, deadline)
            repeated_optimum = candidate in seen
            seen.add(candidate)
            # Une descente peut rejoindre un optimum déjà visité. Dans cette
            # variante, conserver le plateau distinct évite de fusionner les
            # départs ; seul le meilleur cycle est finalement adopté.
            branches[lane] = (plateau if self.retain_plateaus and repeated_optimum
                              else candidate)
            rank = candidate.index(state.apple)
            if rank < best_rank:
                best, best_rank = candidate, rank
                stats["improvements"] += 1
        stats["unique"] = len(seen)
        stats["timed_out"] |= self._expired(deadline)
        self.last_search_cycles = tuple(dict.fromkeys(
            candidate for candidate in [best, *branches]
            if candidate.index(state.apple) == best_rank))
        self.last_search_stats = stats
        self.shield.last_exploration_attempts = stats["candidates"]
        return best

    def _search_perturbed(self, cycle, state, deadline=None):
        """Détours virtuels bornés; seul le meilleur cycle est rendu au jeu."""
        cycle = self._improve(tuple(cycle), state, deadline)
        best, best_rank = cycle, cycle.index(state.apple)
        branches = [cycle] * self.beam_width
        seen = {cycle}
        rng = random.Random(state.steps * 101 + state.score * 31 + cycle_rank(state.apple))
        free_end = BOARD_CELLS - len(state.body)
        count = revisits = 0
        for attempt in range(self.search_budget):
            if best_rank <= 1 or self._expired(deadline):
                break
            lane = attempt % self.beam_width
            parent = branches[lane]
            choices = list(self._moves(parent, parent.index(state.apple), free_end, neutral=None))
            rng.shuffle(choices)
            candidate = None
            for rank, move in choices:
                if rank > best_rank + self.uphill_margin:
                    continue
                proposal = self._apply(parent, move)
                if proposal in seen:
                    revisits += 1
                    continue
                candidate = proposal
                break
            if candidate is None:
                branches[lane] = best
                continue
            self._certify(candidate, state)
            seen.add(candidate)
            count += 1
            plateau = candidate
            candidate = self._improve(candidate, state, deadline)
            repeated = candidate in seen
            seen.add(candidate)
            branches[lane] = plateau if repeated else candidate
            rank = candidate.index(state.apple)
            if rank < best_rank:
                best, best_rank = candidate, rank
        self.last_search_cycles = (best,)
        self.last_search_stats = {"candidates": count, "revisits": revisits,
                                  "unique": len(seen), "deadline_hit": self._expired(deadline)}
        self.shield.last_exploration_attempts = count
        return best

    def _new_tail_connection(self, cycle, state):
        free_end = BOARD_CELLS - len(state.body)
        if state.grow_pending or free_end < 3:
            return False
        index = {cell: rank for rank, cell in enumerate(cycle)}
        for neighbor in _CYCLE_NEIGHBORS[cycle[free_end]]:
            first = index[neighbor]
            if first < free_end - 1 and cycle[free_end + 1] in _CYCLE_NEIGHBORS[cycle[first + 1]]:
                return True
        # Un nouveau bloc Or-opt peut également se terminer à la case libérée.
        if self.or_opt:
            return any(free_end >= length and cycle[free_end + 1] in
                       _CYCLE_NEIGHBORS[cycle[free_end - length]]
                       for length in self.or_lengths)
        return False

    def select_action(self, state):
        if state.terminated:
            raise ValueError("La partie est déjà terminée")
        if state.steps == 0:
            self.shield.reset()
            self._last_score = None
            self._stalled = 0
            self._last_search_step = -1000
        self.shield.validate(state)
        deadline = (None if self.time_limit_ms is None else
                    time.perf_counter() + max(0, self.time_limit_ms) / 1000)
        original = self._normalized(state)
        original_rank = original.index(state.apple)
        cycle = self._improve(original, state, deadline)
        self._stalled = self._stalled + 1 if cycle.index(state.apple) == original_rank else 0
        changed_apple = self._last_score is None or state.score != self._last_score
        adaptive_search = (self.adaptive and self._stalled >= self.stall_steps
                           and state.steps - self._last_search_step >= self.stall_steps
                           and 50 <= state.score <= 150
                           and cycle.index(state.apple) >= self.search_distance
                           and self._new_tail_connection(cycle, state))
        self.last_search_stats = {"candidates": 0, "unique": 1,
                                  "improvements": 0, "timed_out": False}
        self.last_search_cycles = ()
        self.last_decision_kind = "search" if changed_apple or adaptive_search else "ordinary"
        if changed_apple or adaptive_search:
            cycle = self._search(cycle, state, deadline)
            self._last_search_step = state.steps
        self._last_score = state.score
        self.shield._adopt(state, cycle)
        action = None
        if self.lookahead_depth and not self._expired(deadline):
            self.last_decision_kind = "lookahead"
            def optimizer(work, simulated):
                head_rank = work.index[simulated.body[0]]
                normalized = tuple(work.cycle[head_rank:] + work.cycle[:head_rank])
                work._adopt(simulated, self._improve(normalized, simulated, deadline))
            first_cycle, action, look_stats = choose_lookahead(
                self.shield, state, depth=self.lookahead_depth, width=self.beam_width,
                max_nodes=self.lookahead_depth * self.beam_width * 4,
                optimizer=optimizer,
                deadline_ns=None if deadline is None else int(deadline * 1e9),
                initial_cycles=self.last_search_cycles if self.lookahead_roots else ())
            self.shield._adopt(state, first_cycle)
            self.last_search_stats["lookahead"] = look_stats
        if action is None:
            options = self.shield._options(state)
            action = min(options, key=lambda item: (item[2], item[0]))[0]
        self.shield.commit(state, action)
        self.last_search_stats["deadline_hit"] = self._expired(deadline)
        return action


# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, elapsed_time):
    """Affiche le score et le temps écoulé dans le panneau supérieur."""
    
    # Dessiner le panneau de score
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Afficher le score
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))

    # Afficher le temps
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))
    
    # Afficher le taux de remplissage
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 50))

def display_message(surface, font, message, color=BLANC, y_offset=0):
    """
    Affiche un message central sur l'écran, avec un décalage vertical optionnel.
    y_offset permet de positionner plusieurs messages.
    """
    text_surface = font.render(message, True, color)
    # Applique le décalage vertical au centre
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    
    # Dessine un fond pour la lisibilité
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)

# --- BOUCLE PRINCIPALE DU JEU ---

def main(seed=None, manual=False, policy_name=DEFAULT_POLICY, time_limit_ms=20):
    """Agent automatique à 5 mouvements/s ; --manual conserve le clavier."""
    pygame.init()
    
    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Classique - Socle de Base")
    clock = pygame.time.Clock()
    
    # Configuration des polices
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)
    
    # Initialisation des objets du jeu
    game = Game(seed)
    policy = make_policy(policy_name, time_limit_ms=time_limit_ms)
    snake, apple = game.snake, game.apple
    
    # Variables de jeu
    running = True
    game_over = False
    victory = False
    
    # Démarrage du chronomètre
    start_time = time.monotonic()
    elapsed_time = 0

    # --- Boucle de jeu ---
    while running:
        # 1. Gestion des Événements (Contrôles Clavier)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            
            if event.type == pygame.KEYDOWN:
                if game_over:
                    # Logique de redémarrage : seulement si le jeu est terminé
                    if event.key == pygame.K_SPACE:
                        game.reset()
                        policy = make_policy(policy_name, time_limit_ms=time_limit_ms)
                        snake, apple = game.snake, game.apple
                        game_over = False
                        victory = False
                        start_time = time.monotonic()
                        elapsed_time = 0
                elif manual:
                    # Logique de déplacement : seulement si le jeu est en cours
                    if event.key == pygame.K_UP:
                        snake.set_direction(UP)
                    elif event.key == pygame.K_DOWN:
                        snake.set_direction(DOWN)
                    elif event.key == pygame.K_LEFT:
                        snake.set_direction(LEFT)
                    elif event.key == pygame.K_RIGHT:
                        snake.set_direction(RIGHT)
        
        if not running:
            break

        # 2. Logique de Mise à Jour du Jeu
        if not game_over and not victory:
            action = (DIRECTIONS.index(snake.direction) if manual
                      else policy.select_action(game.state()))
            game.step(action)
            elapsed_time = time.monotonic() - start_time
            game_over = game.terminated
            victory = game.completed
        
        # 3. Dessin
        screen.fill(GRIS_FOND) # Fond gris pour la zone de score
        
        # Zone de jeu (décalée par la hauteur du panneau de score)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        
        draw_grid(screen)
        
        # Dessine la pomme et le serpent
        apple.draw(screen)
        snake.draw(screen)
        
        # Affiche le score et le temps
        display_info(screen, font_main, snake, elapsed_time)
        
        # Affichage des messages de fin de jeu
        if game_over:
            if victory:
                # Le premier message est centré (y_offset=0 par défaut)
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
                message_details = "ESPACE pour rejouer."
                # Le deuxième message est décalé vers le bas
                display_message(screen, font_main, message_details, BLANC, y_offset=100) 
            else:
                # Le premier message est centré (y_offset=0 par défaut)
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
                message_details = "ESPACE pour rejouer."
                # Le deuxième message est décalé vers le bas
                display_message(screen, font_main, message_details, BLANC, y_offset=100)
        
        # Mise à jour de l'affichage
        pygame.display.flip()
        
        # Contrôle la vitesse du jeu
        clock.tick(GAME_SPEED)

    pygame.quit()


# --- VALIDATION REPRODUCTIBLE, SANS ATTENTE NI RENDU ---

def latency_summary(samples):
    """Durées en ms ; percentile empirique au rang supérieur."""
    if not samples:
        return {"count": 0, "mean_ms": 0, "p95_ms": 0, "max_ms": 0}
    ordered = sorted(samples)
    return {"count": len(samples), "mean_ms": statistics.fmean(samples),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "max_ms": ordered[-1]}


# Paramètres d'ablation explicites ; les politiques historiques restent intactes.
POLICY_CONFIGS = {
    "bridge_uphill": {"search_budget": 256, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "four_opt": True, "uphill_margin": 8},
    "bridge128": {"search_budget": 128, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "four_opt": True},
    "beam64": {"search_budget": 64, "beam_width": 4},
    "neutral64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True},
    "oropt64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True},
    "adaptive64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "adaptive": True},
    "lookahead64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "adaptive": True, "lookahead_depth": 6},
    "adaptive128": {"search_budget": 128, "beam_width": 4, "apple_neutral": True, "or_opt": True, "adaptive": True},
    "adaptive256": {"search_budget": 256, "beam_width": 8, "apple_neutral": True, "or_opt": True, "adaptive": True},
    "all64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None},
    "alladaptive64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "adaptive": True},
    "all128": {"search_budget": 128, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None},
    "alllookahead64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "lookahead_depth": 6, "lookahead_roots": True},
    "three64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "three_opt": True},
    "threeadaptive64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "three_opt": True, "adaptive": True},
    "three128": {"search_budget": 128, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "three_opt": True},
    "frequent64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "three_opt": True, "adaptive": True, "stall_steps": 2, "search_distance": 8},
    "all256": {"search_budget": 256, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None},
    "all512": {"search_budget": 512, "beam_width": 8, "apple_neutral": True, "or_opt": True, "or_lengths": None},
    "three256": {"search_budget": 256, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "three_opt": True},
    "diverse64": {"search_budget": 64, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "three_opt": True, "retain_plateaus": True},
    "diverse128": {"search_budget": 128, "beam_width": 4, "apple_neutral": True, "or_opt": True, "or_lengths": None, "three_opt": True, "retain_plateaus": True},
}


def make_policy(name, time_limit_ms=None):
    if name == "fixed":
        return FixedCyclePolicy()
    if name == "explored64":
        return ExploredRewiredGreedyPolicy()
    return AdvancedCyclePolicy(**POLICY_CONFIGS[name], time_limit_ms=time_limit_ms)


def simulate(policy_name, seed, max_steps=MAX_SCORE * (BOARD_CELLS - 1),
             verify=True, timings=None, categorized_timings=None, time_limit_ms=None):
    """Même Game.step que l'interface, sans clock.tick ni pommes anticipées.

    Le chronométrage inclut select_action et ses validations internes. La
    création du snapshot, le moteur et les vérifications externes sont exclus.
    Le plafond de pas interrompt l'évaluation, jamais les règles du moteur.
    """
    if max_steps < 1:
        raise ValueError("max_steps doit être positif.")
    policy = make_policy(policy_name, time_limit_ms=time_limit_ms)
    game = Game(seed)
    samples = []
    categories = {"search": [], "ordinary": []}
    phase_steps = {"1-50": 0, "51-100": 0, "101-150": 0, "151-223": 0}
    search_totals = {"deadline_hits": 0, "candidates": 0, "revisits": 0}
    previous_score = None
    last_apple_step = 0
    longest_apple_wait = 0
    wall_start = time.perf_counter()
    if verify:
        policy.shield.validate(game.state())
    while not game.terminated and game.steps < max_steps:
        state = game.state()
        before_distance = policy.shield.distance(state.body[0], state.apple)
        started = time.perf_counter_ns()
        action = policy.select_action(state)
        duration = (time.perf_counter_ns() - started) / 1_000_000
        samples.append(duration)
        default_kind = ("search" if policy_name != "fixed" and state.score != previous_score
                        else "ordinary")
        kind = getattr(policy, "last_decision_kind", default_kind)
        kind = "ordinary" if kind == "ordinary" else "search"
        categories[kind].append(duration)
        previous_score = state.score
        search_stats = getattr(policy, "last_search_stats", {})
        search_totals["deadline_hits"] += bool(search_stats.get("deadline_hit", False))
        search_totals["candidates"] += search_stats.get("candidates", 0)
        search_totals["revisits"] += search_stats.get("revisits", 0)
        phase = ("1-50" if state.score < 50 else "51-100" if state.score < 100
                 else "101-150" if state.score < 150 else "151-223")
        phase_steps[phase] += 1
        if verify:
            policy.shield.validate(state)
            if policy.shield.distance(state.body[0], state.apple) > before_distance:
                raise AssertionError("La recherche a éloigné la pomme.")
            dx, dy = DIRECTIONS[action]
            next_head = ((state.body[0][0] + dx) % GRID_SIZE,
                         (state.body[0][1] + dy) % GRID_SIZE)
            remaining = policy.shield.distance(next_head, state.apple)
            if remaining >= before_distance:
                raise AssertionError("Le mouvement ne progresse pas vers la pomme.")
        event = game.step(action)
        if verify and not event["collision"]:
            policy.shield.validate(game.state())
        if event["ate_apple"]:
            longest_apple_wait = max(longest_apple_wait, game.steps - last_apple_step)
            last_apple_step = game.steps
    wall_seconds = time.perf_counter() - wall_start
    if timings is not None:
        timings.extend(samples)
    if categorized_timings is not None:
        for kind, values in categories.items():
            categorized_timings[kind].extend(values)
    return {
        "policy": policy_name, "seed": seed, "score": game.snake.score,
        "completed": game.completed,
        "collision": game.terminated and not game.completed,
        "interrupted": not game.terminated,
        "steps": game.steps,
        "steps_to_223": game.steps if game.completed and game.snake.score == MAX_SCORE else None,
        "seconds_x1": game.steps / GAME_SPEED,
        "wall_seconds": wall_seconds,
        "max_steps_between_apples": longest_apple_wait,
        "decision": latency_summary(samples),
        "decision_categories": {kind: latency_summary(values) for kind, values in categories.items()},
        "phase_steps": phase_steps, "search_totals": search_totals,
    }


def _benchmark_job(arguments):
    name, seed, max_steps, time_limit_ms = arguments
    samples = []
    categories = {"search": [], "ordinary": []}
    result = simulate(name, seed, max_steps=max_steps, timings=samples,
                      categorized_timings=categories, time_limit_ms=time_limit_ms)
    return result, samples, categories


def _benchmark_jobs(jobs, workers):
    if workers == 1:
        yield from map(_benchmark_job, jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(_benchmark_job, jobs)


def benchmark(seeds, output, max_steps=MAX_SCORE * (BOARD_CELLS - 1),
              policies=("explored64", DEFAULT_POLICY), time_limit_ms=None, workers=1):
    """Compare les mêmes graines ; les corps différents produisent des pommes différentes."""
    records = []
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    benchmark_started = time.perf_counter()
    timings = {name: [] for name in policies}
    categories = {name: {"search": [], "ordinary": []} for name in policies}
    jobs = [(name, seed, max_steps, time_limit_ms) for seed in seeds for name in policies]
    for result, samples, group_samples in _benchmark_jobs(jobs, workers):
        name, seed = result["policy"], result["seed"]
        timings[name].extend(samples)
        for kind, values in group_samples.items():
            categories[name][kind].extend(values)
        records.append(result)
        print(f"{name:10s} seed={seed:3d} score={result['score']:3d} "
              f"pas={result['steps']:5d} x1={result['seconds_x1']:.1f}s "
              f"complet={result['completed']} collision={result['collision']} "
              f"interruption={result['interrupted']}", flush=True)
        # Conserver aussi les parties déjà mesurées si l'évaluation est interrompue.
        Path(str(output) + ".partial.json").write_text(json.dumps({
            "source_sha256": source_hash, "seeds": list(seeds), "policies": list(policies),
            "time_limit_ms": time_limit_ms, "workers": workers, "runs": records,
        }, indent=2) + "\n")
    summary = {}
    for name, samples in timings.items():
        rows = [row for row in records if row["policy"] == name]
        complete = [row for row in rows if row["completed"]]
        best = min(rows, key=lambda row: (-row["score"], row["steps"]))
        summary[name] = {
            "games": len(rows), "scores": [row["score"] for row in rows],
            "completed": len(complete),
            "collisions": sum(row["collision"] for row in rows),
            "interruptions": sum(row["interrupted"] for row in rows),
            "mean_steps_to_223": statistics.fmean(row["steps_to_223"] for row in complete) if complete else None,
            "mean_seconds_x1": statistics.fmean(row["seconds_x1"] for row in complete) if complete else None,
            "decision": latency_summary(samples),
            "decision_categories": {kind: latency_summary(values) for kind, values in categories[name].items()},
            "median_steps_to_223": statistics.median(row["steps"] for row in complete) if complete else None,
            "p95_steps_to_223": sorted(row["steps"] for row in complete)[math.ceil(.95 * len(complete)) - 1] if complete else None,
            "max_steps_to_223": max((row["steps"] for row in complete), default=None),
            "slowest": [{key: row[key] for key in ("seed", "score", "steps", "seconds_x1")}
                        for row in sorted(rows, key=lambda row: row["steps"], reverse=True)[:5]],
            "mean_phase_steps": {phase: statistics.fmean(row["phase_steps"][phase] for row in rows)
                                 for phase in rows[0]["phase_steps"]},
            "search_totals": {key: sum(row["search_totals"][key] for row in rows)
                              for key in rows[0]["search_totals"]},
            "best": {key: best[key] for key in ("seed", "score", "steps", "seconds_x1")},
        }
    fixed, explored = summary[policies[0]], summary[policies[-1]]
    reduction = None
    if fixed["completed"] == explored["completed"] == len(seeds):
        reduction = 100 * (1 - explored["mean_steps_to_223"] / fixed["mean_steps_to_223"])
    report = {
        "protocol": {
            "seeds": list(seeds), "grid_size": GRID_SIZE, "initial_length": 3,
            "growth": "delayed_until_next_move", "max_score": MAX_SCORE,
            "game_speed": GAME_SPEED, "max_steps": max_steps,
            "invariants_checked_each_step": True, "accelerated": True,
            "seconds_x1_definition": "steps / GAME_SPEED; equivalent duration, not a measured GUI run",
            "decision_timing": "select_action including internal cycle validation; excludes snapshot, engine, render, external verification",
            "randomness": "independent seeded apple RNG, x-major free-cell order; policy sees only current immutable public state",
            "python": sys.version.split()[0],
            "source_sha256": source_hash,
            "policies": list(policies),
            "configurations": {name: POLICY_CONFIGS.get(name, {}) for name in policies},
            "time_limit_ms": time_limit_ms,
            "workers": workers,
            "latency_measurement_context": "isolated serial games" if workers == 1 else "concurrent games; confirm latency with workers=1",
            "work_budget_reproducible": True,
        },
        "summary": summary, "movement_reduction_percent": reduction,
        "benchmark_wall_seconds": time.perf_counter() - benchmark_started,
        "runs": records,
    }
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if reduction is not None:
        print(f"Réduction moyenne des déplacements : {reduction:.2f}%")
    print(f"Résultats : {output}")
    return report


def parse_seeds(value):
    """Accepte 1:21 (fin exclue) ou 1,2,3."""
    try:
        if ":" in value:
            start, stop = map(int, value.split(":"))
            seeds = list(range(start, stop))
        else:
            seeds = [int(part) for part in value.split(",")]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError
        return seeds
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Utiliser 1:21 (fin exclue) ou une liste de graines distinctes.") from exc


def cli():
    parser = argparse.ArgumentParser(description="Snake algorithmique : cycle hamiltonien, 2-opt, Or-opt et recherche à quatre branches.")
    parser.add_argument("--seed", type=int, help="Graine de la partie affichée.")
    parser.add_argument("--manual", action="store_true", help="Jouer au clavier.")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="Politique de la partie affichée (défaut all128).")
    parser.add_argument("--benchmark", action="store_true", help="Simulation accélérée, sans fenêtre.")
    parser.add_argument("--seeds", type=parse_seeds, default=list(range(1, 21)),
                        help="Graines du benchmark, défaut 1:21 (1 à 20).")
    parser.add_argument("--output", default="benchmark-results.json")
    parser.add_argument("--policies", default="explored64," + DEFAULT_POLICY, help="Politiques comparées, séparées par des virgules.")
    parser.add_argument("--time-limit-ms", type=float, default=20,
                        help="Budget mural de recherche ; 0 désactive cette limite pour une ablation déterministe.")
    parser.add_argument("--workers", type=int, default=1, help="Processus de simulation (défaut 1 pour mesurer les latences sans concurrence).")
    parser.add_argument("--max-steps", type=int, default=MAX_SCORE * (BOARD_CELLS - 1))
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps doit être positif")
    if args.workers < 1:
        parser.error("--workers doit être positif")
    if args.time_limit_ms < 0:
        parser.error("--time-limit-ms doit être positif ou nul")
    if args.benchmark:
        policies = tuple(args.policies.split(","))
        if (not policies or len(set(policies)) != len(policies)
                or any(name not in {"fixed", "explored64", *POLICY_CONFIGS} for name in policies)):
            parser.error("Politique inconnue")
        report = benchmark(args.seeds, args.output, args.max_steps, policies=policies,
                           time_limit_ms=args.time_limit_ms if args.time_limit_ms > 0 else None,
                           workers=args.workers)
        return 0 if all(row["completed"] for row in report["runs"]) else 1
    if args.policy not in {"fixed", "explored64", *POLICY_CONFIGS}:
        parser.error("Politique inconnue")
    main(seed=args.seed, manual=args.manual, policy_name=args.policy,
         time_limit_ms=args.time_limit_ms if args.time_limit_ms > 0 else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
