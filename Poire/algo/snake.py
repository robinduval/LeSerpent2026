pygame = None  # Import uniquement pour le rendu ; tests sans dépendance graphique.
import random
import time

# --- CONSTANTES DE JEU ---
# Taille de la grille (15x15)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse de jeu (déplacements par seconde)
GAME_SPEED = 5
AUTO_MODE = True
USE_SHORTCUTS = True

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

# --- SOLVEUR : aucune dépendance à Pygame ---
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)


class CollisionError(RuntimeError):
    """Déplacement provoquant une collision."""


def toroidal_step(position, direction, width, height):
    return ((position[0] + direction[0]) % width,
            (position[1] + direction[1]) % height)


def direction_between(source, target, width, height):
    for direction in DIRECTIONS:
        if toroidal_step(source, direction, width, height) == tuple(target):
            return direction
    raise ValueError("Les deux cases ne sont pas voisines.")


class SnakeSolver:
    """Cycle et raccourcis prouvés, pour W,H >= 3.

    Précondition : validate_state() a accepté le serpent, puis seuls les
    mouvements retournés par ce solveur sont appliqués. Les pommes sont libres.
    Ne pas reprendre automatiquement une partie manuelle arbitraire.
    """
    def __init__(self, width=GRID_SIZE, height=GRID_SIZE, shortcuts=True):
        self.width, self.height = width, height
        self.shortcuts = shortcuts
        self.cycle = self.build_cycle(width, height)
        self.size = width * height
        self.index = {position: i for i, position in enumerate(self.cycle)}
        self.validate_cycle()

    @staticmethod
    def build_cycle(width, height):
        if width < 3 or height < 3:
            raise ValueError("Dimensions prises en charge : W,H >= 3.")
        # Chaque ligne est parcourue dans un sens s_y (+1 ou -1).
        # Son arrivée est départ-s_y (mod W). La prochaine ligne part de là.
        # Fermeture du cycle si somme(s_y) = 0 (mod W).
        transpose = height % 2 == 1 and (
            width % 2 == 0 or width > height)
        w, h = (height, width) if transpose else (width, height)
        if h % 2 == 0:
            signs = [1, -1] * (h // 2)  # somme = 0
        else:
            # w,h impairs et h >= w : somme = w.
            positive = (h + w) // 2
            signs = [1] * positive + [-1] * (h - positive)
        cycle, start = [], 0
        for y, sign in enumerate(signs):
            cycle.extend(((start + sign * k) % w, y) for k in range(w))
            start = (start - sign) % w
        return [(y, x) for x, y in cycle] if transpose else cycle

    def validate_cycle(self):
        expected = {(x, y) for x in range(self.width)
                    for y in range(self.height)}
        if len(self.cycle) != self.size or set(self.cycle) != expected:
            raise ValueError("Cycle incomplet ou cases dupliquées.")
        for i, position in enumerate(self.cycle):
            direction_between(position, self.cycle[(i + 1) % self.size],
                              self.width, self.height)

    def distance(self, source, target):
        return (self.index[tuple(target)] - self.index[tuple(source)]) % self.size

    def validate_state(self, body, direction, apple=None, grow_pending=False):
        if grow_pending:
            raise ValueError("Garantie : croissance immédiate, sans dette de croissance.")
        cells = [tuple(cell) for cell in body]
        if not 3 <= len(cells) <= self.size:
            raise ValueError("Longueur prise en charge : 3 à W*H.")
        if len(set(cells)) != len(cells) or any(c not in self.index for c in cells):
            raise ValueError("Corps dupliqué ou hors grille.")
        for a, b in zip(cells, cells[1:]):
            direction_between(a, b, self.width, self.height)
        if direction_between(cells[1], cells[0], self.width, self.height) != direction:
            raise ValueError("Direction incohérente avec le cou et la tête.")
        ordered = cells[::-1]  # queue -> tête
        span = sum(self.distance(a, b) for a, b in zip(ordered, ordered[1:]))
        if span >= self.size:
            raise ValueError("Le corps ne respecte pas l'ordre cyclique.")
        if apple is not None and (tuple(apple) not in self.index or tuple(apple) in cells):
            raise ValueError("La pomme doit être sur une case libre.")

    def initial_body(self, head_index=2, length=3):
        if not 3 <= length <= self.size:
            raise ValueError("Longueur initiale invalide.")
        return [list(self.cycle[(head_index - k) % self.size])
                for k in range(length)]

    def next_direction(self, body, direction, apple, grow_pending=False):
        if grow_pending:
            raise ValueError("Croissance différée non couverte par la garantie.")
        if apple is None or len(body) >= self.size:
            raise ValueError("Aucun déplacement après la victoire.")
        head, tail = body[0], body[-1]
        to_tail = self.distance(head, tail)
        to_apple = self.distance(head, apple)
        reverse = (-direction[0], -direction[1])
        best, best_advance = None, 0
        for candidate in DIRECTIONS:
            if candidate == reverse:
                continue
            target = toroidal_step(head, candidate, self.width, self.height)
            advance = self.distance(head, target)
            if not self.shortcuts and advance != 1:
                continue
            # Ne jamais dépasser la pomme : potentiel strictement décroissant.
            if not 1 <= advance <= to_apple:
                continue
            eating = target == tuple(apple)
            # L'égalité autorise uniquement la queue libérée pendant ce tic.
            if advance > to_tail or (advance == to_tail and eating):
                continue
            if advance > best_advance:
                best, best_advance = candidate, advance
        if best is None:
            raise ValueError("État incompatible avec l'invariant du solveur.")
        return best

class Snake:
    """Serpent d'origine : corps ordonné tête -> queue et rendu conservés."""
    def __init__(self, width=GRID_SIZE, height=GRID_SIZE, body=None, direction=RIGHT):
        self.width, self.height = width, height
        if body is None:
            x, y = width // 4, height // 2
            body = [[(x - k) % width, y] for k in range(3)]
        self.body = [list(cell) for cell in body]
        self.head_pos = self.body[0]
        self.direction = direction
        self._last_direction = direction
        self.grow_pending = False
        self.initial_length = len(self.body)
        self.score = 0

    def set_direction(self, new_dir):
        """Compare au dernier mouvement exécuté, même si plusieurs touches arrivent."""
        if new_dir not in DIRECTIONS:
            raise ValueError("Direction non cardinale.")
        if new_dir != (-self._last_direction[0], -self._last_direction[1]):
            self.direction = new_dir
            return True
        return False

    def next_position(self):
        return toroidal_step(self.head_pos, self.direction, self.width, self.height)

    def preview(self, apple_position=None):
        """Prévoit cible, consommation et collision sans modifier l'état."""
        target = self.next_position()
        eating = apple_position is not None and target == tuple(apple_position)
        growing = eating or self.grow_pending
        obstacles = self.body if growing else self.body[:-1]
        collision = list(target) in obstacles
        return target, eating, growing, collision

    def move(self, apple_position=None):
        """Consommation/croissance atomiques ; queue libérée sauf croissance."""
        target, eating, growing, collision = self.preview(apple_position)
        if collision:
            raise CollisionError("Collision avec le corps.")
        self.body.insert(0, list(target))
        if not growing:
            self.body.pop()
        else:
            self.score += 1
        self.head_pos = self.body[0]
        self.grow_pending = False
        self._last_direction = self.direction
        return eating

    def grow(self):
        """Compatibilité : programme une croissance (hors garantie du solveur).
        Le jeu utilise move(pomme) et ne doit PAS appeler grow() après manger.
        """
        self.grow_pending = True

    def check_wall_collision(self):
        x, y = self.head_pos
        return not (0 <= x < self.width and 0 <= y < self.height)

    def check_self_collision(self):
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        return self.check_wall_collision() or self.check_self_collision()

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
    """Pomme uniforme parmi les cases libres ; None lorsque la grille est pleine."""
    def __init__(self, snake_body, width=GRID_SIZE, height=GRID_SIZE, rng=None):
        self.width, self.height = width, height
        self.rng = rng if rng is not None else random
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        occupied = {tuple(cell) for cell in occupied_positions}
        available = [(x, y) for x in range(self.width) for y in range(self.height)
                     if (x, y) not in occupied]
        return self.rng.choice(available) if available else None

    def relocate(self, snake_body):
        self.position = self.random_position(snake_body)
        return self.position is not None

    def draw(self, surface):
        """Dessine la pomme sur la surface de jeu."""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            # Ajout d'un petit reflet pour un aspect "pomme"
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, start_time, end_time=None):
    """Affiche le score et le temps écoulé dans le panneau supérieur."""
    
    # Dessiner le panneau de score
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Afficher le score
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))

    # Afficher le temps
    elapsed_time = (end_time if end_time is not None else time.perf_counter()) - start_time
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
    
    # Dessine un fond semi-transparent pour la lisibilité
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)

def main(automatic=AUTO_MODE, shortcuts=USE_SHORTCUTS, speed=GAME_SPEED):
    """Affichage d'origine, déplacement autonome ou touches directionnelles."""
    global pygame
    if speed <= 0:
        raise ValueError("La vitesse doit être positive.")
    if pygame is None:
        try:
            import pygame as pygame_module
        except ImportError as exc:
            raise SystemExit("Installez Pygame : python -m pip install pygame-ce") from exc
        pygame = pygame_module
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    mode = "AUTO" if automatic else "MANUEL"
    pygame.display.set_caption(f"Snake toroïdal - {mode}")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 26)
    font_game_over = pygame.font.Font(None, 68)
    solver = SnakeSolver(shortcuts=shortcuts)

    def new_game():
        snake = Snake()  # État initial fourni : tête (3,7), cou (2,7), queue (1,7).
        apple = Apple(snake.body)
        if automatic:
            solver.validate_state(snake.body, snake.direction, apple.position)
        return snake, apple

    snake, apple = new_game()
    running, game_over, victory = True, False, False
    start_time = time.perf_counter()
    end_time = None
    accumulator = 0.0
    period = 1.0 / speed
    key_directions = {pygame.K_UP: UP, pygame.K_DOWN: DOWN,
                      pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT}

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif game_over and event.key == pygame.K_SPACE:
                    snake, apple = new_game()
                    game_over, victory = False, False
                    start_time, end_time = time.perf_counter(), None
                    accumulator = 0.0
                    dt = 0.0
                elif not game_over and not automatic and event.key in key_directions:
                    snake.set_direction(key_directions[event.key])
        if not running:
            break

        if not game_over:
            # Temps réel, indépendant des 60 images/s. Borne de rattrapage :
            # garder la fenêtre réactive si elle a été suspendue.
            accumulator += min(dt, 0.25)
            updates = 0
            while accumulator >= period and not game_over and updates < 100:
                accumulator -= period
                updates += 1
                if automatic:
                    snake.set_direction(solver.next_direction(
                        snake.body, snake.direction, apple.position,
                        snake.grow_pending))
                try:
                    eating = snake.move(apple.position)
                except CollisionError:
                    game_over = True
                else:
                    game_over = snake.is_game_over()
                    if eating and not game_over:
                        victory = not apple.relocate(snake.body)
                        game_over = victory
                if game_over:
                    end_time = time.perf_counter()

        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        draw_grid(screen)
        apple.draw(screen)
        snake.draw(screen)
        display_info(screen, font_main, snake, start_time, end_time)
        if game_over:
            display_message(screen, font_game_over,
                            "VICTOIRE !" if victory else "GAME OVER",
                            VERT if victory else ROUGE)
            display_message(screen, font_main, "ESPACE pour rejouer.",
                            BLANC, y_offset=100)
        pygame.display.flip()
    pygame.quit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Snake toroïdal à victoire garantie.")
    parser.add_argument("--manual", action="store_true", help="Jouer avec les flèches.")
    parser.add_argument("--cycle-only", action="store_true", help="Désactiver les raccourcis.")
    parser.add_argument("--speed", type=float, default=GAME_SPEED,
                        help="Déplacements par seconde (défaut : 5).")
    args = parser.parse_args()
    main(automatic=AUTO_MODE and not args.manual,
         shortcuts=USE_SHORTCUTS and not args.cycle_only, speed=args.speed)
