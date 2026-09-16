"""Bloc 'Game' : l'environnement de jeu, pilotable par un agent.

Repris de serpent-algo.py (meme grille, meme vitesse, meme systeme de score, comme
demande dans le cours : on n'a pas le droit d'y toucher). La seule vraie nouveaute
est SnakeGameAI.play_step(action), qui remplace la boucle clavier par une methode
appelable par l'agent et qui renvoie (reward, game_over, score) a chaque coup.
"""

import pygame
import random
from collections import deque

# --- CONSTANTES DE JEU (identiques a serpent-algo.py : grille, vitesse, ne pas toucher) ---
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
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions (vecteurs (dx, dy)) - ordre du PDF, slide 5
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
ACTIONS = [UP, DOWN, LEFT, RIGHT]  # index de l'action -> direction (HAUT, BAS, GAUCHE, DROITE)


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def bfs_safe_distance(start, blocked_cells, target, grid_size=GRID_SIZE):
    """Longueur du plus court chemin de `start` a `target` qui ne traverse
    aucune case de `blocked_cells` (le corps du serpent) -- "distance
    minimale sans suicide". Parcours en largeur (BFS) sur la grille torique
    (les voisins bouclent avec %, comme Snake.move()).

    Simplification assumee : on bloque tout le corps, y compris la queue,
    qui en realite se librera au prochain mouvement. Resultat legerement
    pessimiste dans de rares cas, largement suffisant pour servir de guide
    a la recompense (ce n'est pas l'agent qui decide, juste un indicateur).

    Renvoie None si aucun chemin sans danger n'existe.
    """
    start = tuple(start)
    target = tuple(target)
    if start == target:
        return 0

    blocked = {tuple(cell) for cell in blocked_cells}
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in ACTIONS:
            nxt = ((x + dx) % grid_size, (y + dy) % grid_size)
            if nxt in visited or nxt in blocked:
                continue
            if nxt == target:
                return dist + 1
            visited.add(nxt)
            queue.append((nxt, dist + 1))

    return None


def flood_fill_size(start, blocked_cells, grid_size=GRID_SIZE):
    """Nombre de cases atteignables depuis `start` sans traverser `blocked_cells`
    (le corps du serpent) -- taille de la "poche" d'espace libre autour de start.

    Utilise pour detecter les culs-de-sac : une case immediatement libre peut
    quand meme mener a une poche minuscule ou le serpent se retrouvera coince
    quelques coups plus tard. bfs_safe_distance seul ne le voit pas (il
    s'arrete des qu'il atteint la pomme) ; flood_fill_size mesure l'espace
    total accessible, donc la "profondeur" de la voie choisie.
    """
    start = tuple(start)
    blocked = {tuple(c) for c in blocked_cells}
    if start in blocked:
        return 0

    visited = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ACTIONS:
            nxt = ((x + dx) % grid_size, (y + dy) % grid_size)
            if nxt in visited or nxt in blocked:
                continue
            visited.add(nxt)
            queue.append(nxt)

    return len(visited)


class Snake:
    """Identique a serpent-algo.py : position, corps, direction, croissance."""

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

    def check_self_collision(self, point=None):
        """Le mur ne tue jamais (move() boucle deja via le modulo) : seule
        l'auto-morsure met fin a la partie."""
        point = point if point is not None else self.head_pos
        return point in self.body[1:]

    def is_game_over(self):
        return self.check_self_collision()

    def draw(self, surface):
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1)

        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2)


class Apple:
    """Identique a serpent-algo.py."""

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


class SnakeGameAI:
    """Environnement pilotable par un agent RL.

    Contrat (cf. PDF slide 4) :
        reward, game_over, score = game.play_step(action)
    ou `action` est un vecteur one-hot de longueur 4 : [HAUT, BAS, GAUCHE, DROITE].

    render=False permet de faire tourner des milliers d'episodes d'entrainement
    sans ouvrir de fenetre ni attendre l'horloge -> l'entrainement va vite.
    render=True (mode demo / evaluation finale) rouvre la fenetre et respecte
    GAME_SPEED comme le jeu de base, pour que le temps affiche reste comparable
    a une partie humaine.
    """

    # Recompenses de base (PDF slide 5)
    REWARD_APPLE_MIN = 10   # plancher : une pomme rapporte toujours au moins ca
    REWARD_DEATH = -20000   # mort encore plus fortement penalisee (demande du groupe)
    REWARD_WIN = 100

    # Recompense de survie, versee a CHAQUE pas (pas seulement a la pomme).
    # Elle grandit elle aussi avec le temps deja survecu, comme le multiplicateur
    # des pommes : rester en vie devient un objectif de plus en plus payant en
    # soi, pas juste un moyen d'atteindre la prochaine pomme.
    REWARD_STEP_BASE = 1.0
    SURVIVAL_STEP_SCALE = 300

    # --- Reglages du bonus "rapidite + survie" sur la recompense pomme ---
    # Plus la pomme est attrapee vite (peu de pas depuis la derniere pomme),
    # plus le bonus est grand. Au-dela de cette fenetre de pas, bonus = 0
    # (mais le plancher REWARD_APPLE_MIN reste garanti).
    SPEED_BONUS_WINDOW = 2 * GRID_SIZE
    # Le multiplicateur de survie grandit avec le nombre total de pas joues
    # dans la partie : plus cette constante est grande, plus il grandit lentement.
    SURVIVAL_SCALE = 200

    # Plus le serpent est long, plus la pomme suivante rapporte. Longueur de
    # depart = 3 (voir Snake.__init__) ; a chaque LENGTH_SCALE segments de
    # plus, le multiplicateur augmente de +1.
    INITIAL_LENGTH = 3
    LENGTH_SCALE = 10

    # Poids de la penalite d'ecart entre l'action jouee et l'action ideale
    # (voir _movement_shaping) : chaque case de detour "inutile" coute ca.
    SHAPING_WEIGHT = 1.0

    # Penalite si le coup joue laisse le serpent dans une poche d'espace
    # libre plus petite que sa propre longueur (signe avant-coureur qu'il va
    # se coincer lui-meme, avant meme que la collision n'arrive).
    TRAP_PENALTY = 2.0

    # Garde-fou d'entrainement : si le serpent tourne en rond sans jamais
    # manger, on arrete l'episode (sinon certaines parties ne terminent jamais).
    # Ce n'est pas une regle du jeu (le score/la grille/l'horloge ne changent
    # pas), c'est juste une limite interne a l'entrainement.
    MAX_STEPS_WITHOUT_FOOD_FACTOR = 100

    def __init__(self, render=True, speed=None):
        self.render_enabled = render
        # `speed` ne modifie QUE l'affichage/test : la constante GAME_SPEED
        # (l'horloge officielle du jeu) n'est jamais touchee. Par defaut on
        # rejoue a GAME_SPEED ; on peut passer un chiffre plus grand pour
        # accelerer visuellement les tests.
        self.speed = speed if speed is not None else GAME_SPEED
        self.screen = None
        self.clock = None
        self.font_main = None
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake IA - Deep Q-Learning")
            self.clock = pygame.time.Clock()
            self.font_main = pygame.font.Font(None, 40)
        self.reset()

    def reset(self):
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.steps_since_food = 0  # pour le bonus de vitesse + le garde-fou anti-boucle
        self.total_steps = 0       # pour le multiplicateur de survie

    def get_elapsed_time(self):
        """Temps de jeu 'equivalent vitesse normale', quelle que soit la
        vitesse d'affichage utilisee pour les tests (self.speed) ou le mode
        (avec/sans fenetre).

        Un pas dure normalement 1/GAME_SPEED secondes. Si on teste 3x plus
        vite (self.speed = 3 x GAME_SPEED), on fait le meme nombre de pas en
        3x moins de temps reel -> pour que le temps affiche reste comparable
        a une partie a vitesse normale, on le calcule a partir du nombre de
        pas joues, pas du temps reel ecoule :

            temps_affiche = total_steps / GAME_SPEED

        Comme ca, peu importe a quelle vitesse on a fait tourner le test
        (ou si on est en headless, sans horloge du tout), le temps reflete
        toujours "combien de temps ca aurait pris a la vitesse officielle".
        """
        return self.total_steps / GAME_SPEED

    def _apple_reward(self):
        """Recompense de capture d'une pomme :
        (minimum + bonus de vitesse) x multiplicateur de survie x multiplicateur de longueur.
        """
        speed_bonus = max(0, self.SPEED_BONUS_WINDOW - self.steps_since_food)
        survival_multiplier = 1 + self.total_steps / self.SURVIVAL_SCALE
        length_multiplier = 1 + (len(self.snake.body) - self.INITIAL_LENGTH) / self.LENGTH_SCALE
        return (self.REWARD_APPLE_MIN + speed_bonus) * survival_multiplier * length_multiplier

    def _movement_shaping(self, distance_before, target):
        """Compare la distance sure realisee par le coup joue a celle qu'aurait
        donnee le coup ideal (le premier pas du plus court chemin sans danger).

        - distance_before : distance sure avant le coup (calculee par l'appelant,
          avant que le serpent ne bouge).
        - Un coup ideal fait TOUJOURS baisser cette distance d'exactement 1
          (definition du plus court chemin) -> ideal_distance = distance_before - 1.
        - ecart = distance_apres_le_coup - ideal_distance
              = 0 si le serpent a joue le coup optimal
              > 0 si son coup l'a eloigne (ou fait stagner) par rapport a l'ideal
        Renvoie une penalite (>= 0) a soustraire de la recompense du pas.
        """
        if distance_before is None:
            return 0.0

        distance_after = bfs_safe_distance(self.snake.head_pos, self.snake.body, target)
        if distance_after is None:
            return 0.0

        ideal_distance = max(0, distance_before - 1)
        gap = distance_after - ideal_distance
        return self.SHAPING_WEIGHT * max(0, gap)

    def play_step(self, action):
        self.steps_since_food += 1
        self.total_steps += 1

        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # Cible et distance "sure" (BFS, sans traverser le corps) mesurees
        # AVANT le coup, pour pouvoir juger apres coup si l'action jouee
        # etait la meilleure possible.
        target = self.apple.position
        distance_before = bfs_safe_distance(self.snake.head_pos, self.snake.body, target)

        # 1. Deplacement selon l'action choisie par l'agent
        action_idx = action.index(1) if isinstance(action, list) else int(action.argmax())
        self.snake.set_direction(ACTIONS[action_idx])
        self.snake.move()

        reward = self.REWARD_STEP_BASE * (1 + self.total_steps / self.SURVIVAL_STEP_SCALE)
        game_over = False

        # 2. Collision (auto-morsure) ou episode d'entrainement trop long
        if self.snake.is_game_over():
            game_over = True
            reward = self.REWARD_DEATH
            return reward, game_over, self.snake.score

        if self.steps_since_food > self.MAX_STEPS_WITHOUT_FOOD_FACTOR * len(self.snake.body):
            game_over = True
            reward = self.REWARD_DEATH
            return reward, game_over, self.snake.score

        # 2bis. Ecart entre le coup joue et le coup ideal -> petite penalite
        # si le serpent s'est eloigne du plus court chemin sans danger.
        reward -= self._movement_shaping(distance_before, target)

        # 2ter. Le coup a-t-il laisse le serpent dans une poche trop petite
        # pour sa propre longueur ? Avertissement precoce avant le piege reel.
        free_space = flood_fill_size(self.snake.head_pos, self.snake.body)
        if free_space < len(self.snake.body):
            reward -= self.TRAP_PENALTY

        # 3. Pomme mangee ?
        if self.snake.head_pos == list(target):
            self.snake.grow()
            reward = self._apple_reward()
            self.steps_since_food = 0
            if not self.apple.relocate(self.snake.body):
                # Plus de case libre : grille remplie -> victoire
                reward = self.REWARD_WIN
                game_over = True

        # 4. Affichage (uniquement en mode demo/evaluation)
        if self.render_enabled:
            self._draw()
            self.clock.tick(self.speed)

        return reward, game_over, self.snake.score

    def _draw(self):
        self.screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.screen, NOIR, game_area_rect)
        draw_grid(self.screen)
        self.apple.draw(self.screen)
        self.snake.draw(self.screen)

        pygame.draw.rect(self.screen, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
        pygame.draw.line(self.screen, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)
        score_text = self.font_main.render(f"Score: {self.snake.score}", True, BLANC)
        self.screen.blit(score_text, (10, 20))

        time_text = self.font_main.render(f"Temps: {format_time(self.get_elapsed_time())}", True, BLANC)
        self.screen.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))

        pygame.display.flip()
