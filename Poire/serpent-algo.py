import pygame
import random
import time

# === AJOUT RL : imports nécessaires au Q-learning tabulaire ===
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Tuple
# === FIN AJOUT ===

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

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

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
        """Vérifie si la tête touche les bords (Game Over si hors grille)."""
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps (Game Over si auto-morsure)."""
        # On vérifie si la position de la tête est dans le reste du corps (body[1:])
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé (mur ou morsure)."""
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

# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, start_time):
    """Affiche le score et le temps écoulé dans le panneau supérieur."""
    
    # Dessiner le panneau de score
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Afficher le score
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))

    # Afficher le temps
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))
    
    # Afficher le taux de remplissage
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 20))

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

# ============================================================
# === AJOUT RL : environnement + Q-learning tabulaire ===
# Tout ce bloc est ajouté autour du jeu de base ci-dessus, qui
# n'est pas modifié : Snake/Apple/draw_* sont réutilisés tels quels.
# ============================================================

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "qtable_poire.pkl")

# Ordre horaire des directions : sert à convertir une action relative
# (tout droit / droite / gauche) en direction absolue (dx, dy).
DIRECTIONS_HORAIRES = [RIGHT, DOWN, LEFT, UP]

State = Tuple[int, int, int, int, int, int, int, int, int, int, int]

STEP_PENALTY = -0.01
DEATH_PENALTY = -10.0
FOOD_REWARD = 10.0


def turn(direction: Tuple[int, int], action: int) -> Tuple[int, int]:
    """action : 0 = tout droit, 1 = tourne à droite, 2 = tourne à gauche."""
    idx = DIRECTIONS_HORAIRES.index(direction)
    if action == 1:
        idx = (idx + 1) % 4
    elif action == 2:
        idx = (idx - 1) % 4
    return DIRECTIONS_HORAIRES[idx]


def get_state(snake, apple) -> State:
    """Vecteur d'état à 11 booléens (danger x3, direction x4, pomme x4).

    Pas de "danger mur" : move() fait un modulo sur la grille (voir
    check_wall_collision, qui ne se déclenche donc jamais), le seul
    vrai danger est de mordre son propre corps.
    """
    head = snake.head_pos
    dx, dy = snake.direction

    def next_pos(direction):
        return [(head[0] + direction[0]) % GRID_SIZE, (head[1] + direction[1]) % GRID_SIZE]

    danger_straight = next_pos(snake.direction) in snake.body[1:]
    danger_right = next_pos(turn(snake.direction, 1)) in snake.body[1:]
    danger_left = next_pos(turn(snake.direction, 2)) in snake.body[1:]

    ax, ay = apple.position

    return (
        int(danger_straight),
        int(danger_right),
        int(danger_left),
        int(dx == -1),
        int(dx == 1),
        int(dy == -1),
        int(dy == 1),
        int(ax < head[0]),
        int(ax > head[0]),
        int(ay < head[1]),
        int(ay > head[1]),
    )


class SnakeEnv:
    """Wrapper headless autour de Snake/Apple : aucune fenêtre pygame
    n'est ouverte pendant l'entraînement, seule la logique est rejouée."""

    def __init__(self, max_idle_steps: int = GRID_SIZE * GRID_SIZE * 4) -> None:
        self.max_idle_steps = max_idle_steps
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.steps_since_food = 0

    def reset(self) -> State:
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.steps_since_food = 0
        return get_state(self.snake, self.apple)

    def step(self, action: int):
        new_dir = turn(self.snake.direction, action)
        self.snake.set_direction(new_dir)
        self.snake.move()

        reward = STEP_PENALTY
        done = False

        if self.snake.check_self_collision():
            reward, done = DEATH_PENALTY, True
        elif self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            reward = FOOD_REWARD
            self.steps_since_food = 0
            if not self.apple.relocate(self.snake.body):
                done = True  # victoire : plus de case libre
        else:
            self.steps_since_food += 1
            if self.steps_since_food > self.max_idle_steps:
                done = True  # timeout : le tore permet de tourner en rond indéfiniment

        return get_state(self.snake, self.apple), reward, done, self.snake.score


QTable = DefaultDict[State, List[float]]


@dataclass
class TrainConfig:
    """Paramètres d'entraînement — à ajuster ici avant de lancer TRAIN = True."""
    episodes: int = 8000
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.998
    log_every: int = 500


@dataclass
class TrainResult:
    episodes: int
    duration_sec: float
    best_score: int
    avg_score_last_100: float
    final_epsilon: float
    states_discovered: int


def train(config: TrainConfig):
    q_table: QTable = defaultdict(lambda: [0.0, 0.0, 0.0])
    env = SnakeEnv()
    epsilon = config.epsilon_start
    scores: List[int] = []
    start_time = time.time()

    for episode in range(1, config.episodes + 1):
        state = env.reset()
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                q_values = q_table[state]
                action = q_values.index(max(q_values))

            next_state, reward, done, score = env.step(action)

            best_next = max(q_table[next_state])
            q_table[state][action] += config.alpha * (
                reward + config.gamma * best_next - q_table[state][action]
            )
            state = next_state

        scores.append(score)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        if episode % config.log_every == 0:
            recent = scores[-100:]
            avg = sum(recent) / len(recent)
            print(
                f"  épisode {episode:>6}/{config.episodes} "
                f"| score moyen (100 derniers) : {avg:5.2f} "
                f"| epsilon : {epsilon:.3f}"
            )

    recent = scores[-100:]
    result = TrainResult(
        episodes=config.episodes,
        duration_sec=time.time() - start_time,
        best_score=max(scores),
        avg_score_last_100=sum(recent) / len(recent),
        final_epsilon=epsilon,
        states_discovered=len(q_table),
    )
    return q_table, result


def save_training(q_table: QTable, config: TrainConfig, result: TrainResult) -> None:
    with open(QTABLE_PATH, "wb") as f:
        pickle.dump({"q_table": dict(q_table), "config": config, "result": result}, f)


def load_training():
    with open(QTABLE_PATH, "rb") as f:
        data = pickle.load(f)
    q_table: QTable = defaultdict(lambda: [0.0, 0.0, 0.0], data["q_table"])
    return q_table, data["config"], data["result"]


def run_training() -> None:
    """TRAIN = True : entraîne un agent headless (pas de fenêtre pygame) et
    affiche les paramètres utilisés puis le résultat en fin d'entraînement."""
    config = TrainConfig()
    print("--- Entraînement Q-learning ---")
    print(
        f"épisodes={config.episodes} alpha={config.alpha} gamma={config.gamma} "
        f"epsilon={config.epsilon_start}->{config.epsilon_end} "
        f"decay={config.epsilon_decay}"
    )

    q_table, result = train(config)
    save_training(q_table, config, result)

    print("--- Résultat ---")
    print(f"durée              : {result.duration_sec:.1f} s")
    print(f"meilleur score     : {result.best_score}")
    print(f"score moyen (100)  : {result.avg_score_last_100:.2f}")
    print(f"epsilon final      : {result.final_epsilon:.4f}")
    print(f"états découverts   : {result.states_discovered}")
    print(f"sauvegardé dans    : {QTABLE_PATH}")


def choose_action(q_table: QTable, state: State) -> int:
    if state not in q_table:
        return 0  # état jamais vu à l'entraînement : on continue tout droit
    q_values = q_table[state]
    return q_values.index(max(q_values))


def run_trained_game() -> None:
    """TRAIN = False : charge le Q-table sauvegardé et laisse l'agent
    jouer seul une partie (pas de contrôle clavier)."""
    if not os.path.exists(QTABLE_PATH):
        print(f"Aucun modèle entraîné trouvé ({QTABLE_PATH}).")
        print("Passe TRAIN = True en bas du fichier pour entraîner un agent d'abord.")
        return

    q_table, _, result = load_training()
    print(
        f"Modèle chargé (score moyen entraînement : {result.avg_score_last_100:.2f}, "
        f"{result.states_discovered} états)."
    )

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - IA Q-learning - Team Poire")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    env = SnakeEnv()
    state = env.reset()
    start_time = time.time()
    running = True
    game_over = False
    victory = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            action = choose_action(q_table, state)
            state, _, done, _ = env.step(action)
            if done:
                game_over = True
                victory = env.apple.position is None

        screen.fill(GRIS_FOND)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        draw_grid(screen)
        env.apple.draw(screen)
        env.snake.draw(screen)
        display_info(screen, font_main, env.snake, start_time)

        if game_over:
            if victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
            display_message(screen, font_main, "Ferme la fenêtre pour quitter.", BLANC, y_offset=100)

        pygame.display.flip()
        clock.tick(GAME_SPEED)

    pygame.quit()

# === FIN AJOUT RL ===
# ============================================================

# --- BOUCLE PRINCIPALE DU JEU (jouable au clavier, inchangée) ---

def main():
    """Fonction principale pour exécuter le jeu Snake Classique."""
    pygame.init()
    
    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Classique - Socle de Base")
    clock = pygame.time.Clock()
    
    # Configuration des polices
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)
    
    # Initialisation des objets du jeu
    snake = Snake()
    apple = Apple(snake.body)
    
    # Variables de jeu
    running = True
    game_over = False
    victory = False
    
    # Démarrage du chronomètre
    start_time = time.time()
    
    # Variable pour la gestion de la vitesse (pour ne bouger qu'une fois par tic)
    move_counter = 0

    # --- Boucle de jeu ---
    while running:
        # 1. Gestion des Événements (Contrôles Clavier)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if game_over:
                    # Logique de redémarrage : seulement si le jeu est terminé
                    if event.key == pygame.K_SPACE:
                        main() # Redémarre le jeu en appelant main()
                        return
                else:
                    # Logique de déplacement : seulement si le jeu est en cours
                    if event.key == pygame.K_UP:
                        snake.set_direction(UP)
                    elif event.key == pygame.K_DOWN:
                        snake.set_direction(DOWN)
                    elif event.key == pygame.K_LEFT:
                        snake.set_direction(LEFT)
                    elif event.key == pygame.K_RIGHT:
                        snake.set_direction(RIGHT)
        
        # 2. Logique de Mise à Jour du Jeu
        if not game_over and not victory:
            # Le serpent se déplace à la vitesse définie
            move_counter += 1
            if move_counter >= GAME_SPEED // 10: # Déplace le serpent à un rythme constant
                snake.move()
                move_counter = 0

                # Vérification des collisions (murs et corps)
                if snake.is_game_over():
                    game_over = True
                    continue # Passe à l'affichage de Game Over

                # Vérification de la pomme mangée
                if snake.head_pos == list(apple.position):
                    snake.grow()
                    
                    # Tente de replacer la pomme, vérifie la Victoire si échec
                    if not apple.relocate(snake.body):
                        victory = True # Plus d'espace pour la pomme
                        game_over = True # Met fin au jeu
        
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
        display_info(screen, font_main, snake, start_time)
        
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

# === AJOUT RL : bascule train/IA au lieu de l'appel direct à main() ===
# True  -> entraîne un agent Q-learning headless et sauvegarde qtable_poire.pkl
# False -> charge qtable_poire.pkl et l'IA joue seule une partie
TRAIN = False

if __name__ == '__main__':
    if TRAIN:
        run_training()
    else:
        run_trained_game()
# === FIN AJOUT ===
