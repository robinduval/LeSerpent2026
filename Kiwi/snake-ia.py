import argparse
import json
import os
import pygame
import random
import time
from collections import defaultdict, deque

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

# --- CONSTANTES DE L'AGENT (Q-LEARNING) ---
# Actions relatives à la direction courante : jamais de demi-tour possible
STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2
ACTIONS = (STRAIGHT, TURN_RIGHT, TURN_LEFT)

# Niveaux de sécurité d'un coup (voir evaluate_move)
DEAD, TRAPPED, ROOMY, TAIL_REACHABLE = 0, 1, 2, 3

# Récompenses
REWARD_FOOD = 10.0
REWARD_DEATH = -100.0
REWARD_VICTORY = 100.0
REWARD_STEP = -0.01
REWARD_CLOSER = 1.0  # bonus (malus) quand le coup suit (ne suit pas) le plus court chemin vers la pomme

# Hyperparamètres
ALPHA = 0.1          # taux d'apprentissage
GAMMA = 0.95         # importance du futur
EPSILON_START = 1.0  # exploration initiale
EPSILON_MIN = 0.001  # exploration finale
TRAIN_EPISODES = 300
EXPLORATION_SHARE = 0.6  # part de l'entraînement pendant laquelle epsilon décroît
CHECKPOINT_EVERY = 20    # après l'exploration, évalue l'agent toutes les N parties...
CHECKPOINT_GAMES = 10    # ...sur ce nombre de parties, et garde la meilleure Q-table

# Nombre de coups sans pomme avant de considérer que l'agent tourne en rond
STARVATION_LIMIT = GRID_SIZE * GRID_SIZE * 2
# Délai avant de relancer automatiquement une partie terminée (secondes)
AUTO_RESTART_DELAY = 3

Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")

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

# --- LOGIQUE D'UN TOUR DE JEU ---

def game_step(snake, apple):
    """
    Exécute un tour de jeu, avec exactement la logique de la boucle principale
    d'origine. Partagée par l'affichage et l'entraînement pour que l'agent
    apprenne sur les mêmes règles que celles du jeu affiché.
    Retourne (game_over, victory, ate).
    """
    snake.move()

    # Vérification des collisions (murs et corps)
    if snake.is_game_over():
        return True, False, False

    # Vérification de la pomme mangée
    if snake.head_pos == list(apple.position):
        snake.grow()
        # Tente de replacer la pomme, Victoire si plus aucune case libre
        if not apple.relocate(snake.body):
            return True, True, True
        return False, False, True

    return False, False, False

# --- PERCEPTION DE L'AGENT ---

def turn(direction, action):
    """Direction absolue obtenue en appliquant une action relative."""
    dx, dy = direction
    if action == TURN_RIGHT:
        return (-dy, dx)  # sens horaire (l'axe y pointe vers le bas)
    if action == TURN_LEFT:
        return (dy, -dx)
    return direction

def clone_snake(snake):
    """Copie indépendante du serpent, pour simuler un coup sans toucher au vrai."""
    clone = Snake.__new__(Snake)
    clone.body = [segment[:] for segment in snake.body]
    clone.head_pos = clone.body[0]
    clone.direction = snake.direction
    clone.grow_pending = snake.grow_pending
    clone.score = snake.score
    return clone

def bfs(start, blocked):
    """Distances depuis start vers toutes les cases atteignables (grille torique)."""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        next_distance = distances[(x, y)] + 1
        for cell in (((x + 1) % GRID_SIZE, y), ((x - 1) % GRID_SIZE, y),
                     (x, (y + 1) % GRID_SIZE), (x, (y - 1) % GRID_SIZE)):
            if cell not in blocked and cell not in distances:
                distances[cell] = next_distance
                queue.append(cell)
    return distances

def evaluate_move(snake, direction, apple_pos):
    """
    Simule un coup avec les vraies méthodes de Snake et retourne
    (niveau de sécurité, distance à la pomme après le coup ou None,
    nombre de cases voisines de la nouvelle tête occupées par le corps).
    """
    clone = clone_snake(snake)
    clone.set_direction(direction)
    clone.move()
    if clone.is_game_over():
        return DEAD, None, 0

    head = tuple(clone.head_pos)
    tail = tuple(clone.body[-1])
    # Si le coup mange la pomme, la queue restera en place au tour suivant
    growing = head == apple_pos
    # La queue n'est pas un obstacle : elle avance en même temps que la tête
    blocked = {tuple(segment) for segment in clone.body[1:-1]}
    distances = bfs(head, blocked)

    tail_distance = distances.get(tail)
    if tail_distance is not None and (not growing or tail_distance > 1):
        # La tête peut toujours suivre sa queue : on ne peut pas s'enfermer
        safety = TAIL_REACHABLE
    elif len(distances) >= len(clone.body):
        # Queue inaccessible, mais assez de place pour tout le corps
        safety = ROOMY
    else:
        # Zone plus petite que le serpent : mort quasi certaine
        safety = TRAPPED

    # Contact : plus il est élevé, plus le serpent reste compact et laisse de la place libre
    x, y = head
    body = {tuple(segment) for segment in clone.body[1:]}
    contact = sum(cell in body for cell in (((x + 1) % GRID_SIZE, y), ((x - 1) % GRID_SIZE, y),
                                            (x, (y + 1) % GRID_SIZE), (x, (y - 1) % GRID_SIZE)))
    return safety, distances.get(apple_pos), contact

def perceive(snake, apple):
    """
    État vu par l'agent : pour chacune des 3 actions, un triplet
    (niveau de sécurité, 1 si l'action suit le plus court chemin vers la pomme, contact).
    """
    apple_pos = tuple(apple.position)
    moves = [evaluate_move(snake, turn(snake.direction, action), apple_pos) for action in ACTIONS]
    reachable = [distance for safety, distance, _ in moves if safety != DEAD and distance is not None]
    best = min(reachable, default=None)
    return tuple((safety, int(safety != DEAD and distance is not None and distance == best), contact)
                 for safety, distance, contact in moves)

# --- AGENT Q-LEARNING ---

class QAgent:
    """
    Agent de Q-Learning. Q(état, action) = q[caractéristiques de l'action] :
    la valeur est partagée entre toutes les situations où un coup a les mêmes
    caractéristiques, ce qui généralise aux états rares de fin de partie.
    """
    def __init__(self, epsilon=0.0):
        self.q = defaultdict(float)
        self.epsilon = epsilon

    def values(self, state):
        return [self.q[state[action]] for action in ACTIONS]

    def choose(self, state):
        """Politique epsilon-greedy : explore au hasard, sinon prend la meilleure action."""
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        values = self.values(state)
        best = max(values)
        return random.choice([action for action in ACTIONS if values[action] == best])

    def act(self, state, stuck):
        """Choix en partie réelle. Si l'agent tourne en rond, il varie parmi les coups les plus sûrs."""
        if stuck:
            safest = max(features[0] for features in state)
            return random.choice([action for action in ACTIONS if state[action][0] == safest])
        return self.choose(state)

    def learn(self, state, action, reward, next_state):
        """Équation de Bellman : Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))."""
        target = reward if next_state is None else reward + GAMMA * max(self.values(next_state))
        features = state[action]
        self.q[features] += ALPHA * (target - self.q[features])

    def save(self, path):
        # Clé lisible "sécurité,pomme,contact" -> valeur Q
        with open(path, "w") as f:
            json.dump({",".join(map(str, features)): value for features, value in sorted(self.q.items())},
                      f, indent=1)

    def load(self, path):
        """Charge une Q-table. Retourne False si le fichier est illisible ou d'un ancien format."""
        try:
            with open(path) as f:
                table = {tuple(int(part) for part in key.split(",")): float(value)
                         for key, value in json.load(f).items()}
        except (OSError, ValueError, TypeError, AttributeError):
            return False
        if any(len(features) != 3 for features in table):
            return False
        self.q.update(table)
        return True

# --- ENTRAÎNEMENT ET ÉVALUATION (sans affichage) ---

def play_episode(agent, learn):
    """Joue une partie complète sans affichage. Retourne (score, victoire, nombre de coups)."""
    snake = Snake()
    apple = Apple(snake.body)
    state = perceive(snake, apple)
    steps_since_food = 0
    steps = 0

    while True:
        stuck = steps_since_food > STARVATION_LIMIT
        action = agent.choose(state) if learn else agent.act(state, stuck)
        snake.set_direction(turn(snake.direction, action))
        game_over, victory, ate = game_step(snake, apple)
        steps += 1
        steps_since_food = 0 if ate else steps_since_food + 1
        # À l'entraînement, tourner en rond trop longtemps compte comme un échec
        starved = learn and steps_since_food > STARVATION_LIMIT

        if victory:
            reward = REWARD_VICTORY
        elif game_over or starved:
            reward = REWARD_DEATH
        elif ate:
            reward = REWARD_FOOD
        else:
            reward = REWARD_STEP + (REWARD_CLOSER if state[action][1] else -REWARD_CLOSER)

        done = game_over or starved or steps_since_food > 10 * STARVATION_LIMIT
        next_state = None if (game_over or starved) else perceive(snake, apple)
        if learn:
            agent.learn(state, action, reward, next_state)
        if done:
            return snake.score, victory, steps
        state = next_state

def format_duration(seconds):
    """Durée au format MM:SS, comme le chronomètre du jeu."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def train(agent, episodes):
    """Entraîne l'agent sur plusieurs parties, avec une exploration décroissante."""
    print(f"Entraînement sur {episodes} parties...")
    train_start = time.time()
    best = 0
    recent = deque(maxlen=20)
    best_checkpoint = None
    best_q = None
    exploration_end = int(episodes * EXPLORATION_SHARE)
    for episode in range(1, episodes + 1):
        # Décroissance linéaire de l'exploration sur le début de l'entraînement
        progress = min(1.0, episode / max(1, exploration_end))
        agent.epsilon = max(EPSILON_MIN, EPSILON_START * (1 - progress))
        score, _, steps = play_episode(agent, learn=True)
        best = max(best, score)
        recent.append(score)
        # Temps de la partie tel que l'afficherait le jeu (un coup par image à GAME_SPEED)
        print(f"  partie {episode:4d} | score {score:3d} | temps de partie {format_duration(steps / GAME_SPEED)} | "
              f"epsilon {agent.epsilon:.3f} | moyenne(20) {sum(recent) / len(recent):6.1f} | meilleur {best}")

        # Point de contrôle : l'apprentissage est bruité, on garde la meilleure politique vue
        if episode >= exploration_end and (episode % CHECKPOINT_EVERY == 0 or episode == episodes):
            epsilon = agent.epsilon
            agent.epsilon = 0.0
            scores = [play_episode(agent, learn=False)[0] for _ in range(CHECKPOINT_GAMES)]
            agent.epsilon = epsilon
            mean = sum(scores) / len(scores)
            improved = best_checkpoint is None or mean > best_checkpoint
            print(f"  point de contrôle : moyenne {mean:.1f} sur {CHECKPOINT_GAMES} parties"
                  f"{' -> meilleure Q-table conservée' if improved else ''}")
            if improved:
                best_checkpoint = mean
                best_q = dict(agent.q)

    if best_q is not None:
        agent.q = defaultdict(float, best_q)
        print(f"Q-table retenue : moyenne {best_checkpoint:.1f} au point de contrôle")
    agent.epsilon = 0.0
    print(f"Entraînement terminé en {format_duration(time.time() - train_start)}")

def evaluate(agent, games):
    """Mesure les performances de la politique apprise (sans exploration)."""
    agent.epsilon = 0.0
    scores = []
    victories = 0
    for game in range(1, games + 1):
        score, victory, steps = play_episode(agent, learn=False)
        scores.append(score)
        victories += victory
        print(f"  partie {game:3d} | score {score:3d} | temps de partie {format_duration(steps / GAME_SPEED)}"
              f"{' (VICTOIRE)' if victory else ''}")
    print(f"Moyenne {sum(scores) / len(scores):.1f} | min {min(scores)} | "
          f"max {max(scores)} | victoires {victories}/{games}")

# --- BOUCLE PRINCIPALE DU JEU ---

def main(agent, speed=1.0):
    """Fonction principale : le jeu Snake Classique, joué par l'agent."""
    pygame.init()

    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Q-Learning")
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
    steps_since_food = 0

    # Chronomètre en temps de jeu : il avance d'un coup par image, comme le serpent,
    # donc il accélère exactement comme lui avec --speed
    frame_count = 0
    game_over_frame = None

    # Variable pour la gestion de la vitesse (pour ne bouger qu'une fois par tic)
    move_counter = 0

    # --- Boucle de jeu ---
    while running:
        # 1. Gestion des Événements (fermeture et redémarrage uniquement)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and game_over and event.key == pygame.K_SPACE:
                main(agent, speed) # Redémarre le jeu en appelant main()
                return

        # Redémarrage automatique après quelques secondes (de temps de jeu)
        if game_over and frame_count - game_over_frame >= AUTO_RESTART_DELAY * GAME_SPEED:
            main(agent, speed)
            return

        # 2. Logique de Mise à Jour du Jeu
        if not game_over and not victory:
            # Le serpent se déplace à la vitesse définie
            move_counter += 1
            if move_counter >= GAME_SPEED // 10: # Déplace le serpent à un rythme constant
                # L'agent choisit la direction à la place du clavier
                state = perceive(snake, apple)
                action = agent.act(state, stuck=steps_since_food > STARVATION_LIMIT)
                snake.set_direction(turn(snake.direction, action))

                game_over, victory, ate = game_step(snake, apple)
                move_counter = 0
                steps_since_food = 0 if ate else steps_since_food + 1

                if game_over:
                    game_over_frame = frame_count
                    print(f"Partie terminée : score {snake.score}{' (VICTOIRE)' if victory else ''}")
                    if not victory:
                        continue # Passe à l'affichage de Game Over

        # 3. Dessin
        screen.fill(GRIS_FOND) # Fond gris pour la zone de score

        # Zone de jeu (décalée par la hauteur du panneau de score)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)

        draw_grid(screen)

        # Dessine la pomme et le serpent
        apple.draw(screen)
        snake.draw(screen)

        # Affiche le score et le temps (display_info mesure time.time() - start_time)
        start_time = time.time() - frame_count / GAME_SPEED
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
        clock.tick(GAME_SPEED * speed)
        frame_count += 1

    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Snake joué par un agent de Q-Learning.")
    parser.add_argument("--train", type=int, metavar="N",
                        help="réentraîne l'agent sur N parties (sinon charge q_table.json s'il existe)")
    parser.add_argument("--eval", type=int, metavar="N",
                        help="évalue l'agent sur N parties sans affichage, puis quitte")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="multiplie la vitesse du serpent et du chronomètre (ex: --speed 10)")
    args = parser.parse_args()

    agent = QAgent()
    if args.speed <= 0:
        parser.error("--speed doit être strictement positif")

    loaded = False
    if args.train is None and os.path.exists(Q_TABLE_PATH):
        loaded = agent.load(Q_TABLE_PATH)
        if loaded:
            print(f"Q-table chargée depuis {Q_TABLE_PATH}")
        else:
            print(f"{Q_TABLE_PATH} est d'un format incompatible : nouvel entraînement.")
    if not loaded:
        train(agent, args.train or TRAIN_EPISODES)
        agent.save(Q_TABLE_PATH)
        print(f"Q-table sauvegardée dans {Q_TABLE_PATH}")

    if args.eval:
        evaluate(agent, args.eval)
    else:
        main(agent, args.speed)
