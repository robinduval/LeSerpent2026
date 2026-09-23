import pygame
import random
import time
from collections import deque

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

# --- IA : CYCLE HAMILTONIEN DYNAMIQUE + RACCOURCIS ---

N_CELLS = GRID_SIZE * GRID_SIZE
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
CELLS = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)]


def wrap(pos, direction):
    """Case voisine dans une direction (la grille reboucle sur les bords)."""
    return ((pos[0] + direction[0]) % GRID_SIZE, (pos[1] + direction[1]) % GRID_SIZE)


NEIGHBORS = {c: [wrap(c, d) for d in DIRECTIONS] for c in CELLS}
NEIGHBOR_SET = {c: set(n) for c, n in NEIGHBORS.items()}


def direction_to(a, b):
    """Direction qui mène de la case a à la case voisine b."""
    for d in DIRECTIONS:
        if wrap(a, d) == b:
            return d
    return None


def cycle_direction(pos):
    """Cycle de départ : 14 pas à droite puis 1 en bas, chaque ligne décalée d'une case.

    Il se referme grâce au rebouclage des bords (impossible en 15x15 avec de vrais murs).
    """
    return DOWN if (pos[0] + pos[1]) % GRID_SIZE == GRID_SIZE - 1 else RIGHT


def blocked_cells(snake):
    """Cases mortelles au prochain pas (la queue se libère, sauf si le serpent grandit)."""
    body = snake.body if snake.grow_pending else snake.body[:-1]
    return {tuple(p) for p in body}


def reachable_area(start, blocked):
    """Nombre de cases atteignables depuis start (flood fill)."""
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in NEIGHBORS[cur]:
            if nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


class _Abort(Exception):
    """Budget du DFS épuisé."""


class HamiltonianShortcutAI:
    """Suit un cycle hamiltonien qu'il réorganise à chaque pomme, avec des raccourcis en début de partie.

    Invariant : en avançant sur le cycle depuis la tête, les cases jusqu'à la queue sont libres.
    Les raccourcis ne dépassent jamais la queue, et le cycle n'est modifié que dans cette zone
    libre : le corps reste rangé dans l'ordre du cycle, le serpent ne peut pas se mordre.
    """

    def __init__(self, shortcut_limit=0.2, margin=2, dfs_budget=500):
        self.shortcut_limit = shortcut_limit  # remplissage au-delà duquel on ne prend plus de raccourcis
        self.margin = margin                  # cases libres gardées devant la queue lors d'un raccourci
        self.dfs_budget = dfs_budget          # nœuds explorés au maximum par reconstruction
        self.fallbacks = 0                    # nombre de fois où la sécurité a dû intervenir
        self.order = []                       # le cycle, sous forme de liste de cases
        cell = (0, 0)
        for _ in range(N_CELLS):
            self.order.append(cell)
            cell = wrap(cell, cycle_direction(cell))
        self._reindex()
        self.last_apple = None
        self.rebuild_pending = False

    # --- Cycle ---

    def _reindex(self):
        self.pos = {c: i for i, c in enumerate(self.order)}

    def _rotate(self, head):
        """Place la tête en tête de liste : le rang d'une case devient son indice."""
        p = self.pos[head]
        if p:
            self.order = self.order[p:] + self.order[:p]
            self._reindex()

    def rank(self, cell, head):
        """Nombre de pas pour aller de la tête à cell en suivant le cycle."""
        return (self.pos[cell] - self.pos[head]) % N_CELLS

    # --- Décision ---

    def choose(self, snake, apple):
        head = tuple(snake.head_pos)
        tail = tuple(snake.body[-1])
        apple = tuple(apple) if apple is not None else None

        # 1. Réorganiser le cycle pour que la pomme arrive le plus tôt possible.
        if apple is not None and (apple != self.last_apple or self.rebuild_pending):
            self.last_apple = apple
            self.rebuild_pending = not self._rebuild(head, tail, apple)
            self._flip(head, tail, apple)

        # 2. Suivre le cycle, ou prendre un raccourci en début de partie.
        direction = direction_to(head, self.order[(self.pos[head] + 1) % N_CELLS])
        if apple is not None and len(snake.body) / N_CELLS < self.shortcut_limit:
            direction = self._shortcut(snake, head, apple) or direction

        # 3. Dernière vérification avant de bouger.
        return self._secure(snake, head, direction)

    def _shortcut(self, snake, head, apple):
        """BFS vers la pomme sur les cases de rang croissant, sans dépasser la queue."""
        tail = tuple(snake.body[-1] if snake.grow_pending else snake.body[-2])
        limit = min(self.rank(apple, head), self.rank(tail, head) - self.margin - 1)
        if limit <= 1:
            return None

        blocked = blocked_cells(snake)
        rank = lambda cell: self.rank(cell, head)
        first_step = {}
        queue = deque()
        # Les plus grands sauts d'abord : à longueur égale, le BFS garde le plus avancé.
        for d in sorted(DIRECTIONS, key=lambda d: -rank(wrap(head, d))):
            nxt = wrap(head, d)
            if 1 <= rank(nxt) <= limit and nxt not in blocked:
                first_step[nxt] = d
                queue.append(nxt)
        if not queue:
            return None
        best_jump = queue[0]

        while queue:
            cur = queue.popleft()
            if cur == apple:
                return first_step[cur]
            r_cur = rank(cur)
            for nxt in NEIGHBORS[cur]:
                if r_cur < rank(nxt) <= limit and nxt not in blocked and nxt not in first_step:
                    first_step[nxt] = first_step[cur]
                    queue.append(nxt)
        # Pomme hors de portée : on avance au plus loin autorisé sur le cycle.
        return first_step[best_jump]

    def _secure(self, snake, head, direction):
        """Vérifie que le coup ne tue pas, sinon prend le coup qui garde le plus d'espace."""
        blocked = blocked_cells(snake)
        if wrap(head, direction) not in blocked:
            return direction
        self.fallbacks += 1
        safe = [d for d in DIRECTIONS if wrap(head, d) not in blocked]
        if not safe:
            return direction
        return max(safe, key=lambda d: reachable_area(wrap(head, d), blocked))

    # --- Réorganisation du cycle (uniquement entre la tête et la queue) ---

    def _rebuild(self, head, tail, apple):
        """Retrace toute la zone libre par DFS : tête → pomme au plus vite → toutes les cases → queue.

        Renvoie False si le DFS n'a pas trouvé mieux (on retentera au prochain coup).
        """
        self._rotate(head)
        rt, ra = self.pos[tail], self.pos[apple]
        if ra >= rt:
            return True  # pomme dans un trou derrière la tête : rien à reconstruire
        free = set(self.order[1:rt])

        # Distance de chaque case libre à la pomme.
        dist = {apple: 0}
        queue = deque([apple])
        while queue:
            cur = queue.popleft()
            for nxt in NEIGHBORS[cur]:
                if nxt in free and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        best_possible = 1 + min((dist[n] for n in NEIGHBORS[head] if n in dist), default=ra)
        if ra <= best_possible:
            return True  # la pomme est déjà au plus tôt

        for greedy in (True, False):
            path = self._dfs(head, tail, apple, free, dist, greedy)
            if path is not None and path.index(apple) + 1 < ra:
                self.order = [head] + path + self.order[rt:]
                self._reindex()
                return True
        return False

    def _dfs(self, head, tail, apple, free, dist, greedy):
        """Chemin hamiltonien sur les cases libres, de la tête jusqu'à une voisine de la queue."""
        unvisited = set(free)
        path = []
        state = {"nodes": 0, "apple": False}
        far = len(free) + 1

        def degree(w, cur):
            # Voisins par lesquels w peut encore être relié : cases libres, tête courante, queue.
            return sum(1 for n in NEIGHBORS[w] if n in unvisited or n == cur or n == tail)

        def still_possible(prev, cur):
            # En quittant prev, ses voisins libres perdent un lien : chacun doit en garder 2.
            return all(degree(w, cur) >= 2 for w in NEIGHBORS[prev] if w in unvisited)

        def explore(cur):
            if not unvisited:
                return tail in NEIGHBOR_SET[cur]
            state["nodes"] += 1
            if state["nodes"] > self.dfs_budget:
                raise _Abort
            candidates = [n for n in NEIGHBORS[cur] if n in unvisited]
            if not state["apple"] and greedy:
                candidates.sort(key=lambda n: (dist.get(n, far), degree(n, n)))  # foncer vers la pomme
            elif not state["apple"]:
                candidates.sort(key=lambda n: (degree(n, n), dist.get(n, far)))
            else:
                candidates.sort(key=lambda n: degree(n, n))  # Warnsdorff : case la plus contrainte d'abord
            for n in candidates:
                unvisited.discard(n)
                path.append(n)
                had_apple = state["apple"]
                state["apple"] = had_apple or n == apple
                if still_possible(cur, n) and explore(n):
                    return True
                state["apple"] = had_apple
                path.pop()
                unvisited.add(n)
            return False

        try:
            return list(path) if explore(head) else None
        except _Abort:
            return None

    def _flip(self, head, tail, apple):
        """Retouches 2-opt : si a→b et c→d sur le cycle avec a voisin de c et b voisin de d,
        on inverse le segment b..c. Répété tant que la pomme remonte dans le cycle."""
        self._rotate(head)
        order = self.order
        while True:
            rt, ra = self.pos[tail], self.pos[apple]
            if ra >= rt:
                return
            best = None
            for i in range(ra):
                a, b = order[i], order[i + 1]
                for c in NEIGHBORS[a]:
                    j = self.pos[c]
                    if ra <= j < rt and order[j + 1] in NEIGHBOR_SET[b]:
                        new_ra = i + 1 + (j - ra)
                        if new_ra < ra and (best is None or new_ra < best[0]):
                            best = (new_ra, i, j)
            if best is None:
                return
            _, i, j = best
            order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
            self._reindex()

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

# --- BOUCLE PRINCIPALE DU JEU ---

def main(ai_class=HamiltonianShortcutAI):
    """Fonction principale pour exécuter le jeu Snake Classique, piloté par une IA."""
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
    ai = ai_class()
    
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
                        main(ai_class) # Redémarre le jeu en appelant main()
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
                snake.set_direction(ai.choose(snake, apple.position))
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

if __name__ == '__main__':
    main()