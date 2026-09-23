"""game.py - Environnement RL pour Snake (wrapper additif autour de serpent-algo.py).

Ce module N'IMPLEMENTE AUCUNE regle de jeu : il reutilise tel quel le socle
(serpent-algo.py) pour Snake/Apple/collisions/vitesse, et se contente
d'exposer une interface de type "Gym" (reset/step) au-dessus, plus le calcul
de l'etat vectoriel (12 features) consomme par l'agent D3QN.

Rappel important herite du socle : Snake.move() applique un modulo GRID_SIZE
=> la grille est un TORE (pas de murs, ils sont traversants). Le seul danger
reel est le corps du serpent. Toutes les distances sont donc calculees sur le
tore (cf. _torus_delta / _apple_distance), jamais en distance "plate".
"""

import importlib.util
import os
import sys
import time

import numpy as np
import pygame

# --- Import du socle de jeu (nom de fichier avec tiret -> import par chemin) ---
_SPEC = importlib.util.spec_from_file_location(
    "serpent_algo", os.path.join(os.path.dirname(os.path.abspath(__file__)), "serpent-algo.py")
)
serpent_algo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(serpent_algo)

# --- Import du module de vision (raycast), meme dossier que ce fichier ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision import raycast

# --- Alias vers le socle : on ne redefinit RIEN, on reutilise les memes objets ---
Snake = serpent_algo.Snake
Apple = serpent_algo.Apple
GRID_SIZE = serpent_algo.GRID_SIZE
GAME_SPEED = serpent_algo.GAME_SPEED
UP = serpent_algo.UP
DOWN = serpent_algo.DOWN
LEFT = serpent_algo.LEFT
RIGHT = serpent_algo.RIGHT
draw_grid = serpent_algo.draw_grid
display_info = serpent_algo.display_info
GRIS_FOND = serpent_algo.GRIS_FOND
NOIR = serpent_algo.NOIR
SCREEN_WIDTH = serpent_algo.SCREEN_WIDTH
SCREEN_HEIGHT = serpent_algo.SCREEN_HEIGHT
SCORE_PANEL_HEIGHT = serpent_algo.SCORE_PANEL_HEIGHT

# --- Constantes de l'environnement RL (additives, ne touchent pas au socle) ---
# Index d'action 0..3 = ordre du one-hot direction dans get_state() (haut, bas, gauche, droite).
ACTIONS = [UP, DOWN, LEFT, RIGHT]

REWARD_EAT = 20.0
REWARD_DEATH = -20.0
REWARD_STEP = -0.5
REWARD_CLOSER = 1.0
REWARD_FARTHER = -1.0

# Troncature d'entrainement : au-dela de ce nombre de pas sans manger, l'episode
# est coupe (evite les episodes infinis pendant l'entrainement).
MAX_STEPS_WITHOUT_FOOD_FACTOR = 100


class SnakeGameRL:
    """Environnement Snake au format RL (reset/play_step) intégrant le socle serpent-algo.py.

    Fournit une interface Gym-like : reset() retourne l'état initial, play_step(action) avance
    d'un pas. L'état vectoriel (12 features) capture la géométrie et les dangers immédiats
    de la grille torique. Les rewards reflètent l'objectif : maximaliser le ratio score/temps
    (trajets efficaces) via gamma=0.95 < 0.99.
    """

    def __init__(self, render=False, fps=None):
        """Initialise l'environnement RL.

        Args:
            render: Si True, affiche la partie en temps réel (pygame). Défaut: False.
            fps: Cadence d'affichage (images/s) utilisée par la clock en mode render.
                None = clock de base du socle (GAME_SPEED), c'est la valeur pour une
                partie lancée manuellement (mode play). Une valeur plus élevée (ou 0 =
                sans limite) ne sert qu'à accélérer l'affichage pendant l'entraînement :
                le jeu avance en pas discrets, l'agent ne voit jamais le temps, donc le
                comportement est identique quelle que soit la cadence.
        """
        self.render_enabled = render
        # GAME_SPEED du socle n'est jamais modifie : on choisit seulement la cadence
        # a laquelle CE wrapper appelle clock.tick().
        self.fps = GAME_SPEED if fps is None else fps
        self.quit_requested = False

        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake RL - D3QN + PER")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 40)

        self.snake = None
        self.apple = None
        self.steps = 0
        self.steps_since_food = 0
        self.start_time = None

        self.reset()

    def reset(self):
        """Réinitialise une nouvelle partie et retourne l'état initial.

        Returns:
            np.ndarray: Vecteur d'état (12,) float32 de la nouvelle partie.
        """
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.steps = 0
        self.steps_since_food = 0
        self.start_time = time.time()
        return self.get_state()

    def _obstacles(self):
        """Cases occupées par le corps au PROCHAIN coup (pour le raycast).

        Après move(), la collision est testée contre body[1:], qui contient
        l'ancienne tête mais PAS l'ancienne queue (sauf en cas de croissance,
        où la queue n'est pas retirée). On reproduit donc ici la même règle
        sur le corps actuel : on exclut la queue sauf si grow_pending.

        Returns:
            set: Ensemble de tuples (x, y) représentant les obstacles.
        """
        body = self.snake.body
        cells = body if self.snake.grow_pending else body[:-1]
        return {tuple(c) for c in cells}

    def _torus_delta(self, a, b):
        """Plus court écart signé de a vers b sur un axe torique de taille GRID_SIZE.

        Calcule la distance signée la plus courte sur le tore, en tenant compte de
        l'enroulement aux bords. Utilisé pour le calcul des distances à la pomme.

        Args:
            a: Coordonnée de départ (entier 0..GRID_SIZE-1).
            b: Coordonnée cible (entier 0..GRID_SIZE-1).

        Returns:
            int: Écart signé dans [-GRID_SIZE//2, GRID_SIZE//2].
        """
        diff = (b - a) % GRID_SIZE
        if diff > GRID_SIZE // 2:
            diff -= GRID_SIZE
        return diff

    def _apple_distance(self):
        """Distance de Manhattan tête->pomme sur le tore (0 si pas de pomme).

        Returns:
            int: Distance Manhattan en utilisant les écarts toriques.
        """
        if self.apple.position is None:
            return 0
        dx = self._torus_delta(self.snake.head_pos[0], self.apple.position[0])
        dy = self._torus_delta(self.snake.head_pos[1], self.apple.position[1])
        return abs(dx) + abs(dy)

    def get_state(self):
        """Construit le vecteur d'état (12,) float32 consommé par l'agent RL.

        Ordre imposé (changement non compatible avec les modèles entraînés) :
          [0:3]   Dangers immédiats (raycast tout droit / gauche / droite).
          [3:7]   Direction one-hot: UP (idx 3), DOWN (4), LEFT (5), RIGHT (6).
          [7:9]   Delta torique tête->pomme normalisée: x/GRID_SIZE, y/GRID_SIZE (0,0 si pas de pomme).
          [9:12]  Distances de vision normalisées (raycast tout droit / gauche / droite).

        Returns:
            np.ndarray: État (12,) float32.
        """
        head = tuple(self.snake.head_pos)
        direction = self.snake.direction
        obstacles = self._obstacles()

        # Un seul appel a raycast : il fournit a la fois dangers et distances.
        dangers, distances = raycast(head, direction, obstacles, GRID_SIZE)

        state = np.zeros(12, dtype=np.float32)
        state[0:3] = dangers

        dir_index = {UP: 3, DOWN: 4, LEFT: 5, RIGHT: 6}
        state[dir_index[direction]] = 1.0

        if self.apple.position is None:
            dx, dy = 0, 0
        else:
            dx = self._torus_delta(self.snake.head_pos[0], self.apple.position[0])
            dy = self._torus_delta(self.snake.head_pos[1], self.apple.position[1])
        state[7] = dx / GRID_SIZE
        state[8] = dy / GRID_SIZE

        state[9:12] = distances
        return state

    def play_step(self, action):
        """Joue une action et avance le jeu d'un pas.

        Applique l'action (index 0..3 : UP, DOWN, LEFT, RIGHT), met à jour l'état du jeu,
        calcule la récompense selon la table (manger +20, mourir -20, pas -0.5, variation
        distance ±1), et détecte la fin de l'épisode (mort, victoire, ou troncature après
        trop de pas sans manger).

        Args:
            action: Index d'action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).

        Returns:
            tuple: (reward (float), done (bool), score (int), info (dict)).
                info contient: steps (int), won (bool), truncated (bool), time_s (float).
        """
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # On signale la demande de fermeture sans interrompre brutalement
                    # une boucle d'entrainement/eval en cours.
                    self.quit_requested = True

        dist_before = self._apple_distance()

        self.snake.set_direction(ACTIONS[action])
        self.snake.move()
        self.steps += 1
        self.steps_since_food += 1

        reward = REWARD_STEP
        done = False
        info = {
            "steps": self.steps,
            "won": False,
            "truncated": False,
            # Temps de jeu nominal (a la vitesse reglementaire GAME_SPEED),
            # equivalent au chrono affiche : c'est la metrique "temps" du ratio score/temps.
            "time_s": self.steps / GAME_SPEED,
        }

        is_dead = self.snake.is_game_over()
        starved = self.steps_since_food > MAX_STEPS_WITHOUT_FOOD_FACTOR * len(self.snake.body)

        if is_dead or starved:
            reward += REWARD_DEATH
            done = True
            info["truncated"] = not is_dead
        elif self.apple.position is not None and self.snake.head_pos == list(self.apple.position):
            self.snake.grow()  # +1 au score affiche, regle inchangee
            self.steps_since_food = 0
            reward += REWARD_EAT
            if not self.apple.relocate(self.snake.body):
                # Plus aucune case libre pour la pomme : victoire (grille pleine).
                done = True
                info["won"] = True
        else:
            dist_after = self._apple_distance()
            if dist_after < dist_before:
                reward += REWARD_CLOSER
            elif dist_after > dist_before:
                reward += REWARD_FARTHER
            # egalite -> pas de bonus/malus de distance

        if self.render_enabled:
            self._draw()
            self.clock.tick(self.fps)  # = GAME_SPEED par defaut (partie manuelle), accelere en entrainement

        return reward, done, self.snake.score, info

    def _draw(self):
        """Reproduit exactement le bloc "3. Dessin" de main() du socle.

        Affiche la grille, la pomme, le serpent et les informations (score, temps).
        """
        self.screen.fill(GRIS_FOND)

        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(self.screen, NOIR, game_area_rect)

        draw_grid(self.screen)

        self.apple.draw(self.screen)
        self.snake.draw(self.screen)

        display_info(self.screen, self.font, self.snake, self.start_time)

        pygame.display.flip()

    def close(self):
        """Ferme pygame si l'environnement était en mode affichage."""
        if self.render_enabled:
            pygame.quit()


if __name__ == "__main__":
    # --- Tests headless (render=False), aucune fenetre requise ---

    # Test 1 : forme/type de l'etat + one-hot direction initiale (RIGHT).
    env = SnakeGameRL(render=False)
    state = env.get_state()
    assert state.shape == (12,), f"Shape incorrecte : {state.shape}"
    assert state.dtype == np.float32, f"Dtype incorrect : {state.dtype}"
    assert state[6] == 1.0, "One-hot direction initiale devrait etre RIGHT (index 6)"
    assert state[3] == 0 and state[4] == 0 and state[5] == 0
    print("Test 1 OK : shape (12,), dtype float32, one-hot RIGHT.")

    # Test 2 : etat initial, corps horizontal de 3, tete en (3,7) direction RIGHT.
    # Danger tout droit = 0 (rien immediatement devant).
    # Note (resolution d'ambiguite) : la vision "tout droit" n'est PAS 1.0 exactement.
    # Sur le tore, en avancant tout droit, on finit par reboucler et croiser
    # l'avant-derniere case du corps (la tete elle-meme et le segment du milieu
    # restent des obstacles "au prochain coup" ; seule la queue est exclue).
    # Avec GRID_SIZE=15 et un corps de longueur 3, cet obstacle est rencontre
    # a exactement GRID_SIZE-1 = 14 cases -> distance normalisee = 14/15.
    assert list(env.snake.head_pos) == [3, 7]
    assert env.snake.direction == RIGHT
    assert state[0] == 0, "Danger tout droit devrait etre 0 (corps derriere la tete)"
    expected_vision_straight = (GRID_SIZE - 1) / GRID_SIZE
    assert abs(state[9] - expected_vision_straight) < 1e-6, (
        f"Vision tout droit attendue {expected_vision_straight}, obtenue {state[9]}"
    )
    print(f"Test 2 OK : danger tout droit = 0, vision tout droit = {state[9]:.4f} (= 14/15, corps rencontre au bout de la boucle torique).")

    # Test 3 : manger une pomme placee juste devant la tete.
    env.reset()
    head = env.snake.head_pos
    env.apple.position = (head[0] + 1, head[1])  # case adjacente devant (direction RIGHT)
    reward, done, score, info = env.play_step(3)  # action RIGHT (index 3)
    assert reward == REWARD_STEP + REWARD_EAT, f"Reward incorrect : {reward}"
    assert score == 1, f"Score incorrect : {score}"
    assert done is False
    print(f"Test 3 OK : manger une pomme -> reward={reward}, score={score}.")

    # Test 4 : auto-collision provoquee (corps en boucle, on fonce dedans).
    env.reset()
    env.snake.head_pos = [5, 5]
    env.snake.body = [[5, 5], [6, 5], [6, 6], [5, 6], [4, 6]]
    env.snake.direction = UP  # pour pouvoir choisir RIGHT ensuite (pas l'inverse de UP)
    env.apple.position = (0, 0)  # hors de portee, pour ne pas interferer
    reward, done, score, info = env.play_step(3)  # action RIGHT -> percute le corps
    assert done is True, "Le serpent aurait du mourir par auto-collision"
    assert reward == REWARD_STEP + REWARD_DEATH, f"Reward incorrect : {reward}"
    assert info["truncated"] is False, "Une vraie collision n'est pas une troncature"
    print(f"Test 4 OK : auto-collision -> done={done}, reward={reward}.")

    # Test 5 : la pomme est loin, une action qui rapproche donne REWARD_CLOSER.
    env.reset()
    env.apple.position = (10, 7)  # loin devant, sur le meme rang que la tete (3,7)
    reward, done, score, info = env.play_step(3)  # RIGHT -> se rapproche
    assert reward == REWARD_STEP + REWARD_CLOSER, f"Reward incorrect : {reward}"
    print(f"Test 5 OK : se rapprocher de la pomme -> reward={reward}.")

    # Test 6 : franchissement du bord droit (tore), pas de mort.
    env.reset()
    env.snake.head_pos = [14, 7]
    env.snake.body = [[14, 7], [13, 7], [12, 7]]
    env.snake.direction = RIGHT
    env.apple.position = (7, 0)  # hors de portee
    reward, done, score, info = env.play_step(3)  # RIGHT -> wrap vers x=0
    assert env.snake.head_pos == [0, 7], f"Wrap incorrect : {env.snake.head_pos}"
    assert done is False, "Le tore ne doit jamais provoquer de mort par mur"
    print("Test 6 OK : wrap du bord droit vers x=0 sans mort.")

    # Test 7 : l'action inverse (LEFT quand direction RIGHT) est ignoree.
    env.reset()
    assert env.snake.direction == RIGHT
    old_head = list(env.snake.head_pos)
    reward, done, score, info = env.play_step(2)  # action LEFT demandee, index 2
    assert env.snake.direction == RIGHT, "La direction ne doit pas s'inverser"
    assert env.snake.head_pos == [old_head[0] + 1, old_head[1]], "Le serpent doit continuer tout droit"
    print("Test 7 OK : demi-tour ignore, le serpent continue tout droit.")

    # Test 8 : troncature apres trop de pas sans manger.
    env.reset()
    env.apple.position = (7, 0)  # hors de la trajectoire (le serpent reste sur le rang 7)
    done = False
    info = {}
    guard = 0
    while not done and guard < 10000:
        reward, done, score, info = env.play_step(3)  # continue tout droit
        guard += 1
    assert done is True, "L'episode aurait du se terminer par troncature"
    assert info["truncated"] is True, f"info incorrecte : {info}"
    print(f"Test 8 OK : troncature apres {info['steps']} pas sans manger, info={info}.")

    env.close()

    # --- Test optionnel : rendu graphique bref, ne doit jamais faire echouer les tests ---
    try:
        render_env = SnakeGameRL(render=True)
        for _ in range(5):
            render_env.play_step(3)
        render_env.close()
        print("Test optionnel OK : 5 pas en mode render=True.")
    except Exception as exc:  # pragma: no cover - environnement d'affichage indisponible
        print(f"Test optionnel ignore (affichage indisponible) : {exc}")

    print("Tous les tests de game.py sont passes avec succes.")
