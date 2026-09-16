"""Environnement d'entraînement sans rendu, reproduisant exactement les règles du moteur.

Règles relevées dans `serpent-algo.py` (non modifié) :
- grille torique : la tête passe d'un bord à l'autre (`% GRID_SIZE`), pas de mort contre un mur ;
- demi-tour ignoré : le serpent continue tout droit ;
- déplacement : nouvelle tête insérée, queue retirée sauf si une croissance est en attente ;
- mort uniquement si la tête touche `body[1:]` après le déplacement ;
- pomme mangée : croissance au déplacement suivant, score +1, pomme replacée ;
- victoire quand la pomme ne trouve plus de case libre.

La taille de grille est un paramètre pour pouvoir s'entraîner sur des grilles plus petites ;
l'évaluation et la démonstration utilisent la taille du moteur (15).
"""
import random

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
# Ordre des sorties du réseau : HAUT, BAS, GAUCHE, DROITE
ACTIONS = (UP, DOWN, LEFT, RIGHT)
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

# Récompenses d'entraînement (le score du jeu reste 1 point par pomme)
REWARD_APPLE = 10.0
REWARD_DEATH = -10.0
REWARD_WIN = 100.0
REWARD_STEP = 0.0  # Pas de pénalité pour encourager l'exploration

# Reward shaping par potentiel (Ng et al. 1999) : ne change pas la politique
# optimale, mais donne un signal dense vers la pomme au lieu d'attendre de la
# trouver par hasard sur 225 cases. Phi(s) = -distance_torique(tête, pomme).
SHAPING_COEF = 0.5


def torus_distance(a, b, g):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return min(dx, g - dx) + min(dy, g - dy)

END_COLLISION = "collision"
END_VICTORY = "victoire"
END_LOOP = "boucle"


class SnakeEnv:
    def __init__(self, grid_size=15, seed=None, max_idle_steps=None):
        self.grid_size = grid_size
        self.rng = random.Random(seed)
        # Coupe les parties où le serpent tourne sans jamais manger. Sur le tore,
        # le plus court chemin entre deux cases fait au plus grid_size (aller-retour
        # sur chaque axe compris) ; 8x grid_size laisse une large marge d'exploration
        # sans laisser une politique quasi aléatoire consommer des centaines de pas
        # (important en tout début d'entraînement, où l'exploration est proche
        # de l'aléatoire et ferait autrement traîner chaque épisode).
        self.max_idle_steps = max_idle_steps or 8 * grid_size
        self.reset()

    def seed(self, seed):
        self.rng.seed(seed)

    def reset(self):
        g = self.grid_size
        head = [g // 4, g // 2]
        self.body = [head, [head[0] - 1, head[1]], [head[0] - 2, head[1]]]
        self.direction = ACTIONS.index(RIGHT)
        self.grow_pending = False
        self.score = 0
        self.steps = 0
        self.idle_steps = 0
        self.done = False
        self.end_cause = None
        self.apple = self._random_position()
        return self

    @property
    def head(self):
        return self.body[0]

    def _random_position(self):
        # Même ordre d'énumération que Apple.random_position pour que la graine
        # produise les mêmes pommes que le moteur.
        g = self.grid_size
        occupied = {tuple(p) for p in self.body}
        available = [(x, y) for x in range(g) for y in range(g) if (x, y) not in occupied]
        if not available:
            return None
        return self.rng.choice(available)

    def legal_mask(self):
        """Actions autorisées : tout sauf le demi-tour (ignoré par le moteur)."""
        mask = [True] * 4
        mask[OPPOSITE[self.direction]] = False
        return mask

    def step(self, action, gamma=0.99, shaping=True):
        """Joue une action absolue. Retourne (reward, done)."""
        assert not self.done
        g = self.grid_size
        phi_before = -torus_distance(tuple(self.body[0]), self.apple, g) if (shaping and self.apple) else 0.0

        if action != OPPOSITE[self.direction]:
            self.direction = action
        dx, dy = ACTIONS[self.direction]
        new_head = [(self.body[0][0] + dx) % g, (self.body[0][1] + dy) % g]
        self.body.insert(0, new_head)
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False
        self.steps += 1
        self.idle_steps += 1

        if new_head in self.body[1:]:
            self.done = True
            self.end_cause = END_COLLISION
            return REWARD_DEATH, True  # Phi(terminal) = 0, pas de shaping ici

        if tuple(new_head) == self.apple:
            self.grow_pending = True
            self.score += 1
            self.idle_steps = 0
            new_apple = self._random_position()
            if new_apple is None:
                self.done = True
                self.end_cause = END_VICTORY
                return REWARD_WIN, True
            self.apple = new_apple
            return REWARD_APPLE, False  # pomme mangée : pas de shaping, plus de distance à mesurer

        reward = REWARD_STEP
        if shaping:
            phi_after = -torus_distance(tuple(new_head), self.apple, g)
            reward += SHAPING_COEF * (gamma * phi_after - phi_before)

        if self.idle_steps >= self.max_idle_steps:
            self.done = True
            self.end_cause = END_LOOP
        return reward, self.done

    def truncated(self):
        """Fin artificielle (boucle) : la valeur future ne doit pas être annulée."""
        return self.end_cause == END_LOOP
