"""Construction du vecteur d'état à 11 valeurs imposé par le cours.

L'ORDRE DES COMPOSANTES EST GELÉ. Les checkpoints, les tests et toutes les
comparaisons entre configurations en dépendent. Toute extension de l'état doit
donner lieu à une variante explicitement nommée, jamais à une réécriture
silencieuse de cette fonction.

    0  danger_straight
    1  danger_right
    2  danger_left
    3  direction_left
    4  direction_right
    5  direction_up
    6  direction_down
    7  food_left
    8  food_right
    9  food_up
    10 food_down
"""

import numpy as np

from . import rules

STATE_SIZE = 11

STATE_LABELS = (
    "danger_straight",
    "danger_right",
    "danger_left",
    "direction_left",
    "direction_right",
    "direction_up",
    "direction_down",
    "food_left",
    "food_right",
    "food_up",
    "food_down",
)


def turn_right(direction):
    """Quart de tour horaire à l'écran (l'axe y pointe vers le bas)."""
    dx, dy = direction
    return (-dy, dx)


def turn_left(direction):
    """Quart de tour anti-horaire à l'écran."""
    dx, dy = direction
    return (dy, -dx)


def _ahead(head, direction):
    return (
        (head[0] + direction[0]) % rules.GRID_SIZE,
        (head[1] + direction[1]) % rules.GRID_SIZE,
    )


def build_state(game):
    """Produit le vecteur d'état de `game` sous forme de `np.float32[11]`."""
    head = game.head
    direction = game.direction

    straight = direction
    right = turn_right(direction)
    left = turn_left(direction)

    food = game.food
    # Grille pleine : plus de pomme. On neutralise les bits de nourriture
    # plutôt que de planter, la partie est de toute façon gagnée.
    if food is None:
        food_left = food_right = food_up = food_down = False
    else:
        food_left = food[0] < head[0]
        food_right = food[0] > head[0]
        food_up = food[1] < head[1]
        food_down = food[1] > head[1]

    return np.array(
        [
            game.is_collision(_ahead(head, straight)),
            game.is_collision(_ahead(head, right)),
            game.is_collision(_ahead(head, left)),
            direction == rules.LEFT,
            direction == rules.RIGHT,
            direction == rules.UP,
            direction == rules.DOWN,
            food_left,
            food_right,
            food_up,
            food_down,
        ],
        dtype=np.float32,
    )
