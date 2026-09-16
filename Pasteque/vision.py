"""
vision.py - Module de vision par raycast pour le serpent RL.

Ce module est autonome (n'importe pas serpent-algo.py) : il reçoit `grid_size`
en paramètre (défaut 15, cf. GRID_SIZE dans serpent-algo.py).

Rappel important : dans le jeu, Snake.move() applique un modulo GRID_SIZE sur
les coordonnées -> la grille est un TORE (pas de murs, ils sont traversants).
Le seul obstacle réel est le corps du serpent.

Convention de coordonnées écran (y vers le bas), comme dans le socle :
    UP    = (0, -1)
    DOWN  = (0, 1)
    LEFT  = (-1, 0)
    RIGHT = (1, 0)
"""


def relative_directions(direction):
    """Convertit une direction absolue en triplet de directions relatives au serpent.

    Pour un serpent orienté selon `direction`, calcule les trois directions relatives :
    (tout droit, gauche, droite). La rotation est faite du point de vue du serpent,
    pas de l'écran, via une transformation 90° dans le plan (dx, dy).

    Args:
        direction: tuple (dx, dy) direction absolue de déplacement.

    Returns:
        tuple (straight, left, right) où chaque élément est un tuple (dx, dy).

    Exemple :
        RIGHT=(1,0)  -> left=(0,-1)=UP,   right=(0,1)=DOWN
        UP=(0,-1)    -> left=(-1,0)=LEFT, right=(1,0)=RIGHT
    """
    dx, dy = direction
    return (dx, dy), (dy, -dx), (-dy, dx)


def raycast(head, direction, obstacles, grid_size=15):
    """Lance 3 rayons (tout droit, gauche, droite) depuis la tête du serpent.

    Donne à l'agent une notion de marge de manœuvre (distance au premier obstacle)
    plutôt qu'un simple danger binaire. Cela aide le DQN à distinguer un chemin
    spacieux d'un couloir étroit, et à planifier d'avance avant de se coincer.

    Args:
        head: tuple (x, y), position de la tête.
        direction: tuple (dx, dy), direction absolue courante du serpent.
        obstacles: set de tuples (x, y), cases occupées par le corps au prochain coup.
        grid_size: int, taille de la grille torique (défaut 15).

    Retourne:
        (dangers, distances) tuple :
            - dangers: [int, int, int] ordre [tout droit, gauche, droite],
              1 si obstacle immédiatement adjacent (dist == 1), sinon 0.
            - distances: [float, float, float] ordre [tout droit, gauche, droite],
              valeur dist / grid_size dans (0, 1], où dist est le nombre de cases
              jusqu'au premier obstacle inclus (max grid_size si pas d'obstacle).
    """
    hx, hy = head
    dangers = []
    distances = []

    for dx, dy in relative_directions(direction):
        dist = grid_size  # valeur par défaut si aucun obstacle rencontré
        for step in range(1, grid_size):
            x = (hx + dx * step) % grid_size
            y = (hy + dy * step) % grid_size
            if (x, y) in obstacles:
                dist = step
                break
        dangers.append(1 if dist == 1 else 0)
        distances.append(dist / grid_size)

    return dangers, distances


if __name__ == "__main__":
    RIGHT = (1, 0)

    # Cas 1 : obstacle immédiatement adjacent tout droit -> danger + distance min.
    dangers, distances = raycast((7, 7), RIGHT, {(8, 7)}, grid_size=15)
    assert dangers[0] == 1
    assert distances[0] == 1 / 15

    # Cas 2 : aucun obstacle -> distance maximale (1.0) sur les 3 directions.
    dangers, distances = raycast((7, 7), RIGHT, set(), grid_size=15)
    assert dangers == [0, 0, 0]
    assert distances == [1.0, 1.0, 1.0]

    # Cas 3 : obstacle à gauche (UP) à 3 cases -> distance = 3/15.
    dangers, distances = raycast((7, 7), RIGHT, {(7, 4)}, grid_size=15)
    assert dangers[1] == 0
    assert distances[1] == 3 / 15

    # Cas 4 : wrap autour du tore, tête proche du bord droit.
    dangers, distances = raycast((14, 7), RIGHT, {(1, 7)}, grid_size=15)
    assert dangers[0] == 0
    assert distances[0] == 2 / 15

    print("Tous les tests vision.py sont passés.")
