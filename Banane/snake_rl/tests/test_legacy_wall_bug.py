"""Démonstration du bug mural du socle fourni par le professeur.

Ces tests documentent le comportement RÉEL de `serpent-algo.py` (non modifié).
Ils ne testent pas le moteur RL : ils servent de preuve reproductible que la
grille du socle se comporte comme un tore et que `check_wall_collision()` est
inatteignable. Ils doivent continuer à passer, car `serpent-algo.py` est
archivé tel quel comme référence.

Note : le socle importe pygame au chargement du module, mais aucun test ici
n'ouvre de fenêtre (on n'appelle jamais `main()`).
"""

import importlib.util
import os
import sys

import pytest

LEGACY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "serpent-algo.py",
)


@pytest.fixture(scope="module")
def legacy():
    """Charge `serpent-algo.py` (nom non importable : tiret) sans l'exécuter."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    spec = importlib.util.spec_from_file_location("serpent_algo_legacy", LEGACY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_constants_are_the_official_ones(legacy):
    """La constante exécutée fait foi : 15, malgré le commentaire qui dit 20x20."""
    assert legacy.GRID_SIZE == 15
    assert legacy.CELL_SIZE == 30
    assert legacy.GAME_SPEED == 5


@pytest.mark.parametrize(
    "start, direction, expected_wrapped_head",
    [
        # tête collée au bord gauche, on avance encore vers la gauche
        ([0, 7], "LEFT", [14, 7]),
        # bord droit
        ([14, 7], "RIGHT", [0, 7]),
        # bord haut
        ([7, 0], "UP", [7, 14]),
        # bord bas
        ([7, 14], "DOWN", [7, 0]),
    ],
)
def test_legacy_snake_teleports_through_walls(
    legacy, start, direction, expected_wrapped_head
):
    """BUG : le modulo de `move()` fait réapparaître le serpent de l'autre côté.

    Attendu par les règles du jeu : sortir de la grille = Game Over.
    Observé sur le socle : la tête est téléportée au bord opposé, vivante.
    """
    snake = legacy.Snake()
    snake.head_pos = list(start)
    snake.body = [list(start)]
    snake.direction = getattr(legacy, direction)

    snake.move()

    assert snake.head_pos == expected_wrapped_head, "le socle devrait wrapper (bug)"
    assert snake.check_wall_collision() is False
    assert snake.is_game_over() is False, (
        "le serpent a traversé un mur sans mourir : la grille est un tore"
    )


def test_legacy_wall_collision_is_unreachable_after_move(legacy):
    """`check_wall_collision()` ne peut JAMAIS retourner True après un `move()`.

    Preuve exhaustive : on balaie toutes les cases de la grille et les quatre
    directions. Le modulo garantit que la tête reste toujours dans
    [0, GRID_SIZE[ sur les deux axes, donc la condition de collision murale
    est mathématiquement morte.
    """
    n = legacy.GRID_SIZE
    directions = [legacy.UP, legacy.DOWN, legacy.LEFT, legacy.RIGHT]

    for x in range(n):
        for y in range(n):
            for direction in directions:
                snake = legacy.Snake()
                snake.head_pos = [x, y]
                snake.body = [[x, y]]
                snake.direction = direction
                snake.move()
                assert snake.check_wall_collision() is False


def test_legacy_move_counter_is_dead_code(legacy):
    """Effet de bord constaté : le seuil du compteur de mouvement vaut 0.

    `move_counter >= GAME_SPEED // 10` vaut `>= 0`, donc toujours vrai.
    Le serpent bouge à chaque frame ; la vitesse réelle vient uniquement de
    `clock.tick(GAME_SPEED)`. Ce n'est pas une règle officielle, c'est du code
    mort que le moteur RL n'a pas à reproduire.
    """
    assert legacy.GAME_SPEED // 10 == 0
