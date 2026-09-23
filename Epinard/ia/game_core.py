"""
Chargement du jeu original SANS le modifier.

serpent-algo.py contient un tiret dans son nom : on ne peut pas faire
`import serpent-algo`. On passe donc par importlib, qui charge le fichier
tel quel. Le `if __name__ == '__main__'` du fichier original empêche
main() de se lancer à l'import : rien ne s'affiche, aucune fenêtre ne s'ouvre.

On récupère ici les classes Snake / Apple et les constantes, telles quelles.
AUCUNE ligne de serpent-algo.py n'est modifiée.
"""
import importlib.util
import os
import sys

# pygame est importé par le fichier original. En environnement headless
# (serveur sans écran), on force le driver "dummy" pour que l'import passe.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORIGINAL = os.path.join(_HERE, "serpent-algo.py")

_spec = importlib.util.spec_from_file_location("serpent_algo_original", _ORIGINAL)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["serpent_algo_original"] = _mod
_spec.loader.exec_module(_mod)

# --- Ce qu'on réutilise du jeu original ---
Snake = _mod.Snake
Apple = _mod.Apple

GRID_SIZE = _mod.GRID_SIZE          # 15
CELL_SIZE = _mod.CELL_SIZE          # 30
GAME_SPEED = _mod.GAME_SPEED        # 5 (la clock : on n'y touche pas)
SCORE_PANEL_HEIGHT = _mod.SCORE_PANEL_HEIGHT
SCREEN_WIDTH = _mod.SCREEN_WIDTH
SCREEN_HEIGHT = _mod.SCREEN_HEIGHT

UP, DOWN, LEFT, RIGHT = _mod.UP, _mod.DOWN, _mod.LEFT, _mod.RIGHT

# Couleurs (pour l'affichage Pygame)
BLANC, NOIR = _mod.BLANC, _mod.NOIR
ORANGE, VERT, ROUGE = _mod.ORANGE, _mod.VERT, _mod.ROUGE
GRIS_FOND, GRIS_GRILLE = _mod.GRIS_FOND, _mod.GRIS_GRILLE

# Ordre horaire des directions : sert à traduire une action relative
# (tout droit / droite / gauche) en direction absolue.
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]


def turn(direction, action_index):
    """
    Traduit une action relative en direction absolue.

    action_index : 0 = tout droit, 1 = tourner à droite, 2 = tourner à gauche.

    On raisonne en relatif (et non en HAUT/BAS/GAUCHE/DROITE absolus) parce que
    le demi-tour est de toute façon refusé par set_direction() ligne 51 du jeu
    original. Avec 3 actions relatives, aucune action n'est jamais gaspillée.
    """
    i = CLOCKWISE.index(direction)
    if action_index == 0:
        return CLOCKWISE[i]
    if action_index == 1:
        return CLOCKWISE[(i + 1) % 4]
    return CLOCKWISE[(i - 1) % 4]


def wrap(x, y):
    """Applique le bouclage torique, comme move() lignes 57-58 du jeu original."""
    return x % GRID_SIZE, y % GRID_SIZE
