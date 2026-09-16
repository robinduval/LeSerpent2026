"""Règles officielles du jeu, reprises telles quelles du socle du professeur.

Ce module est la SOURCE DE VÉRITÉ des constantes non négociables. Aucune
expérimentation d'apprentissage n'a le droit de les modifier : la grille, la
clock et le scoring doivent rester identiques au socle pour que le benchmark
final soit équitable.

Référence : `serpent-algo.py` (conservé intact à la racine du dossier Banane).
"""

# --- Constantes officielles (valeurs réellement exécutées par le socle) ---
# Le commentaire du socle parle de 20x20, mais la constante vaut 15.
# C'est la constante exécutable qui fait foi.
GRID_SIZE = 15
# Identité d'environnement : les anciens résultats sans ce tag sont legacy.
RULESET = "snake-torus-v1"
CELL_SIZE = 30
GAME_SPEED = 5

# Le serpent démarre avec une longueur de 3.
INITIAL_LENGTH = 3

# Dimensions d'affichage (identiques au socle).
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# --- Directions ---
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# --- Actions absolues, dans l'ordre imposé par le cours ---
# 0 = haut, 1 = bas, 2 = gauche, 3 = droite
ACTIONS = (UP, DOWN, LEFT, RIGHT)
N_ACTIONS = len(ACTIONS)

ACTION_NAMES = ("up", "down", "left", "right")

# Nombre de cases total : sert à détecter la victoire (grille pleine).
TOTAL_CELLS = GRID_SIZE * GRID_SIZE

# --- Couleurs (reprises du socle pour que le rendu reste reconnaissable) ---
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)


def opposite(direction):
    """Retourne la direction opposée."""
    return (-direction[0], -direction[1])
