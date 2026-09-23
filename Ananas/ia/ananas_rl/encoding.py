"""Encodage de l'état : toute la grille, centrée sur la tête (la grille est torique).

La tête est toujours en (0, 0) : le réseau convolutif à padding circulaire lit ses
caractéristiques à cet endroit, ce qui rend le modèle indépendant de la taille de grille.
"""
import numpy as np

N_CHANNELS = 10
# 0 tête | 1 corps | 2 temps avant libération d'une case du corps | 3 pomme
# 4 longueur relative | 5 croissance en attente | 6-9 direction courante (one-hot)


def encode(env):
    g = env.grid_size
    obs = np.zeros((N_CHANNELS, g, g), dtype=np.float32)
    hx, hy = env.body[0]
    length = len(env.body)
    cells = g * g
    extra = 1 if env.grow_pending else 0
    for i, (x, y) in enumerate(env.body):
        cx, cy = (x - hx) % g, (y - hy) % g
        if i == 0:
            obs[0, cy, cx] = 1.0
            continue
        obs[1, cy, cx] = 1.0
        # Le segment i quitte sa case dans (length - i + extra) déplacements
        obs[2, cy, cx] = (length - i + extra) / cells
    if env.apple is not None:
        ax, ay = env.apple
        obs[3, (ay - hy) % g, (ax - hx) % g] = 1.0
    obs[4] = length / cells
    obs[5] = float(env.grow_pending)
    obs[6 + env.direction] = 1.0
    return obs


# --- État compact (celui du cours) -------------------------------------------
# L'encodage grille ci-dessus s'est révélé inexploitable sur CPU : 6 canaux sur 10
# sont des plans constants et la pomme n'est qu'une case sur 225, donc après
# convolution + pooling le réseau apprend une constante et ignore l'entrée
# (Q-values identiques à 0.05 près sur des états différents).
# Cet état compact donne au réseau exactement les variables qui décident du coup
# suivant, et s'entraîne en minutes au lieu d'heures.

N_COMPACT = 15

# Rotations (indices d'ACTIONS : 0=HAUT, 1=BAS, 2=GAUCHE, 3=DROITE)
TURN_RIGHT = {0: 3, 3: 1, 1: 2, 2: 0}
TURN_LEFT = {3: 0, 1: 3, 2: 1, 0: 2}


def _danger(env, direction, distance, occupied):
    """1.0 si avancer de `distance` cases dans `direction` tombe sur le corps."""
    from .env import ACTIONS
    g = env.grid_size
    dx, dy = ACTIONS[direction]
    x, y = env.body[0]
    cell = ((x + dx * distance) % g, (y + dy * distance) % g)
    return 1.0 if cell in occupied else 0.0


def encode_compact(env):
    """15 variables : dangers (3 directions x 2 distances), direction courante,
    position relative de la pomme, et remplissage de la grille."""
    g = env.grid_size
    d = env.direction
    right, left = TURN_RIGHT[d], TURN_LEFT[d]
    # La queue libère sa case au prochain déplacement, sauf croissance en attente.
    occupied = {tuple(p) for p in (env.body if env.grow_pending else env.body[:-1])}

    obs = np.zeros(N_COMPACT, dtype=np.float32)
    obs[0] = _danger(env, d, 1, occupied)
    obs[1] = _danger(env, right, 1, occupied)
    obs[2] = _danger(env, left, 1, occupied)
    obs[3] = _danger(env, d, 2, occupied)
    obs[4] = _danger(env, right, 2, occupied)
    obs[5] = _danger(env, left, 2, occupied)

    obs[6 + d] = 1.0  # direction courante (one-hot, canaux 6-9)

    if env.apple is not None:
        hx, hy = env.body[0]
        # Écart le plus court sur le tore, ramené dans [-g/2, g/2]
        dx = ((env.apple[0] - hx + g // 2) % g) - g // 2
        dy = ((env.apple[1] - hy + g // 2) % g) - g // 2
        obs[10] = 1.0 if dx < 0 else 0.0  # pomme à gauche
        obs[11] = 1.0 if dx > 0 else 0.0  # pomme à droite
        obs[12] = 1.0 if dy < 0 else 0.0  # pomme en haut
        obs[13] = 1.0 if dy > 0 else 0.0  # pomme en bas

    obs[14] = len(env.body) / (g * g)
    return obs


# --- État compact v2 : + espace libre accessible -----------------------------
# La v1 plafonne vers 25-28 pommes : ne voyant que 2 cases devant, le serpent
# s'enferme dans des poches dès qu'il est long. On ajoute, pour chacun des 3 coups
# possibles, la fraction de l'espace libre encore accessible après ce coup
# (flood-fill). Le réseau reste seul à décider ; il reçoit juste l'information.

N_COMPACT_V2 = N_COMPACT + 3


def encode_compact_v2(env):
    from .env import ACTIONS

    g = env.grid_size
    d = env.direction
    occupied = {tuple(p) for p in (env.body if env.grow_pending else env.body[:-1])}
    free = max(1, g * g - len(occupied))
    x, y = env.body[0]

    extra = np.zeros(3, dtype=np.float32)
    regions = []  # (cases visitées, taille) : les 3 cases candidates partagent
                  # presque toujours la même région, un seul parcours suffit.
    for k, direction in enumerate((d, TURN_RIGHT[d], TURN_LEFT[d])):
        dx, dy = ACTIONS[direction]
        cell = ((x + dx) % g, (y + dy) % g)
        if cell in occupied:
            continue
        for visited, size in regions:
            if cell in visited:
                extra[k] = size / free
                break
        else:
            visited = _region(cell, occupied, g)
            regions.append((visited, len(visited)))
            extra[k] = len(visited) / free
    return np.concatenate([encode_compact(env), extra])


def _region(start, blocked, g):
    """Ensemble des cases libres accessibles depuis `start` (grille torique)."""
    seen = {start}
    stack = [start]
    while stack:
        cx, cy = stack.pop()
        for nxt in (((cx + 1) % g, cy), ((cx - 1) % g, cy), (cx, (cy + 1) % g), (cx, (cy - 1) % g)):
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                stack.append(nxt)
    return seen
