"""
Filtre de sécurité : flood-fill torique.

Ce n'est PAS du pathfinding. On ne cherche aucun chemin vers la pomme.
On répond à une seule question binaire, pour chaque action candidate :

    "Après ce coup, l'espace libre atteignable est-il assez grand
     pour que le serpent puisse encore y tenir ?"

C'est un remplissage par diffusion (BFS de connexité), pas Dijkstra,
pas A*, pas GBFS. Coût O(nb de cases) = O(225) par action.

Le résultat sert de MASQUE D'ACTIONS pour PPO : les actions jugées
non sûres voient leur logit mis à -inf AVANT le softmax. La politique
échantillonne donc uniquement parmi les coups sûrs, et les log-probs
restent cohérents avec l'action réellement jouée (indispensable pour
que les ratios PPO soient corrects).

OPTIMISATION
------------
`analyse()` calcule danger + masque + espace en UN SEUL passage de
3 flood-fill. Les anciennes fonctions séparées en faisaient 6 par pas
(3 pour l'état, 3 pour le masque) : le profilage montrait 69% du temps
d'entraînement passé là. Le flood-fill utilise une grille plate
d'entiers et un buffer réutilisé, plutôt que des sets de tuples.
"""
from collections import deque

from game_core import GRID_SIZE, turn

N_CELLS = GRID_SIZE * GRID_SIZE

# Décalages des 4 voisins, pré-calculés pour chaque case (bouclage torique).
# Évite de recalculer les modulos à chaque visite.
_NEIGHBORS = []
for _idx in range(N_CELLS):
    _x, _y = _idx % GRID_SIZE, _idx // GRID_SIZE
    _NEIGHBORS.append((
        ((_x) % GRID_SIZE) + ((_y - 1) % GRID_SIZE) * GRID_SIZE,
        ((_x) % GRID_SIZE) + ((_y + 1) % GRID_SIZE) * GRID_SIZE,
        ((_x - 1) % GRID_SIZE) + ((_y) % GRID_SIZE) * GRID_SIZE,
        ((_x + 1) % GRID_SIZE) + ((_y) % GRID_SIZE) * GRID_SIZE,
    ))

# Buffer réutilisé entre appels : on marque avec un compteur croissant
# au lieu de réinitialiser la grille (économise un memset par flood-fill).
_seen = [0] * N_CELLS
_stamp = 0


def _wrap(x, y):
    return x % GRID_SIZE, y % GRID_SIZE


def _idx(x, y):
    return (x % GRID_SIZE) + (y % GRID_SIZE) * GRID_SIZE


def simulate_step(body, direction, action_index):
    """
    Simule un pas SANS toucher au jeu réel.

    Retourne (nouveau_corps, mort, nouvelle_direction).
    Reproduit fidèlement move() du jeu original : insertion en tête,
    bouclage torique, suppression de la queue.
    """
    new_dir = turn(direction, action_index)
    hx, hy = body[0]
    nx, ny = _wrap(hx + new_dir[0], hy + new_dir[1])
    new_head = [nx, ny]
    new_body = [new_head] + body[:-1]
    mort = new_head in new_body[1:]
    return new_body, mort, new_dir


def _flood(head_idx, blocked):
    """
    Flood-fill depuis head_idx. `blocked` est un set d'indices plats.
    Retourne le nombre de cases libres atteignables.
    """
    global _stamp
    _stamp += 1
    stamp = _stamp

    _seen[head_idx] = stamp
    queue = deque((head_idx,))
    count = 0

    while queue:
        cur = queue.popleft()
        for nxt in _NEIGHBORS[cur]:
            if _seen[nxt] == stamp or nxt in blocked:
                continue
            _seen[nxt] = stamp
            count += 1
            queue.append(nxt)

    return count


def analyse(body, direction, margin=1.0):
    """
    UN SEUL passage : renvoie (danger, mask, space) pour les 3 actions.

      danger[a] : 1.0 si l'action tue immédiatement, sinon 0.0
      mask[a]   : True si l'action est sûre (survit ET espace >= longueur)
      space[a]  : espace libre après l'action, normalisé [0,1]

    Remplace les anciens appels séparés à safe_actions() + space_after(),
    qui refaisaient les mêmes flood-fill deux fois par pas de jeu.
    """
    need = len(body) * margin
    danger, mask, space = [], [], []

    for a in range(3):
        new_body, mort, _ = simulate_step(body, direction, a)
        if mort:
            danger.append(1.0)
            mask.append(False)
            space.append(0.0)
            continue

        blocked = {_idx(c[0], c[1]) for c in new_body[1:]}
        head_idx = _idx(new_body[0][0], new_body[0][1])
        free = _flood(head_idx, blocked)

        danger.append(0.0)
        mask.append(free >= need)
        space.append(free / N_CELLS)

    # Aucune action sûre : on n'interdit plus que les morts immédiates,
    # pour laisser la politique choisir plutôt que de la bloquer.
    if not any(mask):
        mask = [d == 0.0 for d in danger]
        if not any(mask):
            mask = [True, True, True]  # condamné quoi qu'il arrive

    return danger, mask, space


def free_space(body):
    """Espace libre atteignable depuis la tête (corps = obstacle, bords = tore)."""
    blocked = {_idx(c[0], c[1]) for c in body[1:]}
    return _flood(_idx(body[0][0], body[0][1]), blocked)


def safe_actions(body, direction, margin=1.0):
    """Masque de sécurité seul. Préférer analyse() dans les boucles chaudes."""
    return analyse(body, direction, margin)[1]


def space_after(body, direction):
    """Espace libre après chaque action. Préférer analyse() dans les boucles chaudes."""
    return analyse(body, direction)[2]


def fallback_action(body, direction):
    """
    Quand aucune action n'est sûre : choisir celle qui laisse le plus
    d'espace. Mourir le plus tard possible, et parfois s'en sortir.
    """
    best_a, best_space = 0, -1
    for a in range(3):
        new_body, mort, _ = simulate_step(body, direction, a)
        s = -1 if mort else free_space(new_body)
        if s > best_space:
            best_a, best_space = a, s
    return best_a
