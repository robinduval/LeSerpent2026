"""Planificateur SafePath : sert à générer des démonstrations pour le DQN.

Ce n'est pas le livrable snake-algo (abandonné sur consigne), seulement un outil
d'entraînement pour pré-remplir le replay buffer avec des parties compétentes,
au lieu de dépendre uniquement de l'exploration aléatoire pour trouver les
premières pommes sur une grille 15x15 (225 cases).

Algorithme (cf. Plan.md) :
1. BFS torique vers la pomme (plus court chemin, coût uniforme).
2. Simule le chemin (croissance différée comprise) et vérifie qu'après l'avoir
   suivi la tête peut encore rejoindre la queue, ou que l'espace libre restant
   est assez grand.
3. Si dangereux, suit un chemin sûr vers la queue pour gagner de l'espace.
4. Sinon, mouvement légal qui maximise la zone accessible (flood-fill), avec
   pénalité pour les culs-de-sac.
"""
from collections import deque

from .env import ACTIONS, OPPOSITE


def _neighbors(pos, g):
    x, y = pos
    for i, (dx, dy) in enumerate(ACTIONS):
        yield i, ((x + dx) % g, (y + dy) % g)


def bfs_path(start, goal, blocked, g):
    """Plus court chemin torique de `start` à `goal`, évitant `blocked`.
    Retourne la liste des indices d'action, ou None si inatteignable."""
    if start == goal:
        return []
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        for action, nxt in _neighbors(pos, g):
            if nxt in visited or nxt in blocked:
                continue
            new_path = path + [action]
            if nxt == goal:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def flood_fill_size(start, blocked, g, limit=None):
    """Taille de la zone accessible depuis `start` (BFS sans destination)."""
    visited = {start}
    queue = deque([start])
    count = 0
    while queue:
        pos = queue.popleft()
        count += 1
        if limit is not None and count >= limit:
            return count
        for _, nxt in _neighbors(pos, g):
            if nxt not in visited and nxt not in blocked:
                visited.add(nxt)
                queue.append(nxt)
    return count


def simulate_body_after_path(body, path, apple, g):
    """Simule le corps du serpent après avoir suivi `path` (indices d'action).
    Reproduit exactement les règles du moteur (croissance différée)."""
    body = [list(p) for p in body]
    grow_pending = False
    for action in path:
        dx, dy = ACTIONS[action]
        new_head = [(body[0][0] + dx) % g, (body[0][1] + dy) % g]
        body.insert(0, new_head)
        if not grow_pending:
            body.pop()
        else:
            grow_pending = False
        if tuple(new_head) == apple:
            grow_pending = True
    return [tuple(p) for p in body], grow_pending


def is_path_safe(body, path, apple, g, min_free_ratio=0.5):
    """Vérifie que suivre `path` laisse le serpent capable de survivre :
    soit la tête peut encore rejoindre la queue après coup, soit l'espace
    libre restant est assez grand par rapport à la longueur du serpent."""
    new_body, grow_pending = simulate_body_after_path(body, path, apple, g)
    head = new_body[0]
    tail = new_body[-1]
    # Le corps bloque tout sauf la queue elle-même (qui va bouger au prochain pas,
    # sauf si une croissance est en attente : dans ce cas elle reste sur place).
    blocked = set(new_body[:-1]) if not grow_pending else set(new_body)
    if head == tail:
        return True
    path_to_tail = bfs_path(head, tail, blocked, g)
    if path_to_tail is not None:
        return True
    free_space = flood_fill_size(head, blocked, g, limit=len(new_body) * 2)
    return free_space >= len(new_body) * min_free_ratio


def choose_action(env, min_free_ratio=0.5):
    """Choisit la prochaine action absolue pour l'environnement `env`."""
    g = env.grid_size
    body = [tuple(p) for p in env.body]
    head = body[0]
    blocked_body = set(body[:-1])  # la queue va bouger, sauf croissance
    if env.grow_pending:
        blocked_body.add(body[-1])
    legal = [i for i in range(4) if i != OPPOSITE[env.direction]]

    # 1. Chemin le plus court vers la pomme
    if env.apple is not None:
        path = bfs_path(head, env.apple, blocked_body, g)
        if path and path[0] in legal:
            if is_path_safe(body, path, env.apple, g, min_free_ratio):
                return path[0]

    # 2. Repli : chemin sûr vers la queue pour gagner de l'espace
    tail = body[-1]
    blocked_wo_tail = set(body[1:-1])
    path_to_tail = bfs_path(head, tail, blocked_wo_tail, g)
    if path_to_tail and path_to_tail[0] in legal:
        return path_to_tail[0]

    # 3. Dernier recours : le mouvement légal qui maximise l'espace accessible
    best_action, best_score = None, -1
    for action in legal:
        dx, dy = ACTIONS[action]
        new_head = ((head[0] + dx) % g, (head[1] + dy) % g)
        if new_head in blocked_body:
            continue
        space = flood_fill_size(new_head, blocked_body, g, limit=g * g)
        if space > best_score:
            best_score, best_action = space, action

    if best_action is not None:
        return best_action
    # Aucun mouvement sûr : on meurt quoi qu'il arrive, autant continuer tout droit
    return legal[0] if legal else env.direction
