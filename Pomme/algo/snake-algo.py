import pygame
import random
import time
import heapq
import logging
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

LOG_PATH = Path(__file__).resolve().with_name('decisions.log')
decision_logger = logging.getLogger('snake.decisions')
decision_logger.setLevel(logging.INFO)
decision_logger.propagate = False


def configure_decision_logging(log_path=LOG_PATH):
    """Ajoute les parties au même fichier, placé à côté du programme."""
    for handler in decision_logger.handlers[:]:
        handler.close()
        decision_logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    decision_logger.addHandler(handler)


def log_death(snake, apple, reason):
    """Un état complet permet de reproduire une mort sans deviner le corps."""
    decision_logger.info('DEATH %s', json.dumps(dict(
        reason=reason, score=snake.score, length=len(snake.body),
        head=snake.head_pos, direction=snake.direction, apple=apple,
        body=snake.body, grow_pending=snake.grow_pending,
        ticks_since_last_apple=snake.ticks_since_last_apple,
        last_mode=snake.last_mode,
        **fragmentation_log(analyze_free_space(snake.body, snake.head_pos))), separators=(',', ':')))


def log_victory(snake, start_time):
    decision_logger.info('VICTORY %s', json.dumps(dict(
        score=snake.score, length=len(snake.body),
        elapsed_time=round(time.time() - start_time, 2)), separators=(',', ':')))

# --- CONSTANTES DE JEU ---
# Taille de la grille (15x15)
GRID_SIZE = 15
# Taille d'une cellule en pixels
CELL_SIZE = 30
# Vitesse de jeu (images par seconde)
GAME_SPEED = 5
SAFETY_RATIO = 0.75
FOOD_LOOKAHEAD_DEPTH = 6
FOOD_BEAM_WIDTH = 12
POST_APPLE_SURVIVAL_DEPTH = 8
POST_APPLE_BEAM_WIDTH = 6
FOOD_GOAL_LIMIT = 4
POCKET_LOOKAHEAD_DEPTH = 6
POCKET_BEAM_WIDTH = 4
STAGNATION_LIMIT = 150
STAGNATION_TICKS = STAGNATION_LIMIT
RECENT_STATES_LIMIT = 512

# Dimensions de l'écran (avec espace pour le score/timer)
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

# Couleurs (en RGB)
BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0) # Tête du serpent
VERT = (0, 200, 0)    # Corps du serpent
ROUGE = (200, 0, 0)   # Pomme
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# --- SAFE TOROIDAL A* ---

def get_toroidal_neighbors(position):
    """Voisins dans l'ordre droite, gauche, bas, haut, bords inclus."""
    x, y = position
    return [((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
            for dx, dy in (RIGHT, LEFT, DOWN, UP)]


GRID_CELLS = frozenset((x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE))
GRID_NEIGHBORS = {p: tuple(get_toroidal_neighbors(p)) for p in GRID_CELLS}


def analyze_free_space(body, head):
    """Composantes libres toriques ; la tête peut accéder à plusieurs composantes."""
    return _analyze_free_space(frozenset(map(tuple, body)), tuple(head))


@lru_cache(maxsize=2048)
def _analyze_free_space(occupied, head):
    free = GRID_CELLS - occupied - {head}
    remaining, components = set(free), []
    accessible, isolated = set(), []
    entrances = set(GRID_NEIGHBORS[head])
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, queue = {start}, deque([start])
        while queue:
            for neighbor in GRID_NEIGHBORS[queue.popleft()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(frozenset(component))
        if component & entrances:
            accessible.update(component)
        else:
            isolated.append(frozenset(component))
    return dict(total_free_space=len(free), accessible_from_head=len(accessible),
                lost_space=len(free) - len(accessible),
                number_of_free_components=len(components),
                isolated_components_sizes=tuple(sorted(map(len, isolated))),
                largest_isolated_component=max(map(len, isolated), default=0),
                isolated_cells=frozenset().union(*isolated),
                accessible_cells=frozenset(accessible), components=tuple(components))


def fragmentation_log(metrics):
    return dict(total_free=metrics['total_free_space'],
                accessible=metrics['accessible_from_head'], lost_space=metrics['lost_space'],
                free_components=metrics['number_of_free_components'],
                isolated_components_sizes=metrics['isolated_components_sizes'],
                largest_isolated_component=metrics['largest_isolated_component'])


def check_dynamic_pocket(before, state, apple=None):
    """Cherche une réouverture des mêmes cases, pas juste une baisse du total perdu.

    Persistante signifie ici « non rouverte dans l'horizon borné », pas une preuve
    d'inaccessibilité éternelle. Aucune pomme future n'est générée.
    """
    after = analyze_free_space(state.body, state.body[0])
    targets = after['isolated_cells'] - before['isolated_cells']
    split = after['number_of_free_components'] > before['number_of_free_components']
    result = dict(new_lost_space=after['lost_space'] - before['lost_space'],
                  new_components=after['number_of_free_components'] - before['number_of_free_components'],
                  pocket_persistent=False, pocket_depth=0,
                  effective_lost_space=after['lost_space'],
                  effective_components=after['number_of_free_components'],
                  persistent_singletons=0)
    if not targets and not split:
        return result
    beam, seen = [simulated_copy(state)], {state_key(state)}
    for depth in range(1, POCKET_LOOKAHEAD_DEPTH + 1):
        children = []
        for parent in beam:
            for child in get_legal_moves(parent, apple).values():
                key = state_key(child)
                if key in seen:
                    continue
                seen.add(key)
                if not legal_directions(child) and len(child.body) < GRID_SIZE * GRID_SIZE:
                    continue
                metrics = analyze_free_space(child.body, child.body[0])
                remaining = len(targets & metrics['isolated_cells'])
                if (remaining == 0 and metrics['lost_space'] <= before['lost_space']
                        and metrics['number_of_free_components'] <= max(1, before['number_of_free_components'])):
                    result.update(pocket_depth=depth,
                                  effective_lost_space=metrics['lost_space'],
                                  effective_components=metrics['number_of_free_components'])
                    return result
                rank = (remaining, metrics['lost_space'], metrics['number_of_free_components'],
                        -len(legal_directions(child)))
                children.append((rank, child))
        if not children:
            break
        children.sort(key=lambda item: item[0])
        beam = [item[1] for item in children[:POCKET_BEAM_WIDTH]]
    result.update(pocket_persistent=True, pocket_depth=POCKET_LOOKAHEAD_DEPTH,
                  persistent_singletons=sum(len(c) == 1 and bool(c & targets)
                                            for c in after['components']))
    return result


def history_key(state, apple):
    state = simulated_copy(state)
    return state.body, state.direction, state.grow_pending, tuple(apple) if apple is not None else None


def geometry_key(state, apple):
    state = simulated_copy(state)
    occupied = set(state.body)
    local_body = tuple(p for p in GRID_NEIGHBORS[state.body[0]] if p in occupied)
    return state.body[0], state.direction, len(state.body), local_body, tuple(apple) if apple is not None else None


def recent_state_penalty(snake, state, apple):
    exact = snake.recent_state_hashes.count(history_key(state, apple))
    similar = snake.recent_geometries.count(geometry_key(state, apple))
    return exact, similar


def update_stagnation(snake, apple):
    repeated, similar = recent_state_penalty(snake, snake, apple)
    detected = (snake.ticks_since_last_apple > STAGNATION_LIMIT
                or repeated >= 2 or (similar >= 4 and snake.ticks_since_last_apple > 30))
    if detected and not snake.anti_stagnation_mode:
        snake.anti_stagnation_mode = True
        decision_logger.info('STAGNATION_DETECTED %s', json.dumps(dict(
            ticks_since_last_apple=snake.ticks_since_last_apple, score=snake.score,
            length=len(snake.body), head=snake.head_pos, apple=apple,
            repeated_states=repeated, similar_states=similar), separators=(',', ':')))
    return repeated


def evaluate_first_moves(snake, apple):
    before = analyze_free_space(snake.body, snake.body[0])
    candidates, ranks = {}, {}
    for direction, child in get_legal_moves(snake, apple).items():
        after = analyze_free_space(child.body, child.body[0])
        pocket = check_dynamic_pocket(before, child, apple)
        future = probe_survival(child)
        legal = len(legal_directions(child))
        exact, similar = recent_state_penalty(snake, child, apple)
        effective_new = pocket['effective_lost_space'] - before['lost_space']
        viable = future['victory'] or future['future_survival'] == POST_APPLE_SURVIVAL_DEPTH
        # Une fois la structure et les sorties suffisantes, départager par la pomme.
        structural = (viable, not pocket['pocket_persistent'],
                      -effective_new,
                      -pocket['persistent_singletons'] if len(snake.body) > 170 else 0,
                      -pocket['effective_components'],
                      min(2, legal),
                      min(SAFETY_RATIO, (after['total_free_space'] - pocket['effective_lost_space'])
                          / after['total_free_space']) if after['total_free_space'] else 1)
        novelty = (-exact, -similar) if snake.anti_stagnation_mode else (0, 0)
        ranks[direction] = structural + novelty
        candidates[direction_name(direction)] = dict(
            collision=False, **fragmentation_log(after), **pocket, legal_moves=legal,
            distance_to_apple=toroidal_manhattan(child.body[0], apple) if apple else None,
            recent_state_penalty=exact, similar_state_penalty=similar,
            future_survival=future['future_survival'], viable=viable)
    for direction in (UP, DOWN, LEFT, RIGHT):
        candidates.setdefault(direction_name(direction), dict(collision=True))
    best = max(ranks.values(), default=None)
    allowed = {direction for direction, rank in ranks.items() if rank == best}
    return before, candidates, allowed


def route_preserves_space(snake, path, apple):
    state = simulated_copy(snake)
    for position in path:
        before = analyze_free_space(state.body, state.body[0])
        child = simulate_path(state, [position], apple)
        if child is None or check_dynamic_pocket(before, child, apple)['pocket_persistent']:
            return False
        state = child
    return True


def toroidal_manhattan(a, b):
    """Heuristique A* : chaque axe peut emprunter le bord le plus proche."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return min(dx, GRID_SIZE - dx) + min(dy, GRID_SIZE - dy)


def direction_to(current, following):
    for direction, neighbor in zip(
            (RIGHT, LEFT, DOWN, UP), get_toroidal_neighbors(current)):
        if neighbor == tuple(following):
            return direction
    return None


def astar(start, target, body, direction=None):
    """A* conservateur : le corps actuel reste un obstacle pendant la recherche.

    Le chemin exclut le départ. None indique qu'aucune route n'existe.
    """
    if target is None:
        return None
    start, target = tuple(start), tuple(target)
    blocked = {tuple(position) for position in body}
    blocked.discard(start)
    frontier = [(toroidal_manhattan(start, target), 0, start)]
    costs, parents = {start: 0}, {}
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if cost != costs[current]:
            continue
        if current == target:
            path = []
            while current != start:
                path.append(current)
                current = parents[current]
            return path[::-1]
        for neighbor in get_toroidal_neighbors(current):
            if neighbor in blocked:
                continue
            if current == start and direction is not None:
                if direction_to(start, neighbor) == (-direction[0], -direction[1]):
                    continue
            next_cost = cost + 1
            if next_cost < costs.get(neighbor, float('inf')):
                costs[neighbor] = next_cost
                parents[neighbor] = current
                heapq.heappush(frontier, (
                    next_cost + toroidal_manhattan(neighbor, target),
                    next_cost, neighbor))
    return None


@dataclass(frozen=True)
class SimulatedSnake:
    body: tuple
    direction: tuple
    grow_pending: bool
    apple_eaten: bool = False


def simulated_copy(snake):
    if isinstance(snake, SimulatedSnake):
        return snake
    return SimulatedSnake(tuple(map(tuple, snake.body)), snake.direction,
                          snake.grow_pending)


def state_key(snake):
    state = simulated_copy(snake)
    return state.body, state.direction, state.grow_pending, state.apple_eaten


def legal_directions(snake):
    state = simulated_copy(snake)
    # La queue est libérée durant ce coup, sauf croissance déjà en attente.
    blocked = set(state.body if state.grow_pending else state.body[:-1])
    reverse = (-state.direction[0], -state.direction[1])
    return {direction: position for direction, position in zip(
        (RIGHT, LEFT, DOWN, UP), get_toroidal_neighbors(state.body[0]))
        if direction != reverse and position not in blocked}


def simulate_path(snake, path, apple_position):
    """Copie légère ; croissance différée exactement comme move() puis grow()."""
    state = simulated_copy(snake)
    for position in path:
        position = tuple(position)
        direction = direction_to(state.body[0], position)
        moves = get_legal_moves(state, apple_position)
        if direction not in moves:
            return None
        state = moves[direction]
    return state


def flood_fill_space(head, body):
    """Compte les cases LIBRES accessibles ; la tête n'entre pas dans le total."""
    head = tuple(head)
    blocked = {tuple(position) for position in body}
    visited, frontier = {head}, deque([head])
    while frontier:
        for neighbor in GRID_NEIGHBORS[frontier.popleft()]:
            if neighbor not in blocked and neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) - 1


def get_legal_moves(snake, apple_position=None):
    """Associe chaque direction légale à son état après un déplacement."""
    state = simulated_copy(snake)
    apple = tuple(apple_position) if apple_position is not None else None
    rest = state.body if state.grow_pending else state.body[:-1]
    return {direction: SimulatedSnake(
        (position,) + rest, direction,
        not state.apple_eaten and position == apple,
        state.apple_eaten or position == apple)
        for direction, position in legal_directions(state).items()}


def space_metrics(state):
    total = GRID_SIZE * GRID_SIZE - len(state.body)
    space = flood_fill_space(state.body[0], state.body)
    return dict(accessible_space=space, total_free_space=total,
                safety_ratio=space / total if total else 1.0)


@dataclass
class SearchBranch:
    state: SimulatedSnake
    path: tuple = ()
    first_move: tuple = None
    min_legal_moves: int = 3
    forced_move_count: int = 0
    revisits: int = 0


def extend_branch(branch, direction, state, recent=()):
    exits = len(legal_directions(state))
    # La grille pleine est terminale : aucune sortie n'est alors nécessaire.
    won = len(state.body) == GRID_SIZE * GRID_SIZE
    return SearchBranch(
        state, branch.path + (state.body[0],), branch.first_move or direction,
        min(branch.min_legal_moves, exits) if not won else branch.min_legal_moves,
        branch.forced_move_count + (exits <= 1 and not won),
        branch.revisits + (state_key(state) in recent))


def probe_survival(state):
    """Beam court après la pomme, sans inventer la prochaine pomme."""
    root = SearchBranch(simulated_copy(state))
    beam, deepest = [root], root
    expanded = 0
    for _ in range(POST_APPLE_SURVIVAL_DEPTH):
        unique = {}
        for branch in beam:
            if len(branch.state.body) == GRID_SIZE * GRID_SIZE:
                return dict(future_survival=len(branch.path), victory=True,
                            min_legal_moves=branch.min_legal_moves,
                            forced_move_count=branch.forced_move_count,
                            states_expanded=expanded)
            for direction, child in get_legal_moves(branch.state).items():
                expanded += 1
                candidate = extend_branch(branch, direction, child)
                key = state_key(child)
                rank = (-candidate.forced_move_count, candidate.min_legal_moves)
                previous = unique.get(key)
                if previous is None or rank > (-previous.forced_move_count, previous.min_legal_moves):
                    unique[key] = candidate
        if not unique:
            break
        beam = sorted(unique.values(), key=lambda b: (
            -b.forced_move_count, b.min_legal_moves), reverse=True)[:POST_APPLE_BEAM_WIDTH]
        deepest = beam[0]
    return dict(future_survival=len(deepest.path),
                victory=len(deepest.state.body) == GRID_SIZE * GRID_SIZE,
                min_legal_moves=deepest.min_legal_moves,
                forced_move_count=deepest.forced_move_count, states_expanded=expanded)


def assess_safety(state):
    metrics = space_metrics(state)
    metrics.update(probe_survival(state))
    exits = len(legal_directions(state))
    metrics['legal_moves_count'] = exits
    metrics['min_legal_moves'] = min(exits, metrics['min_legal_moves'])
    if metrics['victory']:
        reason = None
    elif metrics['future_survival'] < POST_APPLE_SURVIVAL_DEPTH:
        reason = 'post_apple_trap'
    elif metrics['safety_ratio'] < SAFETY_RATIO:
        reason = 'insufficient_space'
        actual = analyze_free_space(state.body, state.body[0])
        baseline = dict(actual, lost_space=0, isolated_cells=frozenset(), number_of_free_components=1)
        if actual['lost_space'] and not check_dynamic_pocket(baseline, state)['pocket_persistent']:
            reason = None
            metrics['temporary_pocket'] = True
    else:
        reason = None
    metrics.update(accepted=reason is None, rejection_reason=reason)
    return metrics


def is_safe_path(simulated):
    return assess_safety(simulated)['accepted']


def choose_safest_move(snake, apple_position):
    """Dernier secours si le beam n'a conservé aucune branche vivante."""
    moves = get_legal_moves(snake, apple_position)
    if not moves:
        return None

    def rank(direction):
        simulated = moves[direction]
        space = flood_fill_space(simulated.body[0], simulated.body)
        distance = (toroidal_manhattan(simulated.body[0], apple_position)
                    if apple_position is not None else 0)
        return len(legal_directions(simulated)), space, -distance

    return max(moves, key=rank)


def food_branch_rank(branch, apple, stagnant=False):
    """Élagage orienté pomme ; les branches mortes ont déjà été exclues."""
    distance = toroidal_manhattan(branch.state.body[0], apple) if apple else 0
    space = analyze_free_space(branch.state.body, branch.state.body[0])
    return (-space['lost_space'], -space['number_of_free_components'],
            -branch.revisits if stagnant else 0, -distance,
            -branch.forced_move_count, branch.min_legal_moves)


def balanced_beam(branches, apple, stagnant):
    """Réserve une part du beam à chaque premier coup encore possible."""
    groups = {}
    for branch in branches:
        groups.setdefault(branch.first_move, []).append(branch)
    selected, rest = [], []
    quota = max(1, FOOD_BEAM_WIDTH // max(1, len(groups)))
    for group in groups.values():
        group.sort(key=lambda b: food_branch_rank(b, apple, stagnant), reverse=True)
        selected.extend(group[:quota])
        rest.extend(group[quota:])
    rest.sort(key=lambda b: food_branch_rank(b, apple, stagnant), reverse=True)
    return (selected + rest[:max(0, FOOD_BEAM_WIDTH - len(selected))])[:FOOD_BEAM_WIDTH]


def branch_report(branch, apple):
    report = space_metrics(branch.state)
    route = astar(branch.state.body[0], apple, branch.state.body, branch.state.direction)
    report.update(depth_reached=len(branch.path), apple_reached=branch.state.apple_eaten,
                  apple_depth=None, post_apple_survival=0,
                  legal_moves_count=len(legal_directions(branch.state)),
                  min_legal_moves=branch.min_legal_moves,
                  forced_move_count=branch.forced_move_count,
                  distance_to_apple=toroidal_manhattan(branch.state.body[0], apple) if apple else None,
                  astar_available_after_simulation=bool(route), revisits=branch.revisits)
    return report


def food_lookahead(snake, apple, allowed_moves=None):
    """Explore le vrai corps ; nourriture sûre > ouverture vers elle > survie."""
    root = SearchBranch(simulated_copy(snake))
    recent = set(getattr(snake, 'recent_states', ()))
    stagnant = getattr(snake, 'ticks_since_last_apple', 0) >= STAGNATION_TICKS
    beam, visited = [root], {state_key(root.state)}
    survivors, goals, rejected = {}, {}, {}
    expanded = post_expanded = evaluations = depth_reached = 0
    for depth in range(1, FOOD_LOOKAHEAD_DEPTH + 1):
        unique, arrivals = {}, []
        for branch in beam:
            for direction, child in get_legal_moves(branch.state, apple).items():
                if depth == 1 and allowed_moves is not None and direction not in allowed_moves:
                    continue
                expanded += 1
                candidate = extend_branch(branch, direction, child, recent)
                key = state_key(child)
                if key in visited:
                    continue
                previous = unique.get(key)
                if previous is None or food_branch_rank(candidate, apple, stagnant) > food_branch_rank(previous, apple, stagnant):
                    unique[key] = candidate
        if not unique:
            break
        depth_reached = depth
        live = []
        for key, branch in unique.items():
            visited.add(key)
            if branch.state.apple_eaten:
                arrivals.append(branch)
            elif legal_directions(branch.state):
                live.append(branch)
        # Borne déterministe : pas de dépendance au temps de la machine.
        arrivals.sort(key=lambda b: (-b.forced_move_count, b.min_legal_moves), reverse=True)
        for branch in arrivals:
            if evaluations >= FOOD_GOAL_LIMIT:
                break
            evaluations += 1
            safety = assess_safety(branch.state)
            if safety['accepted'] and not route_preserves_space(snake, branch.path, apple):
                safety.update(accepted=False, rejection_reason='persistent_pocket')
            post_expanded += safety['states_expanded']
            report = dict(safety, apple_reached=True, apple_depth=depth,
                          post_apple_survival=safety['future_survival'],
                          min_legal_moves=min(branch.min_legal_moves, safety['min_legal_moves']),
                          forced_move_count=branch.forced_move_count + safety['forced_move_count'])
            rank = (-report['forced_move_count'], report['min_legal_moves'],
                    report['safety_ratio'], -depth)
            if safety['accepted']:
                previous = goals.get(branch.first_move)
                if previous is None or rank > previous[0]:
                    goals[branch.first_move] = (rank, branch, report)
            else:
                rejected.setdefault(branch.first_move, report)
        beam = balanced_beam(live, apple, stagnant)
        if not beam:
            break
        for branch in beam:
            survivors.setdefault(branch.first_move, [])
            if (survivors[branch.first_move]
                    and len(survivors[branch.first_move][0].path) < depth):
                survivors[branch.first_move] = []
            survivors[branch.first_move].append(branch)

    reports, finalists = {}, []
    for direction in (UP, DOWN, LEFT, RIGHT):
        name = direction_name(direction)
        if direction not in legal_directions(root.state):
            reports[name] = dict(status='REVERSE_OR_COLLISION')
            continue
        choices = []
        for branch in survivors.get(direction, []):
            report = branch_report(branch, apple)
            # Entre branches viables, rechercher une ouverture vers la pomme.
            progress = (report['astar_available_after_simulation'] or (
                apple is not None and report['distance_to_apple'] < toroidal_manhattan(root.state.body[0], apple)))
            opening = (len(branch.path) == FOOD_LOOKAHEAD_DEPTH and progress
                       and report['safety_ratio'] >= SAFETY_RATIO)
            rank = (opening, len(branch.path), -branch.forced_move_count,
                    branch.min_legal_moves, -branch.revisits if stagnant else 0,
                    report['astar_available_after_simulation'], report['accessible_space'],
                    -(report['distance_to_apple'] or 0))
            choices.append((rank, branch, report))
        if choices:
            best = max(choices, key=lambda item: item[0])
            finalists.append(best)
            reports[name] = dict(best[2], status='OPENING' if best[0][0] else 'SURVIVAL')
        else:
            reports[name] = dict(status='NO_SURVIVING_BRANCH')
        if direction in rejected:
            reports[name]['rejected_apple'] = rejected[direction]
        if direction in goals:
            reports[name] = dict(goals[direction][2], status='SAFE_APPLE',
                                 distance_to_apple=0,
                                 depth_reached=len(goals[direction][1].path))
            if direction in rejected:
                reports[name]['rejected_apple'] = rejected[direction]

    if goals:
        _, chosen, report = max(goals.values(), key=lambda item: item[0])
        mode, reason = 'FOOD_LOOKAHEAD', 'safe_alternative'
    elif finalists:
        rank, chosen, report = max(finalists, key=lambda item: item[0])
        mode = 'FOOD_LOOKAHEAD' if rank[0] else 'TEMP_SURVIVAL'
        reason = 'opens_food_route' if rank[0] else 'no_safe_food_route_within_limits'
    else:
        legal = get_legal_moves(snake, apple)
        permitted = [d for d in legal if allowed_moves is None or d in allowed_moves]
        direction = min(permitted, key=lambda d: toroidal_manhattan(legal[d].body[0], apple)
                        if apple else 0, default=None)
        child = get_legal_moves(root.state, apple).get(direction)
        chosen = extend_branch(root, direction, child, recent) if child else None
        report = branch_report(chosen, apple) if chosen else {}
        mode, reason = 'TEMP_SURVIVAL', 'no_surviving_branch' if child else 'no_legal_move'
    summary = dict(report, depth_requested=FOOD_LOOKAHEAD_DEPTH,
                   depth_reached=len(chosen.path) if chosen else 0,
                   search_depth_reached=depth_reached, states_expanded=expanded,
                   post_states_expanded=post_expanded, food_evaluations=evaluations,
                   goal_limit_reached=evaluations >= FOOD_GOAL_LIMIT,
                   reason=reason,
                   reason_no_food_route=reason if mode == 'TEMP_SURVIVAL' else None,
                   chosen_first_move=direction_name(chosen.first_move) if chosen else '-')
    return chosen, mode, summary, reports


def direction_name(direction):
    return {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}.get(direction, '-')


def choose_ai_direction(snake, apple_position):
    """Repart toujours d'A* ; aucun mode de survie n'est mémorisé comme verrou."""
    started = time.perf_counter()
    apple = tuple(apple_position) if apple_position is not None else None
    repeated = update_stagnation(snake, apple)
    before, move_reports, allowed = evaluate_first_moves(snake, apple)
    path = astar(snake.body[0], apple, snake.body, snake.direction)
    direct = dict(path_length=len(path) if path is not None else None,
                  accepted=False, rejection_reason='no_path' if apple else 'no_apple',
                  safety_ratio=None, accessible_space=None, future_survival=0)
    chosen, mode, search, candidates = None, 'ASTAR_DIRECT', None, None
    if len(snake.body) < GRID_SIZE * GRID_SIZE and path:
        branch = SearchBranch(simulated_copy(snake))
        for position in path:
            direction = direction_to(branch.state.body[0], position)
            child = get_legal_moves(branch.state, apple).get(direction)
            if child is None:
                direct['rejection_reason'] = 'simulated_collision'
                break
            branch = extend_branch(branch, direction, child)
        else:
            direct.update(assess_safety(branch.state))
            direct['min_legal_moves'] = min(branch.min_legal_moves, direct['min_legal_moves'])
            direct['forced_move_count'] += branch.forced_move_count
            if direct['accepted'] and branch.first_move not in allowed:
                report = move_reports[direction_name(branch.first_move)]
                direct.update(accepted=False, rejection_reason=(
                    'persistent_pocket' if report['pocket_persistent']
                    else 'better_structure_or_unvisited_move'))
            if direct['accepted'] and not route_preserves_space(snake, path, apple):
                direct.update(accepted=False, rejection_reason='persistent_pocket')
            if direct['accepted']:
                chosen = branch
    if chosen is None and len(snake.body) < GRID_SIZE * GRID_SIZE:
        chosen, mode, search, candidates = food_lookahead(snake, apple, allowed)
    direction = chosen.first_move if chosen else None
    record = dict(head=tuple(snake.body[0]), apple=apple, length=len(snake.body),
                  direction=direction_name(direction), mode=mode,
                  ticks_since_last_apple=getattr(snake, 'ticks_since_last_apple', 0),
                  astar=direct, fragmentation_before=fragmentation_log(before),
                  repeated_state_count=repeated, anti_stagnation=snake.anti_stagnation_mode,
                  candidates=move_reports, CHOSEN=direction_name(direction))
    if direction is not None:
        record.update({key: value for key, value in move_reports[direction_name(direction)].items()
                       if key != 'collision'})
    if search is not None:
        record.update(search=search, search_candidates=candidates)
    record['decision_ms'] = round((time.perf_counter() - started) * 1000, 2)
    snake.last_mode, snake.last_decision = mode, record
    snake.last_plan = chosen.path if chosen else ()
    if hasattr(snake, 'recent_states'):
        snake.recent_states.append(state_key(snake))
        snake.recent_state_hashes.append(history_key(snake, apple))
        snake.recent_geometries.append(geometry_key(snake, apple))
        snake.last_apple = apple
    if decision_logger.handlers:
        decision_logger.info('%s', json.dumps(record, separators=(',', ':')))
    return direction


# --- CLASSES DU JEU ---

class Snake:
    """Représente le serpent, sa position, sa direction et son corps."""
    def __init__(self):
        # Position initiale au centre
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        # Le corps est une liste de positions (x, y), incluant la tête
        self.body = [self.head_pos, 
                     [self.head_pos[0] - 1, self.head_pos[1]], 
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0
        self.ticks_since_last_apple = 0
        self.recent_states = deque(maxlen=RECENT_STATES_LIMIT)
        self.recent_state_hashes = deque(maxlen=RECENT_STATES_LIMIT)
        self.recent_geometries = deque(maxlen=RECENT_STATES_LIMIT)
        self.anti_stagnation_mode = False
        self.last_apple = None
        self.last_mode = '-'
        self.last_decision = {}
        self.last_plan = ()

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse immédiat."""
        # Vérifie que la nouvelle direction n'est pas l'inverse de l'actuelle
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case dans la direction actuelle."""
        self.ticks_since_last_apple += 1
        # Calcul de la nouvelle position de la tête
        new_head_x = (self.head_pos[0] + self.direction[0]) % GRID_SIZE
        new_head_y = (self.head_pos[1] + self.direction[1]) % GRID_SIZE
        
        # Mettre à jour la tête (la nouvelle position devient la nouvelle tête)
        new_head_pos = [new_head_x, new_head_y]
        self.body.insert(0, new_head_pos)
        self.head_pos = new_head_pos

        # Si le serpent ne doit pas grandir, supprime la queue (mouvement normal)
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False # Réinitialise le drapeau

    def grow(self):
        """Prépare le serpent à grandir au prochain mouvement."""
        self.grow_pending = True
        self.score += 1
        if self.anti_stagnation_mode:
            decision_logger.info('STAGNATION_RESOLVED score=%s length=%s ticks_since_last_apple=%s',
                                 self.score, len(self.body), self.ticks_since_last_apple)
        self.anti_stagnation_mode = False
        self.ticks_since_last_apple = 0
        self.recent_states.clear()
        self.recent_state_hashes.clear()
        self.recent_geometries.clear()

    def check_wall_collision(self):
        """Les déplacements toriques conservent toujours la tête dans la grille."""
        x, y = self.head_pos
        return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

    def check_self_collision(self):
        """Vérifie si la tête touche une partie du corps (Game Over si auto-morsure)."""
        # On vérifie si la position de la tête est dans le reste du corps (body[1:])
        return self.head_pos in self.body[1:]

    def is_game_over(self):
        """Retourne True si le jeu est terminé (mur ou morsure)."""
        return self.check_wall_collision() or self.check_self_collision()

    def draw(self, surface):
        """Dessine le serpent sur la surface de jeu."""
        # Dessine le corps
        for segment in self.body[1:]:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, VERT, rect)
            pygame.draw.rect(surface, NOIR, rect, 1) # Bordure

        # Dessine la tête (couleur différente)
        head_rect = pygame.Rect(self.head_pos[0] * CELL_SIZE, self.head_pos[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, ORANGE, head_rect)
        pygame.draw.rect(surface, NOIR, head_rect, 2) # Bordure plus épaisse

class Apple:
    """Représente la pomme (nourriture) et sa position."""
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        """Trouve une position aléatoire non occupée par le serpent."""
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        occupied = {tuple(position) for position in occupied_positions}
        available_positions = [pos for pos in all_positions if pos not in occupied]
        
        if not available_positions:
            return None # Toutes les cases sont pleines (condition de Victoire)
            
        return random.choice(available_positions)

    def relocate(self, snake_body):
        """Déplace la pomme vers une nouvelle position aléatoire."""
        new_pos = self.random_position(snake_body)
        self.position = new_pos
        return new_pos is not None

    def draw(self, surface):
        """Dessine la pomme sur la surface de jeu."""
        if self.position:
            rect = pygame.Rect(self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
            # Ajout d'un petit reflet pour un aspect "pomme"
            pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)

# --- FONCTIONS D'AFFICHAGE ---

def draw_grid(surface):
    """Dessine la grille pour une meilleure visualisation."""
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))

def display_info(surface, font, snake, start_time):
    """Affiche le score et le temps écoulé dans le panneau supérieur."""
    
    # Dessiner le panneau de score
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    # Afficher le score
    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 20))

    # Afficher le temps
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_text = font.render(f"Temps: {minutes:02d}:{seconds:02d}", True, BLANC)
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 20))
    
    # Afficher le taux de remplissage
    max_cells = GRID_SIZE * GRID_SIZE
    fill_rate = (len(snake.body) / max_cells) * 100
    fill_text = font.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill_text, (SCREEN_WIDTH // 2 - fill_text.get_width() // 2, 20))

def display_message(surface, font, message, color=BLANC, y_offset=0):
    """
    Affiche un message central sur l'écran, avec un décalage vertical optionnel.
    y_offset permet de positionner plusieurs messages.
    """
    text_surface = font.render(message, True, color)
    # Applique le décalage vertical au centre
    center_y = (SCREEN_HEIGHT // 2) + y_offset
    rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, center_y))
    
    # Dessine un fond semi-transparent pour la lisibilité
    padding = 20
    bg_rect = rect.inflate(padding * 2, padding * 2)
    pygame.draw.rect(surface, NOIR, bg_rect, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg_rect, 2, border_radius=10)
    surface.blit(text_surface, rect)

# --- BOUCLE PRINCIPALE DU JEU ---

def main():
    """Lance immédiatement le Snake autonome Safe Toroidal A*."""
    configure_decision_logging()
    decision_logger.info('DEBUT')
    pygame.init()
    
    # Configuration de l'écran
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - Safe Toroidal A*")
    clock = pygame.time.Clock()
    
    # Configuration des polices
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)
    
    # Initialisation des objets du jeu
    snake = Snake()
    apple = Apple(snake.body)
    
    # Variables de jeu
    running = True
    game_over = False
    victory = False
    
    # Démarrage du chronomètre
    start_time = time.time()
    
    # Variable pour la gestion de la vitesse (pour ne bouger qu'une fois par tic)
    move_counter = 0

    # --- Boucle de jeu ---
    while running:
        # 1. Gestion des Événements (Contrôles Clavier)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN and (game_over or victory):
                if event.key == pygame.K_SPACE:
                    decision_logger.info('REJOUER')
                    snake = Snake()
                    apple = Apple(snake.body)
                    game_over = victory = False
                    start_time = time.time()
                    move_counter = 0

        if not running:
            break
        
        # 2. Logique de Mise à Jour du Jeu
        if not game_over and not victory:
            # Le serpent se déplace à la vitesse définie
            move_counter += 1
            if move_counter >= GAME_SPEED // 10: # Déplace le serpent à un rythme constant
                if len(snake.body) == GRID_SIZE * GRID_SIZE:
                    victory = game_over = True
                    log_victory(snake, start_time)
                    continue
                if apple.position is None and not apple.relocate(snake.body):
                    victory = game_over = True
                    log_victory(snake, start_time)
                    continue
                direction = choose_ai_direction(snake, apple.position)
                if direction is None:
                    log_death(snake, apple.position, 'no_legal_move')
                    game_over = True
                    continue
                snake.set_direction(direction)
                snake.move()
                move_counter = 0

                # Vérification des collisions (murs et corps)
                if snake.is_game_over():
                    log_death(snake, apple.position, 'collision')
                    game_over = True
                    continue # Passe à l'affichage de Game Over

                # Vérification de la pomme mangée
                if apple.position is not None and tuple(snake.head_pos) == tuple(apple.position):
                    snake.grow()
                    
                    # Tente de replacer la pomme, vérifie la Victoire si échec
                    if not apple.relocate(snake.body):
                        victory = True # Plus d'espace pour la pomme
                        game_over = True # Met fin au jeu

                if len(snake.body) == GRID_SIZE * GRID_SIZE:
                    victory = game_over = True
                if victory:
                    log_victory(snake, start_time)
        
        # 3. Dessin
        screen.fill(GRIS_FOND) # Fond gris pour la zone de score
        
        # Zone de jeu (décalée par la hauteur du panneau de score)
        game_area_rect = pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH)
        pygame.draw.rect(screen, NOIR, game_area_rect)
        
        draw_grid(screen)
        
        # Dessine la pomme et le serpent
        apple.draw(screen)
        snake.draw(screen)
        
        # Affiche le score et le temps
        display_info(screen, font_main, snake, start_time)
        
        # Affichage des messages de fin de jeu
        if game_over:
            if victory:
                # Le premier message est centré (y_offset=0 par défaut)
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
                message_details = "ESPACE pour rejouer."
                # Le deuxième message est décalé vers le bas
                display_message(screen, font_main, message_details, BLANC, y_offset=100) 
            else:
                # Le premier message est centré (y_offset=0 par défaut)
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
                message_details = "ESPACE pour rejouer."
                # Le deuxième message est décalé vers le bas
                display_message(screen, font_main, message_details, BLANC, y_offset=100)
        
        # Mise à jour de l'affichage
        pygame.display.flip()
        
        # Contrôle la vitesse du jeu
        clock.tick(GAME_SPEED)

    decision_logger.info('FIN score=%s longueur=%s', snake.score, len(snake.body))
    for handler in decision_logger.handlers[:]:
        handler.close()
        decision_logger.removeHandler(handler)
    pygame.quit()

if __name__ == '__main__':
    main()
