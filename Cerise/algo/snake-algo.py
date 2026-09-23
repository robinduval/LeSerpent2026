from __future__ import annotations

from collections import Counter, deque
import heapq
import importlib.util
from pathlib import Path
import random
import time

import pygame


# Chargement direct du socle officiel : les regles ne sont pas reimplementees.
ROOT_DIRECTORY = Path(__file__).resolve().parents[2]
BASE_GAME_PATH = ROOT_DIRECTORY / "serpent-algo.py"
BASE_SPEC = importlib.util.spec_from_file_location("serpent_algo_base", BASE_GAME_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Impossible de charger le jeu de base : {BASE_GAME_PATH}")
BASE_GAME = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE_GAME)

GRID_SIZE = BASE_GAME.GRID_SIZE
CELL_SIZE = BASE_GAME.CELL_SIZE
GAME_SPEED = BASE_GAME.GAME_SPEED
SCREEN_WIDTH = BASE_GAME.SCREEN_WIDTH
SCORE_PANEL_HEIGHT = BASE_GAME.SCORE_PANEL_HEIGHT
SCREEN_HEIGHT = BASE_GAME.SCREEN_HEIGHT

BLANC = BASE_GAME.BLANC
NOIR = BASE_GAME.NOIR
VERT = BASE_GAME.VERT
ROUGE = BASE_GAME.ROUGE
GRIS_FOND = BASE_GAME.GRIS_FOND
GRIS_GRILLE = BASE_GAME.GRIS_GRILLE
BLEU = (0, 190, 255)

UP = BASE_GAME.UP
DOWN = BASE_GAME.DOWN
LEFT = BASE_GAME.LEFT
RIGHT = BASE_GAME.RIGHT
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)

Position = tuple[int, int]
State = tuple[tuple[Position, ...], Position, bool]


def add_position(position: Position, direction: Position) -> Position:
    """Applique un deplacement sur la grille torique du jeu de base."""
    return (
        (position[0] + direction[0]) % GRID_SIZE,
        (position[1] + direction[1]) % GRID_SIZE,
    )


def opposite(direction: Position) -> Position:
    return (-direction[0], -direction[1])


def toroidal_distance(first: Position, second: Position) -> int:
    """Distance de Manhattan tenant compte du passage par les bords."""
    dx = abs(first[0] - second[0])
    dy = abs(first[1] - second[1])
    return min(dx, GRID_SIZE - dx) + min(dy, GRID_SIZE - dy)


def direction_between(first: Position, second: Position) -> Position:
    for direction in DIRECTIONS:
        if add_position(first, direction) == second:
            return direction
    raise ValueError(f"Les cases {first} et {second} ne sont pas voisines")


Snake = BASE_GAME.Snake
Apple = BASE_GAME.Apple


class HamiltonianShortcut:
    """Cycle hamiltonien garanti, accelere par des raccourcis A* dynamiques."""

    def __init__(self) -> None:
        self.cycle = self._build_cycle()
        self.index = {position: index for index, position in enumerate(self.cycle)}
        self.planned_path: deque[Position] = deque()
        self.planned_apple: Position | None = None

    @staticmethod
    def _build_cycle() -> list[Position]:
        # Mot periodique optimise pour une grille torique 15 x 15. Par rapport
        # au serpentin ligne par ligne, ses virages repartis offrent beaucoup
        # plus de raccourcis locaux a A*.
        word = (
            RIGHT,
            RIGHT,
            RIGHT,
            RIGHT,
            RIGHT,
            DOWN,
            DOWN,
            RIGHT,
            RIGHT,
            RIGHT,
            DOWN,
            DOWN,
            DOWN,
            DOWN,
            DOWN,
        )
        position = (0, 0)
        raw_cycle = []
        for _ in range(GRID_SIZE):
            for move in word:
                raw_cycle.append(position)
                position = add_position(position, move)

        # Oriente le cycle dans le sens initial du serpent officiel :
        # (1, 7) -> (2, 7) -> (3, 7).
        cycle = []
        for index, current in enumerate(raw_cycle):
            previous = raw_cycle[(index - 1) % len(raw_cycle)]
            before_previous = raw_cycle[(index - 2) % len(raw_cycle)]
            if (
                add_position(before_previous, RIGHT) == previous
                and add_position(previous, RIGHT) == current
            ):
                dx = (3 - current[0]) % GRID_SIZE
                dy = (7 - current[1]) % GRID_SIZE
                translated = [
                    ((point[0] + dx) % GRID_SIZE, (point[1] + dy) % GRID_SIZE)
                    for point in raw_cycle
                ]
                cycle = translated[index:] + translated[:index]
                break

        if (
            position != (0, 0)
            or len(cycle) != GRID_SIZE * GRID_SIZE
            or len(set(cycle)) != len(cycle)
        ):
            raise RuntimeError("Cycle hamiltonien invalide")
        for index, position in enumerate(cycle):
            following = cycle[(index + 1) % len(cycle)]
            if toroidal_distance(position, following) != 1:
                raise RuntimeError("Deux cases du cycle ne sont pas voisines")
        return cycle

    def _distance_on_cycle(self, start: Position, end: Position) -> int:
        return (self.index[end] - self.index[start]) % len(self.cycle)

    def _shortest_ordered_path(
        self,
        head: Position,
        food: Position,
        direction: Position,
        maximum_progress: int,
    ) -> list[Position] | None:
        """Plus court chemin dont les indices avancent strictement vers la pomme."""
        queue = deque([head])
        parents: dict[Position, Position | None] = {head: None}

        while queue:
            current = queue.popleft()
            if current == food:
                path = []
                cursor = current
                while parents[cursor] is not None:
                    path.append(cursor)
                    cursor = parents[cursor]  # type: ignore[assignment]
                path.reverse()
                return path

            current_progress = self._distance_on_cycle(head, current)
            for move in DIRECTIONS:
                if current == head and move == opposite(direction):
                    continue
                neighbor = add_position(current, move)
                progress = self._distance_on_cycle(head, neighbor)
                if not current_progress < progress <= maximum_progress:
                    continue
                if neighbor in parents:
                    continue
                parents[neighbor] = current
                queue.append(neighbor)
        return None

    def _path_is_safe(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        path: list[Position],
        food: Position,
    ) -> bool:
        virtual_body = list(body)
        virtual_direction = direction
        virtual_growth = grow_pending

        for next_head in path:
            move = direction_between(virtual_body[0], next_head)
            if move == opposite(virtual_direction):
                return False
            virtual_body.insert(0, next_head)
            if virtual_growth:
                virtual_growth = False
            else:
                virtual_body.pop()
            if next_head in virtual_body[1:]:
                return False
            virtual_direction = move
            if next_head == food:
                virtual_growth = True

        # La croissance differee doit encore laisser une reprise sure du cycle.
        for _ in range(3):
            head = virtual_body[0]
            successor = self.cycle[(self.index[head] + 1) % len(self.cycle)]
            move = direction_between(head, successor)
            if move == opposite(virtual_direction):
                return False
            virtual_body.insert(0, successor)
            if virtual_growth:
                virtual_growth = False
            else:
                virtual_body.pop()
            if successor in virtual_body[1:]:
                return False
            virtual_direction = move
        return True

    @staticmethod
    def _transition(
        body: tuple[Position, ...],
        direction: Position,
        grow_pending: bool,
        move: Position,
        food: Position,
    ) -> State | None:
        """Simule exactement un mouvement avec les regles du jeu officiel."""
        if move == opposite(direction):
            return None

        new_head = add_position(body[0], move)
        if grow_pending:
            new_body = (new_head, *body)
        else:
            new_body = (new_head, *body[:-1])
        if new_head in new_body[1:]:
            return None
        return new_body, move, new_head == food

    def _body_is_cycle_ordered(self, body: tuple[Position, ...]) -> bool:
        """Verifie que suivre le cycle liberera toujours les cases dans l'ordre."""
        occupied_arc = 0
        for index in range(1, len(body)):
            gap = (
                self.index[body[index - 1]] - self.index[body[index]]
            ) % len(self.cycle)
            if gap == 0:
                return False
            occupied_arc += gap
            if occupied_arc >= len(self.cycle):
                return False
        return True

    def _can_resume_cycle(self, state: State) -> bool:
        """Valide la croissance puis plusieurs pas du filet hamiltonien."""
        body, direction, grow_pending = state
        if not self._body_is_cycle_ordered(body):
            return False

        virtual_state = state
        # Quatre pas suffisent a valider le raccord. L'ordre hamiltonien prouve
        # ensuite que la queue sera liberee avant le retour de la tete.
        for _ in range(4):
            virtual_body, virtual_direction, virtual_growth = virtual_state
            successor = self.cycle[
                (self.index[virtual_body[0]] + 1) % len(self.cycle)
            ]
            move = direction_between(virtual_body[0], successor)
            next_state = self._transition(
                virtual_body,
                virtual_direction,
                virtual_growth,
                move,
                (-1, -1),
            )
            if next_state is None:
                return False
            virtual_state = next_state
        return True

    @staticmethod
    def _reconstruct_dynamic_path(
        parents: dict[State, tuple[State, Position] | None],
        goal: State,
    ) -> list[Position]:
        path = []
        cursor = goal
        while parents[cursor] is not None:
            previous, position = parents[cursor]  # type: ignore[misc]
            path.append(position)
            cursor = previous
        path.reverse()
        return path

    def _dynamic_astar_shortcut(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        food: Position,
    ) -> list[Position] | None:
        """Plus court raccourci reel qui revient dans l'ordre hamiltonien."""
        start: State = (tuple(body), direction, grow_pending)
        cycle_distance = self._distance_on_cycle(body[0], food)
        if cycle_distance <= 1:
            return None

        # Un chemin aussi long que le cycle n'est pas un raccourci. Cette borne
        # garde A* tres rapide, y compris avec plus de 200 segments.
        maximum_cost = cycle_distance - 1
        expansion_limit = 14_000 if len(body) < 180 else 5_000
        frontier: list[tuple[int, int, int, State]] = []
        order = 0
        heapq.heappush(
            frontier,
            (toroidal_distance(body[0], food), 0, order, start),
        )
        costs = {start: 0}
        parents: dict[State, tuple[State, Position] | None] = {start: None}
        expansions = 0
        first_goal_cost: int | None = None
        best_goal: tuple[tuple[int, int, int], State] | None = None

        while frontier and expansions < expansion_limit:
            estimated_total, cost, _, state = heapq.heappop(frontier)
            if cost != costs[state]:
                continue
            if (
                first_goal_cost is not None
                and estimated_total > first_goal_cost + 6
            ):
                break
            expansions += 1
            current_body, current_direction, current_growth = state

            if current_body[0] == food:
                if self._can_resume_cycle(state):
                    if first_goal_cost is None:
                        first_goal_cost = cost
                    next_apple = self._predict_next_apple(current_body)
                    if next_apple is None:
                        future_distance = 0
                    else:
                        cycle_to_next = self._distance_on_cycle(food, next_apple)
                        tail_distance = self._distance_on_cycle(
                            food, current_body[-1]
                        )
                        # Une pomme situee avant la queue est accessible sans
                        # tour complet; la distance directe favorise les pommes
                        # qui tombent dans l'axe de la tete.
                        if 0 < cycle_to_next < tail_distance:
                            future_distance = min(
                                toroidal_distance(food, next_apple),
                                cycle_to_next,
                            )
                        else:
                            future_distance = cycle_to_next
                    quality = (
                        cost + future_distance,
                        future_distance,
                        cost,
                    )
                    if best_goal is None or quality < best_goal[0]:
                        best_goal = (quality, state)
                continue
            if cost >= maximum_cost:
                continue

            for move in DIRECTIONS:
                next_state = self._transition(
                    current_body,
                    current_direction,
                    current_growth,
                    move,
                    food,
                )
                if next_state is None:
                    continue
                next_cost = cost + 1
                if next_cost > maximum_cost or next_cost >= costs.get(next_state, 10**9):
                    continue
                costs[next_state] = next_cost
                parents[next_state] = (state, next_state[0][0])
                order += 1
                priority = next_cost + toroidal_distance(next_state[0][0], food)
                heapq.heappush(
                    frontier,
                    (priority, next_cost, order, next_state),
                )
        if best_goal is None:
            return None
        return self._reconstruct_dynamic_path(parents, best_goal[1])

    @staticmethod
    def _predict_next_apple(body: tuple[Position, ...]) -> Position | None:
        """Reproduit le prochain tirage sans avancer le hasard du jeu."""
        occupied = set(body)
        available = [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in occupied
        ]
        if not available:
            return None

        predictor = random.Random()
        predictor.setstate(random.getstate())
        return predictor.choice(available)

    @staticmethod
    def _is_legal(
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        move: Position,
    ) -> bool:
        if move == opposite(direction):
            return False
        new_head = add_position(body[0], move)
        next_body = [new_head, *body]
        if not grow_pending:
            next_body.pop()
        return new_head not in next_body[1:]

    def choose_direction(self, snake: Snake, apple: Apple) -> Position:
        body = [tuple(segment) for segment in snake.body]
        head = body[0]
        tail = body[-1]
        direction = tuple(snake.direction)

        if apple.position is None:
            successor = self.cycle[(self.index[head] + 1) % len(self.cycle)]
            return direction_between(head, successor)

        food = tuple(apple.position)

        if self.planned_apple != food:
            self.planned_path.clear()
            self.planned_apple = food

        if self.planned_path:
            next_position = self.planned_path[0]
            try:
                planned_move = direction_between(head, next_position)
            except ValueError:
                self.planned_path.clear()
            else:
                if self._transition(
                    tuple(body),
                    direction,
                    snake.grow_pending,
                    planned_move,
                    food,
                ) is not None:
                    self.planned_path.popleft()
                    return planned_move
                self.planned_path.clear()

        shortcut = self._dynamic_astar_shortcut(
            body,
            direction,
            snake.grow_pending,
            food,
        )
        if shortcut:
            self.planned_path.extend(shortcut)
            return direction_between(head, self.planned_path.popleft())

        distance_to_tail = self._distance_on_cycle(head, tail)
        distance_to_food = self._distance_on_cycle(head, food)
        food_is_ahead = 0 < distance_to_food < distance_to_tail

        if food_is_ahead:
            path = self._shortest_ordered_path(
                head,
                food,
                direction,
                distance_to_food,
            )
            if path and self._path_is_safe(
                body,
                direction,
                snake.grow_pending,
                path,
                food,
            ):
                planned_move = direction_between(head, path[0])
                if self._is_legal(
                    body,
                    direction,
                    snake.grow_pending,
                    planned_move,
                ):
                    return planned_move

        candidates = []

        for move in DIRECTIONS:
            if not self._is_legal(body, direction, snake.grow_pending, move):
                continue
            neighbor = add_position(head, move)
            progress = self._distance_on_cycle(head, neighbor)

            # Ne jamais depasser la queue maintient l'ordre du corps sur le cycle.
            if not 0 < progress < distance_to_tail:
                continue
            # Quand la pomme est devant, ne pas la depasser avec un raccourci.
            if food_is_ahead and progress > distance_to_food:
                continue
            if not self._path_is_safe(
                body,
                direction,
                snake.grow_pending,
                [neighbor],
                food,
            ):
                continue
            candidates.append((progress, move))

        if candidates:
            return max(candidates, key=lambda candidate: candidate[0])[1]

        successor = self.cycle[(self.index[head] + 1) % len(self.cycle)]
        return direction_between(head, successor)


class SafeAStar:
    """A* rapide avec validation de survie et evitement des boucles."""

    def __init__(self) -> None:
        self.planned_path: deque[Position] = deque()
        self.planned_apple: Position | None = None
        self.recent_states: deque[tuple[Position, ...]] = deque(maxlen=5_000)

    @staticmethod
    def _transition(
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        move: Position,
        food: Position | None,
    ) -> tuple[list[Position], Position, bool] | None:
        if move == opposite(direction):
            return None

        new_head = add_position(body[0], move)
        new_body = [new_head, *body]
        new_grow_pending = False
        if not grow_pending:
            new_body.pop()
        if new_head in new_body[1:]:
            return None
        if food is not None and new_head == food:
            new_grow_pending = True
        return new_body, move, new_grow_pending

    def _legal_moves(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        food: Position | None,
    ) -> list[tuple[Position, list[Position], bool]]:
        legal = []
        for move in DIRECTIONS:
            result = self._transition(body, direction, grow_pending, move, food)
            if result is not None:
                next_body, _, next_grow_pending = result
                legal.append((move, next_body, next_grow_pending))
        return legal

    @staticmethod
    def _reconstruct_path(
        parents: dict[Position, Position | None],
        target: Position,
    ) -> list[Position]:
        path = []
        current = target
        while parents[current] is not None:
            path.append(current)
            current = parents[current]  # type: ignore[assignment]
        path.reverse()
        return path

    def _astar(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        target: Position,
    ) -> list[Position] | None:
        """Plus court chemin statique; la simulation finale elimine les faux positifs."""
        start = body[0]
        blocked = set(body if grow_pending else body[:-1])
        blocked.discard(start)
        blocked.discard(target)

        frontier: list[tuple[int, int, int, Position]] = []
        order = 0
        heapq.heappush(frontier, (toroidal_distance(start, target), 0, order, start))
        costs = {start: 0}
        parents: dict[Position, Position | None] = {start: None}

        while frontier:
            _, cost, _, current = heapq.heappop(frontier)
            if cost != costs[current]:
                continue
            if current == target:
                return self._reconstruct_path(parents, target)

            for move in DIRECTIONS:
                if current == start and move == opposite(direction):
                    continue
                neighbor = add_position(current, move)
                if neighbor in blocked:
                    continue
                new_cost = cost + 1
                if new_cost >= costs.get(neighbor, 10**9):
                    continue
                costs[neighbor] = new_cost
                parents[neighbor] = current
                order += 1
                priority = new_cost + toroidal_distance(neighbor, target)
                heapq.heappush(frontier, (priority, new_cost, order, neighbor))
        return None

    def _simulate_path(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        path: list[Position],
        food: Position,
    ) -> tuple[list[Position], Position, bool] | None:
        virtual_body = list(body)
        virtual_direction = direction
        virtual_growth = grow_pending

        for next_position in path:
            move = direction_between(virtual_body[0], next_position)
            result = self._transition(
                virtual_body,
                virtual_direction,
                virtual_growth,
                move,
                food,
            )
            if result is None:
                return None
            virtual_body, virtual_direction, virtual_growth = result
        return virtual_body, virtual_direction, virtual_growth

    @staticmethod
    def _reachable_area(body: list[Position], grow_pending: bool) -> int:
        start = body[0]
        blocked = set(body if grow_pending else body[:-1])
        blocked.discard(start)
        visited = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for move in DIRECTIONS:
                neighbor = add_position(current, move)
                if neighbor in blocked or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return len(visited)

    def _escape_quality(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
    ) -> tuple[bool, int, int]:
        """Mesure la meilleure sortie disponible apres avoir mange une pomme."""
        best = (False, 0, 0)
        for move, next_body, next_growth in self._legal_moves(
            body, direction, grow_pending, None
        ):
            tail_path = self._astar(
                next_body,
                move,
                next_growth,
                next_body[-1],
            )
            area = self._reachable_area(next_body, next_growth)
            mobility = len(self._legal_moves(next_body, move, next_growth, None))
            best = max(best, (tail_path is not None, area, mobility))
        return best

    def _has_escape(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
    ) -> bool:
        return self._escape_quality(body, direction, grow_pending)[0]

    def _safe_food_path(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        food: Position,
    ) -> list[Position] | None:
        path = self._astar(body, direction, grow_pending, food)
        if not path:
            return None
        final_state = self._simulate_path(body, direction, grow_pending, path, food)
        if final_state is None:
            return None
        final_body, final_direction, final_growth = final_state
        if self._has_escape(final_body, final_direction, final_growth):
            return path
        return None

    def _dynamic_safe_food_path(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        food: Position,
        expansion_limit: int = 1_500,
        require_escape: bool = True,
        maximize_escape: bool = False,
    ) -> list[Position] | None:
        """Cherche un detour en tenant compte du deplacement reel de la queue."""
        start: State = (tuple(body), direction, grow_pending)
        frontier: list[tuple[int, int, int, State]] = []
        heapq.heappush(
            frontier,
            (toroidal_distance(body[0], food), 0, 0, start),
        )
        costs = {start: 0}
        parents: dict[State, tuple[State, Position] | None] = {start: None}
        order = 0
        expansions = 0
        best_goal: tuple[tuple[bool, int, int, int], list[Position]] | None = None

        while frontier and expansions < expansion_limit:
            _, cost, _, state = heapq.heappop(frontier)
            if cost != costs[state]:
                continue
            expansions += 1
            body_tuple, current_direction, current_growth = state
            current_body = list(body_tuple)

            if current_body[0] == food:
                quality = self._escape_quality(
                    current_body,
                    current_direction,
                    current_growth,
                )
                if require_escape and not quality[0]:
                    continue
                path = []
                cursor = state
                while parents[cursor] is not None:
                    previous, position = parents[cursor]  # type: ignore[misc]
                    path.append(position)
                    cursor = previous
                path.reverse()
                if not maximize_escape:
                    return path
                goal_score = (*quality, -len(path))
                if best_goal is None or goal_score > best_goal[0]:
                    best_goal = (goal_score, path)
                continue

            for move, next_body, next_growth in self._legal_moves(
                current_body,
                current_direction,
                current_growth,
                food,
            ):
                next_state: State = (tuple(next_body), move, next_growth)
                next_cost = cost + 1
                if next_cost >= costs.get(next_state, 10**9):
                    continue
                costs[next_state] = next_cost
                parents[next_state] = (state, next_body[0])
                order += 1
                priority = next_cost + toroidal_distance(next_body[0], food)
                heapq.heappush(
                    frontier,
                    (priority, next_cost, order, next_state),
                )
        return None if best_goal is None else best_goal[1]

    def _survival_move(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
        food: Position,
    ) -> Position:
        state_counts = Counter(self.recent_states)
        candidates = []
        preferred_move = None
        tail_path = self._astar(body, direction, grow_pending, body[-1])
        if tail_path:
            preferred_move = direction_between(body[0], tail_path[0])

        for move, next_body, next_growth in self._legal_moves(
            body, direction, grow_pending, food
        ):
            ate_food = next_body[0] == food
            if ate_food and not self._has_escape(next_body, move, next_growth):
                continue

            tail_path = self._astar(next_body, move, next_growth, next_body[-1])
            area = self._reachable_area(next_body, next_growth)
            mobility = len(self._legal_moves(next_body, move, next_growth, None))
            repeated = state_counts[tuple(next_body)]
            food_distance = toroidal_distance(next_body[0], food)

            # La survie domine; la proximite de la pomme departage les cases sures.
            score = (
                tail_path is not None,
                move == preferred_move,
                repeated == 0,
                -repeated,
                area,
                mobility,
                -food_distance,
            )
            candidates.append((score, move))

        if not candidates:
            return direction
        return max(candidates, key=lambda candidate: candidate[0])[1]

    def choose_direction(self, snake: Snake, apple: Apple) -> Position:
        if apple.position is None:
            return tuple(snake.direction)

        # Le socle stocke certaines positions sous forme de listes. Le planificateur
        # travaille sur une copie immuable sans modifier les objets officiels.
        body = [tuple(segment) for segment in snake.body]
        head = body[0]
        direction = tuple(snake.direction)
        food = tuple(apple.position)

        self.recent_states.append(tuple(body))

        if self.planned_apple != food:
            self.planned_path.clear()
            self.planned_apple = food

        if self.planned_path:
            next_position = self.planned_path[0]
            try:
                move = direction_between(head, next_position)
            except ValueError:
                self.planned_path.clear()
            else:
                if self._transition(
                    body,
                    direction,
                    snake.grow_pending,
                    move,
                    food,
                ) is not None:
                    self.planned_path.popleft()
                    return move
                self.planned_path.clear()

        safe_path = self._safe_food_path(
            body,
            direction,
            snake.grow_pending,
            food,
        )
        if safe_path and snake.score < 200:
            self.planned_path.extend(safe_path)
            next_position = self.planned_path.popleft()
            return direction_between(head, next_position)

        dynamic_path = self._dynamic_safe_food_path(
            body,
            direction,
            snake.grow_pending,
            food,
            expansion_limit=(
                15_000
                if snake.score >= 215
                else 5_000
                if snake.score >= 200
                else 1_500
            ),
            require_escape=snake.score < 222,
            maximize_escape=200 <= snake.score < 222,
        )
        if dynamic_path:
            self.planned_path.extend(dynamic_path)
            next_position = self.planned_path.popleft()
            return direction_between(head, next_position)

        if safe_path:
            self.planned_path.extend(safe_path)
            next_position = self.planned_path.popleft()
            return direction_between(head, next_position)

        return self._survival_move(
            body,
            direction,
            snake.grow_pending,
            food,
        )


class FastVirtualAStar(SafeAStar):
    """Plus court chemin sur serpent virtuel, sinon detour maximal vers la queue."""

    def _longest_tail_path(
        self,
        body: list[Position],
        direction: Position,
        grow_pending: bool,
    ) -> list[Position] | None:
        shortest = self._astar(body, direction, grow_pending, body[-1])
        if not shortest:
            return None

        path = [body[0], *shortest]
        blocked = set(body[1:-1])
        changed = True
        while changed:
            changed = False
            used = set(path)
            index = 0
            while index < len(path) - 1:
                current = path[index]
                following = path[index + 1]
                step = direction_between(current, following)
                perpendiculars = (UP, DOWN) if step in (LEFT, RIGHT) else (LEFT, RIGHT)
                inserted = False
                for perpendicular in perpendiculars:
                    first = add_position(current, perpendicular)
                    second = add_position(following, perpendicular)
                    if (
                        first in blocked
                        or second in blocked
                        or first in used
                        or second in used
                    ):
                        continue
                    path[index + 1:index + 1] = [first, second]
                    changed = True
                    inserted = True
                    break
                index += 3 if inserted else 1
        return path[1:]

    def choose_direction(self, snake: Snake, apple: Apple) -> Position:
        if apple.position is None:
            return tuple(snake.direction)

        body = [tuple(segment) for segment in snake.body]
        head = body[0]
        direction = tuple(snake.direction)
        food = tuple(apple.position)
        self.recent_states.append(tuple(body))

        if self.planned_apple != food:
            self.planned_path.clear()
            self.planned_apple = food

        if self.planned_path:
            next_position = self.planned_path[0]
            move = direction_between(head, next_position)
            if self._transition(
                body,
                direction,
                snake.grow_pending,
                move,
                food,
            ) is not None:
                self.planned_path.popleft()
                return move
            self.planned_path.clear()

        food_path = self._safe_food_path(
            body,
            direction,
            snake.grow_pending,
            food,
        )

        # En fin de partie, comparer plusieurs arrivees vers la pomme pour ne pas
        # sacrifier l'acces aux quelques cases encore libres.
        if snake.score >= 210:
            dynamic_path = self._dynamic_safe_food_path(
                body,
                direction,
                snake.grow_pending,
                food,
                expansion_limit=5_000,
                require_escape=snake.score < 220,
                maximize_escape=snake.score < 220,
            )
            if dynamic_path:
                self.planned_path.extend(dynamic_path)
                return direction_between(head, self.planned_path.popleft())

        if food_path:
            self.planned_path.extend(food_path)
            return direction_between(head, self.planned_path.popleft())

        tail_path = self._longest_tail_path(body, direction, snake.grow_pending)
        if tail_path:
            move = direction_between(head, tail_path[0])
            if self._transition(
                body,
                direction,
                snake.grow_pending,
                move,
                food,
            ) is not None:
                return move

        return self._survival_move(
            body,
            direction,
            snake.grow_pending,
            food,
        )


def draw_grid(surface: pygame.Surface) -> None:
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def display_info(
    surface: pygame.Surface,
    font: pygame.font.Font,
    snake: Snake,
    start_time: float,
) -> None:
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(
        surface,
        BLANC,
        (0, SCORE_PANEL_HEIGHT - 2),
        (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2),
        2,
    )

    score_text = font.render(f"Score: {snake.score}", True, BLANC)
    surface.blit(score_text, (10, 12))

    elapsed_time = time.time() - start_time
    time_text = font.render(
        f"Temps: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}",
        True,
        BLANC,
    )
    surface.blit(time_text, (SCREEN_WIDTH - time_text.get_width() - 10, 12))

    algorithm_text = pygame.font.Font(None, 24).render(
        "Cerise Hamiltonien optimise + A* dynamique", True, BLEU
    )
    surface.blit(
        algorithm_text,
        (SCREEN_WIDTH // 2 - algorithm_text.get_width() // 2, 50),
    )


def display_message(
    surface: pygame.Surface,
    font: pygame.font.Font,
    message: str,
    color: tuple[int, int, int] = BLANC,
    y_offset: int = 0,
) -> None:
    text_surface = font.render(message, True, color)
    rect = text_surface.get_rect(
        center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + y_offset)
    )
    background = rect.inflate(40, 40)
    pygame.draw.rect(surface, NOIR, background, border_radius=10)
    pygame.draw.rect(surface, BLANC, background, 2, border_radius=10)
    surface.blit(text_surface, rect)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Algorithmique - Cerise Hamiltonien + A*")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 34)
    font_game_over = pygame.font.Font(None, 70)

    snake = Snake()
    apple = Apple(snake.body)
    controller = HamiltonianShortcut()
    start_time = time.time()
    running = True
    game_over = False
    victory = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and game_over and event.key == pygame.K_SPACE:
                main()
                return

        if not game_over:
            direction = controller.choose_direction(snake, apple)
            snake.set_direction(direction)
            snake.move()

            if snake.is_game_over():
                game_over = True
            elif apple.position is not None and snake.head_pos == list(apple.position):
                snake.grow()
                if not apple.relocate(snake.body):
                    victory = True
                    game_over = True

        screen.fill(GRIS_FOND)
        pygame.draw.rect(
            screen,
            NOIR,
            pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH),
        )
        draw_grid(screen)
        apple.draw(screen)
        snake.draw(screen)
        display_info(screen, font_main, snake, start_time)

        if game_over:
            if victory:
                display_message(screen, font_game_over, "VICTOIRE !", VERT)
            else:
                display_message(screen, font_game_over, "GAME OVER", ROUGE)
            display_message(screen, font_main, "ESPACE pour rejouer", BLANC, 90)

        pygame.display.flip()
        clock.tick(GAME_SPEED)

    pygame.quit()


if __name__ == "__main__":
    main()
