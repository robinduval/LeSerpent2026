"""Auditable Hamiltonian-order safety shield for learned action selection.

This module is programmed safety logic, not a learned controller. It exposes
all admissible shortcuts; an RL model can learn which admissible action to
choose. Guarantees apply to reset-aligned, cycle-ordered bodies only. The
included greedy policy is explicitly a non-learned diagnostic comparator.
"""
from __future__ import annotations

from .env import DIRECTIONS, GRID_SIZE, Env

BOARD_CELLS = GRID_SIZE * GRID_SIZE


def cycle_rank(position: tuple[int, int]) -> int:
    x, y = position
    return y * GRID_SIZE + (x + y) % GRID_SIZE


CYCLE = tuple(sorted(((x, y) for x in range(GRID_SIZE)
                      for y in range(GRID_SIZE)), key=cycle_rank))
INDEX = {position: rank for rank, position in enumerate(CYCLE)}


def cycle_distance(start: tuple[int, int], end: tuple[int, int]) -> int:
    return (INDEX[end] - INDEX[start]) % BOARD_CELLS


def successor_action(position: tuple[int, int]) -> int:
    x, y = position
    return 2 if (x + y) % GRID_SIZE == GRID_SIZE - 1 else 1


def is_cycle_ordered(env: Env) -> bool:
    """Whether ranks proceed strictly backwards from head towards tail."""
    if len(set(env.body)) != len(env.body):
        return False
    head = INDEX[env.body[0]]
    distances = [(head - INDEX[cell]) % BOARD_CELLS for cell in env.body]
    return all(a < b for a, b in zip(distances, distances[1:]))


def allowed_actions(env: Env) -> tuple[int, ...]:
    """Experimental fixed-cycle shortcuts (not a total safety guarantee).

    The new head must precede the post-move tail in directed cycle order.
    An eaten apple schedules growth for *the following* move, so eating
    additionally reserves a free cycle successor. At length 225 an apple
    completes the game immediately and requires no additional reserve.
    Restricting progress not to pass the apple guarantees finite progress
    to food even when its cell lies among gaps inside the body arc.

    Consecutive apples can still exhaust the free arc with delayed growth;
    this older heuristic can return an empty set. Use RewiredCycleShield
    for a nonempty admissible set with a completion argument.

    This only reads public current state; it never copies/advances the RNG.
    Return () on terminal states. Callers must not apply the shield to an
    arbitrary body that has not preserved the reset-state cycle invariant.
    """
    if env.terminated:
        return ()
    x, y = env.body[0]
    head_rank = INDEX[(x, y)]
    post_tail = env.body[-1] if env.grow_pending else env.body[-2]
    tail_distance = (INDEX[post_tail] - head_rank) % BOARD_CELLS
    apple_distance = ((INDEX[env.apple] - head_rank) % BOARD_CELLS
                      if env.apple is not None else BOARD_CELLS)
    opposite = (env.direction + 2) % 4
    result = []
    for action, (dx, dy) in enumerate(DIRECTIONS):
        if action == opposite:
            continue
        candidate = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
        advance = (INDEX[candidate] - head_rank) % BOARD_CELLS
        if not 0 < advance < tail_distance or advance > apple_distance:
            continue
        # Reserve a cycle successor for the delayed growth after eating.
        # An already pending growth may fill all 225 cells and finish now.
        eating = candidate == env.apple
        completes = eating and env.grow_pending and len(env.body) == BOARD_CELLS - 1
        if eating and not completes and tail_distance - advance < 2:
            continue
        if successor_action(candidate) == (action + 2) % 4:
            continue
        result.append(action)
    return tuple(result)


class CycleShortcutPolicy:
    """Non-learned comparator: take maximal admissible cycle progress."""

    model_id = "algorithmic_toroidal_cycle_shortcuts"
    learned = False

    def select_action(self, env: Env) -> int:
        actions = allowed_actions(env)
        if not actions:
            raise RuntimeError("No safe action: cycle-order invariant was not preserved")
        x, y = env.body[0]
        return max(actions, key=lambda action: cycle_distance(
            (x, y), ((x + DIRECTIONS[action][0]) % GRID_SIZE,
                     (y + DIRECTIONS[action][1]) % GRID_SIZE)))


class RewiredCyclePolicy:
    """Experimental non-learned 2-opt comparator with a contiguous body cycle.

    Every accepted free-arc reversal keeps one Hamiltonian cycle containing
    all existing body edges. Food's directed distance decreases each move.
    """
    model_id = "algorithmic_rewired_hamiltonian"
    learned = False

    def __init__(self):
        self.cycle = list(CYCLE)
        self.index = dict(INDEX)

    @staticmethod
    def adjacent(a, b):
        return ((a[0] == b[0] and (a[1] - b[1]) % GRID_SIZE in (1, GRID_SIZE - 1))
                or (a[1] == b[1] and (a[0] - b[0]) % GRID_SIZE in (1, GRID_SIZE - 1)))

    def select_action(self, env: Env) -> int:
        if env.steps == 0:
            self.cycle = list(CYCLE)
            self.index = dict(INDEX)
        h = self.index[env.body[0]]
        cycle = self.cycle[h:] + self.cycle[:h]
        indexes = {cell: i for i, cell in enumerate(cycle)}
        apple_i = indexes[env.apple]
        successor = cycle[1]
        # Existing tail may be used only by the ordinary next-cycle move;
        # rewiring never modifies any body edge.
        free_end = BOARD_CELLS - len(env.body)
        choices = []
        x, y = env.body[0]
        for a, (dx, dy) in enumerate(DIRECTIONS):
            if a == (env.direction + 2) % 4:
                continue
            p = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
            k = indexes[p]
            if k == 1:
                choices.append((apple_i - 1, a, 1))
            elif 1 < k <= free_end and self.adjacent(successor, cycle[k + 1]):
                new_distance = k - apple_i if apple_i <= k else apple_i - 1
                if new_distance < apple_i:
                    choices.append((new_distance, a, k))
        if not choices:
            raise RuntimeError("Rewired cycle invariant failed")
        _, a, k = min(choices)
        if k > 1:
            cycle[1:k+1] = reversed(cycle[1:k+1])
        self.cycle = cycle
        self.index = {cell: i for i, cell in enumerate(cycle)}
        return a


class RewiredCycleShield:
    """Hamiltonian safety state for RL choices, with safe free-arc rewiring.

    Start this shield with a reset Env. Call ``allowed_actions(env)`` to
    compute the admissible action set, then ``commit(env, action)`` before
    ``env.step(action)``. ``index`` and ``cycle`` expose the current cycle.
    No method reads future apples or RNG state.

    Unlike the fixed-rank shortcut heuristic, every move retains an entire
    Hamiltonian cycle containing the contiguous body. There is always at
    least the successor action. Rewiring additionally must decrease the
    current food's cycle distance, so any sequence of admissible decisions
    reaches each apple within 224 moves and eventually fills the board.
    """

    def __init__(self):
        self.cycle = list(CYCLE)
        self.index = dict(INDEX)

    def reset(self):
        self.cycle = list(CYCLE)
        self.index = dict(INDEX)

    def distance(self, start, end):
        return (self.index[end] - self.index[start]) % BOARD_CELLS

    def is_aligned(self, env: Env) -> bool:
        h = self.index[env.body[0]]
        return all(self.index[cell] == (h - offset) % BOARD_CELLS
                   for offset, cell in enumerate(env.body))

    def _options(self, env: Env):
        if env.terminated:
            return []
        head = env.body[0]
        h = self.index[head]
        apple_distance = self.distance(head, env.apple)
        successor = self.cycle[(h + 1) % BOARD_CELLS]
        free_end = BOARD_CELLS - len(env.body)
        options = []
        x, y = head
        for action, (dx, dy) in enumerate(DIRECTIONS):
            if action == (env.direction + 2) % 4:
                continue
            candidate = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
            k = self.distance(head, candidate)
            if k == 1:
                options.append((action, 1, apple_distance - 1))
            elif 1 < k <= free_end:
                after_candidate = self.cycle[(self.index[candidate] + 1) % BOARD_CELLS]
                if not RewiredCyclePolicy.adjacent(successor, after_candidate):
                    continue
                next_distance = (k - apple_distance if apple_distance <= k
                                 else apple_distance - 1)
                if next_distance < apple_distance:
                    options.append((action, k, next_distance))
        return options

    def allowed_actions(self, env: Env) -> tuple[int, ...]:
        return tuple(action for action, _, _ in self._options(env))

    def next_food_distance(self, env: Env, action: int) -> int:
        for candidate, _, next_distance in self._options(env):
            if candidate == action:
                return next_distance
        raise ValueError("Action is outside the Hamiltonian safety set")

    def candidate_index(self, env: Env, action: int):
        """Read-only rank map after committing an admissible action."""
        for candidate, k, _ in self._options(env):
            if candidate == action:
                if k == 1:
                    return dict(self.index)
                h = self.index[env.body[0]]
                cycle = self.cycle[h:] + self.cycle[:h]
                cycle[1:k + 1] = reversed(cycle[1:k + 1])
                return {cell: rank for rank, cell in enumerate(cycle)}
        raise ValueError("Action is outside the Hamiltonian safety set")

    def commit(self, env: Env, action: int) -> None:
        for candidate, k, _ in self._options(env):
            if candidate == action:
                if k > 1:
                    h = self.index[env.body[0]]
                    cycle = self.cycle[h:] + self.cycle[:h]
                    cycle[1:k + 1] = reversed(cycle[1:k + 1])
                    self.cycle = cycle
                    self.index = {cell: rank for rank, cell in enumerate(cycle)}
                return
        raise ValueError("Action is outside the Hamiltonian safety set")


class RewiredGreedyPolicy:
    """Explicit programmed comparator selecting nearest food in the new cycle."""
    model_id = "algorithmic_rewired_hamiltonian_greedy"
    learned = False

    def __init__(self):
        self.shield = RewiredCycleShield()

    def select_action(self, env: Env) -> int:
        if env.steps == 0:
            self.shield.reset()
        options = self.shield._options(env)
        action, _, _ = min(options, key=lambda option: (option[2], option[0]))
        self.shield.commit(env, action)
        return action



class OptimizedRewiredGreedyPolicy(RewiredGreedyPolicy):
    """Diagnostic comparator with additional free-arc cycle optimizations."""
    model_id = "algorithmic_rewired_hamiltonian_optimized"

    def select_action(self, env: Env) -> int:
        if env.steps == 0:
            self.shield.reset()
        optimize_free_arc(self.shield, env)
        options = self.shield._options(env)
        action, _, _ = min(options, key=lambda option: (option[2], option[0]))
        self.shield.commit(env, action)
        return action


def optimize_free_arc(shield: RewiredCycleShield, env: Env) -> int:
    """Shorten current food's cycle distance using body-preserving 2-opt.

    This modifies the shield's cycle, never the game. Each accepted reversal
    strictly reduces food rank, so preprocessing also preserves completion.
    Call before choosing an action; candidate feature queries remain pure.
    """
    h = shield.index[env.body[0]]
    cycle = shield.cycle[h:] + shield.cycle[:h]
    index = {cell: rank for rank, cell in enumerate(cycle)}
    free_end = BOARD_CELLS - len(env.body)
    changes = 0
    while True:
        apple_rank = index[env.apple]
        best = None
        # Reversal i+1..j moves apple to i+j+1-apple_rank.
        for i in range(min(apple_rank, free_end - 1)):
            x, y = cycle[i]
            next_i = cycle[i + 1]
            for dx, dy in DIRECTIONS:
                j = index[((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)]
                if not (max(i + 2, apple_rank) <= j <= free_end):
                    continue
                reduction = 2 * apple_rank - i - j - 1
                if reduction <= 0 or (best is not None and reduction <= best[0]):
                    continue
                if RewiredCyclePolicy.adjacent(next_i, cycle[j + 1]):
                    best = (reduction, i, j)
        if best is None:
            break
        _, i, j = best
        cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
        index = {cell: rank for rank, cell in enumerate(cycle)}
        changes += 1
    shield.cycle, shield.index = cycle, index
    return changes


def explore_free_arc(shield: RewiredCycleShield, env: Env, attempts: int = 32) -> int:
    """Diagnostic search over neutral free-arc flips, retaining improvements.

    No environment RNG is used. The small deterministic search seed depends
    only on the current observable state. This is programmed planning.
    """
    import random
    optimize_free_arc(shield, env)
    best_distance = shield.distance(env.body[0], env.apple)
    best_cycle = list(shield.cycle)
    rng = random.Random(env.steps * 101 + env.score * 31 + INDEX[env.apple])
    free_end = BOARD_CELLS - len(env.body)
    improvements = 0
    for _ in range(attempts):
        # optimize_free_arc normalizes head to index zero.
        cycle = shield.cycle
        index = shield.index
        apple_rank = index[env.apple]
        neutral = []
        for i in range(max(0, free_end - 1)):
            x, y = cycle[i]
            for dx, dy in DIRECTIONS:
                j = index[((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)]
                if not i + 2 <= j <= free_end:
                    continue
                if i < apple_rank <= j:
                    continue
                if RewiredCyclePolicy.adjacent(cycle[i + 1], cycle[j + 1]):
                    neutral.append((i, j))
        if not neutral:
            break
        i, j = rng.choice(neutral)
        cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
        shield.index = {cell: rank for rank, cell in enumerate(cycle)}
        optimize_free_arc(shield, env)
        distance = shield.index[env.apple]
        if distance < best_distance:
            best_distance = distance
            best_cycle = list(shield.cycle)
            improvements += 1
    shield.cycle = best_cycle
    shield.index = {cell: rank for rank, cell in enumerate(best_cycle)}
    return improvements


class ExploredRewiredGreedyPolicy(RewiredGreedyPolicy):
    model_id = "algorithmic_rewired_hamiltonian_explored"

    def select_action(self, env: Env) -> int:
        if env.steps == 0:
            self.shield.reset()
        if env.steps == 0 or env.grow_pending:
            explore_free_arc(self.shield, env, attempts=64)
        else:
            optimize_free_arc(self.shield, env)
        options = self.shield._options(env)
        action, _, _ = min(options, key=lambda option: (option[2], option[0]))
        self.shield.commit(env, action)
        return action
