"""Explicitly non-learned comparators; never a fallback for the RL policy.

All policies expose ``select_action(env)`` and return an absolute direction:
up=0, right=1, down=2, left=3. No policy reads future apples or RNG state.
"""

from __future__ import annotations

import random
from typing import Protocol


GRID_SIZE = 15
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class BaselineEnvironment(Protocol):
    body: list[tuple[int, int]]
    direction: int
    apple: tuple[int, int] | None
    grow_pending: bool

    def would_collide(self, action: int) -> bool: ...


def valid_interface_actions(env: BaselineEnvironment) -> tuple[int, ...]:
    """Exclude only the reverse command, which the engine ignores.

    This does not exclude defined actions that would collide with the body.
    """
    opposite = (env.direction + 2) % 4
    return tuple(action for action in range(4) if action != opposite)


class StraightPolicy:
    """Reference for the original game left unattended after reset."""

    model_id = "algorithmic_original_straight"
    learned = False

    def select_action(self, env: BaselineEnvironment) -> int:
        return 1


class RandomPolicy:
    """Seeded random relative movement, with no collision avoidance."""

    model_id = "algorithmic_random"
    learned = False

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select_action(self, env: BaselineEnvironment) -> int:
        return self.rng.choice(valid_interface_actions(env))


class GreedyPolicy:
    """Greedy toroidal food distance with immediate collision avoidance.

    This is a programmed comparator. Local safety does not guarantee future
    viability, freedom from cycles, or completion.
    """

    model_id = "algorithmic_greedy_safe_one_step"
    learned = False

    def select_action(self, env: BaselineEnvironment) -> int:
        actions = valid_interface_actions(env)
        safe = [action for action in actions if not env.would_collide(action)]
        candidates = safe or list(actions)
        head_x, head_y = env.body[0]

        def ordering(action: int) -> tuple[int, bool, int]:
            if env.apple is None:
                return (0, action != env.direction, action)
            dx, dy = DIRECTIONS[action]
            x = (head_x + dx) % GRID_SIZE
            y = (head_y + dy) % GRID_SIZE
            apple_x, apple_y = env.apple
            x_distance = abs(x - apple_x)
            y_distance = abs(y - apple_y)
            distance = min(x_distance, GRID_SIZE - x_distance) + min(
                y_distance, GRID_SIZE - y_distance
            )
            return (distance, action != env.direction, action)

        return min(candidates, key=ordering)


def cycle_rank(position: tuple[int, int]) -> int:
    """Rank on a directed Hamiltonian cycle of the exact 15×15 torus."""
    x, y = position
    return y * GRID_SIZE + (x + y) % GRID_SIZE


class HamiltonianPolicy:
    """Fixed toroidal Hamiltonian cycle, aligned with the reset body.

    Every row starts at x=-y (mod 15), traverses all columns to the right,
    then moves down to the next row. Rank increases by one modulo 225.
    The initial head/body ranks are 115, 114, 113, so the cycle is aligned.
    This is a mathematical comparator, never presented as an RL model.

    Completion safety is established only for episodes starting at the
    engine reset state. Applying this policy to an arbitrary existing body
    does not inherit that guarantee.
    """

    model_id = "algorithmic_toroidal_hamiltonian"
    learned = False

    def select_action(self, env: BaselineEnvironment) -> int:
        x, y = env.body[0]
        return 2 if (x + y) % GRID_SIZE == GRID_SIZE - 1 else 1


POLICIES = {
    "straight": StraightPolicy,
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
    "hamiltonian": HamiltonianPolicy,
}


def make_baseline(name: str, seed: int = 0):
    """Create an explicitly requested comparator, with no implicit fallback."""
    try:
        policy_type = POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown baseline {name!r}; choose {tuple(POLICIES)}") from exc
    return policy_type(seed=seed) if policy_type is RandomPolicy else policy_type()
