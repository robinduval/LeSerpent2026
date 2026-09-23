"""Pure transitions of the supplied Snake game, without a rendering clock.

The displayed application and evaluations must call
``pygame.time.Clock.tick(GAME_SPEED)`` for each transition. Accelerated
training was explicitly authorized separately by the user.
"""
from __future__ import annotations

import copy
import math
import operator
import random

GRID_SIZE = 15
GAME_SPEED = 5
UP, RIGHT, DOWN, LEFT = range(4)
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
INITIAL_BODY = ((3, 7), (2, 7), (1, 7))
THEORETICAL_MAX_SCORE = GRID_SIZE * GRID_SIZE - len(INITIAL_BODY) + 1
RULE_SIGNATURE = "snake15-torus-absolute4-reverse-ignored-delayed-growth-v1"


class Env:
    """Head-first tuple positions and absolute actions in clockwise order.

    Official score counts apples, independently of the RL reward. The source
    defines no RL rewards. ``move_reward`` defaults to one of the course
    values (0.1). Completion uses +100 instead of +10. The user explicitly
    authorized arbitrary training rewards; none change the official score.
    """

    def __init__(self, seed: int | None = None, move_reward: float = 0.1):
        if not math.isfinite(move_reward):
            raise ValueError("move_reward must be finite")
        self.move_reward = float(move_reward)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self, seed: int | None = None) -> Env:
        """Start a game; an explicit seed restarts the independent apple RNG."""
        if seed is not None:
            self.rng.seed(seed)
        self.body = list(INITIAL_BODY)
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0
        self.steps = 0
        self.terminated = False
        self.completed = False
        self.termination_reason: str | None = None
        self.apple: tuple[int, int] | None = self._random_apple()
        return self

    def _random_apple(self) -> tuple[int, int] | None:
        occupied = set(self.body)
        # Preserve the exact original order: x outside, y inside. RNG seed
        # equivalence depends on this even though both orders are uniform.
        available = [(x, y) for x in range(GRID_SIZE)
                     for y in range(GRID_SIZE) if (x, y) not in occupied]
        return self.rng.choice(available) if available else None

    def effective_action(self, action: int) -> int:
        """A reverse request keeps the preceding direction, like the source."""
        try:
            action = operator.index(action)
        except TypeError as exc:
            raise ValueError("action must be an integer in [0, 3]") from exc
        if action not in range(4):
            raise ValueError("action must be an integer in [0, 3]")
        return self.direction if action == (self.direction + 2) % 4 else action

    def would_collide(self, action: int) -> bool:
        """Check one move without mutating body, counters or random state.

        The present tail cell is free on a normal step, but occupied when
        growth was scheduled by the preceding step's apple.
        """
        actual = self.effective_action(action)
        dx, dy = DIRECTIONS[actual]
        x, y = self.body[0]
        head = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
        occupied = self.body if self.grow_pending else self.body[:-1]
        return head in occupied

    def copy(self) -> Env:
        """Copy the whole state including a separate, identical random stream."""
        return copy.deepcopy(self)

    def step(self, action: int) -> tuple[float, bool, dict]:
        """Move, remove/retain tail, check collision, eat, then relocate.

        No timeout, stagnation cutoff, automatic full-grid victory, or reward
        shaping is introduced. An external experimental cutoff must remain
        a truncation, and must not change this terminal flag.
        """
        if self.terminated:
            raise RuntimeError("cannot step a terminated game; call reset()")
        actual = self.effective_action(action)
        requested = operator.index(action)
        self.direction = actual
        dx, dy = DIRECTIONS[actual]
        x, y = self.body[0]
        head = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
        grew = self.grow_pending
        self.body.insert(0, head)
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False
        self.steps += 1
        ate_apple = False
        reward = self.move_reward

        if head in self.body[1:]:
            self.terminated = True
            self.termination_reason = "self_collision"
            reward = -10.0
        elif head == self.apple:
            ate_apple = True
            self.grow_pending = True
            self.score += 1
            reward = 10.0
            new_apple = self._random_apple()
            if new_apple is None:
                self.terminated = True
                self.completed = True
                self.termination_reason = "completed"
                reward = 100.0
                # Apple.relocate() retains its preceding position on failure.
            else:
                self.apple = new_apple

        info = {
            "action_requested": requested, "action_executed": actual,
            "executed_action": actual,
            "ate_apple": ate_apple, "grew": grew, "score": self.score,
            "steps": self.steps, "completed": self.completed,
            "termination_reason": self.termination_reason,
            "final_length": len(self.body),
        }
        return float(reward), self.terminated, info


SnakeEnv = Env
