"""Experimental local safety filter for a learned Q policy.

This is explicitly a hybrid, not a pure RL policy or a completion guarantee.
It uses only the present body, direction and apple. It never copies/steps an
environment or consults its random generator.
"""
from __future__ import annotations

from collections import deque

import torch

from .env import DIRECTIONS, GRID_SIZE
from .state import encode


def reachable(start, occupied):
    """Current empty component, with toroidal adjacency."""
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            cell = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
            if cell not in seen and cell not in occupied:
                seen.add(cell)
                queue.append(cell)
    return seen


class ShieldedPolicy:
    """Reject immediate collisions and prefer actions retaining a tail route.

    A tail route is a heuristic: growth and future body motion can invalidate
    it. The network ranks every remaining action; no shortest path is followed.
    """

    def __init__(self, agent, mode="tail"):
        self.agent = agent
        self.mode = mode
        self.decisions = self.overrides = 0
        self.choice_states = 0
        self.last = {}

    learned = True

    @property
    def config(self):
        return self.agent.config

    @property
    def metadata(self):
        return self.agent.metadata

    @property
    def env_steps(self):
        return self.agent.env_steps

    @property
    def updates(self):
        return self.agent.updates

    @property
    def online(self):
        return self.agent.online

    @property
    def last_decision(self):
        return self.last

    @torch.no_grad()
    def action(self, env):
        state = encode(env, self.agent.config.encoder)
        values = self.agent.online(torch.from_numpy(state).unsqueeze(0))[0].numpy()
        proposed = int(values.argmax())
        candidates = []
        tail_reachable = []
        spaces = {}
        for action in range(4):
            if action == (env.direction + 2) % 4 or env.would_collide(action):
                continue
            candidates.append(action)
            if self.mode == "collision":
                continue
            dx, dy = DIRECTIONS[action]
            x, y = env.body[0]
            head = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
            body = [head] + env.body[:]
            if not env.grow_pending:
                body.pop()
            if self.mode == "tail2" and len(body) < GRID_SIZE * GRID_SIZE:
                next_occupied = set(body if head == env.apple else body[:-1])
                if not any(
                    ((head[0] + ox) % GRID_SIZE,
                     (head[1] + oy) % GRID_SIZE) not in next_occupied
                    for next_action, (ox, oy) in enumerate(DIRECTIONS)
                    if next_action != (action + 2) % 4
                ):
                    candidates.remove(action)
                    continue
            # Eating schedules growth on the NEXT transition. The next tail
            # therefore cannot be treated as free for an immediate exit.
            occupied = set(body[1:] if head == env.apple else body[1:-1])
            region = reachable(head, occupied)
            spaces[action] = len(region)
            if body[-1] in region or head == env.apple and any(
                ((body[-1][0] + ox) % GRID_SIZE,
                 (body[-1][1] + oy) % GRID_SIZE) in region
                for ox, oy in DIRECTIONS
            ):
                tail_reachable.append(action)
        if tail_reachable:
            candidates = tail_reachable
        elif spaces:
            largest = max(spaces.values())
            candidates = [a for a in candidates if spaces[a] == largest]
        executed = max(candidates, key=lambda a: values[a]) if candidates else proposed
        self.decisions += 1
        self.overrides += executed != proposed
        self.choice_states += len(candidates) > 1
        self.last = {"proposed": proposed, "executed": executed,
                     "allowed": candidates, "tail_reachable": tail_reachable,
                     "spaces": spaces}
        return executed

    def select_action(self, env):
        return self.action(env)
