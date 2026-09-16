"""Observable state encoders. Neither reads RNG state nor changes the game."""
from __future__ import annotations

import numpy as np

GRID_SIZE = 15
STATE_DIMS = {"classic11": 11, "ordered": 248}
ENCODER_VERSION = 1


def encode(env, kind: str = "classic11") -> np.ndarray:
    """Course features; optional richer observation requires teacher approval.

    classic11: danger ahead/right/left; direction left/right/up/down;
    apple left/right/up/down using raw coordinates, as in the course.
    ordered adds a head-relative grid of body ranks (tail=1/225), toroidal
    apple offsets, absolute dangers, four body clearances, pending growth and length. Ranks are a release
    ORDER, not exact release times: future growth changes those times.
    """
    if kind not in STATE_DIMS:
        raise ValueError(f"Unknown encoder: {kind!r}")
    x, y = env.body[0]
    direction = int(env.direction)
    apple = env.apple
    ax, ay = apple if apple is not None else (x, y)
    classic = np.array([
        env.would_collide(direction),
        env.would_collide((direction + 1) % 4),
        env.would_collide((direction - 1) % 4),
        direction == 3, direction == 1, direction == 0, direction == 2,
        ax < x, ax > x, ay < y, ay > y,
    ], dtype=np.float32)
    if kind == "classic11":
        return classic
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for index, (bx, by) in enumerate(env.body):
        grid[(by - y) % GRID_SIZE, (bx - x) % GRID_SIZE] = (
            len(env.body) - index
        ) / (GRID_SIZE * GRID_SIZE)
    half = GRID_SIZE // 2
    dx = ((ax - x + half) % GRID_SIZE - half) / half
    dy = ((ay - y + half) % GRID_SIZE - half) / half
    occupied = set(tuple(cell) for cell in env.body[1:])
    clearances = []
    for vx, vy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        free = 0
        for distance in range(1, GRID_SIZE):
            if ((x + vx * distance) % GRID_SIZE, (y + vy * distance) % GRID_SIZE) in occupied:
                break
            free += 1
        clearances.append(free / (GRID_SIZE - 1))
    extras = np.array([
        dx, dy, *(env.would_collide(d) for d in range(4)), *clearances,
        float(env.grow_pending), len(env.body) / (GRID_SIZE * GRID_SIZE),
    ], dtype=np.float32)
    return np.concatenate((classic, grid.ravel(), extras))
