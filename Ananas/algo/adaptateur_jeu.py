"""Adaptateur entre le moteur fourni (serpent-algo.py, non modifié) et les agents.

Le moteur est importé tel quel ; la partie sans affichage reproduit
l'ordre exact de la boucle main() : set_direction -> move ->
is_game_over -> pomme mangée ? grow + relocate (victoire si échec).
"""
import importlib.util
import os
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GAME_FILE = os.path.join(HERE, "serpent-algo.py")

_game = None


def load_game():
    """Importe serpent-algo.py (nom avec tiret, donc via importlib)."""
    global _game
    if _game is None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        spec = importlib.util.spec_from_file_location("serpent_jeu", GAME_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _game = module
    return _game


def observe(snake, apple):
    """État lisible par un agent : (corps, grow_pending, direction, pomme)."""
    body = tuple(tuple(p) for p in snake.body)
    return body, snake.grow_pending, tuple(snake.direction), tuple(apple.position)


def play_headless(agent, seed, max_idle=None):
    """Joue une partie complète sans fenêtre avec les classes du jeu.

    La graine fixe `random`, seule source d'aléa du moteur (placement des
    pommes). Le temps suit l'horloge du jeu : un déplacement par tick de
    clock.tick(GAME_SPEED), allongé si l'agent réfléchit plus d'un tick.
    `max_idle` arrête une partie où le serpent tourne sans manger (le vrai
    jeu ne se terminerait jamais).
    """
    game = load_game()
    n = game.GRID_SIZE
    if max_idle is None:
        max_idle = 20 * n * n
    random.seed(seed)
    snake = game.Snake()
    apple = game.Apple(snake.body)
    agent.reset()

    tick = 1 / game.GAME_SPEED
    clock = last_apple = 0.0
    idle = 0
    decision_max = 0.0
    cause = None
    while cause is None:
        body, grow, direction, apple_pos = observe(snake, apple)
        t0 = time.perf_counter()
        d = agent.decide(body, grow, direction, apple_pos, n)
        decision = time.perf_counter() - t0
        decision_max = max(decision_max, decision)
        clock += max(tick, decision)

        snake.set_direction(d)
        snake.move()
        idle += 1
        if snake.is_game_over():
            cause = "collision"
            break
        if snake.head_pos == list(apple.position):
            snake.grow()
            idle = 0
            last_apple = clock
            if not apple.relocate(snake.body):
                cause = "victoire"
                break
        if idle >= max_idle:
            cause = "boucle"

    return {
        "agent": agent.name,
        "seed": seed,
        "score": snake.score,
        "last_apple_s": last_apple,
        "end_s": clock,
        "cause": cause,
        "decision_max_ms": 1000 * decision_max,
    }
