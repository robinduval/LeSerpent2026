"""Fait jouer snake-algo.py (la vraie boucle pygame, fenêtre factice, horloge
neutralisée pour aller vite) jusqu'à la fin de la partie et affiche le résultat.
Usage : python test_playable.py [stratégie] [graine]"""
import importlib.util
import os
import random
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
strategy = sys.argv[1] if len(sys.argv) > 1 else "search-plus"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.argv = ["snake-algo.py", strategy]
random.seed(seed)

import pygame
spec = importlib.util.spec_from_file_location("snake_algo", os.path.join(os.path.dirname(os.path.abspath(__file__)), "snake-algo.py"))
game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game)


class Done(Exception):
    pass


class FastClock:
    def tick(self, fps):
        return 0


state = {}
orig_dir = game.agent_direction
worst = [0.0]


def spy(agent, snake, apple):
    state["snake"], state["apple"] = snake, apple
    t = time.perf_counter()
    d = orig_dir(agent, snake, apple)
    worst[0] = max(worst[0], time.perf_counter() - t)
    return d


def flip():
    s = state.get("snake")
    if s and (s.is_game_over() or s.score >= game.GRID_SIZE ** 2 - 2):
        raise Done


game.agent_direction = spy
pygame.time.Clock = FastClock
pygame.display.flip = flip
t0 = time.time()
try:
    game.main()
except Done:
    s = state["snake"]
    win = s.score >= game.GRID_SIZE ** 2 - 2
    print(f"{strategy} graine {seed} : {'VICTOIRE' if win else 'perdu'} score={s.score} "
          f"(pire coup {1000 * worst[0]:.0f} ms, {time.time() - t0:.0f} s de calcul)")
