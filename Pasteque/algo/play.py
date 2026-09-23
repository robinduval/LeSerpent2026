"""Partie visuelle : le planner joue dans la fenêtre Pygame du moteur.

Par défaut la partie continue jusqu'à la grille pleine (pas d'arrêt à 100).

    python play.py                 # 20 ticks/s, jusqu'au plateau plein
    python play.py --fps 60 --seed 3
    python play.py --target 100    # s'arrête à 100
"""

from __future__ import annotations

import argparse
import os
import random
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from bench import engine  # noqa: E402
from planner import SnakePlanner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20, help="ticks par seconde")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--target", type=int, default=0, help="score d'arrêt (0 = grille pleine)")
    a = ap.parse_args()

    eng = engine()
    if a.seed is not None:
        random.seed(a.seed)
    pygame.init()
    screen = pygame.display.set_mode((eng.SCREEN_WIDTH, eng.SCREEN_HEIGHT))
    pygame.display.set_caption("Pastèque - planner")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    font_big = pygame.font.Font(None, 70)

    snake = eng.Snake()
    apple = eng.Apple(snake.body)
    planner = SnakePlanner(eng.GRID_SIZE, eng.GRID_SIZE)
    start = time.time()
    ticks = 0
    tick_100 = None
    result = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        if result is None:
            d = planner.choose_action(snake.body, snake.direction, apple.position,
                                      snake.score, snake.grow_pending)
            snake.set_direction(d)
            snake.move()
            ticks += 1
            if snake.is_game_over():
                result = "GAME OVER"
            elif snake.head_pos == list(apple.position):
                snake.grow()
                if snake.score == 100 and tick_100 is None:
                    tick_100 = ticks
                    print(f"score 100 atteint en {ticks} ticks")
                if a.target and snake.score >= a.target:
                    result = f"{a.target} !"
                elif not apple.relocate(snake.body):
                    apple.position = None
                    result = "GRILLE PLEINE !"
            if result:
                print(f"{result} score {snake.score}, {ticks} ticks, longueur {len(snake.body)}")

        screen.fill(eng.GRIS_FOND)
        pygame.draw.rect(screen, eng.NOIR, (0, eng.SCORE_PANEL_HEIGHT, eng.SCREEN_WIDTH, eng.SCREEN_WIDTH))
        eng.draw_grid(screen)
        apple.draw(screen)
        snake.draw(screen)
        eng.display_info(screen, font, snake, start)
        mode = planner.last_info.get("mode", "")
        extra = f"tick {ticks}  mode {mode}" + (f"  100 @ {tick_100}" if tick_100 else "")
        screen.blit(font.render(extra, True, eng.BLANC), (10, 52))
        if result:
            color = eng.ROUGE if result == "GAME OVER" else eng.VERT
            eng.display_message(screen, font_big, result, color)
        pygame.display.flip()
        clock.tick(a.fps if result is None else 10)

    pygame.quit()


if __name__ == "__main__":
    main()
