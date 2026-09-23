"""Snake joué par un agent algorithmique (circuit dynamique par défaut).

    python3 snake-algo.py                   # circuit dynamique (100 % de victoires)
    python3 snake-algo.py --algo safepath   # étape 2 : SafePath
    python3 snake-algo.py --algo bfs        # étape 1 : BFS + simulation + repli
    python3 snake-algo.py --seed 7          # rejouer une partie précise
    python3 snake-algo.py --vitesse 10      # affichage 10x plus rapide (0 = max)

Reprend la boucle main() de serpent-algo.py sans la modifier : mêmes
classes Snake/Apple, même affichage, même clock.tick(GAME_SPEED). Seul le
clavier est remplacé par l'agent. ESPACE relance une partie à la fin.
Chaque partie terminée ajoute une ligne à RESULTATS.md.
"""
import argparse
import datetime
import os
import random
import re
import time

from adaptateur_jeu import load_game, observe
from serpent_agents import BFSAgent, DynamicCycleAgent, SafePathAgent

AGENTS = {"dynamique": DynamicCycleAgent, "safepath": SafePathAgent, "bfs": BFSAgent}
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTATS.md")


def next_run_number():
    """Numéro du run = plus grand numéro déjà présent dans RESULTATS.md + 1."""
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            runs = [int(m.group(1)) for m in re.finditer(r"^\| (\d+) \|", f.read(), re.M)]
    except OSError:
        runs = []
    return max(runs, default=0) + 1


def record(run, algo, seed, score, last_apple_s, cause):
    m, s = divmod(round(last_apple_s), 60)
    row = (f"| {run} | {datetime.datetime.now():%H:%M} | {algo} | {seed} "
           f"| {score} | {m}:{s:02d} | {cause} |\n")
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(row)
    print(row.strip())


def main(default_algo="dynamique"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=AGENTS, default=default_algo)
    ap.add_argument("--seed", type=int, default=None,
                    help="graine de la première partie (aléatoire par défaut)")
    ap.add_argument("--vitesse", type=float, default=1,
                    help="multiplicateur d'affichage (1 = vitesse du jeu, 0 = maximum) ;"
                         " le chrono affiché reste en temps de jeu")
    args = ap.parse_args()

    game = load_game()
    pygame = game.pygame
    agent = AGENTS[args.algo]()
    run = next_run_number()
    # Au-delà de ce nombre de déplacements sans pomme, le serpent tourne en
    # rond : on note la partie comme « boucle » (le jeu, lui, continue).
    max_idle = 20 * game.GRID_SIZE * game.GRID_SIZE

    pygame.init()
    screen = pygame.display.set_mode((game.SCREEN_WIDTH, game.SCREEN_HEIGHT))
    pygame.display.set_caption(f"Snake Algo - {agent.name}")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_game_over = pygame.font.Font(None, 80)

    seed = args.seed
    running = True
    while running:
        # --- Nouvelle partie ---
        if seed is None:
            seed = random.SystemRandom().randrange(1_000_000)
        random.seed(seed)
        snake = game.Snake()
        apple = game.Apple(snake.body)
        agent.reset()
        game_over = victory = recorded = False
        frames = 0          # horloge du jeu : une image = 1 / GAME_SPEED s
        last_apple = 0.0
        idle = 0
        move_counter = 0
        restart = False

        while running and not restart:
            # 1. Événements : fermer la fenêtre, ESPACE pour rejouer.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif (event.type == pygame.KEYDOWN and game_over
                      and event.key == pygame.K_SPACE):
                    restart = True

            # 2. Mise à jour, dans l'ordre exact de serpent-algo.py.
            if not game_over and not victory:
                move_counter += 1
                if move_counter >= game.GAME_SPEED // 10:
                    body, grow, direction, apple_pos = observe(snake, apple)
                    snake.set_direction(agent.decide(body, grow, direction,
                                                     apple_pos, game.GRID_SIZE))
                    snake.move()
                    move_counter = 0
                    idle += 1

                    if snake.is_game_over():
                        game_over = True
                        record(run, agent.name, seed, snake.score,
                               last_apple, "collision")
                        recorded = True
                        continue

                    if snake.head_pos == list(apple.position):
                        snake.grow()
                        idle = 0
                        last_apple = (frames + 1) / game.GAME_SPEED
                        if not apple.relocate(snake.body):
                            victory = True
                            game_over = True
                            record(run, agent.name, seed, snake.score,
                                   last_apple, "victoire")
                            recorded = True

                    if idle == max_idle and not recorded:
                        record(run, agent.name, seed, snake.score,
                               last_apple, "boucle")
                        recorded = True

            # 3. Dessin, identique au jeu de base.
            screen.fill(game.GRIS_FOND)
            game_area_rect = pygame.Rect(0, game.SCORE_PANEL_HEIGHT,
                                         game.SCREEN_WIDTH, game.SCREEN_WIDTH)
            pygame.draw.rect(screen, game.NOIR, game_area_rect)
            game.draw_grid(screen)
            apple.draw(screen)
            snake.draw(screen)
            # display_info calcule time.time() - start_time : on lui passe un
            # départ fictif pour qu'il affiche le temps de jeu, pas le temps réel.
            game_time = frames / game.GAME_SPEED
            game.display_info(screen, font_main, snake, time.time() - game_time)
            if game_over:
                if victory:
                    game.display_message(screen, font_game_over, "VICTOIRE !", game.VERT)
                else:
                    game.display_message(screen, font_game_over, "GAME OVER", game.ROUGE)
                game.display_message(screen, font_main, "ESPACE pour rejouer.",
                                     game.BLANC, y_offset=100)
            pygame.display.flip()
            if not game_over:
                frames += 1
            clock.tick(game.GAME_SPEED * args.vitesse)

        if not recorded:
            record(run, agent.name, seed, snake.score, last_apple, "interrompue")
        seed = None

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
