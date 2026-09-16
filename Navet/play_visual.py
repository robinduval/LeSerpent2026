# Visualisation pygame d'un modèle entraîné : le réseau joue, on regarde.
# Réutilise SnakeGame (game.py) et Agent.get_state (agent.py) tels quels :
# ce qui s'affiche est exactement ce que mesure evaluate.py, aucune logique dupliquée.
# Rendu calqué sur serpent-algo.py (couleurs, panneau score, GAME_SPEED=5 par défaut).
import argparse
import time

import pygame
import torch

from game import SnakeGame, GRID_SIZE
from agent import Agent

CELL_SIZE = 30
GAME_SPEED = 5  # clock du socle, intacte par défaut (--speed pour accélérer la visu)

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)


def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH + 1, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT + 1, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def draw_snake(surface, game):
    for segment in game.body[1:]:
        rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                           CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, VERT, rect)
        pygame.draw.rect(surface, NOIR, rect, 1)
    head = pygame.Rect(game.head[0] * CELL_SIZE, game.head[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                       CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, ORANGE, head)
    pygame.draw.rect(surface, NOIR, head, 2)


def draw_apple(surface, game):
    if game.apple is None:
        return
    rect = pygame.Rect(game.apple[0] * CELL_SIZE, game.apple[1] * CELL_SIZE + SCORE_PANEL_HEIGHT,
                       CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, ROUGE, rect, border_radius=5)
    pygame.draw.circle(surface, BLANC, (rect.x + CELL_SIZE * 0.7, rect.y + CELL_SIZE * 0.3), CELL_SIZE // 8)


def draw_panel(surface, font, small, game, episode, episodes, elapsed, best, q_values):
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    surface.blit(font.render(f"Score: {game.score}", True, BLANC), (10, 12))

    fill_rate = len(game.body) / (GRID_SIZE * GRID_SIZE) * 100
    fill = small.render(f"Remplissage: {fill_rate:.1f}%", True, BLANC)
    surface.blit(fill, (SCREEN_WIDTH // 2 - fill.get_width() // 2, 14))

    # temps de JEU (steps / GAME_SPEED), independant de --speed : c'est la duree
    # qu'aurait la partie a la clock du socle, donc le chiffre a donner au prof.
    jeu = game.steps / GAME_SPEED
    m, s = divmod(int(jeu), 60)
    t = small.render(f"Temps jeu: {m:02d}:{s:02d}", True, BLANC)
    surface.blit(t, (SCREEN_WIDTH - t.get_width() - 10, 14))

    # ligne du bas : contexte de la démo + décision du réseau
    left = small.render(f"Partie {episode}/{episodes}  best {best}", True, BLANC)
    surface.blit(left, (10, 48))
    if q_values is not None:
        labels = ("tout droit", "droite", "gauche")
        choice = max(range(3), key=lambda i: q_values[i])
        txt = small.render(f"Q: {labels[choice]}", True, ORANGE)
        surface.blit(txt, (SCREEN_WIDTH - txt.get_width() - 10, 48))


def draw_message(surface, font, message, color):
    text = font.render(message, True, color)
    rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    bg = rect.inflate(40, 20)
    pygame.draw.rect(surface, NOIR, bg, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg, 2, border_radius=10)
    surface.blit(text, rect)


def main():
    ap = argparse.ArgumentParser(description="Regarder un modèle entraîné jouer une vraie partie.")
    ap.add_argument("--model", default="model_fix6_s2_best.pth")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=999, help="même seed que evaluate.py par défaut")
    ap.add_argument("--speed", type=float, default=GAME_SPEED, help="FPS de la visu (5 = clock du socle)")
    ap.add_argument("--hunger-mode", default="truncated", choices=["loeber", "truncated"])
    ap.add_argument("--hunger-k", type=float, default=2.0)
    ap.add_argument("--step-reward", type=float, default=-0.01)
    ap.add_argument("--torus-food", action="store_true", help="Fix 5/5 : doit matcher l'entrainement du modele")
    ap.add_argument("--rich-state", action="store_true", help="Fix 6/6 : doit matcher l'entrainement du modele")
    args = ap.parse_args()

    agent = Agent(torus_food=args.torus_food, rich_state=args.rich_state)
    agent.model.load_state_dict(torch.load(args.model, map_location="cpu"))
    agent.model.eval()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Navet — {args.model}")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 26)
    font_big = pygame.font.Font(None, 70)

    speed = args.speed
    paused = False
    best = 0
    scores = []
    running = True
    episode = 1

    while running and episode <= args.episodes:
        game = SnakeGame(seed=args.seed + episode, step_reward=args.step_reward,
                         hunger_mode=args.hunger_mode, hunger_k=args.hunger_k)
        start = time.time()
        info = {}
        q_values = None

        while not game.done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key in (pygame.K_UP, pygame.K_PLUS, pygame.K_EQUALS):
                        speed = min(240.0, speed * 2)
                    elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                        speed = max(1.0, speed / 2)

            if not running:
                break

            if not paused:
                state = agent.get_state(game)
                with torch.no_grad():
                    pred = agent.model(torch.tensor(state, dtype=torch.float))
                q_values = pred.tolist()
                move = [0, 0, 0]
                move[int(torch.argmax(pred).item())] = 1
                _, _, _, info = game.play_step(move)

            screen.fill(NOIR)
            pygame.draw.rect(screen, NOIR, pygame.Rect(0, SCORE_PANEL_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH))
            draw_grid(screen)
            draw_apple(screen, game)
            draw_snake(screen, game)
            draw_panel(screen, font_main, font_small, game, episode, args.episodes,
                       time.time() - start, best, q_values)
            if paused:
                draw_message(screen, font_main, "PAUSE", BLANC)
            pygame.display.flip()
            clock.tick(speed)

        if not running:
            break

        best = max(best, game.score)
        scores.append(game.score)
        cause = info.get("cause", "?")
        print(f"partie {episode}: score={game.score} steps={game.steps} cause={cause}")

        if game.victory:
            draw_message(screen, font_big, "VICTOIRE !", VERT)
        else:
            draw_message(screen, font_big, "GAME OVER", ROUGE)
        pygame.display.flip()
        # petite pause lisible entre deux parties, interruptible
        end_at = time.time() + 1.5
        while time.time() < end_at and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    running = False
            clock.tick(30)

        episode += 1

    pygame.quit()
    if scores:
        print(f"\n{len(scores)} parties : mean={sum(scores)/len(scores):.1f} max={max(scores)}")


if __name__ == "__main__":
    main()
