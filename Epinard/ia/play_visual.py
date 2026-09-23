"""
Interface 1 : fenêtre Pygame — regarder l'agent PPO jouer en direct.

Affiche score, temps, ratio score/temps, remplissage, et l'état du
masque de sécurité (combien d'actions sont sûres à cet instant).

Lancement :
    python3 play_visual.py                  # modèle entraîné, 5 FPS (la clock du jeu)
    python3 play_visual.py --fps 20         # accéléré, pour voir plus vite
    python3 play_visual.py --essais 10      # enchaîne 10 parties
    python3 play_visual.py --stochastique   # échantillonne au lieu de prendre le max

Touches : ESPACE = partie suivante, ÉCHAP = quitter.
"""
import argparse
import os
import sys

# Retirer le driver headless éventuellement posé par game_core
os.environ.pop("SDL_VIDEODRIVER", None)

import numpy as np
import torch

import game_core as g
from agent import get_mask, get_state, Config
from env_wrapper import SnakeEnv
from model import ActorCritic
from safety import fallback_action

import pygame


def draw_panel(screen, font, small, env, record, essai, n_essais, n_safe):
    """Panneau d'information en haut de la fenêtre."""
    pygame.draw.rect(screen, g.GRIS_FOND, (0, 0, g.SCREEN_WIDTH, g.SCORE_PANEL_HEIGHT))
    pygame.draw.line(
        screen, g.BLANC,
        (0, g.SCORE_PANEL_HEIGHT - 2), (g.SCREEN_WIDTH, g.SCORE_PANEL_HEIGHT - 2), 2
    )

    t = env.elapsed
    ratio = env.score / t if t > 0 else 0.0

    # Ligne 1 : score, temps, ratio
    screen.blit(font.render(f"Score: {env.score}", True, g.BLANC), (10, 8))
    mn, sec = int(t // 60), int(t % 60)
    tx = font.render(f"{mn:02d}:{sec:02d}", True, g.BLANC)
    screen.blit(tx, (g.SCREEN_WIDTH - tx.get_width() - 10, 8))
    rt = font.render(f"{ratio:.2f} pt/s", True, (120, 220, 255))
    screen.blit(rt, (g.SCREEN_WIDTH // 2 - rt.get_width() // 2, 8))

    # Ligne 2 : record, essai, remplissage, sécurité
    fill = 100 * len(env.snake.body) / (g.GRID_SIZE ** 2)
    screen.blit(small.render(f"Record: {record}", True, (180, 180, 180)), (10, 46))
    info = small.render(f"Essai {essai}/{n_essais}", True, (180, 180, 180))
    screen.blit(info, (g.SCREEN_WIDTH // 2 - info.get_width() // 2, 46))

    col = (100, 230, 100) if n_safe >= 2 else (240, 180, 60) if n_safe == 1 else (240, 90, 90)
    sf = small.render(f"Sûr: {n_safe}/3  |  {fill:.0f}%", True, col)
    screen.blit(sf, (g.SCREEN_WIDTH - sf.get_width() - 10, 46))


def draw_game(screen, env):
    """Zone de jeu : grille, pomme, serpent."""
    area = pygame.Rect(0, g.SCORE_PANEL_HEIGHT, g.SCREEN_WIDTH, g.SCREEN_WIDTH)
    pygame.draw.rect(screen, g.NOIR, area)

    for x in range(0, g.SCREEN_WIDTH, g.CELL_SIZE):
        pygame.draw.line(screen, g.GRIS_GRILLE, (x, g.SCORE_PANEL_HEIGHT), (x, g.SCREEN_HEIGHT))
    for y in range(g.SCORE_PANEL_HEIGHT, g.SCREEN_HEIGHT, g.CELL_SIZE):
        pygame.draw.line(screen, g.GRIS_GRILLE, (0, y), (g.SCREEN_WIDTH, y))

    # Pomme
    ax, ay = env.apple.position
    r = pygame.Rect(ax * g.CELL_SIZE, ay * g.CELL_SIZE + g.SCORE_PANEL_HEIGHT,
                    g.CELL_SIZE, g.CELL_SIZE)
    pygame.draw.rect(screen, g.ROUGE, r, border_radius=5)
    pygame.draw.circle(screen, g.BLANC,
                       (r.x + g.CELL_SIZE * 0.7, r.y + g.CELL_SIZE * 0.3), g.CELL_SIZE // 8)

    # Corps puis tête
    for seg in env.snake.body[1:]:
        rc = pygame.Rect(seg[0] * g.CELL_SIZE, seg[1] * g.CELL_SIZE + g.SCORE_PANEL_HEIGHT,
                         g.CELL_SIZE, g.CELL_SIZE)
        pygame.draw.rect(screen, g.VERT, rc)
        pygame.draw.rect(screen, g.NOIR, rc, 1)

    h = env.snake.head_pos
    hr = pygame.Rect(h[0] * g.CELL_SIZE, h[1] * g.CELL_SIZE + g.SCORE_PANEL_HEIGHT,
                     g.CELL_SIZE, g.CELL_SIZE)
    pygame.draw.rect(screen, g.ORANGE, hr)
    pygame.draw.rect(screen, g.NOIR, hr, 2)


def message(screen, font, text, color, y_offset=0):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(g.SCREEN_WIDTH // 2, g.SCREEN_HEIGHT // 2 + y_offset))
    bg = rect.inflate(40, 40)
    pygame.draw.rect(screen, g.NOIR, bg, border_radius=10)
    pygame.draw.rect(screen, g.BLANC, bg, 2, border_radius=10)
    screen.blit(surf, rect)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ppo_snake.pth")
    p.add_argument("--fps", type=int, default=g.GAME_SPEED,
                   help=f"défaut {g.GAME_SPEED} = la clock du jeu original")
    p.add_argument("--essais", type=int, default=5)
    p.add_argument("--greedy", action="store_true",
                   help="mode deterministe (deconseille : la politique PPO est stochastique)")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Modèle introuvable : {args.model}")
        print("Lance d'abord l'entraînement : python3 agent.py")
        sys.exit(1)

    model = ActorCritic()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    pygame.init()
    screen = pygame.display.set_mode((g.SCREEN_WIDTH, g.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake PPO — agent entraîné")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small = pygame.font.Font(None, 26)
    big = pygame.font.Font(None, 64)

    env = SnakeEnv()
    cfg = Config()
    record = 0
    essai = 1
    resultats = []
    running = True

    while running and essai <= args.essais:
        env.reset()

        while not env.game_over and running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False

            state = get_state(env)
            mask = get_mask(env, cfg.safety_margin)
            st = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mk = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)

            with torch.no_grad():
                if args.greedy:
                    a = int(model.greedy(st, mk).item())
                else:
                    action, _, _, _ = model.act(st, mk)
                    a = int(action.item())

            # Si le masque n'autorisait rien, on prend le coup qui survit le plus longtemps
            if not mask[a]:
                a = fallback_action(env.snake.body, env.snake.direction)

            env.play_step(a)
            record = max(record, env.score)

            draw_game(screen, env)
            draw_panel(screen, font, small, env, record, essai, args.essais, int(mask.sum()))
            pygame.display.flip()
            clock.tick(args.fps)

        if not running:
            break

        s = env.stats()
        resultats.append(s)
        print(f"Essai {essai}: score={s['score']} temps={s['temps']}s "
              f"ratio={s['ratio_score_temps']} pt/s fin={s['fin']}")

        # Écran de fin
        draw_game(screen, env)
        draw_panel(screen, font, small, env, record, essai, args.essais, 0)
        if env.victory:
            message(screen, big, "VICTOIRE !", g.VERT)
        else:
            message(screen, big, f"Score {env.score}", g.ROUGE)
        message(screen, small, "ESPACE = suivant  |  ECHAP = quitter", g.BLANC, y_offset=80)
        pygame.display.flip()

        waiting = True
        while waiting and running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running, waiting = False, False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running, waiting = False, False
                    elif e.key == pygame.K_SPACE:
                        waiting = False
            clock.tick(30)

        essai += 1

    pygame.quit()

    if resultats:
        scores = [r["score"] for r in resultats]
        ratios = [r["ratio_score_temps"] for r in resultats]
        print("\n--- Récapitulatif ---")
        print(f"Essais          : {len(resultats)}")
        print(f"Score moyen     : {sum(scores)/len(scores):.1f}")
        print(f"Record          : {max(scores)}")
        print(f"Ratio moyen     : {sum(ratios)/len(ratios):.3f} pt/s")


if __name__ == "__main__":
    main()
