"""Rendu Pygame et lecteur de replay.

Ce module est le SEUL à importer Pygame, et il est toujours importé
tardivement. Un entraînement headless ne doit jamais le charger.

Le lecteur rejoue une trajectoire enregistrée. Il ne fait pas rejouer le
modèle : on voit donc exactement la partie qui a produit le score mesuré,
image par image, et un replay s'ouvre sans PyTorch ni checkpoint.

La cadence d'affichage est purement visuelle (60 FPS par défaut).
Elle ne touche ni `GAME_SPEED` dans le moteur, ni le score, ni
aucune métrique.
"""

import argparse

import pygame

from . import rules
from .metrics import load_replay


def draw_board(surface, frame, font, header_lines):
    """Dessine une image du replay, avec la même charte que le socle."""
    surface.fill(rules.GRIS_FOND)

    board = pygame.Rect(
        0, rules.SCORE_PANEL_HEIGHT, rules.SCREEN_WIDTH, rules.SCREEN_WIDTH
    )
    pygame.draw.rect(surface, rules.NOIR, board)

    # Grille
    for x in range(0, rules.SCREEN_WIDTH + 1, rules.CELL_SIZE):
        pygame.draw.line(
            surface,
            rules.GRIS_GRILLE,
            (x, rules.SCORE_PANEL_HEIGHT),
            (x, rules.SCREEN_HEIGHT),
        )
    for y in range(rules.SCORE_PANEL_HEIGHT, rules.SCREEN_HEIGHT + 1, rules.CELL_SIZE):
        pygame.draw.line(
            surface, rules.GRIS_GRILLE, (0, y), (rules.SCREEN_WIDTH, y)
        )

    def cell_rect(position):
        return pygame.Rect(
            position[0] * rules.CELL_SIZE,
            position[1] * rules.CELL_SIZE + rules.SCORE_PANEL_HEIGHT,
            rules.CELL_SIZE,
            rules.CELL_SIZE,
        )

    if frame.get("food"):
        rect = cell_rect(frame["food"])
        pygame.draw.rect(surface, rules.ROUGE, rect, border_radius=5)
        pygame.draw.circle(
            surface,
            rules.BLANC,
            (rect.x + rules.CELL_SIZE * 0.7, rect.y + rules.CELL_SIZE * 0.3),
            rules.CELL_SIZE // 8,
        )

    body = frame["body"]
    for segment in body[1:]:
        rect = cell_rect(segment)
        pygame.draw.rect(surface, rules.VERT, rect)
        pygame.draw.rect(surface, rules.NOIR, rect, 1)

    head_rect = cell_rect(body[0])
    pygame.draw.rect(surface, rules.ORANGE, head_rect)
    pygame.draw.rect(surface, rules.NOIR, head_rect, 2)

    # Bandeau d'informations
    pygame.draw.rect(
        surface, rules.GRIS_FOND, (0, 0, rules.SCREEN_WIDTH, rules.SCORE_PANEL_HEIGHT)
    )
    pygame.draw.line(
        surface,
        rules.BLANC,
        (0, rules.SCORE_PANEL_HEIGHT - 2),
        (rules.SCREEN_WIDTH, rules.SCORE_PANEL_HEIGHT - 2),
        2,
    )
    for i, line in enumerate(header_lines):
        surface.blit(font.render(line, True, rules.BLANC), (8, 6 + i * 18))


def replay_frames(
    replay,
    speed_multiplier=None,
    close_when_done=True,
    linger_seconds=1.5,
    max_real_seconds=None,
    fps=60,
):
    """Rejoue une trajectoire enregistrée dans une fenêtre Pygame.

    Commandes : espace pause, flèche droite pas à pas, + accélérer,
    - ralentir, échap fermer.

    Args:
        replay: dictionnaire chargé depuis un fichier de replay.
        speed_multiplier: ancien facteur visuel explicite, conservé pour
            compatibilité. S'il est fourni, il prend priorité sur fps.
        fps: cadence graphique uniquement (défaut 60).
        max_real_seconds: coupe-circuit pour les tests automatisés.
    """
    frames = replay["frames"]
    if not frames:
        return 0

    if type(fps) is not int or fps < 1:
        raise ValueError("fps doit être un entier >= 1")
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((rules.SCREEN_WIDTH, rules.SCREEN_HEIGHT))
    pygame.display.set_caption(
        f"Replay {replay.get('run_id', '?')} — score {replay.get('score', '?')}"
    )
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    index = 0
    paused = False
    frame_rate = fps if speed_multiplier is None else rules.GAME_SPEED * max(1, int(speed_multiplier))
    running = True
    announced = False
    started = pygame.time.get_ticks()

    while running and index < len(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    index = min(index + 1, len(frames) - 1)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    frame_rate = min(frame_rate + 10, 360)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    frame_rate = max(1, frame_rate - 10)

        frame = frames[index]
        draw_board(
            screen,
            frame,
            font,
            [
                f"{replay.get('run_id', '?')} / {replay.get('algorithm', '?')}"
                f"   bloc {replay.get('episode', '-')}   seed {replay.get('seed', '-')}",
                f"Score {frame.get('score', 0)}   Record global "
                f"{replay.get('record', replay.get('score', 0))}"
                f"   Step {frame.get('steps', index)} / {len(frames) - 1}",
                f"Vitesse visuelle {frame_rate} FPS"
                + ("   [PAUSE]" if paused else "")
                + "   espace pause, droite pas a pas, +/- vitesse, echap quitter",
            ],
        )
        pygame.display.flip()
        if not announced:
            print(f"REPLAY WINDOW OPEN | driver {pygame.display.get_driver()} | fps {frame_rate} | frames {len(frames)}",
                  flush=True)
            announced = True

        if not paused:
            index += 1

        clock.tick(frame_rate)

        if (
            max_real_seconds is not None
            and (pygame.time.get_ticks() - started) / 1000 > max_real_seconds
        ):
            break

    played = index
    if running and close_when_done and linger_seconds:
        pygame.time.wait(int(linger_seconds * 1000))
    pygame.quit()
    print(f"REPLAY WINDOW CLOSED | frames {played}/{len(frames)}", flush=True)
    return played


def replay_file(path, **kwargs):
    """Charge un fichier de replay et le rejoue."""
    return replay_frames(load_replay(path), **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rejoue une partie enregistrée, sans charger de modèle"
    )
    parser.add_argument("replay", help="chemin d'un fichier de replay JSON")
    cadence = parser.add_mutually_exclusive_group()
    cadence.add_argument("--fps", type=int, default=60, help="cadence graphique (défaut : 60 FPS)")
    cadence.add_argument("--speed", type=int, help="ancien multiplicateur visuel (compatibilité)")
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args(argv)

    replay = load_replay(args.replay)
    print(
        f"Replay {replay['run_id']} / {replay['algorithm']} | seed {replay['seed']} "
        f"| score {replay['score']} | {len(replay['frames'])} images"
    )
    replay_frames(
        replay,
        speed_multiplier=args.speed,
        fps=args.fps,
        linger_seconds=3.0 if args.keep_open else 1.5,
    )


if __name__ == "__main__":
    main()
