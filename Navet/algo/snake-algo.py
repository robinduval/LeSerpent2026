# Visualisation pygame : l'algorithme joue, on regarde. Calqué sur ia/play_visual.py
# (mêmes couleurs, panneau, temps de jeu), sans torch. Le jeu est ia/game.py tel quel.
#
#   python3 main.py                    -> meilleur réglage : Dijkstra prudent (tol 5), switch à 60 %,
#                                         cycle le plus large (4 candidats, +2 pas), vitesse x1
#   python3 main.py --cycle helice --switch 0.3   -> cycle fixe en hélice (référence)
#   python3 main.py --x 20             -> 20x plus rapide
#   python3 main.py --switch 0.4 --episodes 3
#
# Touches : 1 = x1, 2 = x5, 3 = x20, 4 = max ; espace = pause ; c = afficher le cycle ; q = quitter
import argparse
import time

import pygame

from player import AlgoPlayer, CYCLE_MODES, N, LOOP_LIMIT, new_game, xy
from game import GRID_SIZE

CELL_SIZE = 30
GAME_SPEED = 5  # clock du socle = x1
SPEEDS = {pygame.K_1: 1, pygame.K_2: 5, pygame.K_3: 20, pygame.K_4: 0}  # 0 = max

SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCORE_PANEL_HEIGHT = 80
SCREEN_HEIGHT = SCREEN_WIDTH + SCORE_PANEL_HEIGHT

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
BLEU = (80, 160, 255)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)
GRIS_CYCLE = (45, 45, 70)
MODE_COLORS = {"dijkstra": BLEU, "queue": ROUGE, "transition": ORANGE, "secours": ROUGE, "evite": (120, 200, 255),
               "hamilton": VERT, "raccourci": (180, 255, 120)}


def center(p):
    x, y = xy(p)
    return x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2 + SCORE_PANEL_HEIGHT


def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH + 1, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, SCREEN_HEIGHT))
    for y in range(SCORE_PANEL_HEIGHT, SCREEN_HEIGHT + 1, CELL_SIZE):
        pygame.draw.line(surface, GRIS_GRILLE, (0, y), (SCREEN_WIDTH, y))


def draw_cycle(surface, cycle):
    # segments entre cellules voisines non wrappées (les sauts de bord ne sont pas tracés)
    for i in range(N):
        a, b = cycle.order[i], cycle.order[(i + 1) % N]
        (ax, ay), (bx, by) = xy(a), xy(b)
        if abs(ax - bx) + abs(ay - by) == 1:
            pygame.draw.line(surface, GRIS_CYCLE, center(a), center(b), 3)


def draw_path(surface, player):
    color = MODE_COLORS.get(player.mode, BLANC)
    for p in player.path:
        pygame.draw.circle(surface, color, center(p), 3)


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


def draw_panel(surface, font, small, game, player, episode, episodes, best, mult):
    pygame.draw.rect(surface, GRIS_FOND, (0, 0, SCREEN_WIDTH, SCORE_PANEL_HEIGHT))
    pygame.draw.line(surface, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (SCREEN_WIDTH, SCORE_PANEL_HEIGHT - 2), 2)

    surface.blit(font.render(f"Score: {game.score}", True, BLANC), (10, 12))
    fill = small.render(f"{len(game.body) / N * 100:.0f}% rempli", True, BLANC)
    surface.blit(fill, (SCREEN_WIDTH // 2 - fill.get_width() // 2, 18))

    # temps de JEU (steps / GAME_SPEED), indépendant de la vitesse d'affichage
    m, s = divmod(int(game.steps / GAME_SPEED), 60)
    t = small.render(f"Temps jeu: {m:02d}:{s:02d}", True, BLANC)
    surface.blit(t, (SCREEN_WIDTH - t.get_width() - 10, 14))

    vit = "max" if mult == 0 else f"x{mult}"
    left = small.render(f"Partie {episode}/{episodes}  best {best}  {vit}", True, BLANC)
    surface.blit(left, (10, 48))
    mode = small.render(player.mode.upper(), True, MODE_COLORS.get(player.mode, BLANC))
    surface.blit(mode, (SCREEN_WIDTH - mode.get_width() - 10, 48))


def draw_message(surface, font, message, color):
    text = font.render(message, True, color)
    rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    bg = rect.inflate(40, 20)
    pygame.draw.rect(surface, NOIR, bg, border_radius=10)
    pygame.draw.rect(surface, BLANC, bg, 2, border_radius=10)
    surface.blit(text, rect)


def main():
    ap = argparse.ArgumentParser(description="Regarder l'algorithme (Dijkstra + flood-fill -> Hamilton) jouer.")
    ap.add_argument("--switch", type=float, default=0.6,
                    help="remplissage (0..1) déclenchant Hamilton ; > 1 = jamais")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=999, help="même convention que ia/play_visual.py")
    ap.add_argument("--cycle", default="prudent", choices=CYCLE_MODES,
                    help="prudent = Dijkstra prudent + seuil tardif (meilleur, 21:10) ; "
                         "helice = cycle fixe (référence) ; reconstruit = option A ; temporel = option B")
    ap.add_argument("--tol", type=int, default=5,
                    help="(prudent/ilots) îlots bordés par les tol derniers segments ignorés (meilleur : 5)")
    ap.add_argument("--width-k", type=int, default=4,
                    help="cycles d'A candidats à chaque reconstruction (1 = pas de choix ; meilleur : 4)")
    ap.add_argument("--width-slack", type=int, default=2,
                    help="pas de plus tolérés vers la pomme pour un cycle plus large (meilleur : 2)")
    ap.add_argument("--cycle-trap", dest="trap", action="store_true",
                    help="(temporel) switch aussi juste avant d'être coincé")
    ap.add_argument("--x", default="1", choices=["1", "5", "20", "max"], help="accélération")
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    sw = "jamais" if args.switch > 1 else f"{args.switch:.0%}"
    pygame.display.set_caption(f"Navet — algo, switch Hamilton à {sw}, cycle {args.cycle}")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 26)
    font_big = pygame.font.Font(None, 70)

    mult = 0 if args.x == "max" else int(args.x)
    paused = False
    show_cycle = True
    best = 0
    results = []
    running = True
    episode = 1

    while running and episode <= args.episodes:
        game = new_game(args.seed + episode)
        player = AlgoPlayer(args.switch, args.cycle, args.trap, tol=args.tol,
                            width_k=args.width_k, width_slack=args.width_slack)
        info = {}

        while not game.done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_c:
                        show_cycle = not show_cycle
                    elif event.key in SPEEDS:
                        mult = SPEEDS[event.key]
            if not running:
                break

            if not paused:
                if game.steps_since_apple > LOOP_LIMIT:
                    info = {"cause": "boucle"}
                    break
                _, _, _, info = game.play_step(player.get_action(game))

            screen.fill(NOIR)
            if show_cycle and player.hamilton:
                draw_cycle(screen, player.cycle)
            draw_grid(screen)
            draw_path(screen, player)
            draw_apple(screen, game)
            draw_snake(screen, game)
            draw_panel(screen, font_main, font_small, game, player, episode, args.episodes, best, mult)
            if paused:
                draw_message(screen, font_main, "PAUSE", BLANC)
            pygame.display.flip()
            clock.tick(GAME_SPEED * mult if mult else 0)

        if not running:
            break

        best = max(best, game.score)
        cause = info.get("cause", "?")
        results.append((game.score, game.steps, game.victory))
        st = player.stats
        sw_txt = f"switch au pas {st['switch_step']}" if st["switch_step"] is not None else "pas de switch"
        print(f"partie {episode}: score={game.score} steps={game.steps} cause={cause} ({sw_txt})")

        if game.victory:
            draw_message(screen, font_big, "VICTOIRE !", VERT)
        else:
            draw_message(screen, font_big, "GAME OVER", ROUGE)
        pygame.display.flip()
        end_at = time.time() + 1.5
        while time.time() < end_at and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    running = False
            clock.tick(30)

        episode += 1

    pygame.quit()
    if results:
        scores = [r[0] for r in results]
        wins = [r for r in results if r[2]]
        print(f"\n{len(results)} parties : victoires {len(wins)}/{len(results)}  "
              f"score moyen {sum(scores) / len(scores):.1f}  max {max(scores)}")
        if wins:
            mean_steps = sum(r[1] for r in wins) / len(wins)
            m, s = divmod(int(mean_steps / GAME_SPEED), 60)
            print(f"victoire en {mean_steps:.0f} pas en moyenne ({m:02d}:{s:02d} de jeu à la clock du socle)")


if __name__ == "__main__":
    main()
