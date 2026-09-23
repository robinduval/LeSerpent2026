"""Baseline algorithmique : cycle hamiltonien sur le tore 15 x 15.

Lancement : python snake-algo.py. Aucun apprentissage, aucune clock accélérée.
Le moteur original reste la référence pour les déplacements et la croissance.
"""
import argparse
from collections import deque
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import random
import time

import pygame

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('snake_rules', ROOT.parent / 'ia' / 'serpent-algo.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Cycle:
    def __init__(self, bidirectional=False):
        n = base.GRID_SIZE
        # Chaque ligne avance n-1 fois à droite, puis descend. Son origine
        # recule donc d'une colonne. Après n lignes, le cycle se referme.
        self.cells = [((1-r+k) % n, (n//2+r) % n)
                      for r in range(n) for k in range(n)]
        if bidirectional:
            # 2-opt sur des paires de lignes disjointes : remplacer a-b et
            # c-d par a-c et b-d, en inversant b..c. Les nouvelles arêtes sont
            # verticales voisines sur le tore. Un seul cycle est conservé.
            # La première ligne (et donc le corps initial) reste intacte.
            for r in range(1, n-1, 2):
                i, j = r*n+5, (r+1)*n+6
                self.cells[i+1:j+1] = reversed(self.cells[i+1:j+1])
        self.index = {cell: i for i, cell in enumerate(self.cells)}

    def next(self, cell):
        return self.cells[(self.index[tuple(cell)]+1) % len(self.cells)]

    def distance(self, start, end):
        return (self.index[tuple(end)]-self.index[tuple(start)]) % len(self.cells)


class Game:
    def __init__(self, seed, algorithm='hamiltonian'):
        if algorithm not in ('hamiltonian', 'bfs-safe', 'shortcut'):
            raise ValueError('Algorithme inconnu')
        self.algorithm = algorithm
        self.decision = 'cycle'
        self.rng = random.Random(seed)
        self.snake = base.Snake()
        self.cycle = Cycle(bidirectional=algorithm == 'shortcut')
        self.apple = base.Apple.__new__(base.Apple)
        self.apple.position = self.place_apple()
        self.steps = self.without_food = self.wraps = 0
        self.done = False
        self.reason = 'running'
        self.decision_ms = 0.0
        self.max_decision_ms = 0.0
        self.first_ten_seconds = None
        self.direction_counts = {'right': 0, 'left': 0, 'up': 0, 'down': 0}

    def place_apple(self):
        occupied = set(map(tuple, self.snake.body))
        free = [(x, y) for x in range(base.GRID_SIZE)
                for y in range(base.GRID_SIZE) if (x, y) not in occupied]
        return self.rng.choice(free) if free else None

    def step(self):
        if self.done:
            return
        before = tuple(self.snake.head_pos)
        started = time.perf_counter()
        target = self.choose_target()
        n = base.GRID_SIZE
        direction = next((dx, dy) for dx, dy in
                         (base.RIGHT, base.DOWN, base.LEFT, base.UP)
                         if ((before[0]+dx) % n, (before[1]+dy) % n) == target)
        self.decision_ms = (time.perf_counter()-started)*1000
        self.max_decision_ms = max(self.max_decision_ms, self.decision_ms)
        self.snake.set_direction(direction)
        self.snake.move()
        name = {base.RIGHT: 'right', base.LEFT: 'left', base.UP: 'up', base.DOWN: 'down'}[self.snake.direction]
        self.direction_counts[name] += 1
        self.steps += 1
        self.without_food += 1
        self.wraps += int(abs(before[0]-target[0])+abs(before[1]-target[1]) > 1)
        if self.snake.is_game_over():
            self.done, self.reason = True, 'collision'
        elif tuple(self.snake.head_pos) == self.apple.position:
            self.snake.grow()
            self.without_food = 0
            self.apple.position = self.place_apple()
            if self.apple.position is None:
                self.done, self.reason = True, 'victory'

    def choose_target(self):
        head = tuple(self.snake.head_pos)
        fallback = self.cycle.next(head)
        self.decision = 'cycle'
        if self.algorithm == 'hamiltonian':
            return fallback
        if self.algorithm == 'shortcut':
            # En fin de remplissage, des pommes consécutives peuvent retenir
            # la queue plusieurs tours. Revenir tôt au cycle pour résorber
            # les trous laissés par les raccourcis.
            if len(self.snake.body) >= len(self.cycle.cells)//2:
                return fallback
            # Raccourci glouton : maximiser l'avance sans dépasser la pomme.
            # Ne pas anticiper la libération de la queue : avec des pommes
            # consécutives, cela supprimait la réserve nécessaire au repli.
            space = self.cycle.distance(head, self.snake.body[-1]) - 1 - int(self.snake.grow_pending)
            food = self.cycle.distance(head, self.apple.position)
            occupied = set(map(tuple, self.snake.body if self.snake.grow_pending else self.snake.body[:-1]))
            candidates = []
            for dx, dy in (base.RIGHT, base.DOWN, base.LEFT, base.UP):
                if (dx, dy) == tuple(-v for v in self.snake.direction):
                    continue
                target = ((head[0]+dx)%base.GRID_SIZE, (head[1]+dy)%base.GRID_SIZE)
                advance = self.cycle.distance(head, target)
                reserve = int(target == self.apple.position)
                if target not in occupied and 0 < advance <= food and advance < space-reserve:
                    candidates.append((advance, target))
            if candidates:
                target = max(candidates)[1]
                self.decision = 'raccourci dynamique' if target != fallback else 'cycle'
                return target
            return fallback
        # Aucun saut au-delà de la queue ni de la pomme. Réserve conservatrice
        # pour la croissance différée ; le successeur reste le repli.
        food_distance = self.cycle.distance(head, self.apple.position)
        limit = min(food_distance, self.cycle.distance(head, self.snake.body[-1])
                    - 2 - int(self.snake.grow_pending))
        if limit < 2:
            return fallback
        n = base.GRID_SIZE
        ranks = {cell: self.cycle.distance(head, cell) for cell in self.cycle.cells}
        queue = deque([head])
        parent = {head: None}
        best = head
        while queue:
            cell = queue.popleft()
            if cell == self.apple.position:
                best = cell
                break
            for dx, dy in (base.RIGHT, base.DOWN, base.LEFT, base.UP):
                target = ((cell[0]+dx) % n, (cell[1]+dy) % n)
                if target not in parent and ranks[cell] < ranks[target] <= limit:
                    parent[target] = cell
                    queue.append(target)
                    if ranks[target] > ranks[best]:
                        best = target
        if best == head:
            return fallback
        while parent[best] != head:
            best = parent[best]
        self.decision = 'BFS / raccourci' if best != fallback else 'BFS / cycle'
        return best

    def metrics(self, elapsed):
        score = self.snake.score
        if score >= 10 and self.first_ten_seconds is None:
            self.first_ten_seconds = elapsed
        return dict(algorithm=self.algorithm, decision=self.decision, score=score, steps=self.steps,
                    length=len(self.snake.body), occupancy=len(self.snake.body)/225,
                    real_seconds=elapsed, score_per_real_second=score/elapsed if elapsed else 0,
                    steps_per_apple=self.steps/score if score else None,
                    without_food=self.without_food, wraps=self.wraps,
                    cycle_distance_to_apple=self.cycle.distance(self.snake.head_pos, self.apple.position)
                    if self.apple.position is not None else 0,
                    decision_ms=self.decision_ms, max_decision_ms=self.max_decision_ms,
                    first_ten_seconds=self.first_ten_seconds, clock_hz=base.GAME_SPEED,
                    status=self.reason, completed=self.done,
                    **{f'moves_{key}': value for key, value in self.direction_counts.items()})


class Recorder:
    def __init__(self, folder, seed, algorithm='hamiltonian'):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'config.json').write_text(json.dumps(dict(seed=seed, algorithm=algorithm,
            grid_size=base.GRID_SIZE, clock_hz=base.GAME_SPEED), indent=2))

    def write(self, data):
        path = self.folder/'steps.csv'
        new = not path.exists()
        with path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data))
            if new:
                writer.writeheader()
            writer.writerow(data)
        temporary = self.folder/'metrics.tmp'
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(self.folder/'metrics.json')


class View:
    """Rendu du socle, comme snake-ia.py ; instruments hors du plateau."""
    def __init__(self):
        self.screen = pygame.display.set_mode((850, base.SCREEN_HEIGHT))
        pygame.display.set_caption('Citron | Algo : cycle torique | 5 Hz')
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 22)

    def text(self, value, x, y, color=(226, 236, 241), font=None):
        self.screen.blit((font or self.font).render(str(value), True, color), (x, y))

    def draw(self, game, data, history):
        s = self.screen
        s.fill(base.GRIS_FOND)
        pygame.draw.rect(s, base.NOIR, (0, base.SCORE_PANEL_HEIGHT,
                                       base.SCREEN_WIDTH, base.SCREEN_WIDTH))
        base.draw_grid(s)
        game.apple.draw(s)
        game.snake.draw(s)
        elapsed = data['real_seconds']
        self.text(f"Score : {data['score']} | {elapsed:.1f}s", 15, 25, base.BLANC)
        rows = [f'Partie 1 | algo : {game.algorithm}',
                f"Score {data['score']} / objectif 10",
                f"Temps réel : {int(elapsed)//60:02}:{elapsed%60:04.1f}",
                f"Score / seconde réelle : {data['score_per_real_second']:.3f}",
                f"Occupation : {data['length']} / 225 ({data['occupancy']:.1%})",
                f"Mouvements : {data['steps']}  |  traversées : {data['wraps']}",
                f"Sans pomme : {data['without_food']} mouvements",
                f"Pomme dans : {data['cycle_distance_to_apple']} pas du cycle",
                f"Pas / pomme : {data['steps_per_apple']:.1f}" if data['steps_per_apple'] else 'Pas / pomme : en attente',
                f"Calcul décision : {data['decision_ms']:.3f} ms",
                f'Décision : {game.decision}',
                f"État : {data['status']} | Clock : 5 Hz",
                f"Directions G {data['moves_left']} D {data['moves_right']} H {data['moves_up']} B {data['moves_down']}"]
        for i, row in enumerate(rows):
            self.text(row, 460, 20+i*27, base.BLANC, self.small)
        self.text('Score / temps réel', 460, 377, base.BLANC, self.small)
        pygame.draw.line(s, base.BLANC, (460, 476), (830, 476))
        if len(history) > 1:
            maximum = max(10, data['score'])
            points = [(460+int(t/max(elapsed, .001)*370), 473-int(score/maximum*65))
                      for t, score in history]
            pygame.draw.lines(s, base.ORANGE, False, points)
        self.text('Échap : terminer | Export CSV/JSON : runs/', 460, 500, base.BLANC, self.small)
        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algorithm', choices=['hamiltonian', 'bfs-safe', 'shortcut'], default='shortcut')
    parser.add_argument('--max-steps', type=int, default=0, help='Limite de diagnostic ; partie marquée interrompue')
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error('--max-steps doit être positif')
    pygame.init()
    view = View()
    game = Game(args.seed, args.algorithm)
    pygame.display.set_caption(f'Citron | {args.algorithm} | 5 Hz')
    folder = ROOT/'runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    recorder = Recorder(folder, args.seed, args.algorithm)
    print(f'Métriques : {folder}', flush=True)
    clock = pygame.time.Clock()
    started = time.perf_counter()
    history = [(0, 0)]
    data = game.metrics(0)
    recorder.write(data)
    view.draw(game, data, history)
    try:
        running = True
        while running:
            clock.tick(base.GAME_SPEED)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            if not running:
                break
            if not game.done:
                game.step()
                data = game.metrics(time.perf_counter()-started)
                history.append((data['real_seconds'], data['score']))
                recorder.write(data)
            view.draw(game, data, history)
            if args.max_steps and game.steps >= args.max_steps:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not game.done:
            game.reason = 'interrupted'
            data = game.metrics(time.perf_counter()-started)
        recorder.write(data)
        print(json.dumps(data, indent=2), flush=True)
        pygame.quit()


if __name__ == '__main__':
    main()
