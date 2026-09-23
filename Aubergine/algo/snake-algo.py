"""
Snake-algo : bot « cycle hamiltonien + raccourcis » pour le vrai jeu (serpent-algo.py).

Le fichier du jeu n'est PAS modifié ni recopié : on l'importe et on lance sa vraie
fonction main(). Le bot s'y branche par sous-classes de Snake et d'Apple (le jeu les
cherche dans ses globals) : à chaque déplacement, le bot choisit la direction avec le
set_direction() du jeu, puis Snake.move() du jeu s'exécute tel quel.

Algorithme (en couches) :
  1. Cycle hamiltonien sur le tore : idx[case] dans [0, N), rel(c) = (idx[c] - idx[tête]) mod N.
  2. Invariant : en avançant depuis la queue le long du cycle, on croise le corps dans l'ordre.
     Toutes les cases de rel 1 à rel(queue)-1 sont donc vides. On n'ose que rel <= limite.
  3. Plus court chemin vers la pomme dans le graphe sans cycle « rel croissant », par
     programmation dynamique en O(N), recalculé à chaque coup.

Usage :
  python snake-algo.py                      # fenêtre, vitesse 30 images/s
  python snake-algo.py --fps 5              # vitesse d'origine du jeu
  python snake-algo.py --fps 0              # aussi vite que possible
  python snake-algo.py --headless --games 20 --jobs 8 --debug   # vrai jeu sans fenêtre
"""
import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

INF = 10 ** 9


# --- COUCHE 1 : CYCLE HAMILTONIEN ---

def _zigzag(width, height):
    """Colonne 0 réservée au retour, aller-retours sur les colonnes 1..width-1.
    Sans passage de mur si height est pair ; si height est impair (et width impair),
    la dernière ligne finit en (width-1, height-1) et se referme par un mur."""
    cycle = [(x, 0) for x in range(width)]
    for y in range(1, height):
        xs = range(width - 1, 0, -1) if y % 2 == 1 else range(1, width)
        cycle += [(x, y) for x in xs]
    cycle += [(0, y) for y in range(height - 1, 0, -1)]
    return cycle


def build_cycle(width, height, wrap=True):
    """Construit un cycle hamiltonien de la grille width x height (liste de (x, y))."""
    if width < 3 or height < 3:
        raise ValueError("Grille trop petite (minimum 3x3).")
    if height % 2 == 0:
        cycle = _zigzag(width, height)
    elif width % 2 == 0:
        cycle = [(x, y) for (y, x) in _zigzag(height, width)]  # transposée
    elif wrap:
        cycle = _zigzag(width, height)  # deux dimensions impaires : le mur sert d'arête
    else:
        raise ValueError(
            f"Grille {width}x{height} : deux dimensions impaires, pas de cycle hamiltonien "
            "sans traverser les murs (nombre de cases impair)."
        )
    validate_cycle(cycle, width, height, wrap)
    return cycle


def _adjacent(a, b, width, height, wrap):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    if wrap:
        dx, dy = min(dx, width - dx), min(dy, height - dy)
    return dx + dy == 1


def validate_cycle(cycle, width, height, wrap=True):
    """Chaque case une seule fois, cases consécutives voisines (dernière -> première incluse)."""
    if len(cycle) != width * height or len(set(cycle)) != width * height:
        raise ValueError("Cycle invalide : cases manquantes ou en double.")
    for i, cell in enumerate(cycle):
        if not _adjacent(cell, cycle[(i + 1) % len(cycle)], width, height, wrap):
            raise ValueError(f"Cycle invalide : {cell} et {cycle[(i + 1) % len(cycle)]} ne sont pas voisines.")


def _orient_cycle(cycle, width, height, body):
    """Choisit la symétrie/sens du cycle dans lequel le corps de départ (queue -> tête)
    est déjà rangé dans l'ordre du cycle (invariant vrai dès le premier coup)."""
    tail_to_head = [tuple(p) for p in reversed(body)]
    for flip_x in (False, True):
        for flip_y in (False, True):
            base = [((width - 1 - x) if flip_x else x, (height - 1 - y) if flip_y else y) for x, y in cycle]
            for candidate in (base, base[::-1]):
                pos = {c: i for i, c in enumerate(candidate)}
                if all(c in pos for c in tail_to_head) and all(
                    (pos[b] - pos[a]) % len(candidate) == 1 for a, b in zip(tail_to_head, tail_to_head[1:])
                ):
                    return candidate
    raise ValueError("Aucune orientation du cycle ne contient le serpent de départ dans l'ordre.")


# --- COUCHES 2 ET 3 : SÉCURITÉ ET PLUS COURT CHEMIN CONTRAINT ---

class CycleBot:
    """Choisit la direction à chaque coup. Toutes les cases sont des entiers x + y * width."""

    def __init__(self, width, height, initial_body, wrap=True, growth=1, margin=2, debug=False):
        self.width, self.height, self.wrap = width, height, wrap
        self.n = width * height
        self.growth = growth
        self.margin = margin
        self.debug = debug
        cycle = _orient_cycle(build_cycle(width, height, wrap), width, height, initial_body)
        self.cyc = [x + y * width for x, y in cycle]
        self.idx = [0] * self.n
        for i, c in enumerate(self.cyc):
            self.idx[c] = i
        self.nbrs = [self._neighbors(c) for c in range(self.n)]
        self.fallbacks = 0  # nombre de fois où le filet de sécurité final a dû intervenir

    def _neighbors(self, c):
        x, y = c % self.width, c // self.width
        out = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if self.wrap:
                nx, ny = nx % self.width, ny % self.height
            elif not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            out.append(nx + ny * self.width)
        return out

    def cell(self, pos):
        return pos[0] + pos[1] * self.width

    def check_invariant(self, body):
        """Mode debug : corps rangé dans l'ordre du cycle, sans collision, longueur == cases occupées."""
        ih = self.idx[self.cell(body[0])]
        rels = [(self.idx[self.cell(p)] - ih) % self.n for p in body]
        assert len({self.cell(p) for p in body}) == len(body), "collision ou case comptée deux fois"
        assert all(rels[i] > rels[i + 1] for i in range(1, len(rels) - 1)), "invariant du cycle rompu"

    def choose(self, body, pending, apple):
        """Retourne la direction (dx, dy) à donner au serpent."""
        n, idx, cyc, nbrs = self.n, self.idx, self.cyc, self.nbrs
        h = self.cell(body[0])
        ih = idx[h]
        tail_rel = n if len(body) == 1 else (idx[self.cell(body[-1])] - ih) % n
        free_ahead = tail_rel - 1                 # cases rel 1..tail_rel-1 : toutes vides
        p = 1 if pending else 0                   # croissance en attente
        limit_free = free_ahead - p - self.margin
        limit_apple = limit_free - self.growth    # manger ajoute une croissance en attente
        apple_cell = self.cell(apple) if apple else None
        apple_rel = (idx[apple_cell] - ih) % n if apple else None

        target = None
        if apple_rel is not None and 1 <= apple_rel <= limit_apple:
            target = self._first_step(h, ih, apple_rel)
        if target is None:
            # pomme trop loin / trop risquée : voisin autorisé au plus grand rel, sinon case suivante du cycle
            best_rel = 0
            for v in nbrs[h]:
                r = (idx[v] - ih) % n
                if 1 <= r <= limit_free and v != apple_cell and r > best_rel:
                    target, best_rel = v, r
        if target is None:
            target = cyc[(ih + 1) % n]

        if not self._is_legal(target, body, p):
            # filet de sécurité (ne doit jamais arriver si l'invariant tient)
            self.fallbacks += 1
            if self.debug:
                raise AssertionError("mouvement du bot illégal dans le vrai jeu")
            legal = [v for v in nbrs[h] if self._is_legal(v, body, p)]
            if legal:
                target = legal[0]
        return self._direction(body[0], target)

    def _first_step(self, h, ih, apple_rel):
        """Plus court chemin tête -> pomme dans le graphe « rel croissant » (DP à rebours).
        Premier pas ; à longueur égale, le voisin de plus petit rel (saute moins de cases)."""
        n, idx, cyc, nbrs = self.n, self.idx, self.cyc, self.nbrs
        dist = [INF] * (apple_rel + 1)
        dist[apple_rel] = 0
        for k in range(apple_rel - 1, 0, -1):
            best = INF
            for v in nbrs[cyc[(ih + k) % n]]:
                r = (idx[v] - ih) % n
                if k < r <= apple_rel and dist[r] < best:
                    best = dist[r]
            dist[k] = best + 1 if best < INF else INF
        target, key = None, (INF, INF)
        for v in nbrs[h]:
            r = (idx[v] - ih) % n
            if 1 <= r <= apple_rel and dist[r] < INF and (dist[r], r) < key:
                target, key = v, (dist[r], r)
        return target

    def _is_legal(self, cell, body, pending):
        """Vraie règle du jeu : la case ne doit pas être dans le corps une fois la queue retirée."""
        occupied = body[:-1] if not pending else body
        return all(self.cell(p) != cell for p in occupied[1:]) if len(body) > 1 else True

    def _direction(self, head, cell):
        dx, dy = cell % self.width - head[0], cell // self.width - head[1]
        if self.wrap:
            if dx in (self.width - 1, -(self.width - 1)):
                dx = -1 if dx > 0 else 1
            if dy in (self.height - 1, -(self.height - 1)):
                dy = -1 if dy > 0 else 1
        return (dx, dy)


# --- BRANCHEMENT SUR LE VRAI JEU ---

def load_game():
    """Importe serpent-algo.py tel quel (le nom contient un tiret : import par chemin)."""
    path = Path(__file__).with_name("serpent-algo.py")
    spec = importlib.util.spec_from_file_location("serpent_algo", path)
    game = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game)
    return game


class Run:
    """État de la partie en cours (partagé entre Snake, Apple et l'horloge)."""
    def __init__(self):
        self.snake = None
        self.apple = None
        self.victory = False
        self.reported = False


def install_bot(game, args, results):
    """Remplace Snake et Apple du jeu par des sous-classes pilotées par le bot, et l'horloge
    par une horloge qui règle la vitesse et met fin à la partie (mode --headless)."""
    import pygame
    run = Run()
    cap = 10 * game.GRID_SIZE ** 2 * game.GRID_SIZE ** 2  # limite de coups : détecte une boucle infinie
    real_clock = pygame.time.Clock

    class BotSnake(game.Snake):
        def __init__(self):
            super().__init__()
            run.snake, run.victory, run.reported = self, False, False
            self.moves = 0
            self.bot = CycleBot(game.GRID_SIZE, game.GRID_SIZE, self.body, wrap=True,
                                growth=1, margin=args.margin, debug=args.debug)

        def move(self):
            if args.debug:
                self.bot.check_invariant(self.body)
            self.set_direction(self.bot.choose(self.body, self.grow_pending, run.apple.position))
            super().move()
            self.moves += 1
            if args.debug and not self.is_game_over():
                self.bot.check_invariant(self.body)

    class BotApple(game.Apple):
        def __init__(self, snake_body):
            super().__init__(snake_body)
            run.apple = self

        def relocate(self, snake_body):
            moved = super().relocate(snake_body)
            if not moved:
                run.victory = True
            return moved

    class BotClock:
        def __init__(self):
            self._clock = real_clock()

        def tick(self, _fps=0):
            s = run.snake
            if s is not None and not run.reported:
                outcome = "victoire" if run.victory else ("mort" if s.is_game_over() else None)
                if outcome is None and s.moves > cap:
                    outcome = "boucle"
                if outcome:
                    run.reported = True
                    results.append({"issue": outcome, "score": s.score, "coups": s.moves,
                                    "longueur": len(s.body), "replis": s.bot.fallbacks})
                    if not args.headless:
                        print(f"{outcome.upper()} : score {s.score}, {s.moves} coups, longueur {len(s.body)}")
                    if args.headless:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
            return self._clock.tick(args.fps)

    game.Snake, game.Apple = BotSnake, BotApple
    pygame.time.Clock = BotClock


def play(args, games=1, seed=None):
    """Lance `games` fois la vraie fonction main() du jeu ; retourne la liste des résultats."""
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    game = load_game()
    results = []
    install_bot(game, args, results)
    for i in range(games):
        if seed is not None:
            random.seed(seed + i)
        game.main()
    return results


def _worker(job):
    args, games, seed = job
    return play(args, games, seed)


def summarize(results, elapsed):
    n = len(results)
    wins = sum(r["issue"] == "victoire" for r in results)
    print(f"\n{n} parties du vrai jeu en {elapsed:.1f}s : {wins} victoires ({100 * wins / n:.1f} %), "
          f"{sum(r['issue'] == 'mort' for r in results)} morts, {sum(r['issue'] == 'boucle' for r in results)} boucles, "
          f"{sum(r['replis'] for r in results)} replis de sécurité")
    scores = [r["score"] for r in results]
    coups = [r["coups"] for r in results]
    print(f"score  : min {min(scores)}, moyenne {sum(scores) / n:.1f}, max {max(scores)}")
    print(f"coups  : min {min(coups)}, moyenne {sum(coups) / n:.0f}, max {max(coups)}")
    print(f"coups par pomme (moyenne) : {sum(coups) / max(1, sum(scores)):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Bot hamiltonien + raccourcis sur le vrai jeu Snake.")
    parser.add_argument("--fps", type=int, default=30, help="images/s (5 = vitesse d'origine, 0 = illimité)")
    parser.add_argument("--margin", type=int, default=2, help="marge de sécurité de la limite de raccourci")
    parser.add_argument("--headless", action="store_true", help="sans fenêtre, enchaîne --games parties puis résume")
    parser.add_argument("--games", type=int, default=1, help="nombre de parties (avec --headless)")
    parser.add_argument("--jobs", type=int, default=1, help="processus parallèles (avec --headless)")
    parser.add_argument("--seed", type=int, default=None, help="graine aléatoire de la première partie")
    parser.add_argument("--debug", action="store_true", help="vérifie l'invariant à chaque coup")
    args = parser.parse_args()

    if not args.headless:
        play(args)
        return
    args.fps = 0
    start = time.time()
    seed = 0 if args.seed is None else args.seed
    if args.jobs <= 1:
        results = play(args, args.games, seed)
    else:
        import multiprocessing
        chunks = [len(range(j, args.games, args.jobs)) for j in range(args.jobs)]
        offsets = [sum(chunks[:j]) for j in range(args.jobs)]
        with multiprocessing.Pool(args.jobs) as pool:
            parts = pool.map(_worker, [(args, c, seed + o) for c, o in zip(chunks, offsets) if c])
        results = [r for part in parts for r in part]
    summarize(results, time.time() - start)
    sys.exit(0 if all(r["issue"] == "victoire" for r in results) else 1)


if __name__ == "__main__":
    main()
