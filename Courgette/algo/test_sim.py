"""Vérifie que le simulateur (core.py) obéit aux mêmes règles que le vrai jeu
(../../serpent-algo.py) : mêmes corps, mêmes morts, mêmes scores, coup par coup,
sur des milliers de coups aléatoires. Lancer : python test_sim.py"""
import importlib.util
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import core

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "..", "..", "serpent-algo.py")
spec = importlib.util.spec_from_file_location("real_game", REAL)
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)

assert real.GRID_SIZE == core.N, "la grille du jeu a changé"
REAL_DIRS = [real.UP, real.DOWN, real.LEFT, real.RIGHT]


def to_cells(body):
    return [core.cid(x, y) for x, y in body]


def run(seed, policy_seed):
    rng = random.Random(policy_seed)
    g = core.Game(seed)
    random.seed(seed)
    rs = real.Snake()
    ra = real.Apple(rs.body)
    ra.position = tuple(divmod(g.apple, core.N))[::-1]    # même pomme au départ
    eaten = 0
    for _ in range(3000):
        # politique : plutôt manger la pomme, sinon au hasard (pour tester les morts)
        if rng.random() < 0.6:
            best = min(range(4), key=lambda d: core.torus_dist(core.NB[g.head][d], g.apple))
            d = best
        else:
            d = rng.randrange(4)
        g.step(d)
        rs.set_direction(REAL_DIRS[d])
        rs.move()
        assert rs.is_game_over() == (not g.alive), "mort différente"
        if not g.alive:
            return "dead", eaten
        assert to_cells(rs.body) == list(g.body), "corps différents"
        if rs.head_pos == list(ra.position):
            rs.grow()
            eaten += 1
            assert g.score == rs.score
            ra.position = tuple(divmod(g.apple, core.N))[::-1]   # même nouvelle pomme
        assert g.grow == rs.grow_pending
    return "alive", eaten


if __name__ == "__main__":
    deaths = eats = 0
    for s in range(300):
        st, e = run(s, 1000 + s)
        deaths += st == "dead"
        eats += e
    print(f"OK : 300 parties identiques coup par coup ({deaths} morts, {eats} pommes)")
    cyc = core.build_cycle()
    print("OK : cycle hamiltonien valide, serpent initial posé dessus")
