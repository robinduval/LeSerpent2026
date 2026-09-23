"""Vérifie le cell-tree : (1) tout arbre couvrant donne un cycle hamiltonien de 225
cases ; (2) pendant de vraies parties, après CHAQUE coup, le cycle reconstruit est
hamiltonien et contient toutes les arêtes du corps (=> victoire garantie).
Usage : python test_celltree.py [nb_parties]"""
import random
import sys

import celltree as ct
import core


def random_tree(rng):
    parent = list(range(ct.NC))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = list(range(ct.NE))
    rng.shuffle(edges)
    t = bytearray(ct.NE)
    for e in edges:
        a, b = ct.EDGE_CELLS[e]
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            t[e] = 1
    assert sum(t) == ct.NC - 1
    return t


def check_state(agent, g):
    order = ct.cycle_order(agent.tree, g.head)          # hamiltonien ?
    body = list(g.body)
    for a, b in zip(body, body[1:]):
        assert ct.present(agent.tree, a, b), f"arête du corps {a}-{b} absente du cycle"


if __name__ == "__main__":
    rng = random.Random(1)
    for _ in range(300):
        ct.cycle_order(random_tree(rng), 0)
    ct.cycle_order(ct.comb_tree(), 0)
    print("OK : 300 arbres aléatoires -> cycles hamiltoniens de 225 cases")
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for seed in range(games):
        g = core.Game(seed)
        a = ct.CellTree()
        a.reset(g)
        check_state(a, g)
        while not g.over:
            g.step(a.choose(g))
            if g.alive:
                check_state(a, g)
        print(f"OK : partie {seed} : {'victoire' if g.won else 'PERDU'} score {g.score} en {g.moves} coups, invariant vérifié à chaque coup")
        assert g.won
