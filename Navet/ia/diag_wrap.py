# Diagnostic : le modele exploite-t-il le tore ?
# IMPORTANT : passer --torus-food si le modele a ete entraine avec (sinon on lui
# donne une entree qu'il n'a jamais vue et la mesure ne veut rien dire).
# 1) frequence des wraps reels  2) % de pommes ou le chemin court passe par le mur
#    et ou l'etat 11 bits pointe dans la mauvaise direction
import argparse, torch
from game import SnakeGame, GRID_SIZE
from agent import Agent

ap = argparse.ArgumentParser(description="Le modele exploite-t-il le tore ?")
ap.add_argument("--model", default="model_fix4_best.pth")
ap.add_argument("--torus-food", action="store_true", help="doit matcher l'entrainement du modele")
ap.add_argument("--rich-state", action="store_true", help="doit matcher l'entrainement du modele")
ap.add_argument("--episodes", type=int, default=30)
args = ap.parse_args()

agent = Agent(torus_food=args.torus_food, rich_state=args.rich_state)
agent.model.load_state_dict(torch.load(args.model, map_location="cpu"))
agent.model.eval()

wraps = 0; steps_tot = 0; apples = 0
short_via_wrap = 0   # pommes dont le chemin court passe par un mur
state_lies = 0       # ... et dont l'etat 11 bits indique la mauvaise direction
for ep in range(1, args.episodes + 1):
    g = SnakeGame(seed=999+ep, step_reward=-0.01, hunger_mode="truncated")
    prev = g.head[:]
    while not g.done:
        # analyse de la pomme courante (une fois par pomme)
        ax, ay = g.apple; hx, hy = g.head
        dx = (ax-hx) % GRID_SIZE; dy = (ay-hy) % GRID_SIZE
        # distance torique vs distance directe, par axe
        wrap_x = min(dx, GRID_SIZE-dx) < abs(ax-hx)
        wrap_y = min(dy, GRID_SIZE-dy) < abs(ay-hy)
        if wrap_x or wrap_y:
            short_via_wrap += 1
            # L'etat REELLEMENT donne au reseau ment-il ? Depend de --torus-food :
            # sans le flag, les bits viennent des coordonnees brutes et pointent a
            # l'oppose ici par construction ; avec le flag ils sont corrects.
            if not args.torus_food:
                state_lies += 1
        s = agent.get_state(g)
        with torch.no_grad():
            m = int(torch.argmax(agent.model(torch.tensor(s, dtype=torch.float))).item())
        mv=[0,0,0]; mv[m]=1
        _,_,_,info = g.play_step(mv)
        steps_tot += 1
        # wrap reel : saut de plus d'une case en coordonnees brutes
        if abs(g.head[0]-prev[0]) > 1 or abs(g.head[1]-prev[1]) > 1:
            wraps += 1
        prev = g.head[:]
        if info.get("cause")=="pomme": apples += 1

print(f"{args.episodes} parties : {steps_tot} pas, {apples} pommes")
print(f"wraps reels : {wraps}  ({100*wraps/steps_tot:.1f}% des pas)")
print(f"situations ou le chemin court passe par un mur : {short_via_wrap} ({100*short_via_wrap/steps_tot:.1f}% des pas)")
print(f"  ... dont l'etat 11 bits REELLEMENT vu par ce modele pointe a l'oppose : "
      f"{state_lies} ({100*state_lies/max(1,short_via_wrap):.1f}% de ces cas)")
print(f"pas par pomme : {steps_tot/max(1,apples):.2f}")
