# Point d'entree du rendu : lance le meilleur modele en visualisation pygame.
#
#   python3 main.py                 -> meilleur modele, clock du socle (GAME_SPEED=5)
#   python3 main.py --speed 20      -> 4x plus rapide pour une demo courte
#   python3 main.py --episodes 3    -> 3 parties
#   python3 main.py --model X.pth --torus-food --rich-state  -> un autre modele
#
# Le modele par defaut et ses flags d'etat sont declares ici en un seul endroit :
# un modele DOIT etre joue avec l'etat sur lequel il a ete entraine, sinon on lui
# donne une entree qu'il n'a jamais vue (c'est le piege qui nous a fait croire que
# fix5 restait "timide" alors qu'on regardait fix4 sans --torus-food).
import sys

# meilleur modele au 2026-09-16 22:12 : eval eps=0 50 parties -> mean 88.1, max 119
BEST_MODEL = "model_fix6_s1_best.pth"
BEST_FLAGS = ["--torus-food", "--rich-state"]  # etat 16 bits (Fix 5 + Fix 6)

from play_visual import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    # si l'utilisateur ne precise pas de modele, on injecte le meilleur + ses flags
    if not any(a == "--model" or a.startswith("--model=") for a in argv):
        sys.argv = [sys.argv[0], "--model", BEST_MODEL] + BEST_FLAGS + argv
    main()
