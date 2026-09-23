"""
replay_best_ratio.py — Trouve, parmi N parties jouées par un modèle, celle
avec le MEILLEUR ratio score/temps sur une seule partie, puis la rejoue en
fenêtre (vitesse officielle du jeu) pour pouvoir la regarder.

Le tirage aléatoire (position des pommes) est fixé par une seed au départ,
donc relancer ce script avec la même seed retrouve exactement la même
partie "record" à chaque fois.

Usage :
    python replay_best_ratio.py --model results_compare/astar-shaped_seed100_ep600_best.pth --games 1000
"""

import argparse
import importlib.util
import os
import random

import numpy as np
import torch

# Charge snake-ia.py comme un module (nom de fichier avec un tiret, donc pas
# importable directement avec `import`).
_spec = importlib.util.spec_from_file_location("snake_ia", os.path.join(os.path.dirname(__file__), "snake-ia.py"))
snake_ia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snake_ia)


def find_best_ratio_game(model_path, n_games, seed):
    """Rejoue exactement la même séquence de parties que snake_ia.evaluate()
    (DQN seul, epsilon=0, headless), mais en gardant l'index de la partie au
    meilleur ratio score/temps, pour pouvoir la rejouer ensuite à l'identique."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = snake_ia.DQNConfig()
    agent = snake_ia.DQNAgent(cfg, device=torch.device("cpu"))
    agent.load(model_path)
    agent.policy_net.eval()

    best_ratio = -1.0
    best_idx = None
    best_info = None

    for i in range(n_games):
        game = snake_ia.SnakeGameAI(render=False, stall_limit_factor=cfg.stall_limit_factor)
        state = snake_ia.get_state(game)
        done = False
        while not done:
            action = agent.act(state, epsilon=0.0)
            reward, done, score = game.play_step(action)
            state = snake_ia.get_state(game)

        elapsed = game.elapsed_time
        ratio = score / elapsed if elapsed > 0 else 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
            best_info = (score, elapsed)

    return best_idx, best_ratio, best_info


def replay_game_at_index(model_path, target_idx, seed):
    """Refait EXACTEMENT la même séquence de random (même seed), rejoue les
    parties une à une sans affichage jusqu'à target_idx, puis rejoue CETTE
    partie-là avec une fenêtre (vitesse officielle du jeu)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = snake_ia.DQNConfig()
    agent = snake_ia.DQNAgent(cfg, device=torch.device("cpu"))
    agent.load(model_path)
    agent.policy_net.eval()

    for i in range(target_idx):
        game = snake_ia.SnakeGameAI(render=False, stall_limit_factor=cfg.stall_limit_factor)
        state = snake_ia.get_state(game)
        done = False
        while not done:
            action = agent.act(state, epsilon=0.0)
            _, done, _ = game.play_step(action)
            state = snake_ia.get_state(game)

    print(f"Partie #{target_idx} (celle du meilleur ratio) : fenêtre en cours d'ouverture...")
    game = snake_ia.SnakeGameAI(render=True, stall_limit_factor=cfg.stall_limit_factor)
    state = snake_ia.get_state(game)
    done = False
    while not done:
        action = agent.act(state, epsilon=0.0)
        _, done, score = game.play_step(action)
        state = snake_ia.get_state(game)
        if game.quit_requested:
            break
    game.pause_on_game_over(seconds=3)
    import pygame
    pygame.quit()
    print(f"Terminé : score={score} temps={game.elapsed_time:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Recherche du meilleur ratio score/temps sur {args.games} parties "
          f"(modèle : {args.model}, seed={args.seed})...")
    idx, ratio, (score, elapsed) = find_best_ratio_game(args.model, args.games, args.seed)
    print(f"Trouvé : partie #{idx} -> score={score:.0f} temps={elapsed:.1f}s ratio={ratio:.4f}")

    replay_game_at_index(args.model, idx, args.seed)


if __name__ == "__main__":
    main()
