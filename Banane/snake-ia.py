#!/usr/bin/env python3
"""Livrable Snake RL : démonstration autonome des modules snake_rl existants."""

import argparse
import json
from pathlib import Path
import time

import torch

from snake_rl import rules
from snake_rl.agent import Agent
from snake_rl.config import Config
from snake_rl.game import SnakeGame
from snake_rl.state import build_state

BASE = Path(__file__).resolve().parent


def load_agent(checkpoint):
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute() and not path.exists():
            path = BASE / path
        candidates = [path]
    else:
        candidates = list((BASE / 'runs').glob('dqn_long_live*/best_mean.pt'))
    available = []
    for path in candidates:
        try:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload.get('ruleset') != rules.RULESET:
                raise ValueError('checkpoint legacy : règles toriques non attestées')
            available.append((float(payload.get('eval_mean_score', 0)), path, payload))
        except (OSError, ValueError, RuntimeError, EOFError) as error:
            if checkpoint:
                raise ValueError(f'Checkpoint inutilisable : {path}: {error}') from error
    if not available:
        raise ValueError('Aucun best_mean torique de run longue disponible. Utiliser --checkpoint CHEMIN.')
    mean, path, payload = max(available, key=lambda item: item[0])
    config = Config.from_dict(payload['config'])
    agent = Agent(config, device=torch.device('cpu'))
    agent.load_state_dict(payload, load_optimizer=False)
    agent.policy_net.eval().requires_grad_(False)
    agent.target_net.eval().requires_grad_(False)
    print(f'MODEL LOADED | {path.resolve()} | training episode {payload.get("episode")} '
          f'| eval mean {mean} | epsilon 0 | no learning', flush=True)
    return agent


def action(agent, game):
    with torch.inference_mode():
        choice = agent.act(build_state(game), mask=game.legal_action_mask(), greedy=True)
    agent.policy_net.eval()
    return choice


def demonstrate(agent, seed, speed):
    import pygame
    from snake_rl.render import draw_board

    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((rules.SCREEN_WIDTH, rules.SCREEN_HEIGHT))
        pygame.display.set_caption('Snake IA — Reinforcement Learning — Banane')
        font = pygame.font.Font(None, 22)
        clock = pygame.time.Clock()
        game = SnakeGame(seed=seed)  # Fins officielles uniquement, pas de time limit.
        started = time.monotonic()
        finished_time = None
        announced = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                if game.done and event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_r):
                    game = SnakeGame(seed=seed)
                    started = time.monotonic()
                    finished_time = None
            if not game.done:
                result = game.step(action(agent, game))
                if result.ate:
                    print(f'APPLE | score {game.score} | steps {game.steps}', flush=True)
                if result.done:
                    finished_time = time.monotonic() - started
                    print(f'GAME END | score {game.score} | steps {game.steps} | {result.info}', flush=True)
            elapsed = finished_time if finished_time is not None else time.monotonic() - started
            official = game.steps / rules.GAME_SPEED
            ratio = game.score / official if official else 0
            draw_board(screen, game.snapshot(), font, [
                f'Score : {game.score}   Temps : {elapsed:.1f} s',
                f'Temps officiel steps/5 : {official:.1f} s   Score/s : {ratio:.3f}',
                f'IA autonome / epsilon 0 / {speed} FPS'
                + (' OFFICIEL' if speed == rules.GAME_SPEED else ' EXPERIMENTAL'),
                'FIN : Espace ou R pour rejouer / Echap quitter' if game.done else 'Jeu torique 15x15 / aucun apprentissage / Echap quitter',
            ])
            pygame.display.flip()
            if not announced:
                print(f'DEMO WINDOW OPEN | driver {pygame.display.get_driver()} | FPS {speed} '
                      f'| GAME_SPEED {rules.GAME_SPEED} | GRID_SIZE {rules.GRID_SIZE}', flush=True)
                announced = True
            clock.tick(speed)
    finally:
        pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='checkpoint torique, sinon meilleur best_mean de run longue')
    parser.add_argument('--speed', type=int, default=rules.GAME_SPEED, help='FPS visuels ; 5 = mode officiel')
    parser.add_argument('--seed', type=int, help='seed des pommes, reproductible à politique figée')
    parser.add_argument('--headless', action='store_true', help='une partie sans affichage ni apprentissage')
    args = parser.parse_args(argv)
    if args.speed < 1:
        parser.error('--speed doit être >= 1')
    torch.set_num_threads(1)  # Ne pas monopoliser le CPU de la run en cours.
    try:
        agent = load_agent(args.checkpoint)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    before = agent.learn_steps, agent.episodes_done, len(agent.memory)
    try:
        if args.headless:
            game = SnakeGame(seed=args.seed)
            while not game.done:
                game.step(action(agent, game))
            seconds = game.steps / rules.GAME_SPEED
            print(json.dumps({'score': game.score, 'steps': game.steps,
                              'official_time_seconds': seconds,
                              'score_per_second': game.score / seconds if seconds else 0,
                              'terminated': game.terminated, 'truncated': game.truncated}))
        else:
            demonstrate(agent, args.seed, args.speed)
    finally:
        assert before == (agent.learn_steps, agent.episodes_done, len(agent.memory))
        print('NO LEARNING VERIFIED | counters unchanged | replay buffer empty', flush=True)


if __name__ == '__main__':
    main()
