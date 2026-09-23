#!/usr/bin/env python
"""Bounded accelerated DDQN training with explicit heuristic warm-start cost."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from snake_rl.env import Env
from snake_rl.hybrid_agent import HybridAgent, HybridConfig, action_features
from snake_rl.cycle_shield import RewiredGreedyPolicy, OptimizedRewiredGreedyPolicy, ExploredRewiredGreedyPolicy, optimize_free_arc, explore_free_arc, cycle_distance


def validate(agent, seeds):
    rows = []
    for seed in seeds:
        env = Env(seed=seed)
        choices_before = agent.choice_states
        while not env.terminated and env.steps < 40000:
            env.step(agent.select_action(env))
        rows.append({'seed': seed, 'score': env.score, 'steps': env.steps,
                     'completed': env.completed, 'choice_states': agent.choice_states-choices_before,
                     'timing': 'accelerated_training_validation_not_official_5Hz'})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--reward', choices=['time','progress'], default='time')
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'runs'/'hybrid_sprint')
    parser.add_argument('--save-every', type=int, default=25000)
    parser.add_argument('--update-every', type=int, default=8)
    parser.add_argument('--warmstart-steps', type=int, default=6000)
    parser.add_argument('--warmstart-updates', type=int, default=500)
    parser.add_argument('--optimize-free-arc', action='store_true')
    parser.add_argument('--explore-attempts', type=int, choices=[0,64], default=0)
    parser.add_argument('--functional-check', action='store_true', help='Run accelerated invariant checks; not official evaluation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    agent = HybridAgent(HybridConfig(seed=args.seed, optimize_free_arc=args.optimize_free_arc, explore_attempts=args.explore_attempts))
    start = time.perf_counter()
    # Explicit supervised initialization. Labels rank candidate next positions
    # by their remaining cycle distance to the CURRENT apple, never future RNG.
    examples = []
    expert = ExploredRewiredGreedyPolicy() if args.explore_attempts else (OptimizedRewiredGreedyPolicy() if args.optimize_free_arc else RewiredGreedyPolicy())
    env = Env(seed=args.seed)
    for _ in range(args.warmstart_steps):
        if args.explore_attempts and (env.steps == 0 or env.grow_pending):
            explore_free_arc(expert.shield, env, attempts=args.explore_attempts)
        elif args.optimize_free_arc:
            optimize_free_arc(expert.shield, env)
        features, mask = action_features(env, expert.shield)
        examples.extend(features[mask])
        env.step(expert.select_action(env))
        if env.terminated:
            env.reset()
            expert.shield.reset()
    examples = np.asarray(examples, dtype=np.float32)
    for _ in range(args.warmstart_updates):
        idx = agent.rng.integers(len(examples), size=256)
        batch = torch.from_numpy(examples[idx])
        # Heuristic state-action value initialization; this is not claimed as RL.
        labels = -batch[:,3] * 15
        loss = F.mse_loss(agent.online(batch), labels)
        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        agent.optimizer.step()
    agent.target.load_state_dict(agent.online.state_dict())
    agent.refresh_weights()
    warmstart_elapsed = time.perf_counter()-start
    print(json.dumps({'phase': 'warmstart', 'seconds': warmstart_elapsed,
                      'transitions': args.warmstart_steps, 'updates': args.warmstart_updates}), flush=True)
    env = Env(seed=args.seed+10000, move_reward=agent.config.move_reward)
    features, mask = agent.encode(env)
    episodes = []
    total_return = 0.
    last_loss = None
    with (args.output/'episodes.jsonl').open('w') as episode_log:
        for step in range(1, args.steps+1):
            epsilon = max(.015, .2*(1-step/max(1,args.steps*.7)))
            action = agent.select_features(features, mask, epsilon)
            before_distance = agent.shield.distance(env.body[0], env.apple)
            after_distance = agent.shield.next_food_distance(env, action)
            agent.commit(env, action)
            _, terminal, info = env.step(action)
            # Potential shaping computed against the same apple avoids RNG leakage.
            reward = -.05
            if args.reward == 'progress':
                reward += (before_distance-after_distance)/15
            if info['ate_apple']:
                reward += 2.
            if terminal:
                reward += 100. if env.completed else -100.
                next_features = np.zeros_like(features)
                next_mask = np.ones(4, bool)
            else:
                next_features, next_mask = agent.encode(env)
            agent.observe(features, action, reward, next_features, next_mask, terminal)
            total_return += reward
            if step % args.update_every == 0:
                last_loss = agent.train_step()
            features, mask = next_features, next_mask
            if terminal:
                row = {'training_transitions': step, 'score': env.score, 'steps': env.steps,
                       'completed': env.completed, 'rl_return': total_return}
                episodes.append(row)
                episode_log.write(json.dumps(row)+'\n')
                episode_log.flush()
                print(json.dumps(row), flush=True)
                env.reset()
                features, mask = agent.encode(env)
                total_return = 0.
            if step % args.save_every == 0 or step == args.steps:
                metadata = {'algorithm': 'action-conditioned masked Double DQN',
                            'safety': 'programmed Hamiltonian cycle rewiring shield v1',
                            'warmstart': 'supervised current-apple cycle-distance heuristic values',
                            'warmstart_transitions': args.warmstart_steps,
                            'warmstart_updates': args.warmstart_updates,
                            'warmstart_wall_seconds': warmstart_elapsed,
                            'training_seed': args.seed, 'training_wall_seconds': time.perf_counter()-start,
                            'planner_optimize_free_arc': args.optimize_free_arc,
                            'planner_explore_attempts': args.explore_attempts,
                            'reward': ('-0.05 + 2/apple + 100/completion' if args.reward == 'time' else '-0.05 + cycle-distance progress/15 + 2/apple + 100/completion'),
                            'last_loss': last_loss}
                path = args.output/f'candidate_{step:09d}.pt'
                agent.save(path, metadata)
                print(json.dumps({'checkpoint': str(path), 'updates': agent.updates,
                                  'elapsed': time.perf_counter()-start}), flush=True)
        validation = validate(agent, [41001,41002,41003]) if args.functional_check else []
        summary = {'training_transitions': agent.env_steps, 'updates': agent.updates,
                   'wall_seconds': time.perf_counter()-start,
                   'training_completed_episodes': episodes, 'functional_checks': validation}
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
