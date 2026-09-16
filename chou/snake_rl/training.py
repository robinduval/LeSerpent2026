"""Bounded accelerated RL, authorized by the user's September 16 clarification."""
from __future__ import annotations

import json
from pathlib import Path
import time
import random
import numpy as np
import torch

from .env import Env, GRID_SIZE
from .agent import DQNAgent, DQNConfig
from .state import encode


def potential(env, coefficient):
    if env.terminated or env.apple is None:
        return 0.0
    x, y = env.body[0]
    ax, ay = env.apple
    dx, dy = abs(x - ax), abs(y - ay)
    return -coefficient * (min(dx, GRID_SIZE-dx) + min(dy, GRID_SIZE-dy))


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def train(args, base):
    torch.set_num_threads(1)
    run = base / 'runs' / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    if (run/'config.json').exists() and not args.resume:
        raise ValueError(f'Run already exists: {run}; choose another --run-id or --resume')
    if args.resume:
        agent = DQNAgent.load(args.resume, require_resume=True)
        agent.end_episode()
        previous_shaping=agent.metadata.get('shaping_coefficient',args.shaping)
        if args.shaping != previous_shaping:
            raise ValueError('Resume requires the same --shaping coefficient as its checkpoint')
    else:
        agent = DQNAgent(DQNConfig(encoder=args.encoder, seed=args.seed,
            gamma=args.gamma, lr=args.lr, batch_size=args.batch_size, capacity=args.capacity,
            warmup=args.warmup, epsilon_decay_steps=args.epsilon_decay,
            n_step=args.n_step, prioritized=args.per, move_reward=args.reward_step))
    cfg = vars(agent.config)
    settings = dict(vars(args))
    settings.update(agent_config=cfg, started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    training_accelerated=True, evaluation_hz=5)
    (run/'config.json').write_text(json.dumps(settings, indent=2, default=str)+'\n')
    env = Env(seed=args.seed, move_reward=agent.config.move_reward)
    start_steps = agent.env_steps
    end_steps = start_steps + args.transitions
    episode = 0
    start = time.perf_counter()
    previous_wall_time=agent.metadata.get('training_wall_time',0.0)
    episode_started = start
    episode_seed = agent.metadata.get('next_episode_seed', args.seed * 1_000_000)
    env.reset(seed=episode_seed)
    episode_return = learned_return = 0.0
    last_apple = 0
    latency_sum=0.0
    reverse_requests=0
    metrics = {}
    # Resume at an explicit episode boundary. Replay/optimizer/RNG are restored;
    # this is statistical continuation, not a claim of bitwise whole-run replay.
    def save(name, replay=False):
        path = run/name
        agent.save(path, metadata={
            'run_id': args.run_id, 'training_wall_time': previous_wall_time+time.perf_counter()-start,
            'session_training_wall_time': time.perf_counter()-start,
            'training_transitions': agent.env_steps, 'training_updates': agent.updates,
            'shaping_coefficient': args.shaping, 'demonstration_cost': 0,
            'simulation_cost': 0, 'qualification': 'not yet evaluated',
            'next_episode_seed': episode_seed+1,
            'resume_boundary': 'new episode; model, optimizer, replay and learner RNG restored',
        }, include_replay=replay)
        return path
    try:
        while agent.env_steps < end_steps:
            state = encode(env, agent.config.encoder)
            old_phi = potential(env, args.shaping)
            began=time.perf_counter()
            action = agent.select_action(state, explore=True)
            latency_sum+=time.perf_counter()-began
            before = env.score
            reward, terminal, info = env.step(action)
            reverse_requests+=action!=info['action_executed']
            if env.score != before:
                last_apple = env.steps
            truncated = not terminal and (env.steps >= args.episode_limit or
                         env.steps-last_apple >= args.stagnation_limit or agent.env_steps+1 >= end_steps)
            learner_reward = reward + agent.config.gamma * potential(env, args.shaping) - old_phi
            next_state = encode(env, agent.config.encoder)
            agent.observe(state, action, learner_reward, next_state, terminal, truncated=truncated)
            if agent.env_steps % args.update_every == 0:
                for _ in range(args.updates_per_step):
                    metrics = agent.train_step() or metrics
            episode_return += reward
            learned_return += learner_reward
            if terminal or truncated:
                now = time.perf_counter()
                append_json(run/'episodes.jsonl', {
                    'model_id': args.run_id, 'configuration_id': args.run_id,
                    'seed': episode_seed, 'official_score': env.score,
                    'official_time_seconds': None, 'training_episode_wall_seconds': now-episode_started,
                    'rl_return': episode_return, 'learning_return': learned_return,
                    'apples': env.score, 'steps': env.steps, 'completed': env.completed,
                    'termination_reason': env.termination_reason if terminal else 'training_interruption',
                    'final_length': len(env.body), 'decision_latency': latency_sum/env.steps,
                    'reverse_requests': reverse_requests,
                    'training_transitions': agent.env_steps, 'training_updates': agent.updates,
                    'training_wall_time': previous_wall_time+now-start, 'session_training_wall_time': now-start,
                    'demonstration_cost': 0, 'simulation_cost': 0,
                })
                episode += 1
                episode_seed += 1
                env.reset(seed=episode_seed)
                episode_return = learned_return = 0.0
                episode_started = now
                last_apple = 0
                latency_sum=0.0
                reverse_requests=0
            if agent.env_steps % 1000 == 0:
                row = {'transitions': agent.env_steps, 'updates': agent.updates, 'episodes': episode,
                       'epsilon': agent.epsilon, 'wall_seconds': time.perf_counter()-start, **metrics}
                append_json(run/'learning.jsonl', row)
                print(json.dumps(row), flush=True)
            if agent.env_steps % args.save_every == 0:
                save(f'candidate_{agent.env_steps:09d}.pt')
    except KeyboardInterrupt:
        print('Training interrupted: saving a resumable checkpoint.', flush=True)
    finally:
        agent.end_episode()
        save('resume.pt', replay=True)
        final = save(f'candidate_{agent.env_steps:09d}.pt')
        print(f'Saved {final}; not selected until equal-budget 5 Hz validation.', flush=True)
    return final
