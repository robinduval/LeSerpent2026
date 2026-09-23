"""Real-clock evaluation. No accelerated evaluation option is provided."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import pygame
import torch

from .env import Env, GAME_SPEED
from .agent import DQNAgent
from .state import encode
from .metrics import summarize
from .training import append_json
from .policy_io import load_policy, policy_kind


def checkpoint_id(path):
    return Path(path).stem + ':' + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def action_for(policy, env):
    if isinstance(policy, DQNAgent):
        return policy.select_action(encode(env, policy.config.encoder), explore=False)
    return policy.select_action(env)


def play_episode(policy, seed, model_id, max_steps=1000, render=None, trace_path=None):
    env = Env(seed=seed, move_reward=getattr(getattr(policy, 'config', None), 'move_reward', 0.0))
    if hasattr(policy, 'reset_episode'):
        policy.reset_episode()
    initial_overrides = getattr(policy, 'overrides', 0)
    initial_choices = getattr(policy, 'choice_states', 0)
    clock = pygame.time.Clock()
    start_time = time.time()
    start_perf = time.perf_counter()
    latencies = []
    rl_return = 0.0
    reason = None
    timing = []
    elapsed = 0.0
    while True:
        if pygame.display.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    reason = 'user_quit'
            if reason:
                elapsed = time.time()-start_time
                break
        began = time.perf_counter()
        action = action_for(policy,env)
        latencies.append(time.perf_counter()-began)
        tick_time = time.perf_counter()-start_perf
        reward, terminal, info = env.step(action)
        timing.append(tick_time)
        rl_return += reward
        elapsed = time.time()-start_time
        if trace_path:
            append_json(trace_path,dict(seed=seed,step=env.steps,action_proposed=action,
                action_executed=info['action_executed'],reward=reward,score=env.score,
                elapsed_seconds=elapsed, policy_decision=getattr(policy, 'last_decision', None)))
        if render:
            render(env,start_time,False)
        if terminal:
            reason = env.termination_reason
            break
        if max_steps and env.steps >= max_steps:
            reason = 'external_step_limit'
            break
        clock.tick(GAME_SPEED)
    metadata=getattr(policy, 'metadata', {})
    return dict(model_id=model_id,configuration_id=metadata.get('run_id',model_id),seed=seed,
                official_score=env.score,official_time_seconds=elapsed,
                time_convention='wall time at terminal event or external cutoff; source display never freezes',
                rl_return=rl_return,apples=env.score,steps=env.steps,completed=env.completed,
                termination_reason=reason,final_length=len(env.body),
                decision_latency=statistics.mean(latencies) if latencies else 0,
                decision_latency_p95=float(np.percentile(latencies,95)) if latencies else 0,
                evaluation_step_budget=max_steps,clock_hz=GAME_SPEED,
                observed_move_hz=(len(timing)-1)/(timing[-1]-timing[0]) if len(timing)>1 else None,
                training_transitions=getattr(policy, 'env_steps', 0),
                training_updates=getattr(policy, 'updates', 0),
                training_wall_time=metadata.get('training_wall_time'),
                demonstration_cost=metadata.get('demonstration_cost', 0), simulation_cost=metadata.get('simulation_cost', 0),
                policy_kind=policy_kind(policy), safety_filter=metadata.get('safety_filter'),
                filter_overrides=getattr(policy, 'overrides', 0)-initial_overrides,
                learned_choice_states=getattr(policy, 'choice_states', 0)-initial_choices,
                classability='not established for externally interrupted games' if reason in ('external_step_limit','user_quit') else 'terminal game; official multi-trial aggregation unknown')


def evaluate(args,base):
    torch.set_num_threads(1)
    out = base/'runs'/args.run_id
    out.mkdir(parents=True,exist_ok=True)
    path = out/'evaluation.jsonl'
    if path.exists():
        raise ValueError(f'Evaluation already exists: {path}; choose a new --run-id')
    (out/'config.json').write_text(json.dumps(dict(vars(args)),indent=2,default=str)+'\n')
    if args.baseline:
        from .baselines import make_baseline
        policy = make_baseline(args.baseline, seed=args.seed)
        model_id = 'algorithmic_'+args.baseline
    else:
        policy = load_policy(args.checkpoint)
        model_id = checkpoint_id(args.checkpoint)
    render = None
    if args.display:
        from .ui import Display
        display = Display(model_id)
        render = display.draw
    results = []
    try:
        for seed in range(args.seed,args.seed+args.episodes):
            if args.baseline:
                policy = make_baseline(args.baseline, seed=seed)
            row = play_episode(policy,seed,model_id,args.max_steps,render,
                               out/'actions.jsonl' if args.trace else None)
            append_json(path,row)
            results.append(row)
            print(json.dumps(row),flush=True)
            if row['termination_reason'] == 'user_quit':
                break
    finally:
        pygame.quit()
    summary = summarize(results)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if results:
        with (out/'evaluation.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
    print(json.dumps(summary,indent=2),flush=True)
    return results
