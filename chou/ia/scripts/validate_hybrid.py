#!/usr/bin/env python
"""Accelerated validation only: never reports these wall times as game time."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from snake_rl.env import Env
from snake_rl.hybrid_agent import HybridAgent
from snake_rl.cycle_shield import RewiredCycleShield, RewiredGreedyPolicy

class RandomSafePolicy:
    def __init__(self):
        self.rng = np.random.default_rng(991)
        self.shield = RewiredCycleShield()
    def select_action(self, env):
        if env.steps == 0:
            self.shield.reset()
        action = int(self.rng.choice(self.shield.allowed_actions(env)))
        self.shield.commit(env, action)
        return action

def evaluate(policy, seeds):
    records=[]
    for seed in seeds:
        env=Env(seed=seed)
        choice_start=getattr(policy,'choice_states',0)
        while not env.terminated and env.steps<40000:
            env.step(policy.select_action(env))
        records.append({'seed':seed,'score':env.score,'steps':env.steps,'completed':env.completed,
                        'choice_states':getattr(policy,'choice_states',0)-choice_start})
    return {'records':records,'mean_score':float(np.mean([r['score'] for r in records])),
            'mean_steps':float(np.mean([r['steps'] for r in records])),
            'min_score':min(r['score'] for r in records),
            'completion_count':sum(r['completed'] for r in records)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoints',nargs='+',type=Path)
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--seed-start',type=int,default=43000)
    parser.add_argument('--output',type=Path,default=Path('runs/hybrid_rewired/ablation.json'))
    args=parser.parse_args()
    seeds=range(args.seed_start,args.seed_start+args.seeds)
    results={'timing':'accelerated_diagnostics_not_official_5Hz','seed_start':args.seed_start,'seeds':args.seeds,'models':{}}
    for path in args.checkpoints:
        agent=HybridAgent.load(path)
        row=evaluate(agent,seeds)
        row.update(training_transitions=agent.env_steps,training_updates=agent.updates)
        results['models'][str(path)]=row
        print(json.dumps({'model':str(path),**{k:v for k,v in row.items() if k!='records'}}),flush=True)
    for name,policy in [('programmed_greedy',RewiredGreedyPolicy()),('uniform_safe_random',RandomSafePolicy()),('untrained_network',HybridAgent())]:
        row=evaluate(policy,seeds)
        results['models'][name]=row
        print(json.dumps({'model':name,**{k:v for k,v in row.items() if k!='records'}}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':
    main()
