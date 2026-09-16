#!/usr/bin/env python
"""Professor entry point. No argument: load the delivered RL agent and play."""
from pathlib import Path
import argparse
import os
import sys

BASE_DIR=Path(__file__).resolve().parent
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')


def parser():
    p=argparse.ArgumentParser(description='Snake RL: score first, elapsed time only at equal score.')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--train',action='store_true')
    modes.add_argument('--evaluate',action='store_true')
    modes.add_argument('--benchmark',action='store_true')
    modes.add_argument('--smoke-test',action='store_true')
    p.add_argument('--checkpoint',type=Path,default=BASE_DIR/'checkpoints'/'selected.pt')
    p.add_argument('--run-id',default=None)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--max-steps',type=int,default=1000,help='External evaluation budget; 0 means no cutoff. Not a game timeout.')
    p.add_argument('--baseline',choices=['straight','random','greedy','hamiltonian'])
    p.add_argument('--display',action='store_true')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--transitions',type=int,default=100000)
    p.add_argument('--encoder',choices=['classic11','ordered'],default='classic11')
    p.add_argument('--per',action='store_true')
    p.add_argument('--n-step',type=int,default=1)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--capacity',type=int,default=50000)
    p.add_argument('--warmup',type=int,default=1000)
    p.add_argument('--epsilon-decay',type=int,default=80000)
    p.add_argument('--episode-limit',type=int,default=2000,help='Training rollout interruption, not game rule')
    p.add_argument('--stagnation-limit',type=int,default=500,help='Training rollout interruption, not game rule')
    p.add_argument('--reward-step',type=float,default=-.02)
    p.add_argument('--shaping',type=float,default=.2,help='Potential coefficient used only in learner reward')
    p.add_argument('--update-every',type=int,default=4)
    p.add_argument('--updates-per-step',type=int,default=1)
    p.add_argument('--save-every',type=int,default=50000)
    p.add_argument('--resume',type=Path)
    return p


def main():
    p=parser(); args=p.parse_args()
    if sys.version_info[:2] != (3,13):
        print('This project requires Python 3.13. Activate chou/.venv before running.',file=sys.stderr)
        return 2
    if args.run_id is None:
        import time
        args.run_id=('train' if args.train else 'eval')+'_'+time.strftime('%Y%m%d_%H%M%S')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.','..'):
        p.error('--run-id must be a simple directory name inside chou/runs')
    for name in ['episodes','transitions','batch_size','capacity','warmup','epsilon_decay','episode_limit','stagnation_limit','update_every','updates_per_step','save_every','n_step']:
        if getattr(args,name)<=0:
            p.error(f'--{name.replace("_","-")} must be positive')
    if args.max_steps<0:
        p.error('--max-steps must be nonnegative')
    try:
        import numpy,pygame,torch
    except ImportError as exc:
        print(f'Missing dependency: {exc}. Install explicitly: uv pip install --python chou/.venv/bin/python -r chou/requirements.txt',file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    try:
        if args.train:
            from snake_rl.training import train
            train(args,BASE_DIR)
        elif args.evaluate:
            from snake_rl.evaluation import evaluate
            evaluate(args,BASE_DIR)
        elif args.benchmark:
            from snake_rl.benchmark import benchmark
            benchmark(BASE_DIR)
        else:
            from snake_rl.ui import play
            play(args.checkpoint,BASE_DIR,seed=args.seed,smoke_steps=8 if args.smoke_test else 0)
    except (ValueError,FileNotFoundError,RuntimeError,KeyError,FloatingPointError) as exc:
        print(f'Snake RL: {exc}',file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pygame.quit()
        print('Stopped cleanly.',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
