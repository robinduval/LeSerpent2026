"""Independent 5 Hz qualification of one frozen experimental hybrid."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from snake_rl.agent import DQNAgent
from snake_rl.evaluation import play_episode
from snake_rl.shielded_policy import ShieldedPolicy


def run(task):
    seed, checkpoint, destination, max_steps = task
    agent = DQNAgent.load(checkpoint)
    agent.online.eval()
    policy = ShieldedPolicy(agent, "tail2")
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    row = play_episode(policy, seed, "tail2:" + digest[:12], max_steps=max_steps,
                       trace_path=Path(destination)/f"trace_{seed}.jsonl")
    row.update(policy="learned DQN plus local tail/2step safety filter",
               training_transitions=agent.env_steps, training_updates=agent.updates,
               overrides=policy.overrides, decisions=policy.decisions,
               checkpoint_sha256=digest)
    (Path(destination)/f"result_{seed}.json").write_text(json.dumps(row, indent=2)+"\n")
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[880001,880002,880003])
    parser.add_argument("--max-steps", type=int, default=4500)
    args = parser.parse_args()
    checkpoint = BASE/"runs/safety_sprint_tail2/candidate.pt"
    destination = BASE/"runs/safety_sprint_tail2_validation"
    destination.mkdir(exist_ok=False)
    (destination/"config.json").write_text(json.dumps(dict(
        seeds=args.seeds, max_steps=args.max_steps, clock_hz=5,
        purpose="frozen checkpoint hybrid validation; no training during evaluation",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")), indent=2)+"\n")
    with ProcessPoolExecutor(max_workers=len(args.seeds)) as pool:
        results = list(pool.map(run, [(s, checkpoint, destination, args.max_steps)
                                    for s in args.seeds]))
    (destination/"evaluation.json").write_text(json.dumps(results, indent=2)+"\n")
