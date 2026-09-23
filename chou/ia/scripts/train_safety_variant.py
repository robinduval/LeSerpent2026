"""Accelerated, bounded online RL experiment with an explicit local filter."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from snake_rl.agent import DQNAgent
from snake_rl.env import Env
from snake_rl.shielded_policy import ShieldedPolicy
from snake_rl.state import encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tail", "tail2", "collision"], default="tail")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    source = BASE / "checkpoints/selected_v1_pure.pt"
    agent = DQNAgent.load(source)
    shield = ShieldedPolicy(agent, args.mode)
    run_id = args.run_id or ("safety_sprint_" + args.mode)
    if Path(run_id).name != run_id:
        parser.error("--run-id must be a single directory name")
    out = BASE / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    steps = updates = 0
    with (out / "training.jsonl").open("w") as stream:
        for episode in range(args.episodes):
            env = Env(seed=args.seed + episode, move_reward=agent.config.move_reward)
            while not env.terminated and env.steps < args.max_steps:
                state = encode(env, agent.config.encoder)
                action = shield.action(env)
                reward, done, _ = env.step(action)
                agent.observe(state, action, reward, encode(env, agent.config.encoder),
                              done, truncated=env.steps == args.max_steps)
                steps += 1
                if steps % 16 == 0:
                    updates += agent.train_step() is not None
                if time.monotonic() - started > args.seconds:
                    break
            agent.end_episode()
            row = dict(kind="accelerated_training_episode", seed=args.seed + episode,
                       score=env.score, steps=env.steps, completed=env.completed,
                       terminated=env.terminated, training_transitions=steps,
                       training_updates=updates, wall_seconds=time.monotonic()-started,
                       overrides=shield.overrides, decisions=shield.decisions)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)
            if time.monotonic() - started > args.seconds:
                break
    agent.save(out / "candidate.pt", metadata={"qualification":"unqualified hybrid online training",
               "run_id":run_id, "training_transitions":agent.env_steps,
               "training_updates":agent.updates,
               "training_wall_time":agent.metadata.get("training_wall_time",0)+time.monotonic()-started,
               "session_training_wall_time":time.monotonic()-started,
               "source_checkpoint":str(source.relative_to(BASE)),
               "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
               "shaping_coefficient":0.0,
               "safety_filter":args.mode, "training_transitions_sprint":steps,
               "training_updates_sprint":updates}, include_replay=False)


if __name__ == "__main__":
    main()
