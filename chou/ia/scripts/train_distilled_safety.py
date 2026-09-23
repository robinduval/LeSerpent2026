"""Bounded RL plus large-margin demonstration learning; pure-network export.

The demonstrator is the frozen learned Q network plus the local tail2 filter.
This is a DQfD-inspired experiment, not a reproduction of that publication.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from snake_rl.agent import DQNAgent
from snake_rl.env import Env
from snake_rl.shielded_policy import ShieldedPolicy
from snake_rl.state import encode


def main():
    output = BASE / "runs/distillation_sprint"
    output.mkdir(exist_ok=False)
    teacher_agent = DQNAgent.load(BASE/"runs/safety_sprint_tail2/candidate.pt")
    teacher = ShieldedPolicy(teacher_agent, "tail2")
    student = DQNAgent.load(BASE/"checkpoints/selected_v1_pure.pt")
    student.metadata.pop("safety_filter", None)
    student.config.lr = .0002
    for group in student.optimizer.param_groups:
        group["lr"] = student.config.lr
    rng = np.random.default_rng(14917)
    states = np.empty((60000, student.state_dim), dtype=np.float32)
    labels = np.empty(60000, dtype=np.int64)
    count = transitions = updates = imitation_updates = 0
    started = time.monotonic()
    rows = []
    while time.monotonic() - started < 75 and transitions < 200000:
        seed = 930000 + len(rows)
        env = Env(seed=seed, move_reward=student.config.move_reward)
        while not env.terminated and env.steps < 10000:
            state = encode(env, student.config.encoder)
            action = teacher.action(env)
            states[count % len(states)] = state
            labels[count % len(states)] = action
            count += 1
            # DAgger-like mixture: expose teacher to occasional student choices.
            # Training may collide; terminal rules remain untouched.
            executed = student.select_action(state) if rng.random() < .15 else action
            reward, done, _ = env.step(executed)
            student.observe(state, executed, reward, encode(env, student.config.encoder),
                            done, truncated=env.steps == 10000)
            transitions += 1
            if transitions % 16 == 0:
                updates += student.train_step() is not None
            if count >= 128 and transitions % 4 == 0:
                chosen = rng.integers(min(count, len(states)), size=128)
                observations = torch.from_numpy(states[chosen])
                target_actions = torch.from_numpy(labels[chosen])
                q = student.online(observations)
                margin = torch.full_like(q, .8)
                margin.scatter_(1, target_actions[:, None], 0.)
                loss = ((q + margin).max(dim=1).values
                        - q.gather(1, target_actions[:, None]).squeeze(1)).mean()
                student.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.online.parameters(), 10.)
                student.optimizer.step()
                imitation_updates += 1
            if transitions % 1000 == 0 and time.monotonic() - started > 75:
                break
        student.end_episode()
        row = dict(kind="accelerated_training", seed=seed, score=env.score,
                   steps=env.steps, completed=env.completed, terminated=env.terminated,
                   transitions=transitions, rl_updates=updates,
                   imitation_updates=imitation_updates, wall_seconds=time.monotonic()-started)
        rows.append(row)
        with (output/"training.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        if len(rows) % 20 == 0:
            print(json.dumps(row), flush=True)
    student.target.load_state_dict(student.online.state_dict())
    student.save(output/"candidate.pt", metadata=dict(
        qualification="pure RL network, unqualified after training with safety demonstrations",
        run_id="distillation_sprint", demonstration_cost=count,
        demonstration_policy="frozen tail2 hybrid, training only",
        imitation_updates=imitation_updates, sprint_rl_updates=updates,
        sprint_transitions=transitions, sprint_wall_seconds=time.monotonic()-started,
        shaping_coefficient=0.0), include_replay=False)
    print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
