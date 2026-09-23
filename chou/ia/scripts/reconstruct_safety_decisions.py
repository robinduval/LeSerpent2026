"""Replay recorded observations/actions to audit an already timed episode.

This does not produce an evaluation score or a simulated official time. It
reconstructs historical decisions and requires exact agreement with the raw
trace at every step before reporting filter counters.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from snake_rl.agent import DQNAgent
from snake_rl.env import Env
from snake_rl.shielded_policy import ShieldedPolicy


def reconstruct(seed):
    folder = BASE/"runs/safety_sprint_tail2_validation"
    source = folder/f"trace_{seed}.jsonl"
    checkpoint = BASE/"runs/safety_sprint_tail2/candidate.pt"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if not rows:
        raise ValueError("Empty trace")
    agent = DQNAgent.load(checkpoint)
    policy = ShieldedPolicy(agent, "tail2")
    env = Env(seed=seed, move_reward=0.0)  # historical old evaluation setting
    reconstructed = []
    for row in rows:
        selected = policy.action(env)
        if selected != row["action_executed"]:
            raise AssertionError(f"Policy mismatch at recorded step {row['step']}")
        decision = dict(policy.last)
        _, _, info = env.step(selected)
        if env.steps != row["step"] or env.score != row["score"]:
            raise AssertionError(f"State mismatch at recorded step {row['step']}")
        if info["action_executed"] != row["action_executed"]:
            raise AssertionError("Recorded game action did not match")
        reconstructed.append(dict(kind="offline_log_reconstruction", seed=seed,
            step=env.steps, score=env.score,
            recorded_elapsed_seconds=row["elapsed_seconds"],
            action_proposed_unfiltered=decision["proposed"],
            action_executed=selected, policy_decision=decision))
    output = folder/f"decisions_reconstructed_{seed}.jsonl"
    with output.open("x") as stream:
        stream.write("".join(json.dumps(row)+"\n" for row in reconstructed))
    summary = dict(kind="offline_log_reconstruction", seed=seed,
        steps_verified=len(rows), all_actions_match=True, all_scores_match=True,
        filter_overrides=policy.overrides, learned_choice_states=policy.choice_states,
        final_score_from_recorded_trace=rows[-1]["score"],
        recorded_time_seconds=rows[-1]["elapsed_seconds"],
        terminal_from_reconstructed_recorded_actions=env.terminated,
        source_trace_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (folder/f"reconstruction_summary_{seed}.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="+", type=int)
    args = parser.parse_args()
    for seed in args.seeds:
        reconstruct(seed)
