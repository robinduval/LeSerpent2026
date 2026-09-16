"""CPU cost measurements on fixed observations, never a scored evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from .agent import DQNAgent, DQNConfig
from .env import Env
from .state import STATE_DIMS, encode


def _elapsed_ms(function, repetitions):
    started = time.perf_counter()
    for _ in range(repetitions):
        function()
    return (time.perf_counter() - started) * 1000 / repetitions


def _hardware():
    memory = None
    if sys.platform == "darwin":
        try:
            memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    elif hasattr(os, "sysconf"):
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": memory, "python": platform.python_version(),
            "python_executable": sys.executable, "torch": str(torch.__version__),
            "numpy": str(np.__version__), "torch_threads": torch.get_num_threads(),
            "device": "cpu", "load_average_at_start": list(os.getloadavg())}


def benchmark(base: Path | None = None, *, inferences: int = 1000, updates: int = 100):
    """Write runs/hardware.json and return costs, not gameplay performance.

    Estimates assume one batch-128 update every four training interactions.
    They exclude logging, checkpoint serialization, apple allocation and resets.
    The 5-Hz evaluation clock remains a separate constraint.
    """
    if min(inferences, updates) < 1:
        raise ValueError("Benchmark repetition counts must be positive")
    base = Path(base) if base else Path(__file__).resolve().parents[1]
    torch.set_num_threads(1)
    result = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "kind": "fixed-state CPU cost benchmark; not score or official time",
              "hardware": _hardware(), "inferences_per_encoder": inferences,
              "updates_per_encoder": updates, "batch_size": 128,
              "assumed_update_every": 4, "models": {}}
    fixed_env = Env(seed=123, move_reward=0.)
    # Each instance executes ONE identical transition. No scored rollout or
    # model evaluation is performed; deep-copy setup is outside the timer.
    isolated = [fixed_env.copy() for _ in range(inferences)]
    started = time.perf_counter()
    for env in isolated:
        env.step(1)
    env_step_ms = (time.perf_counter() - started) * 1000 / inferences
    result["isolated_environment_step_ms"] = env_step_ms
    for encoder in STATE_DIMS:
        agent = DQNAgent(DQNConfig(encoder=encoder, batch_size=128,
                                  capacity=512, warmup=128, seed=123))
        state = encode(fixed_env, encoder)
        for index in range(256):
            # Synthetic training data only times matrix/replay work. Never save
            # these weights or treat them as a trained Snake candidate.
            next_state = state.copy()
            agent.observe(state, index % 4, float(index % 3 - 1), next_state,
                          index % 17 == 0)
        for _ in range(10):
            agent.select_action(state)
        for _ in range(5):
            agent.train_step()
        inference_ms = _elapsed_ms(lambda: agent.select_action(state), inferences)
        encoding_ms = _elapsed_ms(lambda: encode(fixed_env, encoder), inferences)
        update_ms = _elapsed_ms(agent.train_step, updates)
        # The simple estimate intentionally assumes greedy inference every step,
        # although epsilon exploration skips some network calls during training.
        estimated_step_ms = inference_ms + 2 * encoding_ms + env_step_ms + update_ms / 4
        result["models"][encoder] = {
            "state_dim": len(state),
            "parameters": sum(parameter.numel() for parameter in agent.online.parameters()),
            "inference_mean_ms": inference_ms, "encoding_mean_ms": encoding_ms,
            "batch128_update_mean_ms": update_ms,
            "estimated_100k_training_transitions_seconds": estimated_step_ms * 100,
            "estimated_1m_training_transitions_seconds": estimated_step_ms * 1000,
            "evaluation_tick_budget_ms": 200.,
            "finite_last_loss": bool(np.isfinite(agent.last_stats["loss"])),
        }
    result["limitations"] = (
        "Fixed initial state; long bodies and prioritized replay cost more. "
        "Training estimates exclude replay insertion, logging, checkpoints, "
        "resets and apple placement. System load can change timings. "
        "Faster inference is not evidence of faster official game time."
    )
    output = base / "runs" / "hardware.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
