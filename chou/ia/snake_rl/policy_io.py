"""Explicit checkpoint dispatch: learned policies never silently fall back."""
from pathlib import Path
import pickle

import torch

from .agent import DQNAgent


def load_policy(path):
    try:
        data = torch.load(Path(path), map_location='cpu', weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, IndexError, pickle.UnpicklingError) as exc:
        raise ValueError(f'Cannot load checkpoint {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError('Checkpoint must contain a dictionary')
    if 'hybrid_version' in data:
        from .hybrid_agent import HybridAgent
        policy = HybridAgent.load(path)
        # Normalize telemetry names without rewriting the immutable checkpoint.
        policy.metadata.setdefault('safety_filter', policy.config.shield)
        policy.metadata.setdefault('training_wall_time', policy.metadata.get('training_wall_seconds'))
        policy.metadata.setdefault('demonstration_cost', {
            'transitions': policy.metadata.get('warmstart_transitions', 0),
            'supervised_updates': policy.metadata.get('warmstart_updates', 0),
            'wall_seconds': policy.metadata.get('warmstart_wall_seconds', 0),
        })
    else:
        policy = DQNAgent.load(path)
        mode = policy.metadata.get('safety_filter')
        if mode:
            if mode not in ('tail', 'tail2', 'collision'):
                raise ValueError(f'Unknown safety filter: {mode}')
            from .shielded_policy import ShieldedPolicy
            policy = ShieldedPolicy(policy, mode)
    policy.online.eval()
    return policy


def policy_kind(policy):
    if isinstance(policy, DQNAgent):
        return 'pure_rl'
    return 'rl_with_safety_filter' if getattr(policy, 'learned', False) else 'algorithmic'
