"""Compact Double DQN: four absolute actions, no planning or safety override."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import operator
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .replay import ReplayBuffer
from .state import ENCODER_VERSION, STATE_DIMS

CHECKPOINT_VERSION = 1
RULE_SIGNATURE = "snake15-torus-absolute4-reverse-ignored-delayed-growth-v1"


@dataclass
class DQNConfig:
    encoder: str = "classic11"
    hidden: int = 128
    gamma: float = 0.95
    lr: float = 0.001
    batch_size: int = 64
    capacity: int = 20000
    warmup: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 80000
    target_update: int = 200
    gradient_clip: float = 10.0
    n_step: int = 1
    prioritized: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 200000
    priority_epsilon: float = 1e-5
    move_reward: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.encoder not in STATE_DIMS:
            raise ValueError(f"Unknown encoder {self.encoder!r}")
        for name in ("hidden", "batch_size", "capacity", "warmup", "epsilon_decay_steps",
                     "target_update", "n_step", "per_beta_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > self.capacity or self.warmup > self.capacity:
            raise ValueError("Replay capacity must cover batch size and warmup")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Invalid discount or exploration range")
        if not 0 <= self.per_alpha <= 1 or not 0 <= self.per_beta_start <= 1:
            raise ValueError("Invalid PER parameters")
        for name in ("lr", "gradient_clip", "priority_epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.move_reward):
            raise ValueError("move_reward must be finite")


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, state):
        return self.layers(state)


@torch.no_grad()
def double_dqn_targets(online, target, next_states, rewards, discounts, terminated):
    """Online network selects; independent target network evaluates; no gradients.

    discounts is gamma**k for each replay transition. Only genuine termination
    suppresses the bootstrap; a time-limit truncation keeps it.
    """
    actions = online(next_states).argmax(dim=1, keepdim=True)
    next_values = target(next_states).gather(1, actions).squeeze(1)
    return rewards + discounts * (~terminated.bool()).float() * next_values


class DQNAgent:
    def __init__(self, config: DQNConfig | None = None):
        self.config = config or DQNConfig()
        self.state_dim = STATE_DIMS[self.config.encoder]
        # Small batches on this model are faster and reproducible on one CPU thread.
        torch.set_num_threads(1)
        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.online = QNetwork(self.state_dim, self.config.hidden).cpu()
        self.target = QNetwork(self.state_dim, self.config.hidden).cpu()
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.replay = ReplayBuffer(self.config.capacity, self.state_dim,
                                   prioritized=self.config.prioritized,
                                   alpha=self.config.per_alpha,
                                   priority_epsilon=self.config.priority_epsilon)
        self.pending = deque()
        self.env_steps, self.updates = 0, 0
        self.metadata = {}
        self.last_stats = None

    @property
    def epsilon(self):
        fraction = min(1., self.env_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    @property
    def training_transitions(self):
        return self.env_steps

    @property
    def training_updates(self):
        return self.updates

    def _state(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (self.state_dim,) or not np.isfinite(value).all():
            raise ValueError(f"Expected finite state of shape ({self.state_dim},)")
        return value

    @torch.no_grad()
    def select_action(self, state, explore: bool = False) -> int:
        state = self._state(state)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        values = self.online(torch.from_numpy(state).unsqueeze(0))
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite action values")
        return int(values.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, terminated, truncated=False):
        state, next_state = self._state(state).copy(), self._state(next_state).copy()
        try:
            if isinstance(action, bool):
                raise TypeError("boolean action")
            action = operator.index(action)
        except TypeError as error:
            raise ValueError("Action must be an integer in [0, 3]") from error
        if not 0 <= action < 4:
            raise ValueError("Action must be an integer in [0, 3]")
        if not math.isfinite(float(reward)):
            raise ValueError("Reward must be finite")
        self.pending.append((state, int(action), float(reward), next_state,
                             bool(terminated), bool(truncated)))
        self.env_steps += 1
        if len(self.pending) >= self.config.n_step:
            self._emit_pending()
        if terminated or truncated:
            # Flush shorter suffixes: never borrow transitions from the next episode.
            while self.pending:
                self._emit_pending()

    def end_episode(self):
        """Flush an external interruption before resetting the environment.

        Stored next observations remain valid bootstrap states. This adds no
        transition and does not label an interruption as an actual game loss.
        Useful when KeyboardInterrupt arrives between environment steps.
        """
        while self.pending:
            self._emit_pending()

    def _emit_pending(self):
        total_reward, discount = 0., 1.
        end = None
        for transition in list(self.pending)[:self.config.n_step]:
            total_reward += discount * transition[2]
            discount *= self.config.gamma
            end = transition
            if transition[4] or transition[5]:
                break
        start = self.pending.popleft()
        self.replay.add(start[0], start[1], total_reward, end[3], end[4], discount)

    def train_step(self):
        if len(self.replay) < max(self.config.batch_size, self.config.warmup):
            return None
        beta = self.config.per_beta_start + (1 - self.config.per_beta_start) * min(
            1., self.env_steps / self.config.per_beta_steps)
        batch = self.replay.sample(self.config.batch_size, self.rng, beta)
        predicted = self.online(batch["states"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        targets = double_dqn_targets(self.online, self.target, batch["next_states"],
                                     batch["rewards"], batch["discounts"], batch["terminated"])
        errors = targets - predicted
        loss = (F.smooth_l1_loss(predicted, targets, reduction="none") * batch["weights"]).mean()
        if not torch.isfinite(loss) or not torch.isfinite(targets).all():
            raise FloatingPointError("Non-finite DDQN targets or loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip,
                                             error_if_nonfinite=True)
        self.optimizer.step()
        if self.config.prioritized:
            self.replay.update_priorities(batch["indices"], errors.detach().numpy())
        self.updates += 1
        if self.updates % self.config.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_stats = {"loss": float(loss.detach()), "q_mean": float(predicted.detach().mean()),
                           "q_abs_max": float(predicted.detach().abs().max()),
                           "target_mean": float(targets.mean()), "gradient_norm": float(norm),
                           "td_abs_mean": float(errors.detach().abs().mean()),
                           "epsilon": self.epsilon, "per_beta": beta,
                           "updates": self.updates, "transitions": self.env_steps}
        return self.last_stats.copy()

    def save(self, path, metadata=None, include_replay=True):
        """Atomic, CPU-only checkpoint; all objects accepted by weights_only=True.

        include_replay=False produces an inference checkpoint. A training resume
        must use an include_replay=True checkpoint and restore its environment too.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored_metadata = self.metadata.copy()
        if metadata is not None:
            stored_metadata.update(metadata)
        # Keep metadata safe and portable; reject Python objects and non-finite floats.
        stored_metadata = json.loads(json.dumps(stored_metadata, allow_nan=False))
        payload = {"format_version": CHECKPOINT_VERSION, "rule_signature": RULE_SIGNATURE,
                   "encoder_version": ENCODER_VERSION, "config": asdict(self.config),
                   "state_dim": self.state_dim, "actions": 4,
                   "online": self.online.state_dict(), "target": self.target.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "env_steps": self.env_steps,
                   "updates": self.updates, "rng": self.rng.bit_generator.state,
                   "torch_rng": torch.get_rng_state(), "metadata": stored_metadata,
                   "resumable": bool(include_replay)}
        if include_replay:
            payload["replay"] = self.replay.state_dict()
            payload["pending"] = [
                [torch.from_numpy(s.copy()), a, r, torch.from_numpy(ns.copy()), term, trunc]
                for s, a, r, ns, term, trunc in self.pending]
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path, *, expected_encoder=None, expected_move_reward=None, require_resume=False):
        """Load only tensors and builtins on CPU, with strict architecture checks."""
        try:
            data = torch.load(Path(path), map_location="cpu", weights_only=True)
            if data.get("format_version") != CHECKPOINT_VERSION:
                raise ValueError("Unsupported checkpoint version")
            if data.get("rule_signature") != RULE_SIGNATURE or data.get("encoder_version") != ENCODER_VERSION:
                raise ValueError("Checkpoint rules/encoder are incompatible")
            config = DQNConfig(**data["config"])
            if data.get("state_dim") != STATE_DIMS[config.encoder] or data.get("actions") != 4:
                raise ValueError("Checkpoint architecture metadata is incompatible")
            if expected_encoder is not None and config.encoder != expected_encoder:
                raise ValueError("Checkpoint encoder does not match the requested encoder")
            if expected_move_reward is not None and config.move_reward != expected_move_reward:
                raise ValueError("Checkpoint move reward does not match the requested reward")
            if require_resume and not data.get("resumable"):
                raise ValueError("Inference-only checkpoint cannot exactly resume training")
            agent = cls(config)
            for name in ("online", "target"):
                for tensor in data[name].values():
                    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                        raise ValueError("Checkpoint contains invalid model weights")
                getattr(agent, name).load_state_dict(data[name], strict=True)
            agent.optimizer.load_state_dict(data["optimizer"])
            agent.env_steps, agent.updates = int(data["env_steps"]), int(data["updates"])
            if agent.env_steps < 0 or agent.updates < 0:
                raise ValueError("Invalid checkpoint counters")
            agent.rng.bit_generator.state = data["rng"]
            torch.set_rng_state(data["torch_rng"])
            if data.get("resumable"):
                agent.replay.load_state_dict(data["replay"])
                for s, a, r, ns, term, trunc in data["pending"]:
                    if term or trunc or not math.isfinite(r) or a not in range(4):
                        raise ValueError("Invalid pending replay transition")
                    agent.pending.append((agent._state(s.numpy()).copy(), a, r,
                                          agent._state(ns.numpy()).copy(), term, trunc))
                if len(agent.pending) >= config.n_step:
                    raise ValueError("Invalid pending n-step queue")
            agent.metadata = data.get("metadata", {})
            return agent
        except (FileNotFoundError, PermissionError):
            raise
        except Exception as error:
            raise ValueError(f"Cannot load compatible checkpoint {path}: {error}") from error


# A concise alias for scripts that name the algorithm rather than the agent.
Agent = DQNAgent
