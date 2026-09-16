"""Owned replay transitions, proportional PER and safe checkpoint conversion."""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, *, prioritized=False,
                 alpha=0.6, priority_epsilon=1e-5):
        if capacity < 1 or state_dim < 1:
            raise ValueError("Replay dimensions must be positive")
        self.capacity, self.state_dim = int(capacity), int(state_dim)
        self.prioritized = bool(prioritized)
        self.alpha, self.priority_epsilon = float(alpha), float(priority_epsilon)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.size, self.position = 0, 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, terminated, discount):
        index = self.position
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminated[index] = terminated
        self.discounts[index] = discount
        self.priorities[index] = self.priorities[:self.size].max() if self.size else 1.
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng, beta=0.4):
        if self.size < batch_size:
            raise ValueError("Replay has fewer transitions than batch_size")
        if self.prioritized:
            mass = np.power(self.priorities[:self.size].astype(np.float64), self.alpha)
            probabilities = mass / mass.sum()
            indices = rng.choice(self.size, size=batch_size, replace=True, p=probabilities)
            weights = np.power(self.size * probabilities[indices], -beta)
            # Global normalization, not minibatch normalization.
            weights /= np.power(self.size * probabilities.min(), -beta)
        else:
            indices = rng.choice(self.size, size=batch_size, replace=False)
            probabilities = np.full(self.size, 1. / self.size)
            weights = np.ones(batch_size)
        batch = {name: torch.from_numpy(getattr(self, name)[indices].copy())
                 for name in ("states", "actions", "rewards", "next_states", "terminated", "discounts")}
        batch.update(indices=indices, weights=torch.tensor(weights, dtype=torch.float32),
                     probabilities=probabilities[indices].copy())
        return batch

    def update_priorities(self, indices, td_errors):
        errors = np.asarray(td_errors, dtype=np.float64)
        if not np.isfinite(errors).all():
            raise FloatingPointError("Non-finite replay priorities")
        values = np.abs(errors) + self.priority_epsilon
        # A sampled index may repeat. Give it the largest observed error.
        for index in np.unique(indices):
            self.priorities[int(index)] = values[np.asarray(indices) == index].max()

    def state_dict(self):
        result = {"size": self.size, "position": self.position,
                  "capacity": self.capacity, "state_dim": self.state_dim}
        for name in ("states", "actions", "rewards", "next_states", "terminated", "discounts", "priorities"):
            result[name] = torch.from_numpy(getattr(self, name)[:self.size].copy())
        return result

    def load_state_dict(self, data):
        if data["capacity"] != self.capacity or data["state_dim"] != self.state_dim:
            raise ValueError("Replay shape does not match checkpoint config")
        size, position = int(data["size"]), int(data["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("Invalid replay cursor")
        if size < self.capacity and position != size:
            raise ValueError("Invalid partial replay cursor")
        for name in ("states", "actions", "rewards", "next_states", "terminated", "discounts", "priorities"):
            target = getattr(self, name)[:size]
            source = data[name]
            if not isinstance(source, torch.Tensor) or tuple(source.shape) != target.shape:
                raise ValueError(f"Invalid replay array: {name}")
            array = source.cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite replay array: {name}")
            target[:] = array
        if size and (np.any(self.actions[:size] < 0) or np.any(self.actions[:size] > 3)
                     or np.any(self.priorities[:size] <= 0)
                     or np.any(self.discounts[:size] < 0) or np.any(self.discounts[:size] > 1)):
            raise ValueError("Invalid replay contents")
        self.size, self.position = size, position
