"""Replay buffer avec priorité (PER) pour Rainbow DQN."""
import numpy as np
from collections import deque


class SumTree:
    """Arbre de segment pour l'échantillonnage efficace des priorités."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = deque(maxlen=capacity)
        self.ptr = 0

    def add(self, priority, transition):
        idx = self.capacity - 1 + self.ptr
        self.data.append(transition)
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity

    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def sample(self, batch_size, rng):
        samples = []
        indices = []
        priorities = []
        total = self.tree[0]
        segment = total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = rng.uniform(a, b)
            idx = 0
            while 2 * idx + 1 < len(self.tree):
                left, right = 2 * idx + 1, 2 * idx + 2
                if s < self.tree[left]:
                    idx = left
                else:
                    s -= self.tree[left]
                    idx = right
            data_idx = idx - (self.capacity - 1)
            samples.append(self.data[data_idx])
            indices.append(idx)
            priorities.append(self.tree[idx])
        return samples, indices, np.array(priorities) / total


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, rng=None):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.rng = rng if rng is not None else np.random.RandomState()
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done, truncated=False):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done, truncated))

    def sample(self, batch_size):
        samples, indices, weights = self.tree.sample(batch_size, self.rng)
        states, actions, rewards, next_states, dones, truncateds = zip(*samples)
        weights = np.minimum(1.0, weights) ** (-self.beta)
        weights /= weights.max()
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            np.array(weights, dtype=np.float32),
            indices,
        )

    def update_priorities(self, indices, tds):
        for idx, td in zip(indices, tds):
            self.tree.update(idx, (np.abs(td) + 1e-6) ** self.alpha)
            self.max_priority = max(self.max_priority, np.abs(td) + 1e-6)
