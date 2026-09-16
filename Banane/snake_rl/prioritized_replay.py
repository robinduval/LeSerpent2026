"""Prioritized Experience Replay (Schaul et al., 2016).

Principe : rejouer plus souvent les transitions sur lesquelles le réseau se
trompe le plus. La priorité d'une transition est son erreur TD.

    P(i) = p_i^alpha / somme_k p_k^alpha

Cet échantillonnage est biaisé : on ne tire plus selon la distribution réelle
des transitions. On compense par des poids d'importance

    w_i = (1 / (N * P(i)))^beta

normalisés par leur maximum. `beta` monte de `beta_start` vers 1 pendant
l'entraînement : le biais est toléré au début, quand les estimations sont de
toute façon grossières, puis corrigé quand la politique se stabilise.

L'arbre de sommes donne un tirage et une mise à jour en O(log n), là où une
recherche linéaire sur 100 000 priorités coûterait bien trop cher par batch.

Ne pas activer avant d'avoir une baseline DDQN mesurée.
"""

import numpy as np
import torch

from .rules import N_ACTIONS
from .state import STATE_SIZE


class SumTree:
    """Arbre binaire de sommes : tirage proportionnel aux priorités."""

    def __init__(self, capacity):
        self.capacity = capacity
        # Arbre complet stocké à plat. Les feuilles occupent la seconde moitié.
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total(self):
        return float(self.tree[1])

    def update(self, index, priority):
        """Fixe la priorité d'une feuille et remonte la somme jusqu'à la racine."""
        node = index + self.capacity
        self.tree[node] = priority
        node //= 2
        while node >= 1:
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
            node //= 2

    def find(self, value):
        """Descend l'arbre et renvoie la feuille dont l'intervalle contient `value`."""
        node = 1
        while node < self.capacity:
            left = 2 * node
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node = left + 1
        return node - self.capacity

    def max_leaf(self):
        leaves = self.tree[self.capacity:]
        return float(leaves.max()) if leaves.size else 0.0


class PrioritizedReplayBuffer:
    """Mémoire de replay à échantillonnage prioritaire.

    L'interface reste celle de `ReplayBuffer`, avec deux ajouts : `sample`
    renvoie aussi les indices et les poids d'importance, et
    `update_priorities` doit être appelée après chaque mise à jour.
    """

    def __init__(
        self,
        capacity=100_000,
        state_size=STATE_SIZE,
        seed=None,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_steps=100_000,
        priority_epsilon=1e-6,
    ):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = max(1, beta_steps)
        self.priority_epsilon = priority_epsilon

        self._rng = np.random.default_rng(seed)
        self._tree = SumTree(self.capacity)

        self.states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones((self.capacity, N_ACTIONS), dtype=bool)

        self._position = 0
        self._size = 0
        self._steps = 0
        self._max_priority = 1.0

    def __len__(self):
        return self._size

    @property
    def is_full(self):
        return self._size == self.capacity

    @property
    def beta(self):
        """Montée linéaire de beta_start vers beta_end."""
        progress = min(1.0, self._steps / self.beta_steps)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done, next_mask=None):
        """Insère une transition avec la priorité maximale observée.

        Priorité maximale et non nulle : une transition jamais rejouée n'a pas
        d'erreur TD connue, il faut donc lui garantir au moins un passage.
        """
        i = self._position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.next_masks[i] = (
            np.ones(N_ACTIONS, dtype=bool) if next_mask is None else next_mask
        )

        self._tree.update(i, self._max_priority**self.alpha)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, device=None):
        """Tirage stratifié proportionnel aux priorités.

        On découpe [0, total] en `batch_size` segments et on tire une valeur
        dans chacun : cela couvre tout le spectre des priorités au lieu de
        concentrer le batch sur quelques transitions très prioritaires.
        """
        if self._size == 0:
            raise ValueError("replay buffer vide")

        self._steps += 1
        total = self._tree.total
        segment = total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            value = self._rng.uniform(segment * i, segment * (i + 1))
            index = self._tree.find(value)
            # Une feuille jamais remplie peut sortir en bord d'intervalle.
            indices[i] = min(index, self._size - 1)

        priorities = self._tree.tree[indices + self.capacity]
        probabilities = priorities / max(total, 1e-12)
        weights = (self._size * np.maximum(probabilities, 1e-12)) ** (-self.beta)
        weights /= max(weights.max(), 1e-12)  # normalisation par le maximum

        device = device or torch.device("cpu")
        return (
            torch.from_numpy(self.states[indices]).to(device),
            torch.from_numpy(self.actions[indices]).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.next_states[indices]).to(device),
            torch.from_numpy(self.dones[indices]).to(device),
            torch.from_numpy(self.next_masks[indices]).to(device),
            indices,
            torch.from_numpy(weights.astype(np.float32)).to(device),
        )

    def update_priorities(self, indices, td_errors):
        """Réajuste les priorités à partir des erreurs TD du dernier batch."""
        priorities = np.abs(np.asarray(td_errors, dtype=np.float64))
        priorities += self.priority_epsilon  # jamais zéro, sinon jamais rejouée
        self._max_priority = max(self._max_priority, float(priorities.max()))
        for index, priority in zip(np.asarray(indices), priorities):
            self._tree.update(int(index), priority**self.alpha)


def build_replay_buffer(config):
    """Choisit la mémoire correspondant à l'algorithme configuré."""
    if config.uses_per:
        return PrioritizedReplayBuffer(
            capacity=config.replay_capacity,
            seed=config.seed,
            alpha=config.per_alpha,
            beta_start=config.per_beta_start,
            beta_end=config.per_beta_end,
            beta_steps=config.per_beta_steps,
            priority_epsilon=config.priority_epsilon,
        )
    from .replay_buffer import ReplayBuffer

    return ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)
