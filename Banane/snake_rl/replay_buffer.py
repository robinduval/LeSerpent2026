"""Mémoire de replay uniforme.

Implémentation sur tableaux NumPy préalloués plutôt que sur une `deque` de
tuples : à capacité 100 000, l'échantillonnage devient nettement moins coûteux
et la conversion vers PyTorch se fait en une seule copie par batch.

On mémorise aussi le masque d'actions légales de l'état suivant. Il est
indispensable au calcul correct de la cible : le max sur les Q suivantes ne
doit pas pouvoir choisir un demi-tour, que le jeu refuserait.
"""

import numpy as np
import torch

from .rules import N_ACTIONS
from .state import STATE_SIZE


class ReplayBuffer:
    """Tampon circulaire de transitions, échantillonné uniformément."""

    def __init__(self, capacity=100_000, state_size=STATE_SIZE, seed=None):
        self.capacity = int(capacity)
        self.state_size = state_size
        self._rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones((self.capacity, N_ACTIONS), dtype=bool)

        self._position = 0
        self._size = 0

    def __len__(self):
        return self._size

    @property
    def is_full(self):
        return self._size == self.capacity

    def push(self, state, action, reward, next_state, done, next_mask=None):
        """Ajoute une transition, en écrasant la plus ancienne si nécessaire."""
        i = self._position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.next_masks[i] = (
            np.ones(N_ACTIONS, dtype=bool) if next_mask is None else next_mask
        )

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, device=None):
        """Tire `batch_size` transitions et les renvoie en tenseurs.

        L'échantillonnage est avec remise : c'est le comportement usuel du DQN
        et cela évite un cas d'erreur quand le tampon est plus petit que le
        batch en début d'entraînement.
        """
        if self._size == 0:
            raise ValueError("replay buffer vide")
        idx = self._rng.integers(0, self._size, size=batch_size)
        device = device or torch.device("cpu")

        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
            torch.from_numpy(self.next_masks[idx]).to(device),
        )
