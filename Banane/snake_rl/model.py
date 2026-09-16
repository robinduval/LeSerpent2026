"""Architectures de réseau. Entrée 11 valeurs, sortie 4 valeurs Q.

Deux architectures sélectionnables par configuration :

A. `DQN`        — perceptron de référence, celui du cours.
B. `DuelingDQN` — tronc partagé, puis une branche valeur d'état et une branche
                  avantage par action (Wang et al., 2016).

Double DQN n'est pas une architecture : c'est une façon de calculer la cible.
Il vit donc dans l'agent, pas ici.
"""

import torch
import torch.nn as nn

from .rules import N_ACTIONS
from .state import STATE_SIZE


class DQN(nn.Module):
    """Perceptron à deux couches cachées : 11 -> h -> h -> 4."""

    def __init__(self, hidden_size=256, input_size=STATE_SIZE, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    """Tronc partagé, branche valeur et branche avantage.

    Q(s, a) = V(s) + A(s, a) - moyenne_a' A(s, a')

    Le retrait de la moyenne lève l'ambiguïté d'identifiabilité : sans lui, on
    pourrait ajouter une constante à V et la retrancher à A sans changer Q, ce
    qui rend l'entraînement instable.
    """

    def __init__(self, hidden_size=256, input_size=STATE_SIZE, n_actions=N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.advantage_head = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        features = self.trunk(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def build_model(config):
    """Instancie l'architecture demandée par la configuration."""
    if config.uses_dueling:
        return DuelingDQN(hidden_size=config.hidden_size)
    return DQN(hidden_size=config.hidden_size)


def resolve_device(name="auto"):
    """Traduit `auto` en `cuda` si disponible, sinon `cpu`."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
