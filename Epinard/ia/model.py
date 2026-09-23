"""
Réseau acteur-critique pour PPO.

Tronc partagé 14 -> 128 -> 128, puis deux têtes :
  - acteur  : 3 logits (tout droit / droite / gauche)
  - critique: 1 valeur d'état V(s)

Le tronc est partagé parce que les deux têtes ont besoin des mêmes
features (danger, direction de la pomme, espace libre).

ACTION MASKING
--------------
Les logits des actions non sûres sont mis à -inf AVANT le softmax.
La politique n'échantillonne donc que parmi les coups sûrs, et les
log-probs servant au ratio PPO correspondent bien à l'action jouée.
(Un veto appliqué APRÈS échantillonnage fausserait les ratios et
dégraderait l'entraînement silencieusement.)
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

MASK_VALUE = -1e8  # -inf numérique : évite les NaN dans le softmax


def _init(layer, std=2.0 ** 0.5, bias=0.0):
    """Initialisation orthogonale — le standard pour PPO, stabilise le début."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, n_features=17, n_actions=3, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            _init(nn.Linear(n_features, hidden)), nn.Tanh(),
            _init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        # gain faible sur la tête politique : démarrage quasi uniforme
        self.actor = _init(nn.Linear(hidden, n_actions), std=0.01)
        self.critic = _init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x, mask=None):
        h = self.trunk(x)
        logits = self.actor(h)
        if mask is not None:
            # mask : True = action autorisée
            logits = torch.where(mask, logits, torch.full_like(logits, MASK_VALUE))
        return logits, self.critic(h).squeeze(-1)

    def act(self, x, mask=None):
        """Échantillonne une action. Utilisé pendant la collecte de rollouts."""
        logits, value = self.forward(x, mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, x, actions, mask=None):
        """Ré-évalue des actions passées. Utilisé pendant les époques PPO."""
        logits, value = self.forward(x, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def greedy(self, x, mask=None):
        """Action la plus probable, sans échantillonnage. Pour l'évaluation."""
        logits, _ = self.forward(x, mask)
        return torch.argmax(logits, dim=-1)
