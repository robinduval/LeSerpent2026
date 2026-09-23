"""Réseau Rainbow DQN compacte : CNN + Dueling + NoisyNets."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .encoding import N_CHANNELS


class NoisyLinear(nn.Module):
    """Couche linéaire avec bruit d'exploration (NoisyNet)."""

    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma

        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma / math.sqrt(in_features)))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma / math.sqrt(1)))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

    def sample_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """Le support de distribution doit couvrir tout le retour cumulé possible :
    avec gamma=0.99 et un score cible >= 10 pommes (+10 chacune) plus la victoire
    (+100), un retour de -15 à 150 dans [v_min, v_max] évite un écrêtage qui
    empêcherait le réseau de distinguer les bonnes séquences de plusieurs pommes.
    """

    def __init__(self, grid_size=15, n_actions=4, n_atoms=11, v_min=-12.0, v_max=60.0):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)

        # CNN : feature extraction de (B, 10, grid_size, grid_size)
        self.conv1 = nn.Conv2d(N_CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Réduit la carte spatiale à une taille fixe avant l'aplatissement :
        # rend le coût CPU indépendant de grid_size (15x15 -> 225 cases est
        # trop lourd à aplatir directement) tout en gardant l'info autour de
        # la tête, toujours centrée en (0, 0) par l'encodage.
        self.pool_size = 5
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

        conv_out = 64 * self.pool_size * self.pool_size
        hidden = 128

        # Value stream (dueling)
        self.fc_v_base = nn.Linear(conv_out, hidden)
        self.fc_v_base_noise = NoisyLinear(hidden, hidden)
        self.fc_v = NoisyLinear(hidden, n_atoms)

        # Advantage stream (dueling)
        self.fc_a_base = nn.Linear(conv_out, hidden)
        self.fc_a_base_noise = NoisyLinear(hidden, hidden)
        self.fc_a = NoisyLinear(hidden, n_actions * n_atoms)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Value stream
        v = F.relu(self.fc_v_base(x))
        v = F.relu(self.fc_v_base_noise(v))
        v = self.fc_v(v)  # (B, n_atoms)

        # Advantage stream
        a = F.relu(self.fc_a_base(x))
        a = F.relu(self.fc_a_base_noise(a))
        a = self.fc_a(a)  # (B, n_actions * n_atoms)
        a = a.view(a.size(0), self.n_actions, self.n_atoms)

        # Dueling : Z(s,a) = V(s) + (A(s,a) - mean(A))
        a_mean = a.mean(dim=1, keepdim=True)
        z = v.unsqueeze(1) + (a - a_mean)  # (B, n_actions, n_atoms)

        # Softmax sur le support : (B, n_actions, n_atoms)
        return F.softmax(z, dim=2)

    def sample_noise(self):
        self.fc_v_base_noise.sample_noise()
        self.fc_a_base_noise.sample_noise()
        self.fc_v.sample_noise()
        self.fc_a.sample_noise()

    def eval_noise(self):
        """Éteint le bruit pour l'évaluation."""
        self.fc_v_base_noise.weight_epsilon.zero_()
        self.fc_v_base_noise.bias_epsilon.zero_()
        self.fc_a_base_noise.weight_epsilon.zero_()
        self.fc_a_base_noise.bias_epsilon.zero_()
        self.fc_v.weight_epsilon.zero_()
        self.fc_v.bias_epsilon.zero_()
        self.fc_a.weight_epsilon.zero_()
        self.fc_a.bias_epsilon.zero_()


class RainbowMLP(nn.Module):
    """Rainbow sur l'état compact : MLP dueling + distributionnel + NoisyNets.

    Remplace RainbowDQN (CNN) pour l'entraînement réel : sur CPU, le CNN plein
    grille n'arrivait pas à différencier les états (Q identiques à 0.05 près) et
    ne convergeait pas dans le temps disponible.
    """

    def __init__(self, n_inputs=None, n_actions=4, n_atoms=51, v_min=-15.0, v_max=80.0, hidden=256):
        super().__init__()
        from .encoding import N_COMPACT
        n_inputs = n_inputs or N_COMPACT
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)

        self.feature = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_v = NoisyLinear(hidden, n_atoms)
        self.fc_a = NoisyLinear(hidden, n_actions * n_atoms)

    def forward(self, x):
        h = self.feature(x)
        v = self.fc_v(h).unsqueeze(1)                                   # (B, 1, atoms)
        a = self.fc_a(h).view(x.size(0), self.n_actions, self.n_atoms)  # (B, A, atoms)
        z = v + (a - a.mean(dim=1, keepdim=True))
        return F.softmax(z, dim=2)

    def sample_noise(self):
        self.fc_v.sample_noise()
        self.fc_a.sample_noise()

    def eval_noise(self):
        for layer in (self.fc_v, self.fc_a):
            layer.weight_epsilon.zero_()
            layer.bias_epsilon.zero_()
