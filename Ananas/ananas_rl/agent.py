"""Agent Rainbow DQN : Double DQN + Dueling + distributionnel (C51) + NoisyNets.

L'état est l'état compact (cf. encoding.encode_compact) et le réseau un MLP :
l'encodage plein-grille + CNN ne convergeait pas sur CPU dans le temps imparti
(le réseau sortait des Q identiques quel que soit l'état).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .model import RainbowMLP
from .encoding import encode_compact, N_COMPACT, encode_compact_v2, N_COMPACT_V2

# Jeux de variables d'entrée. Le nom est enregistré dans le checkpoint pour que
# le chargement reconstruise le bon réseau.
FEATURES = {
    "v1": (encode_compact, N_COMPACT),       # état compact du cours
    "v2": (encode_compact_v2, N_COMPACT_V2),  # + espace libre accessible
}


class RainbowAgent:
    def __init__(self, grid_size=15, device="cpu", lr=1e-3, gamma=0.99, n_atoms=51, features="v2"):
        self.grid_size = grid_size
        self.device = device
        self.gamma = gamma
        self.n_atoms = n_atoms
        self.features = features
        self.encode, n_inputs = FEATURES[features]

        self.q_net = RainbowMLP(n_inputs, n_atoms=n_atoms).to(device)
        self.target_net = RainbowMLP(n_inputs, n_atoms=n_atoms).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.support = self.q_net.support.to(device)
        self.delta_z = (self.q_net.v_max - self.q_net.v_min) / (n_atoms - 1)

    def _q_values(self, distributions):
        """(B, A, atoms) -> (B, A) : espérance sur le support."""
        return torch.sum(distributions * self.support.view(1, 1, -1), dim=2)

    def choose_action(self, env, training=True, epsilon=0.0, obs=None):
        """`obs` : observation déjà encodée, pour éviter de la recalculer."""
        mask = np.array(env.legal_mask())
        if training and np.random.rand() < epsilon:
            return int(np.random.choice(np.flatnonzero(mask)))
        if obs is None:
            obs = self.encode(env)
        with torch.no_grad():
            q = self._q_values(self.q_net(torch.from_numpy(obs).unsqueeze(0).to(self.device)))[0].cpu().numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def update(self, batch, weights):
        """batch : (states, actions, rewards, next_states, dones, truncateds)."""
        states, actions, rewards, next_states, dones, truncateds = batch
        B = len(states)

        states = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.asarray(actions, dtype=np.int64)).to(self.device)
        rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).to(self.device)
        # Une fin par "boucle" est une coupure artificielle : la valeur future ne
        # doit pas être annulée, contrairement à une vraie mort.
        terminal = torch.from_numpy(
            (np.asarray(dones, dtype=bool) & ~np.asarray(truncateds, dtype=bool)).astype(np.float32)
        ).to(self.device)
        weights_t = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(self.device)

        self.q_net.sample_noise()
        self.target_net.sample_noise()

        dist = self.q_net(states)
        dist_a = dist[torch.arange(B), actions]

        with torch.no_grad():
            best_actions = torch.argmax(self._q_values(self.q_net(next_states)), dim=1)  # Double DQN
            next_dist_a = self.target_net(next_states)[torch.arange(B), best_actions]

            # Projection catégorielle (C51)
            tz = rewards.unsqueeze(1) + (1.0 - terminal).unsqueeze(1) * self.gamma * self.support.unsqueeze(0)
            tz = tz.clamp(self.q_net.v_min, self.q_net.v_max)
            b = (tz - self.q_net.v_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()
            # Si b tombe pile sur un atome, l == u : toute la masse va sur cet atome,
            # sinon elle se perdrait (les deux poids valant 0).
            eq = (l == u)
            w_l = torch.where(eq, torch.ones_like(b), u.float() - b)
            w_u = torch.where(eq, torch.zeros_like(b), b - l.float())

            target_dist = torch.zeros(B, self.n_atoms, device=self.device)
            target_dist.scatter_add_(1, l.clamp(0, self.n_atoms - 1), w_l * next_dist_a)
            target_dist.scatter_add_(1, u.clamp(0, self.n_atoms - 1), w_u * next_dist_a)

        per_sample = -torch.sum(target_dist * torch.log(dist_a + 1e-8), dim=1)
        loss = (per_sample * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        return per_sample.detach().cpu().numpy()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": self.features, "n_atoms": self.n_atoms,
                    "state_dict": self.q_net.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path, grid_size=15, device="cpu"):
        """Reconstruit l'agent avec le bon jeu de variables et le met en mode jeu."""
        data = torch.load(path, map_location=device)
        if isinstance(data, dict) and "state_dict" in data:
            features, n_atoms, state_dict = data["features"], data["n_atoms"], data["state_dict"]
        else:
            # Premiers checkpoints : state_dict brut, entraînés sur l'état v1.
            features, n_atoms, state_dict = "v1", 51, data
        agent = cls(grid_size=grid_size, device=device, n_atoms=n_atoms, features=features)
        agent.q_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)
        # Politique déterministe : sans ça le bruit NoisyNet chargé depuis le
        # checkpoint reste actif et rend le jeu incohérent.
        agent.q_net.eval()
        agent.q_net.eval_noise()
        return agent
