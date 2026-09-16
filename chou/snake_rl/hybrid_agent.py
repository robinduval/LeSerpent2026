"""Learned Double DQN decisions constrained by a cyclic safety shield.

The shield supplies admissible actions; a trained action-value network ranks
those actions. No hand-written action selector is used by this agent.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .env import DIRECTIONS, GRID_SIZE, RULE_SIGNATURE
from . import cycle_shield

FEATURE_DIM = 20

@dataclass
class HybridConfig:
    hidden: int = 64
    gamma: float = 0.97
    lr: float = 0.001
    batch_size: int = 128
    capacity: int = 40000
    move_reward: float = -0.05
    seed: int = 731
    encoder: str = 'cycle_action20'
    shield: str = 'rewired-cycle-v1'
    optimize_free_arc: bool = False
    explore_attempts: int = 0

class ActionQ(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, value):
        return self.layers(value).squeeze(-1)

def action_features(env, shield=None):
    """Pure board-derived observations; never inspect or copy the apple RNG."""
    index = cycle_shield.INDEX if shield is None else shield.index
    admissible = cycle_shield.allowed_actions(env) if shield is None else shield.allowed_actions(env)
    size = GRID_SIZE * GRID_SIZE
    head = env.body[0]
    hi = index[head]
    apple = env.apple if env.apple is not None else head
    apple_distance = (index[apple] - hi) % size
    tail_distance = (index[env.body[-1]] - hi) % size
    def torus(a, b):
        dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
        return min(dx, GRID_SIZE-dx) + min(dy, GRID_SIZE-dy)
    distance_before = torus(head, apple)
    result = np.empty((4, FEATURE_DIM), dtype=np.float32)
    for action, (dx, dy) in enumerate(DIRECTIONS):
        nxt = ((head[0]+dx) % GRID_SIZE, (head[1]+dy) % GRID_SIZE)
        advance = (index[nxt]-hi) % size
        remaining = (index[apple]-index[nxt]) % size
        if shield is not None and action in admissible:
            remaining = shield.next_food_distance(env, action)
        distance_after = torus(nxt, apple)
        result[action] = [
            apple_distance / size, tail_distance / size, advance / size,
            remaining / size, (apple_distance-remaining) / size,
            distance_before / 14, distance_after / 14,
            (distance_before-distance_after) / 14,
            len(env.body) / size, float(env.grow_pending), float(nxt == apple),
            float(advance <= apple_distance), float(action == env.direction),
            float(action == (env.direction+2) % 4),
            max(0, tail_distance-advance) / size,
            float(action == 0), float(action == 1), float(action == 2),
            float(action == 3), 1.,
        ]
    mask = np.zeros(4, dtype=bool)
    mask[list(admissible)] = True
    return result, mask

class HybridAgent:
    learned = True
    def __init__(self, config=None):
        self.config = config or HybridConfig()
        torch.set_num_threads(1)
        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.online = ActionQ(self.config.hidden)
        self.target = ActionQ(self.config.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.env_steps = self.updates = self.neural_choice_count = self.choice_states = 0
        self.metadata = {}
        self.last_decision = None
        self._numpy_layers = []
        self.refresh_weights()
        self._replay = None
        self._count = 0
        if self.config.shield not in ('rewired-cycle-v1', 'static-cycle-v1'):
            raise ValueError('Unknown safety shield')
        self.shield = cycle_shield.RewiredCycleShield() if self.config.shield == 'rewired-cycle-v1' else None
        self._last_env = None
        self._last_env_step = -1

    @property
    def training_transitions(self):
        return self.env_steps
    @property
    def training_updates(self):
        return self.updates

    def refresh_weights(self):
        self._numpy_layers = [(layer.weight.detach().numpy().copy(), layer.bias.detach().numpy().copy())
                              for layer in self.online.layers if isinstance(layer, nn.Linear)]

    def values(self, features):
        hidden = features
        for i, (weight, bias) in enumerate(self._numpy_layers):
            hidden = hidden @ weight.T + bias
            if i < len(self._numpy_layers)-1:
                hidden = np.maximum(hidden, 0)
        return hidden[..., 0]

    def select_features(self, features, mask, epsilon=0.):
        values = self.values(features)
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite hybrid network output')
        allowed = np.flatnonzero(mask)
        if not len(allowed):
            raise RuntimeError('Safety shield returned no admissible action')
        proposed = int(values.argmax())
        neural = int(np.where(mask, values, -np.inf).argmax())
        action = int(self.rng.choice(allowed)) if epsilon and self.rng.random() < epsilon else neural
        if len(allowed) > 1:
            self.choice_states += 1
            self.neural_choice_count += int(action == neural)
        self.last_decision = {'proposed_unmasked': proposed, 'executed': action,
                              'mask': mask.tolist(), 'q_values': values.tolist(),
                              'neural_choice': neural, 'admissible_count': len(allowed)}
        return action

    def encode(self, env):
        if self.shield is not None and (self._last_env is not env or env.steps == 0 or env.steps < self._last_env_step):
            self.shield.reset()
        self._last_env = env
        self._last_env_step = env.steps
        if self.config.explore_attempts and self.shield is not None and (env.steps == 0 or env.grow_pending):
            cycle_shield.explore_free_arc(self.shield, env, attempts=self.config.explore_attempts)
        elif self.config.optimize_free_arc and self.shield is not None:
            cycle_shield.optimize_free_arc(self.shield, env)
        return action_features(env, self.shield)

    def commit(self, env, action):
        if self.shield is not None:
            self.shield.commit(env, action)

    def select_action(self, env, explore=False):
        features, mask = self.encode(env)
        action = self.select_features(features, mask, epsilon=.05 if explore else 0.)
        self.commit(env, action)
        return action

    def observe(self, features, action, reward, next_features, next_mask, terminated):
        if self._replay is None:
            n = self.config.capacity
            self._replay = (np.empty((n, FEATURE_DIM), np.float32), np.empty(n, np.float32),
                            np.empty((n, 4, FEATURE_DIM), np.float32), np.empty((n,4), bool),
                            np.empty(n, bool))
        i = self.env_steps % self.config.capacity
        states, rewards, successors, masks, terminal = self._replay
        states[i] = features[action]
        rewards[i] = reward
        successors[i] = next_features
        masks[i] = next_mask
        terminal[i] = terminated
        self.env_steps += 1
        self._count = min(self._count + 1, self.config.capacity)

    def train_step(self):
        if self._count < self.config.batch_size:
            return None
        idx = self.rng.integers(self._count, size=self.config.batch_size)
        states, rewards, successors, masks, terminal = [torch.from_numpy(x[idx]) for x in self._replay]
        prediction = self.online(states)
        with torch.no_grad():
            selected = self.online(successors).masked_fill(~masks, -1e9).argmax(1)
            next_value = self.target(successors).gather(1, selected[:,None]).squeeze(1)
            targets = rewards + self.config.gamma * (~terminal).float() * next_value
        loss = F.smooth_l1_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite DDQN loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 250 == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.refresh_weights()
        return float(loss.detach())

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {'hybrid_version': 1, 'rule_signature': RULE_SIGNATURE,
                'config': asdict(self.config), 'online': self.online.state_dict(),
                'target': self.target.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'env_steps': self.env_steps, 'updates': self.updates,
                'metadata': {**self.metadata, **(metadata or {})}}
        json.dumps(data['metadata'], allow_nan=False)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(data, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, **kwargs):
        data = torch.load(Path(path), map_location='cpu', weights_only=True)
        if data.get('hybrid_version') != 1 or data.get('rule_signature') != RULE_SIGNATURE:
            raise ValueError('Incompatible hybrid checkpoint')
        config = dict(data['config'])
        config.setdefault('shield', 'static-cycle-v1')
        agent = cls(HybridConfig(**config))
        for name in ('online', 'target'):
            if any(not torch.isfinite(t).all() for t in data[name].values()):
                raise ValueError('Non-finite hybrid weights')
            getattr(agent, name).load_state_dict(data[name], strict=True)
        agent.optimizer.load_state_dict(data['optimizer'])
        agent.env_steps, agent.updates = int(data['env_steps']), int(data['updates'])
        if agent.env_steps <= 0 or agent.updates <= 0:
            raise ValueError('Checkpoint has not undergone reinforcement learning')
        agent.metadata = data['metadata']
        agent.refresh_weights()
        return agent
