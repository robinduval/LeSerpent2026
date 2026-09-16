"""Agent DQN : politique, mémoire, apprentissage et checkpoints.

L'agent orchestre les trois blocs du cours. Il lit l'état produit par `Game`,
interroge `Model`, mémorise la transition et déclenche l'apprentissage.

Toutes les variantes (target network, Double DQN, Dueling) sont pilotées par
la configuration. Par défaut, `algorithm="dqn"` donne la baseline du cours :
on mesure d'abord, on améliore ensuite.
"""

import numpy as np
import torch
import torch.nn as nn

from .model import build_model, resolve_device
from .prioritized_replay import build_replay_buffer
from .rules import N_ACTIONS, RULESET

# Valeur utilisée pour éteindre une action interdite avant un argmax.
# On n'utilise pas -inf : une ligne entièrement masquée produirait des NaN.
MASKED_Q = -1e9


class Agent:
    """Agent DQN avec replay buffer, epsilon greedy et target network."""

    def __init__(self, config, device=None):
        self.config = config
        self.device = device or resolve_device(config.device)

        self.policy_net = build_model(config).to(self.device)
        self.target_net = build_model(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # jamais de gradient côté cible

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )
        # Réduction manuelle : PER doit pondérer chaque transition par son poids
        # d'importance avant de moyenner.
        self.criterion = (
            nn.SmoothL1Loss(reduction="none")
            if config.loss == "huber"
            else nn.MSELoss(reduction="none")
        )

        # Mémoire uniforme, ou prioritaire si l'algorithme se termine par _per.
        self.memory = build_replay_buffer(config)
        self._rng = np.random.default_rng(config.seed)

        self.episodes_done = 0
        self.learn_steps = 0

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    @property
    def epsilon(self):
        """Décroissance linéaire de epsilon_start à epsilon_end."""
        cfg = self.config
        if cfg.epsilon_decay_episodes <= 0:
            return cfg.epsilon_end
        progress = min(1.0, self.episodes_done / cfg.epsilon_decay_episodes)
        return cfg.epsilon_start + progress * (cfg.epsilon_end - cfg.epsilon_start)

    # ------------------------------------------------------------------
    # Décision
    # ------------------------------------------------------------------

    def act(self, state, mask=None, greedy=False):
        """Choisit une action.

        Args:
            state: vecteur d'état (11 valeurs).
            mask: masque des actions légales. Le demi-tour est exclu ici plutôt
                qu'ignoré par le jeu, pour que le réseau ne gaspille pas une
                sortie sur une action sans effet.
            greedy: True en évaluation. Aucune exploration, epsilon vaut 0.

        Returns:
            L'indice de l'action choisie.
        """
        mask = np.ones(N_ACTIONS, dtype=bool) if mask is None else np.asarray(mask)
        legal = np.flatnonzero(mask)
        if legal.size == 0:  # ne devrait pas arriver, filet de sécurité
            legal = np.arange(N_ACTIONS)

        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.choice(legal))

        return int(self.q_values(state, mask).argmax())

    def q_values(self, state, mask=None):
        """Valeurs Q d'un état, actions interdites éteintes. Sans gradient."""
        self.policy_net.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(
                np.asarray(state, dtype=np.float32), device=self.device
            ).unsqueeze(0)
            q = self.policy_net(tensor).squeeze(0).cpu().numpy()
        self.policy_net.train()
        if mask is not None:
            q = np.where(np.asarray(mask), q, MASKED_Q)
        return q

    # ------------------------------------------------------------------
    # Mémoire
    # ------------------------------------------------------------------

    def remember(self, state, action, reward, next_state, done, next_mask=None):
        """`done` est uniquement terminated, jamais une fin par time limit."""
        self.memory.push(state, action, reward, next_state, done, next_mask)

    def diagnostic_td_loss(self, state, action, result, next_state, next_mask):
        """Erreur TD d'une transition d'évaluation, sans gradient/apprentissage.

        Ce diagnostic n'est pas la loss des batches d'entraînement. Les
        troncatures conservent le bootstrap, comme dans learn().
        """
        with torch.no_grad():
            current = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            following = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.as_tensor(next_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            q_taken = self.policy_net(current)[0, action]
            target_q = self.target_net(following)
            if self.config.uses_double:
                best = self.policy_net(following).masked_fill(~mask, MASKED_Q).argmax(dim=1, keepdim=True)
                next_value = target_q.gather(1, best).squeeze()
            else:
                next_value = target_q.masked_fill(~mask, MASKED_Q).max()
            target = result.reward + self.config.gamma * next_value * (not result.terminated)
            return float(self.criterion(q_taken, target).item())

    # ------------------------------------------------------------------
    # Apprentissage
    # ------------------------------------------------------------------

    def learn(self):
        """Une mise à jour sur un batch. Retourne (loss, q_moyen) ou None.

        Retourne None tant que la mémoire n'a pas atteint `learning_starts` :
        apprendre sur quelques dizaines de transitions très corrélées revient
        surtout à mémoriser du bruit.
        """
        cfg = self.config
        if len(self.memory) < max(cfg.learning_starts, cfg.batch_size):
            return None

        batch = self.memory.sample(cfg.batch_size, device=self.device)
        if cfg.uses_per:
            (states, actions, rewards, next_states, dones, next_masks,
             indices, weights) = batch
        else:
            states, actions, rewards, next_states, dones, next_masks = batch
            indices, weights = None, None

        # Q(s, a) pour les actions réellement jouées.
        q_taken = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_target = self.target_net(next_states)

            if cfg.uses_double:
                # Double DQN : le policy net CHOISIT l'action, le target net
                # l'ÉVALUE. Découpler les deux réduit la surestimation des Q.
                next_q_policy = self.policy_net(next_states)
                next_q_policy = next_q_policy.masked_fill(~next_masks, MASKED_Q)
                best_actions = next_q_policy.argmax(dim=1, keepdim=True)
                next_value = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                next_q_target = next_q_target.masked_fill(~next_masks, MASKED_Q)
                next_value = next_q_target.max(dim=1).values

            targets = rewards + cfg.gamma * next_value * (1.0 - dones)

        per_sample_loss = self.criterion(q_taken, targets)
        if weights is None:
            loss = per_sample_loss.mean()
        else:
            # Les transitions sur-échantillonnées pèsent moins dans le gradient,
            # ce qui corrige le biais introduit par le tirage prioritaire.
            loss = (per_sample_loss * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), cfg.gradient_clip
            )
        self.optimizer.step()

        if indices is not None:
            # Erreur TD APRÈS la mise à jour du réseau : c'est la priorité que
            # méritera la transition au prochain tirage.
            with torch.no_grad():
                td_errors = (q_taken.detach() - targets).cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

        self.learn_steps += 1
        self._sync_target()

        return float(loss.item()), float(q_taken.mean().item())

    def _sync_target(self):
        """Met à jour le réseau cible, en hard update ou en soft update."""
        cfg = self.config
        if cfg.tau > 0:
            with torch.no_grad():
                for target_param, param in zip(
                    self.target_net.parameters(), self.policy_net.parameters()
                ):
                    target_param.mul_(1 - cfg.tau).add_(param, alpha=cfg.tau)
        elif (
            cfg.target_update_interval > 0
            and self.learn_steps % cfg.target_update_interval == 0
        ):
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def state_dict(self, **extra):
        """Contenu complet d'un checkpoint, configuration comprise."""
        from .seed import capture_rng_state

        payload = {
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_done": self.episodes_done,
            "learn_steps": self.learn_steps,
            "epsilon": self.epsilon,
            "config": self.config.to_dict(),
            "rng": capture_rng_state(),
        }
        payload.update(extra)
        payload["ruleset"] = RULESET
        return payload

    def save(self, path, **extra):
        torch.save(self.state_dict(**extra), path)

    def load_state_dict(self, payload, load_optimizer=True):
        if payload.get("ruleset") != RULESET:
            raise ValueError("checkpoint legacy/incompatible : règles toriques non attestées")
        self.policy_net.load_state_dict(payload["policy"])
        self.target_net.load_state_dict(payload["target"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.episodes_done = payload.get("episodes_done", 0)
        self.learn_steps = payload.get("learn_steps", 0)
        return self

    @classmethod
    def load(cls, path, config=None, device=None, load_optimizer=True):
        """Recharge un agent. La config stockée fait foi si aucune n'est donnée.

        Le replay buffer n'est pas sauvegardé : une reprise d'entraînement
        repart donc d'une mémoire vide et ne sera pas bit à bit identique à un
        run continu. C'est documenté et assumé.
        """
        from .config import Config

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("ruleset") != RULESET:
            raise ValueError("checkpoint legacy/incompatible : règles toriques non attestées")
        config = config or Config.from_dict(payload["config"])
        agent = cls(config, device=device)
        agent.load_state_dict(payload, load_optimizer=load_optimizer)
        return agent
