"""agent.py - Agent D3QN (Dueling Double DQN + PER) pour Snake RL.

Contient la classe Agent (réseaux DQN dueling, mémorisation PER, epsilon-greedy),
ainsi que les boucles d'entraînement (train) et d'évaluation (play).

Objectif : maximiser le ratio score/temps (trajectoires courtes et efficaces).
Hyperparamètres clés : gamma=0.95 (impatient, pas 0.99), epsilon-decay exponentiel
par épisode (×0.98, plancher 0.01), PER avec poids d'importance dans la loss Huber,
synchronisation dure de la cible tous les 1000 pas d'apprentissage.

Fichiers produits par train() : training_log.csv, best_model.pth, last_model.pth,
victory_model.pth, training_plot.png.
"""

import csv
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

# --- Rendre le dossier du script importable, quel que soit le cwd d'appel ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import DuelingDQN
from replay_buffer import PrioritizedReplayBuffer
from game import SnakeGameRL, GAME_SPEED

# Cadence d'affichage utilisee UNIQUEMENT par train(--render) : accelere la clock
# pendant l'entrainement. play() n'utilise pas cette constante et reste a GAME_SPEED
# (clock de base du socle). 0 = aucune limite de FPS.
TRAINING_FPS = 60

# --- Interface figee avec helper.py (peut ne pas encore exister) ---
try:
    from helper import TrainingLogger, plot_training, summarize
    HELPER_AVAILABLE = True
except ImportError:
    HELPER_AVAILABLE = False

    _CSV_HEADER = [
        "episode", "score", "steps", "time_s", "ratio",
        "total_reward", "epsilon", "won", "truncated", "wall_time_s",
    ]

    class TrainingLogger:
        """Logger CSV de secours (meme schema que helper.TrainingLogger)."""

        def __init__(self, csv_path):
            self.csv_path = csv_path
            dossier = os.path.dirname(csv_path)
            if dossier and not os.path.exists(dossier):
                os.makedirs(dossier, exist_ok=True)
            self._file = open(csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(_CSV_HEADER)
            self._file.flush()

        def log(self, episode, score, steps, time_s, ratio, total_reward,
                 epsilon, won, truncated, wall_time_s):
            self._writer.writerow([
                episode, score, steps, f"{time_s:.6f}", f"{ratio:.6f}",
                f"{total_reward:.6f}", f"{epsilon:.6f}",
                int(bool(won)), int(bool(truncated)), f"{wall_time_s:.6f}",
            ])
            self._file.flush()

        def close(self):
            self._file.close()

    def plot_training(csv_path, png_path):
        """Pas de graphique possible sans helper.py : no-op silencieux."""
        return None

    def summarize(csv_path):
        """Resume minimal a partir du CSV, sans dependance a helper.py."""
        best_score = 0
        best_ratio = 0.0
        wins = 0
        n = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n += 1
                best_score = max(best_score, int(row["score"]))
                best_ratio = max(best_ratio, float(row["ratio"]))
                wins += int(row["won"])
        result = {
            "episodes": n,
            "best_score": best_score,
            "best_ratio": best_ratio,
            "wins": wins,
        }
        print(result)
        return result


def _seed_all(seed):
    """Seed random/numpy/torch pour la reproductibilite (no-op si seed None)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Agent:
    """Agent D3QN : Dueling DQN + Double DQN + Prioritized Experience Replay.

    Utilise un réseau principal DuelingDQN pour la sélection epsilon-greedy, un réseau
    cible pour évaluer les Q-values cibles (Double DQN), et un buffer de replay priorité
    (PER) qui suréchantionne les expériences à erreur TD élevée. La loss Huber pondérée
    par les poids d'importance réduit la surestimation de Q-learning.

    Entraînable sur n'importe quel état vectoriel, disposant de get_action() / remember() /
    train_step() pour intégration dans une boucle RL (cf. train()).
    """

    def __init__(self, gamma=0.95, lr=1e-3, batch_size=128, buffer_capacity=100_000,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.98, target_sync_every=1000,
                 learn_start=1000, per_alpha=0.6, per_beta_start=0.4,
                 per_beta_frames=100_000, seed=None):
        """Initialise l'agent D3QN.

        Args:
            gamma: Facteur de discount (défaut 0.95 < 0.99 pour impatience).
            lr: Learning rate du réseau principal (défaut 1e-3).
            batch_size: Taille des minibatch (défaut 128).
            buffer_capacity: Capacité du replay buffer (défaut 100_000).
            eps_start: Epsilon initial (défaut 1.0).
            eps_end: Epsilon minimal (plancher, défaut 0.01).
            eps_decay: Facteur de décroissance exponentiel par épisode (défaut 0.98).
            target_sync_every: Synchronisation dure tous les N pas d'apprentissage (défaut 1000).
            learn_start: Nombre de transitions avant de commencer l'apprentissage (défaut 1000).
            per_alpha: PER exponent α pour la priorité (défaut 0.6).
            per_beta_start: Poids d'importance β initial (défaut 0.4).
            per_beta_frames: Nombre de frames pour annéler β de beta_start à 1.0 (défaut 100_000).
            seed: Graine pour reproductibilité (défaut None).
        """
        _seed_all(seed)

        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_sync_every = target_sync_every
        self.learn_start = learn_start

        self.model = DuelingDQN(input_size=12, hidden_size=256, output_size=4)
        self.target_model = DuelingDQN(input_size=12, hidden_size=256, output_size=4)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(
            capacity=buffer_capacity, alpha=per_alpha,
            beta_start=per_beta_start, beta_frames=per_beta_frames,
        )

        self.n_games = 0
        self.learn_steps = 0

    @property
    def epsilon(self):
        """Epsilon décroissant exponentiellement PAR ÉPISODE (pas par pas).

        Avec eps_decay=0.98 et ~230 épisodes, sature rapidement vers eps_end.
        Découplé de n_steps pour un exploration cohérente par partie (pas par transition).

        Returns:
            float: Valeur courante d'epsilon, bornée à [eps_end, eps_start].
        """
        return max(self.eps_end, self.eps_start * (self.eps_decay ** self.n_games))

    def get_action(self, state, greedy=False):
        """Sélectionne une action epsilon-greedy (ou greedy si demandé).

        Args:
            state: État (np.ndarray (12,) float32 pour Snake RL).
            greedy: Si True, ignore epsilon et retourne toujours argmax Q-value (défaut False).

        Returns:
            int: Indice d'action (0..3).
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 3)

        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32))
        with torch.no_grad():
            q_values = self.model(state_t)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        """Ajoute une transition au replay buffer (PER).

        Args:
            state: État actuel.
            action: Action exécutée.
            reward: Récompense reçue.
            next_state: État suivant.
            done: Si True, épisode terminé.
        """
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self):
        """Effectue une étape de Double DQN + PER sur un minibatch.

        Sélectionne le minibatch avec PER, utilise le réseau principal pour choisir l'action
        suivante (Double DQN) et le réseau cible pour l'évaluation, calcule la loss Huber
        pondérée par les poids d'importance, et met à jour les priorités du buffer.
        Synchronise la cible tous les target_sync_every pas.

        Returns:
            float ou None: Loss Huber scalaire, ou None si le buffer est trop petit.
        """
        if len(self.memory) < max(self.batch_size, self.learn_start):
            return None

        states, actions, rewards, next_states, dones, tree_indices, is_weights = \
            self.memory.sample(self.batch_size)

        states_t = torch.from_numpy(states)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1)
        next_states_t = torch.from_numpy(next_states)
        dones_t = torch.from_numpy(dones).unsqueeze(1)
        is_weights_t = torch.from_numpy(is_weights).unsqueeze(1)

        q = self.model(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN : le reseau principal choisit l'action, la cible l'evalue.
            next_actions = self.model(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_states_t).gather(1, next_actions)
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        td_errors = target - q
        elementwise_loss = F.smooth_l1_loss(q, target, reduction="none")
        loss = (is_weights_t * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update_priorities(
            tree_indices, td_errors.detach().squeeze(1).abs().numpy()
        )

        self.learn_steps += 1
        if self.learn_steps % self.target_sync_every == 0:
            self.sync_target()

        return loss.item()

    def sync_target(self):
        """Synchronisation dure : copie tous les poids du réseau principal vers la cible.

        Appelée tous les target_sync_every pas d'apprentissage dans train_step(),
        ou manuellement après load() pour initialiser la cible.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        """Sauvegarde le réseau principal sur disque.

        Args:
            path: Chemin du fichier .pth de sortie.
        """
        self.model.save(path)

    def load(self, path):
        """Charge un réseau principal depuis disque et synchronise la cible.

        Args:
            path: Chemin du fichier .pth à charger.
        """
        self.model.load(path)
        self.sync_target()


def train(n_episodes=600, render=False, results_dir=None, seed=None, log_every=10,
          fps=TRAINING_FPS):
    """Boucle d'entraînement complète : crée l'agent, l'environnement et exécute n_episodes.

    Sauvegarde les modèles best_model.pth (meilleur score), victory_model.pth (grille remplie),
    et last_model.pth. Écrit un journal training_log.csv avec toutes les métriques par épisode.
    Génère training_plot.png tous les 100 épisodes (score et ratio score/temps).

    Args:
        n_episodes: Nombre d'épisodes d'entraînement (défaut 600).
        render: Si True, affiche chaque partie (défaut False).
        fps: Cadence d'affichage en mode render (défaut TRAINING_FPS, clock accélérée).
            Sans effet en headless. Le mode play reste à GAME_SPEED.
        results_dir: Dossier de sortie (défaut <dossier_script>/results).
        seed: Graine pour reproductibilité (défaut None).
        log_every: Affiche le résumé tous les N épisodes dans la console (défaut 10).

    Returns:
        dict: Résumé final avec keys : episodes, best_score, best_ratio, wins.
    """
    if results_dir is None:
        results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "training_log.csv")
    best_model_path = os.path.join(results_dir, "best_model.pth")
    last_model_path = os.path.join(results_dir, "last_model.pth")
    victory_model_path = os.path.join(results_dir, "victory_model.pth")
    plot_path = os.path.join(results_dir, "training_plot.png")

    env = SnakeGameRL(render=render, fps=fps)
    agent = Agent(seed=seed)
    logger = TrainingLogger(csv_path)

    best_score = -1
    scores_window = deque(maxlen=50)
    quit_early = False

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        episode_losses = []
        t0 = time.time()
        info = {}

        while not done:
            action = agent.get_action(state)
            reward, done, score, info = env.play_step(action)
            next_state = env.get_state()
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            state = next_state
            total_reward += reward

            if env.quit_requested:
                quit_early = True
                break

        agent.n_games += 1
        wall_time_s = time.time() - t0
        time_s = info.get("time_s", 0.0)
        ratio = score / time_s if time_s > 0 else 0.0
        won = info.get("won", False)
        truncated = info.get("truncated", False)

        logger.log(
            agent.n_games, score, info.get("steps", 0), time_s, ratio,
            total_reward, agent.epsilon, won, truncated, wall_time_s,
        )

        if score > best_score:
            best_score = score
            agent.save(best_model_path)
        if won:
            agent.save(victory_model_path)

        scores_window.append(score)

        if quit_early:
            print(f"Arret demande (fenetre fermee) a l'episode {agent.n_games}.")
            break

        if agent.n_games % log_every == 0:
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
            moving_avg = sum(scores_window) / len(scores_window)
            print(
                f"Episode {agent.n_games}/{n_episodes} | score={score} | best={best_score} | "
                f"ratio={ratio:.3f} | moy50={moving_avg:.2f} | eps={agent.epsilon:.3f} | "
                f"buffer={len(agent.memory)} | loss_moy={avg_loss:.4f}"
            )

        if agent.n_games % 100 == 0:
            try:
                plot_training(csv_path, plot_path)
            except Exception as exc:  # pragma: no cover - le plot ne doit jamais casser l'entrainement
                print(f"Avertissement : plot_training a echoue ({exc}).")

    agent.save(last_model_path)
    try:
        plot_training(csv_path, plot_path)
    except Exception as exc:  # pragma: no cover
        print(f"Avertissement : plot_training a echoue ({exc}).")

    try:
        summary = summarize(csv_path)
    except Exception as exc:  # pragma: no cover
        print(f"Avertissement : summarize a echoue ({exc}).")
        summary = {"episodes": agent.n_games, "best_score": best_score}

    env.close()
    return summary


def play(model_path, n_games=5, render=True):
    """Charge un modèle et joue n_games parties en mode greedy pur (epsilon=0).

    Affiche le score, les steps, le temps et le ratio pour chaque partie en console.
    Respecte la demande de fermeture (quit_requested) si l'utilisateur ferme la fenêtre.

    Args:
        model_path: Chemin du fichier .pth du modèle entraîné.
        n_games: Nombre de parties à jouer (défaut 5).
        render: Si True, affiche les parties (défaut True).

    Returns:
        list[dict]: Liste des résultats avec keys : score, steps, time_s, ratio, won.
    """
    env = SnakeGameRL(render=render)
    agent = Agent()
    agent.load(model_path)

    results = []
    for i in range(n_games):
        state = env.reset()
        done = False
        info = {}
        score = 0
        while not done:
            action = agent.get_action(state, greedy=True)
            reward, done, score, info = env.play_step(action)
            state = env.get_state()
            if env.quit_requested:
                done = True

        time_s = info.get("time_s", 0.0)
        ratio = score / time_s if time_s > 0 else 0.0
        result = {
            "score": score,
            "steps": info.get("steps", 0),
            "time_s": time_s,
            "ratio": ratio,
            "won": info.get("won", False),
        }
        results.append(result)
        print(
            f"Partie {i + 1}/{n_games} | score={result['score']} | steps={result['steps']} | "
            f"time_s={result['time_s']:.2f} | ratio={result['ratio']:.3f} | won={result['won']}"
        )

        if env.quit_requested:
            print("Arret demande (fenetre fermee).")
            break

    env.close()
    return results


if __name__ == "__main__":
    print("=== Tests unitaires agent.py ===")

    # Test 1 : train_step effectue bien un pas de Double DQN + PER apres remplissage du buffer.
    test_agent = Agent(
        gamma=0.95, batch_size=16, buffer_capacity=200,
        learn_start=16, target_sync_every=5, seed=123,
    )
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    for _ in range(64):
        s = np_rng.standard_normal(12).astype(np.float32)
        a = rng.randint(0, 3)
        r = rng.uniform(-1.0, 1.0)
        s2 = np_rng.standard_normal(12).astype(np.float32)
        d = rng.random() < 0.1
        test_agent.remember(s, a, r, s2, d)

    total_before = test_agent.memory.tree.total
    loss = test_agent.train_step()
    assert loss is not None, "train_step aurait du retourner une loss (buffer assez rempli)"
    assert isinstance(loss, float), f"loss devrait etre un float, obtenu {type(loss)}"
    total_after = test_agent.memory.tree.total
    assert total_before != total_after, "Les priorites du SumTree auraient du changer apres train_step"
    print(f"Test 1 OK : loss={loss:.4f}, total priorites {total_before:.4f} -> {total_after:.4f}")

    # Test 2 : pas assez de transitions -> train_step retourne None.
    empty_agent = Agent(batch_size=32, learn_start=32, seed=0)
    assert empty_agent.train_step() is None, "train_step devrait retourner None si le buffer est trop petit"
    print("Test 2 OK : train_step retourne None si len(memory) < max(batch_size, learn_start).")

    # Test 3 : sync_target aligne bien les poids de la cible sur le modele principal.
    test_agent.sync_target()
    for p_online, p_target in zip(test_agent.model.parameters(), test_agent.target_model.parameters()):
        assert torch.allclose(p_online, p_target), "sync_target n'a pas copie les poids correctement"
    print("Test 3 OK : sync_target synchronise bien la cible sur le reseau principal.")

    # Test 4 : epsilon decroit avec n_games et sature a eps_end.
    eps_agent = Agent(eps_start=1.0, eps_end=0.01, eps_decay=0.98, seed=0)
    eps0 = eps_agent.epsilon
    eps_agent.n_games = 500
    eps1 = eps_agent.epsilon
    assert eps0 == 1.0
    assert abs(eps1 - 0.01) < 1e-9
    print(f"Test 4 OK : epsilon {eps0} -> {eps1} (sature a eps_end).")

    print(f"HELPER_AVAILABLE = {HELPER_AVAILABLE}")
    print("Tous les tests d'agent.py sont passes avec succes.")
