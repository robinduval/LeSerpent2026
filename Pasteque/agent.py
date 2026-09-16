"""agent.py - Agent D3QN (Dueling Double DQN + PER) pour Snake RL.

Contient la classe Agent (réseaux DQN dueling, mémorisation PER, epsilon-greedy),
ainsi que les boucles d'entraînement (train) et d'évaluation (play).

Objectif : maximiser le ratio score/temps (trajectoires courtes et efficaces).
Hyperparamètres clés : gamma=0.95 (impatient, pas 0.99), epsilon-decay exponentiel
par épisode (×0.985, plancher 0.02), PER avec poids d'importance dans la loss Huber.

Optimisations (diagnostic run de reference, 363 episodes, seed 42 : plateau ~40
pommes des l'ep. 200, epsilon au plancher a l'ep. ~230, 15-20 ms/pas car un pas de
gradient a CHAQUE pas de jeu, morts par auto-enfermement mal anticipees) :
- train_every : un pas de gradient tous les N pas de jeu seulement (x3-4 plus rapide).
- n-step returns : propage plus vite la penalite de mort vers les decisions en amont.
- Polyak (tau) : cible mise a jour en douceur a chaque pas de gradient, moins de sauts.
- eps_end/eps_decay releves : garde un minimum d'exploration plus longtemps.
- scheduler de lr lineaire par episode : stabilise la phase de plateau.

Fichiers produits par train() : training_log.csv, config.json, best_model.pth,
last_model.pth, victory_model.pth, training_plot.png.
"""

import csv
import json
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
                 eps_start=1.0, eps_end=0.02, eps_decay=0.985, target_sync_every=1000,
                 learn_start=1000, per_alpha=0.6, per_beta_start=0.4,
                 per_beta_frames=100_000, seed=None,
                 train_every=4, n_step=3, tau=0.005, threads=None,
                 lr_min=1e-4, lr_decay_episodes=400):
        """Initialise l'agent D3QN.

        Args:
            gamma: Facteur de discount (défaut 0.95 < 0.99 pour impatience).
            lr: Learning rate initial du réseau principal (défaut 1e-3).
            batch_size: Taille des minibatch (défaut 128).
            buffer_capacity: Capacité du replay buffer (défaut 100_000).
            eps_start: Epsilon initial (défaut 1.0).
            eps_end: Epsilon minimal (plancher, défaut 0.02 : garde un peu d'exploration
                plus longtemps que l'ancien 0.01, utile une fois le plateau atteint).
            eps_decay: Facteur de décroissance exponentiel par épisode (défaut 0.985 :
                plancher atteint vers l'ép. ~260 au lieu de ~230 avec 0.98).
            target_sync_every: Synchronisation dure tous les N pas d'apprentissage, utilisée
                uniquement si tau est None/0 (défaut 1000).
            learn_start: Nombre de transitions avant de commencer l'apprentissage (défaut 1000).
            per_alpha: PER exponent α pour la priorité (défaut 0.6).
            per_beta_start: Poids d'importance β initial (défaut 0.4).
            per_beta_frames: Nombre de frames pour annéler β de beta_start à 1.0 (défaut 100_000).
            seed: Graine pour reproductibilité (défaut None).
            train_every: N'effectue un pas de gradient que tous les train_every appels à
                train_step() (défaut 4). Pratique DQN standard : ×3-4 plus rapide (moins de
                pas de gradient redondants sur des transitions très corrélées d'un même pas
                à l'autre) pour une efficacité d'apprentissage quasi identique.
            n_step: Horizon des retours n-step (défaut 3). Propage plus vite la pénalité de
                mort (-20) vers les décisions qui y ont mené (auto-enfermement), au lieu
                d'attendre une chaîne de mises à jour Bellman à 1 pas.
            tau: Coefficient de mise à jour douce (Polyak) de la cible, appliquée à chaque
                pas de gradient (défaut 0.005). Si None ou 0, retombe sur l'ancien hard sync
                tous les target_sync_every pas. Une cible plus stable réduit les sauts de
                la fonction de valeur, donc les décisions erratiques en phase de plateau.
            threads: Si fourni, fixe torch.set_num_threads(threads) (défaut None = défaut
                torch). Utile pour lancer plusieurs seeds en parallèle sans contention CPU.
            lr_min: Learning rate plancher du scheduler linéaire par épisode (défaut 1e-4).
            lr_decay_episodes: Nombre d'épisodes sur lesquels le lr décroît linéairement de
                lr à lr_min (défaut 400). Stabilise la phase de plateau (moins d'oscillations
                de la loss une fois la politique proche de sa performance asymptotique).
        """
        _seed_all(seed)

        if threads is not None:
            torch.set_num_threads(threads)
        self.threads = threads

        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_sync_every = target_sync_every
        self.learn_start = learn_start

        # --- Retours n-step : deque bornee, videe episode par episode (cf. flush_n_step). ---
        self.n_step = n_step
        self.gamma_n = gamma ** n_step
        self._nstep_buffer = deque(maxlen=n_step)

        # --- Cadence des pas de gradient (train_every) et cible (Polyak si tau). ---
        self.train_every = train_every
        self.env_steps = 0
        self.tau = tau

        # --- Scheduler de lr lineaire par episode, applique dans on_episode_end(). ---
        self.lr_start = lr
        self.lr_min = lr_min
        self.lr_decay_episodes = lr_decay_episodes

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
        """Empile une transition dans le tampon n-step, et pousse au buffer PER si plein.

        La transition brute est ajoutée à une deque(maxlen=n_step). Dès que celle-ci
        atteint n_step éléments, on calcule le retour n-step R = Σ_{k<n} γ^k r_k en
        partant du plus ancien élément (troncature à la première transition terminale
        rencontrée), et on pousse (s_0, a_0, R, s_n, done) au replay buffer PER. On
        retire ensuite le plus ancien élément pour faire glisser la fenêtre d'un pas.
        En fin d'épisode, flush_n_step() doit être appelé pour vider les restes.

        Args:
            state: État actuel.
            action: Action exécutée.
            reward: Récompense reçue.
            next_state: État suivant.
            done: Si True, épisode terminé.
        """
        self._nstep_buffer.append((state, action, reward, next_state, done))
        if len(self._nstep_buffer) >= self.n_step:
            self._emit_nstep_transition()
            self._nstep_buffer.popleft()

    def _emit_nstep_transition(self):
        """Calcule et pousse au buffer PER la transition n-step depuis le plus ancien élément.

        Parcourt le tampon n-step à partir de l'indice 0, cumule les récompenses
        actualisées par γ^k, et s'arrête à la première transition terminale (troncature :
        R ne doit pas inclure de récompenses au-delà de la fin d'un épisode).
        """
        buf = self._nstep_buffer
        state0, action0 = buf[0][0], buf[0][1]
        R = 0.0
        next_state = buf[0][3]
        done_flag = False
        for k, (_, _, r, s2, d) in enumerate(buf):
            R += (self.gamma ** k) * r
            next_state = s2
            if d:
                done_flag = True
                break
        self.memory.add(state0, action0, R, next_state, done_flag)

    def flush_n_step(self):
        """Vide le tampon n-step en fin d'épisode, en émettant toutes les sous-séquences restantes.

        À appeler par la boucle d'entraînement quand done=True (ou en cas d'arrêt anticipé),
        pour ne pas perdre les derniers pas de l'épisode ni mélanger des états de deux
        épisodes différents dans le tampon n-step suivant.
        """
        while self._nstep_buffer:
            self._emit_nstep_transition()
            self._nstep_buffer.popleft()

    def train_step(self):
        """Effectue une étape de Double DQN + PER sur un minibatch (tous les train_every appels).

        Sélectionne le minibatch avec PER, utilise le réseau principal pour choisir l'action
        suivante (Double DQN) et le réseau cible pour l'évaluation, calcule la loss Huber
        pondérée par les poids d'importance, et met à jour les priorités du buffer. La cible
        du Bellman utilise l'horizon n-step (gamma_n) puisque les transitions stockées sont
        déjà des retours n-step. Met à jour la cible : Polyak à chaque pas si tau est défini,
        sinon hard sync tous les target_sync_every pas.

        Returns:
            float ou None: Loss Huber scalaire, ou None si pas de pas de gradient ce coup-ci
                (train_every non atteint, ou buffer trop petit).
        """
        self.env_steps += 1
        if self.env_steps % self.train_every != 0:
            return None

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
            target = rewards_t + self.gamma_n * next_q * (1.0 - dones_t)

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
        if self.tau:
            self._soft_update_target()
        elif self.learn_steps % self.target_sync_every == 0:
            self.sync_target()

        return loss.item()

    def _soft_update_target(self):
        """Mise à jour douce (Polyak) : θ_target ← τ·θ_online + (1−τ)·θ_target.

        Appliquée à chaque pas de gradient (au lieu d'une copie brutale périodique) :
        la cible bouge en continu mais lentement, ce qui évite les sauts de la fonction
        de valeur observés avec le hard sync et stabilise l'apprentissage.
        """
        with torch.no_grad():
            for p_online, p_target in zip(self.model.parameters(), self.target_model.parameters()):
                p_target.mul_(1.0 - self.tau).add_(p_online, alpha=self.tau)

    def sync_target(self):
        """Synchronisation dure : copie tous les poids du réseau principal vers la cible.

        Appelée tous les target_sync_every pas d'apprentissage dans train_step() (seulement
        si tau est None/0), ou manuellement après load() pour initialiser la cible.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def on_episode_end(self):
        """Décroît linéairement le lr de lr_start vers lr_min sur lr_decay_episodes épisodes.

        À appeler une fois par épisode (après incrémentation de n_games). Stabilise la
        phase de plateau où un lr encore élevé fait osciller la loss sans plus améliorer
        la politique.
        """
        if self.lr_decay_episodes > 0:
            progress = min(1.0, self.n_games / float(self.lr_decay_episodes))
        else:
            progress = 1.0
        new_lr = self.lr_start - progress * (self.lr_start - self.lr_min)
        self.optimizer.param_groups[0]["lr"] = new_lr

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
          fps=TRAINING_FPS, train_every=4, n_step=3, tau=0.005, threads=None,
          eps_end=0.02, eps_decay=0.985):
    """Boucle d'entraînement complète : crée l'agent, l'environnement et exécute n_episodes.

    Sauvegarde les modèles best_model.pth (meilleur score), victory_model.pth (grille remplie),
    et last_model.pth. Écrit un journal training_log.csv avec toutes les métriques par épisode,
    ainsi qu'un config.json (hyperparamètres effectifs) en début d'entraînement, utile pour
    comparer des runs (cf. helper.compare_runs). Génère training_plot.png tous les 100 épisodes.

    Args:
        n_episodes: Nombre d'épisodes d'entraînement (défaut 600).
        render: Si True, affiche chaque partie (défaut False).
        fps: Cadence d'affichage en mode render (défaut TRAINING_FPS, clock accélérée).
            Sans effet en headless. Le mode play reste à GAME_SPEED.
        results_dir: Dossier de sortie (défaut <dossier_script>/results).
        seed: Graine pour reproductibilité (défaut None).
        log_every: Affiche le résumé tous les N épisodes dans la console (défaut 10).
        train_every: Transmis à Agent (défaut 4) : un pas de gradient tous les N pas de jeu.
        n_step: Transmis à Agent (défaut 3) : horizon des retours n-step.
        tau: Transmis à Agent (défaut 0.005) : coefficient de mise à jour douce de la cible.
        threads: Transmis à Agent (défaut None) : torch.set_num_threads si fourni.
        eps_end: Transmis à Agent (défaut 0.02) : plancher d'exploration.
        eps_decay: Transmis à Agent (défaut 0.985) : décroissance d'epsilon par épisode.

    Returns:
        dict: Résumé final avec keys : episodes, best_score, best_ratio, wins.
    """
    if results_dir is None:
        results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "training_log.csv")
    config_path = os.path.join(results_dir, "config.json")
    best_model_path = os.path.join(results_dir, "best_model.pth")
    last_model_path = os.path.join(results_dir, "last_model.pth")
    victory_model_path = os.path.join(results_dir, "victory_model.pth")
    plot_path = os.path.join(results_dir, "training_plot.png")

    env = SnakeGameRL(render=render, fps=fps)
    agent = Agent(
        seed=seed, train_every=train_every, n_step=n_step, tau=tau,
        threads=threads, eps_end=eps_end, eps_decay=eps_decay,
    )
    logger = TrainingLogger(csv_path)

    # Config effective ecrite en debut d'entrainement, pour comparer des runs a posteriori.
    config = {
        "n_episodes": n_episodes,
        "seed": seed,
        "fps": fps,
        "gamma": agent.gamma,
        "gamma_n": agent.gamma_n,
        "lr": agent.lr_start,
        "lr_min": agent.lr_min,
        "lr_decay_episodes": agent.lr_decay_episodes,
        "batch_size": agent.batch_size,
        "eps_start": agent.eps_start,
        "eps_end": agent.eps_end,
        "eps_decay": agent.eps_decay,
        "train_every": agent.train_every,
        "n_step": agent.n_step,
        "tau": agent.tau,
        "target_sync_every": agent.target_sync_every,
        "learn_start": agent.learn_start,
        "threads": agent.threads,
    }
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except OSError as exc:  # pragma: no cover - ne doit jamais casser l'entrainement
        print(f"Avertissement : ecriture de config.json a echoue ({exc}).")

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

        # Vide le tampon n-step (dernieres sous-sequences de l'episode, y compris en cas
        # d'arret anticipe) avant de passer a l'episode suivant, pour ne pas melanger
        # d'etats entre deux episodes dans le tampon n-step.
        agent.flush_n_step()
        agent.n_games += 1
        agent.on_episode_end()
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
            current_lr = agent.optimizer.param_groups[0]["lr"]
            print(
                f"Episode {agent.n_games}/{n_episodes} | score={score} | best={best_score} | "
                f"ratio={ratio:.3f} | moy50={moving_avg:.2f} | eps={agent.epsilon:.3f} | "
                f"buffer={len(agent.memory)} | loss_moy={avg_loss:.4f} | lr={current_lr:.2e} | "
                f"grad_steps={len(episode_losses)}"
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
    # train_every=1 et tau=None : isole ce test de la cadence et du Polyak (testes a part).
    test_agent = Agent(
        gamma=0.95, batch_size=16, buffer_capacity=200,
        learn_start=16, target_sync_every=5, seed=123,
        train_every=1, tau=None,
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

    # Test 5 (a) : retour n-step nominal (gamma=0.5, n=3), pas de done.
    # R attendu = 1 + 0.5*2 + 0.25*4 = 3.0, next_state = etat apres la 3e transition.
    s0 = np.zeros(12, dtype=np.float32)
    s1 = np.ones(12, dtype=np.float32)
    s2 = np.full(12, 2.0, dtype=np.float32)
    s3 = np.full(12, 3.0, dtype=np.float32)
    nstep_agent = Agent(gamma=0.5, n_step=3, batch_size=4, buffer_capacity=50,
                         learn_start=1000, seed=10)
    nstep_agent.remember(s0, 0, 1.0, s1, False)
    nstep_agent.remember(s1, 1, 2.0, s2, False)
    assert len(nstep_agent.memory) == 0, "Aucun push tant que le tampon n-step n'est pas plein"
    nstep_agent.remember(s2, 2, 4.0, s3, False)
    assert len(nstep_agent.memory) == 1, "Le tampon plein doit emettre exactement une transition"
    state_stored, action_stored, R_stored, next_state_stored, done_stored = nstep_agent.memory.tree.data[0]
    assert action_stored == 0
    assert abs(R_stored - 3.0) < 1e-6, f"R attendu 3.0, obtenu {R_stored}"
    assert np.allclose(next_state_stored, s3), "next_state doit etre l'etat apres la 3e transition"
    assert done_stored == 0.0
    print(f"Test 5a OK : n-step nominal, R={R_stored} (attendu 3.0), next_state=s3.")

    # Test 5 (b) : done au milieu -> troncature (via flush_n_step, tampon jamais plein a n_step=3).
    mid_agent = Agent(gamma=0.5, n_step=3, batch_size=4, buffer_capacity=50,
                       learn_start=1000, seed=11)
    mid_agent.remember(s0, 0, 1.0, s1, False)
    mid_agent.remember(s1, 1, 2.0, s2, True)
    assert len(mid_agent.memory) == 0, "Tampon a 2/3 : pas encore d'emission automatique"
    mid_agent.flush_n_step()
    assert len(mid_agent.memory) == 2, "flush_n_step doit emettre les 2 sous-sequences restantes"
    _, _, R0, next0, done0 = mid_agent.memory.tree.data[0]
    assert abs(R0 - 2.0) < 1e-6, f"R tronque attendu 2.0 (1 + 0.5*2), obtenu {R0}"
    assert done0 == 1.0
    assert np.allclose(next0, s2)
    _, _, R1, next1, done1 = mid_agent.memory.tree.data[1]
    assert abs(R1 - 2.0) < 1e-6, f"R (transition seule, reward=2) attendu 2.0, obtenu {R1}"
    assert done1 == 1.0
    print(f"Test 5b OK : done au milieu, R tronque={R0} (attendu 2.0), done={bool(done0)}.")

    # Test 5 (c) : flush_n_step en fin d'episode emet bien les restes non encore pousses.
    flush_agent = Agent(gamma=0.9, n_step=3, batch_size=4, buffer_capacity=50,
                         learn_start=1000, seed=12)
    flush_agent.remember(s0, 0, 1.0, s1, False)          # tampon 1/3
    flush_agent.remember(s1, 1, 1.0, s2, False)          # tampon 2/3
    flush_agent.remember(s2, 2, 1.0, s3, False)          # tampon plein -> 1 emission automatique
    assert len(flush_agent.memory) == 1
    flush_agent.remember(s3, 3, 1.0, s0, True)           # tampon plein -> 1 emission automatique
    assert len(flush_agent.memory) == 2
    flush_agent.flush_n_step()                           # vide les 2 restants
    assert len(flush_agent.memory) == 4, "flush_n_step doit emettre les sous-sequences restantes"
    assert len(flush_agent._nstep_buffer) == 0, "Le tampon n-step doit etre vide apres flush"
    print(f"Test 5c OK : flush_n_step emet les {len(flush_agent.memory)} transitions restantes en fin d'episode.")

    # Test 5 (d) : mise a jour Polyak (tau) : la cible bouge de tau vers le modele en ligne.
    tau_agent = Agent(gamma=0.95, tau=0.1, n_step=1, train_every=1, batch_size=8,
                       buffer_capacity=100, learn_start=8, seed=13)
    rng_tau = random.Random(1)
    np_rng_tau = np.random.default_rng(1)
    for _ in range(20):
        s = np_rng_tau.standard_normal(12).astype(np.float32)
        a = rng_tau.randint(0, 3)
        r = rng_tau.uniform(-1.0, 1.0)
        s2_ = np_rng_tau.standard_normal(12).astype(np.float32)
        tau_agent.remember(s, a, r, s2_, False)

    target_before = [p.clone() for p in tau_agent.target_model.parameters()]
    loss_tau = tau_agent.train_step()
    assert loss_tau is not None
    moved = False
    for p_before, p_online, p_target in zip(
        target_before, tau_agent.model.parameters(), tau_agent.target_model.parameters()
    ):
        expected = p_before * (1.0 - tau_agent.tau) + p_online.detach() * tau_agent.tau
        assert torch.allclose(p_target, expected, atol=1e-6), "La cible ne suit pas la formule Polyak"
        if not torch.allclose(p_before, p_target):
            moved = True
    assert moved, "La cible aurait du bouger vers le modele en ligne"
    print("Test 5d OK : mise a jour Polyak conforme (theta_target <- tau*theta + (1-tau)*theta_target).")

    # Test 5 (e) : train_every=4 -> exactement 2 pas de gradient sur 8 appels (buffer deja plein).
    te_agent = Agent(gamma=0.95, train_every=4, n_step=1, batch_size=8,
                      buffer_capacity=100, learn_start=8, seed=14)
    rng_te = random.Random(2)
    np_rng_te = np.random.default_rng(2)
    for _ in range(20):
        s = np_rng_te.standard_normal(12).astype(np.float32)
        a = rng_te.randint(0, 3)
        r = rng_te.uniform(-1.0, 1.0)
        s2_ = np_rng_te.standard_normal(12).astype(np.float32)
        te_agent.remember(s, a, r, s2_, False)

    n_grad_steps = sum(1 for _ in range(8) if te_agent.train_step() is not None)
    assert n_grad_steps == 2, f"train_every=4 sur 8 appels : attendu 2 pas de gradient, obtenu {n_grad_steps}"
    print(f"Test 5e OK : train_every=4 -> {n_grad_steps}/8 appels ont fait un pas de gradient.")

    print(f"HELPER_AVAILABLE = {HELPER_AVAILABLE}")
    print("Tous les tests d'agent.py sont passes avec succes.")
