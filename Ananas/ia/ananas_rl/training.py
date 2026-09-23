"""Entraînement du Rainbow DQN avec curriculum, démonstrations SafePath et courbe live."""
import numpy as np
import json
import torch
from pathlib import Path

from .env import SnakeEnv
from .replay import PrioritizedReplayBuffer
from .agent import RainbowAgent
from . import safepath

# Sur CPU, les tenseurs manipulés ici sont petits (batch <= 64, grille <= 15x15) :
# le multi-threading par défaut de PyTorch synchronise plus qu'il n'accélère.
# Mesuré : ~1.8x plus rapide avec 2 threads qu'avec les 12 threads par défaut.
torch.set_num_threads(2)


def run_safepath_episode(env, replay, gamma, encode):
    """Joue un épisode entier avec le planificateur SafePath et stocke les
    transitions dans le replay. Sert à amorcer l'apprentissage : sans cela,
    sur une grille 15x15 (225 cases), une politique quasi aléatoire met très
    longtemps à trouver ne serait-ce qu'une pomme."""
    env.reset()
    state = encode(env)
    while not env.done:
        action = safepath.choose_action(env)
        reward, done = env.step(action, gamma=gamma)
        next_state = encode(env)
        replay.add(state, action, reward, next_state, done, env.truncated())
        state = next_state
    return env.score


def plot_learning_curve(history, out_path):
    """Écrit une courbe PNG : score par épisode + moyenne glissante."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    episodes = history["episode"]
    scores = history["score"]
    if not episodes:
        return

    window = min(30, max(1, len(scores) // 10))
    rolling = np.convolve(scores, np.ones(window) / window, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.scatter(episodes, scores, s=6, alpha=0.3, color="#4C72B0", label="Score par partie")
    ax1.plot(episodes[window - 1:], rolling, color="#C44E52", linewidth=2, label=f"Moyenne glissante ({window})")
    ax1.axhline(10, color="gray", linestyle="--", linewidth=1, label="Objectif (10 pommes)")
    ax1.set_ylabel("Score (pommes)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"Apprentissage Rainbow DQN — {len(scores)} parties jouées")
    ax1.grid(alpha=0.3)

    if any(g is not None for g in history.get("grid_size", [])):
        ax2.plot(episodes, history["grid_size"], color="#55A868", linewidth=1.5)
        ax2.set_ylabel("Taille de grille (curriculum)")
        ax2.set_xlabel("Épisode")
        ax2.grid(alpha=0.3)
    else:
        ax2.plot(episodes, history["epsilon"], color="#8172B2", linewidth=1.5)
        ax2.set_ylabel("Epsilon")
        ax2.set_xlabel("Épisode")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def train(
    curriculum=None,
    grid_size=15,
    num_episodes=500,
    batch_size=32,
    update_freq=4,
    target_update_freq=1000,
    capacity=50000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=500,
    demo_warmup_episodes=40,
    demo_refresh_every=20,
    device="cpu",
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    seed=42,
    gamma=0.99,
    plot_every=10,
    features="v2",
):
    """Entraîne l'agent Rainbow DQN.

    curriculum : liste de (grid_size, num_episodes) pour l'entraînement progressif
    (ex: [(8, 150), (15, 450)]). Si None, entraîne directement sur `grid_size`
    pendant `num_episodes`. Le réseau est indépendant de la taille de grille
    (pooling adaptatif), donc les poids se transfèrent d'une étape à l'autre.
    """
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if curriculum is None:
        curriculum = [(grid_size, num_episodes)]

    final_grid = curriculum[-1][0]
    np.random.seed(seed)
    agent = RainbowAgent(grid_size=final_grid, device=device, gamma=gamma, features=features)
    encode = agent.encode
    # Un replay buffer par taille de grille : les observations de tailles
    # différentes n'ont pas la même forme et ne peuvent pas être empilées dans
    # un même mini-batch. Le réseau (pooling adaptatif) reste, lui, partagé
    # d'une étape de curriculum à l'autre.
    replay = None

    history = {
        "episode": [], "reward": [], "score": [], "steps": [],
        "loss": [], "epsilon": [], "grid_size": [],
    }

    total_steps = 0
    total_episodes = 0
    best_score = 0
    best_mean_50 = -1.0

    for stage_idx, (stage_grid, stage_episodes) in enumerate(curriculum):
        print(f"=== Étape {stage_idx + 1}/{len(curriculum)} : grille {stage_grid}x{stage_grid}, {stage_episodes} épisodes ===")
        env = SnakeEnv(grid_size=stage_grid, seed=seed + stage_idx * 100000)
        replay = PrioritizedReplayBuffer(
            capacity=capacity, alpha=0.6, beta=0.4,
            rng=np.random.RandomState(seed + stage_idx * 100000),
        )

        # Amorçage : quelques parties jouées par SafePath pour peupler le replay
        # de transitions utiles (le planificateur atteint ~100 pommes en moyenne
        # sur 15x15 ; sans ça l'exploration aléatoire met très longtemps à
        # trouver ne serait-ce qu'une pomme sur une grande grille).
        if demo_warmup_episodes > 0:
            demo_scores = [run_safepath_episode(env, replay, gamma, agent.encode) for _ in range(demo_warmup_episodes)]
            print(f"  Amorçage SafePath : {demo_warmup_episodes} parties, score moyen {np.mean(demo_scores):.1f}")

        for local_ep in range(stage_episodes):
            episode = total_episodes
            total_episodes += 1

            # Réinjecte régulièrement une démonstration fraîche : garde de bonnes
            # trajectoires dans le replay tout au long de l'entraînement, pas
            # seulement au début.
            if demo_refresh_every > 0 and local_ep % demo_refresh_every == 0 and local_ep > 0:
                run_safepath_episode(env, replay, gamma, agent.encode)

            env.reset()
            state_arr = encode(env)
            episode_reward = 0.0
            episode_loss = []
            epsilon = max(epsilon_end, epsilon_start - (local_ep / epsilon_decay))

            while not env.done:
                action = agent.choose_action(env, training=True, epsilon=epsilon, obs=state_arr)
                reward, done = env.step(action, gamma=gamma)
                next_state_arr = encode(env)
                truncated = env.truncated()

                replay.add(state_arr, action, reward, next_state_arr, done, truncated)
                episode_reward += reward
                total_steps += 1

                if total_steps % update_freq == 0 and total_steps > batch_size:
                    batch = replay.sample(batch_size)
                    tds = agent.update(batch[:6], batch[6])
                    replay.update_priorities(batch[7], tds)
                    episode_loss.append(tds.mean())

                if total_steps % target_update_freq == 0:
                    agent.sync_target()

                state_arr = next_state_arr

            avg_loss = float(np.mean(episode_loss)) if episode_loss else 0.0
            history["episode"].append(int(episode))
            history["reward"].append(float(episode_reward))
            history["score"].append(int(env.score))
            history["steps"].append(int(env.steps))
            history["loss"].append(avg_loss)
            history["epsilon"].append(float(epsilon))
            history["grid_size"].append(int(stage_grid))

            if env.score > best_score:
                best_score = env.score
                agent.save(checkpoint_dir / "best.pt")

            mean_50 = float(np.mean(history["score"][-50:]))
            if len(history["score"]) >= 20 and mean_50 > best_mean_50:
                best_mean_50 = mean_50
                agent.save(checkpoint_dir / "best_mean.pt")

            if (episode + 1) % 25 == 0:
                print(f"  Ep {episode + 1} (grille {stage_grid}) : score={env.score}, moyenne50={mean_50:.2f}, "
                      f"meilleur={best_score}, eps={epsilon:.3f}")

            if (episode + 1) % plot_every == 0:
                with open(log_dir / "history_live.json", "w") as f:
                    json.dump(history, f)
                plot_learning_curve(history, checkpoint_dir / "learning_curve.png")

            if (episode + 1) % 50 == 0:
                agent.save(checkpoint_dir / f"checkpoint_{episode + 1}.pt")

    agent.save(checkpoint_dir / "final.pt")
    with open(log_dir / "history_final.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_learning_curve(history, checkpoint_dir / "learning_curve.png")

    return agent, history
