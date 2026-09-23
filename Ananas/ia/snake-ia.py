#!/usr/bin/env python3
"""Le Serpent 2026 — agent Rainbow DQN (groupe Ananas).

Lancement de la démonstration (comportement par défaut, aucun flag requis) :

    python snake-ia.py

L'agent charge `modele_final.pt` et joue seul, avec le visuel et la clock
du jeu de base. Autres modes :

    python snake-ia.py --train        # entraînement (sans rendu)
    python snake-ia.py --eval         # évaluation chiffrée sur N parties
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ananas_rl.env import SnakeEnv
from ananas_rl.agent import RainbowAgent
from ananas_rl.training import train

# Constantes reprises du moteur fourni (serpent-algo.py), non modifiées.
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 5
SCORE_PANEL_HEIGHT = 80
BLANC, NOIR = (255, 255, 255), (0, 0, 0)
ORANGE, VERT, ROUGE = (255, 165, 0), (0, 200, 0), (200, 0, 0)
GRIS_FOND, GRIS_GRILLE = (50, 50, 50), (80, 80, 80)

# Modèle figé après évaluation sur 30 seeds (33.7 pommes de moyenne) : il ne
# dépend pas des checkpoints que réécrit un entraînement en cours.
DEFAULT_CHECKPOINT = Path(__file__).parent / "modele_final.pt"


# --- Démonstration graphique -------------------------------------------------

def play(checkpoint, grid_size=GRID_SIZE, speed=1, seed=None, auto_restart=True):
    """Joue avec le rendu PyGame. `speed` accélère la partie ; le temps affiché
    est multiplié d'autant, donc il reste proportionnel à la clock du jeu."""
    import pygame

    screen_width = grid_size * CELL_SIZE
    screen_height = screen_width + SCORE_PANEL_HEIGHT

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Snake IA — Rainbow DQN (Ananas)")
    clock = pygame.time.Clock()
    font_main = pygame.font.Font(None, 40)
    font_over = pygame.font.Font(None, 80)

    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        agent = RainbowAgent.from_checkpoint(checkpoint, grid_size=grid_size)
        model_label = f"{checkpoint} (état {agent.features})"
    else:
        agent = RainbowAgent(grid_size=grid_size)
        print(f"ATTENTION : aucun modèle trouvé à {checkpoint}. "
              f"Le réseau n'est pas entraîné — lancez d'abord `python snake-ia.py --train`.")
        agent.q_net.eval()
        agent.q_net.eval_noise()
        model_label = "NON ENTRAÎNÉ"

    env = SnakeEnv(grid_size=grid_size, seed=seed)
    start = time.time()
    running, game_over_at = True, None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and env.done:
                env.reset()
                start, game_over_at = time.time(), None

        if not env.done:
            env.step(agent.choose_action(env, training=False))
            if env.done:
                game_over_at = time.time()
        elif auto_restart and game_over_at and time.time() - game_over_at > 2.0:
            env.reset()
            start, game_over_at = time.time(), None

        screen.fill(GRIS_FOND)
        pygame.draw.rect(screen, NOIR, pygame.Rect(0, SCORE_PANEL_HEIGHT, screen_width, screen_width))
        for i in range(grid_size + 1):
            x = i * CELL_SIZE
            pygame.draw.line(screen, GRIS_GRILLE, (x, SCORE_PANEL_HEIGHT), (x, screen_height))
            pygame.draw.line(screen, GRIS_GRILLE, (0, SCORE_PANEL_HEIGHT + x), (screen_width, SCORE_PANEL_HEIGHT + x))

        if env.apple:
            ax, ay = env.apple
            pygame.draw.rect(screen, ROUGE, (ax * CELL_SIZE, ay * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE))
        for i, (x, y) in enumerate(env.body):
            rect = pygame.Rect(x % grid_size * CELL_SIZE, y % grid_size * CELL_SIZE + SCORE_PANEL_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, ORANGE if i == 0 else VERT, rect)
            pygame.draw.rect(screen, NOIR, rect, 2 if i == 0 else 1)

        # Temps de jeu : le temps réel multiplié par le facteur de vitesse, pour
        # rester proportionnel à la clock du jeu de base.
        elapsed = (time.time() - start) * speed
        screen.blit(font_main.render(f"Score: {env.score}", True, BLANC), (10, 20))
        t = font_main.render(f"Temps: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}", True, BLANC)
        screen.blit(t, (screen_width - t.get_width() - 10, 20))
        fill = font_main.render(f"Remplissage: {len(env.body) / (grid_size ** 2) * 100:.1f}%", True, BLANC)
        screen.blit(fill, (screen_width // 2 - fill.get_width() // 2, 20))

        if env.done:
            victory = env.end_cause == "victoire"
            msg = font_over.render("VICTOIRE !" if victory else "GAME OVER", True, VERT if victory else ROUGE)
            rect = msg.get_rect(center=(screen_width // 2, screen_height // 2))
            pygame.draw.rect(screen, NOIR, rect.inflate(40, 40), border_radius=10)
            pygame.draw.rect(screen, BLANC, rect.inflate(40, 40), 2, border_radius=10)
            screen.blit(msg, rect)
            sub = font_main.render(f"{env.end_cause} — ESPACE pour rejouer", True, BLANC)
            screen.blit(sub, sub.get_rect(center=(screen_width // 2, screen_height // 2 + 100)))

        pygame.display.flip()
        clock.tick(GAME_SPEED * speed)

    pygame.quit()
    print(f"Modèle : {model_label} | dernier score : {env.score}")


# --- Évaluation --------------------------------------------------------------

def evaluate(checkpoint, grid_size, num_runs, seed=42, speed=1):
    """Métriques du protocole d'évaluation (cf. CLAUDE.md) sur des seeds fixes."""
    agent = RainbowAgent.from_checkpoint(checkpoint, grid_size=grid_size)

    results = []
    for i in range(num_runs):
        run_seed = seed + i
        env = SnakeEnv(grid_size=grid_size, seed=run_seed)
        decisions = []
        while not env.done:
            t0 = time.perf_counter()
            action = agent.choose_action(env, training=False)
            decisions.append(time.perf_counter() - t0)
            env.step(action)
        results.append({
            "seed": run_seed,
            "score": env.score,
            "steps": env.steps,
            "cause": env.end_cause,
            "duree_jeu_s": env.steps / GAME_SPEED,
            "decision_moy_ms": float(np.mean(decisions) * 1000),
            "decision_max_ms": float(np.max(decisions) * 1000),
            "pas_par_pomme": env.steps / env.score if env.score else None,
        })

    scores = [r["score"] for r in results]
    spp = [r["pas_par_pomme"] for r in results if r["pas_par_pomme"]]
    stats = {
        "agent": "snake-ia (Rainbow DQN)",
        "checkpoint": str(checkpoint),
        "etat": agent.features,
        "grille": grid_size,
        "parties": num_runs,
        "seeds": f"{seed}..{seed + num_runs - 1}",
        "score_moyen": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_ecart_type": float(np.std(scores)),
        "score_min": int(np.min(scores)),
        "score_max": int(np.max(scores)),
        "pas_par_pomme_median": float(np.median(spp)) if spp else None,
        "decision_moy_ms": float(np.mean([r["decision_moy_ms"] for r in results])),
        "decision_max_ms": float(np.max([r["decision_max_ms"] for r in results])),
        "taux_survie": sum(1 for r in results if r["cause"] != "collision") / num_runs,
        "collisions": sum(1 for r in results if r["cause"] == "collision"),
        "victoires": sum(1 for r in results if r["cause"] == "victoire"),
        "boucles": sum(1 for r in results if r["cause"] == "boucle"),
        "objectif_moyenne_10": bool(np.mean(scores) >= 10),
    }
    return stats, results


# --- CLI ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Le Serpent 2026 — agent Rainbow DQN (Ananas)")
    p.add_argument("--train", action="store_true", help="Entraîner l'agent (sans rendu)")
    p.add_argument("--eval", action="store_true", help="Évaluer l'agent sur plusieurs parties")

    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--capacity", type=int, default=100000)
    p.add_argument("--update_freq", type=int, default=2)
    p.add_argument("--target_update_freq", type=int, default=1000)
    p.add_argument("--epsilon_decay", type=int, default=300)
    p.add_argument("--curriculum", type=str, default=None, help="ex : '8:200,15:800'")
    p.add_argument("--demo_warmup", type=int, default=50)
    p.add_argument("--demo_refresh", type=int, default=15)
    p.add_argument("--features", choices=("v1", "v2"), default="v2",
                   help="v1 : état compact du cours ; v2 : + espace libre accessible")

    p.add_argument("--grid", type=int, default=GRID_SIZE)
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--speed", type=int, default=1, help="Accélère le jeu ; le temps affiché suit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    if args.train:
        curriculum = None
        if args.curriculum:
            curriculum = [(int(g), int(n)) for g, n in (s.split(":") for s in args.curriculum.split(","))]
        train(
            curriculum=curriculum, grid_size=args.grid, num_episodes=args.episodes,
            batch_size=args.batch_size, update_freq=args.update_freq,
            target_update_freq=args.target_update_freq, capacity=args.capacity,
            epsilon_decay=args.epsilon_decay, demo_warmup_episodes=args.demo_warmup,
            demo_refresh_every=args.demo_refresh, device=args.device,
            checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir, seed=args.seed,
            features=args.features,
        )
    elif args.eval:
        stats, results = evaluate(args.checkpoint, args.grid, args.runs, args.seed, args.speed)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        out = Path(args.log_dir) / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"stats": stats, "parties": results}, indent=2, ensure_ascii=False))
        print(f"\nDétail écrit dans {out}")
    else:
        # Comportement par défaut : la démonstration, sans aucun flag.
        play(args.checkpoint, args.grid, args.speed, args.seed)


if __name__ == "__main__":
    main()
