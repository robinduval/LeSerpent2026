"""Boucle d'entraînement headless.

    python -m snake_rl.train --episodes 500 --algorithm ddqn --run-id essai
    python -m snake_rl.train --config configs/baseline.json

Aucun affichage pendant l'entraînement : ni fenêtre, ni `clock.tick`, ni
`sleep`. La seule chose qui ouvre éventuellement une fenêtre est le replay
d'un bloc d'évaluation, et il se coupe avec `--no-window`.
"""

import argparse
import json
import os
import statistics
import time

from .agent import Agent
from .config import Config
from .evaluate import evaluate, is_better
from .game import COURSE_REWARDS, RewardProfile, SnakeGame
from .metrics import MetricsLogger, environment_info, save_replay
from .seed import set_global_seed
from .state import build_state

# Profil expérimental : garde le barème du cours mais supprime la prime de
# survie, qui peut encourager des boucles sans fin. Il ne doit jamais être
# présenté comme la règle officielle.
REWARD_PROFILES = {
    "course": COURSE_REWARDS,
    "experimental": RewardProfile(name="experimental", step=0.0),
}


class Trainer:
    """Orchestre entraînement, évaluations périodiques et checkpoints."""

    def __init__(self, config, run_dir=None, show_window=None):
        self.config = config
        self.run_dir = run_dir or os.path.join(config.output_dir, config.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.show_window = (
            config.show_replay_window if show_window is None else show_window
        )
        self.reward_profile = REWARD_PROFILES[config.reward_profile]
        self.max_steps_without_food = config.max_steps_without_food or None

        set_global_seed(config.seed)
        self.agent = Agent(config)

        self.train_scores = []
        self.record_train = 0
        self.best_eval = None
        self.best_record_eval = 0
        self.started_at = None

        config.save(os.path.join(self.run_dir, "config.json"))

    # ------------------------------------------------------------------

    def run_training_episode(self, episode_index):
        """Une partie d'entraînement : exploration active, apprentissage actif."""
        game = SnakeGame(
            # Les seeds d'entraînement tournent sur une plage distincte de
            # celles d'évaluation, pour qu'aucune partie mesurée n'ait été vue.
            seed=self.config.seed * 100_000 + episode_index,
            reward_profile=self.reward_profile,
            max_steps_without_food=self.max_steps_without_food,
        )

        total_reward = 0.0
        losses, q_values = [], []

        while not game.done:
            state = build_state(game)
            mask = game.legal_action_mask()
            action = self.agent.act(state, mask=mask)

            result = game.step(action)
            next_state = build_state(game)
            next_mask = game.legal_action_mask()

            # Une troncature n'est PAS un état terminal : l'épisode est coupé
            # par une limite de temps, pas par une règle du jeu. On garde donc
            # le bootstrap sur l'état suivant, sinon on apprendrait au réseau
            # que ces situations ne valent rien.
            terminal = result.done and not result.truncated
            self.agent.remember(
                state, action, result.reward, next_state, terminal, next_mask
            )
            total_reward += result.reward

            learned = self.agent.learn()
            if learned is not None:
                losses.append(learned[0])
                q_values.append(learned[1])

        self.agent.episodes_done += 1
        return {
            "score": game.score,
            "reward": total_reward,
            "steps": game.steps,
            "won": game.won,
            "loss_mean": statistics.fmean(losses) if losses else None,
            "q_mean": statistics.fmean(q_values) if q_values else None,
        }

    # ------------------------------------------------------------------

    def run_evaluation(self, episode_index, logger):
        """Bloc d'évaluation : modèle figé, epsilon nul, aucun apprentissage."""
        block = evaluate(
            self.agent,
            self.config.eval_seeds(),
            record_best_frames=self.config.replay_best_after_eval,
            max_steps_without_food=self.max_steps_without_food,
        )

        eval_dir = os.path.join(self.run_dir, "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        summary = {key: value for key, value in block.items() if key != "best_replay"}
        summary["episode"] = episode_index
        with open(
            os.path.join(eval_dir, f"evaluation_{episode_index:04d}.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        replay_path = None
        if block["best_replay"] is not None:
            replay_path = save_replay(
                os.path.join(
                    self.run_dir, "replays", f"best_replay_eval_{episode_index:04d}.json"
                ),
                block["best_replay"],
                self.config,
                self.config.algorithm,
                block=episode_index,
                record=max(self.best_record_eval, block["record"]),
            )

        self.best_record_eval = max(self.best_record_eval, block["record"])

        # Sélection du champion : score moyen d'abord, jamais le record.
        if is_better(block, self.best_eval):
            self.best_eval = block
            self.agent.save(
                os.path.join(self.run_dir, "best_mean.pt"),
                episode=episode_index,
                eval_mean_score=block["mean_score"],
                eval_summary=summary,
            )

        logger.log(
            run_id=self.config.run_id,
            episode=episode_index,
            phase="eval",
            eval_mean_score=block["mean_score"],
            eval_median_score=block["median_score"],
            eval_std_score=block["std_score"],
            eval_p10_score=block["p10_score"],
            eval_p90_score=block["p90_score"],
            eval_record=block["record"],
            eval_mean_steps=block["mean_steps"],
            eval_win_rate=block["win_rate"],
            eval_truncation_rate=block["truncation_rate"],
            eval_mean_decision_seconds=block["mean_decision_seconds"],
            wall_time_seconds=time.perf_counter() - self.started_at,
        )

        if self.show_window and replay_path:
            # Import tardif : pygame ne doit jamais être chargé pendant un
            # entraînement headless ou une grid search.
            from .render import replay_file

            replay_file(replay_path, speed_multiplier=self.config.replay_speed_multiplier)

        return block

    # ------------------------------------------------------------------

    def train(self, verbose=True):
        """Lance l'entraînement complet. Retourne le résumé du run."""
        cfg = self.config
        self.started_at = time.perf_counter()

        with MetricsLogger(self.run_dir) as logger:
            for episode in range(1, cfg.episodes + 1):
                stats = self.run_training_episode(episode)
                self.train_scores.append(stats["score"])
                self.record_train = max(self.record_train, stats["score"])

                logger.log(
                    run_id=cfg.run_id,
                    episode=episode,
                    phase="train",
                    train_score=stats["score"],
                    train_reward=round(stats["reward"], 3),
                    episode_steps=stats["steps"],
                    epsilon=round(self.agent.epsilon, 4),
                    loss_mean=stats["loss_mean"],
                    q_mean=stats["q_mean"],
                    replay_size=len(self.agent.memory),
                    learning_rate=cfg.learning_rate,
                    record_train=self.record_train,
                    wall_time_seconds=round(
                        time.perf_counter() - self.started_at, 3
                    ),
                )

                if cfg.eval_interval > 0 and episode % cfg.eval_interval == 0:
                    block = self.run_evaluation(episode, logger)
                    if verbose:
                        print(
                            f"[{cfg.run_id}] ep {episode:5d} "
                            f"| eps {self.agent.epsilon:.3f} "
                            f"| train moy50 {mean_tail(self.train_scores, 50):5.2f} "
                            f"| EVAL moy {block['mean_score']:5.2f} "
                            f"med {block['median_score']:4.1f} "
                            f"p10 {block['p10_score']:4.1f} "
                            f"max {block['record']:3d} "
                            f"| {episode / (time.perf_counter() - self.started_at):.1f} ep/s",
                            flush=True,
                        )

                if cfg.checkpoint_every > 0 and episode % cfg.checkpoint_every == 0:
                    self.agent.save(
                        os.path.join(self.run_dir, f"milestone_{episode:05d}.pt"),
                        episode=episode,
                    )

            self.agent.save(os.path.join(self.run_dir, "latest.pt"), episode=cfg.episodes)

        # Import tardif : matplotlib ne doit pas être chargé si on ne trace rien.
        from .plotting import plot_run

        plot_run(self.run_dir, title=f"{cfg.run_id} — {cfg.algorithm}")
        return self.write_summary()

    # ------------------------------------------------------------------

    def write_summary(self):
        """Résumé du run, suffisant pour le comparer à un autre."""
        duration = time.perf_counter() - self.started_at
        best = self.best_eval or {}
        summary = {
            "run_id": self.config.run_id,
            "algorithm": self.config.algorithm,
            "config": self.config.to_dict(),
            "episodes": self.config.episodes,
            "training_seconds": round(duration, 2),
            "episodes_per_second": round(self.config.episodes / duration, 2),
            "record_train": self.record_train,
            "train_mean_score": (
                statistics.fmean(self.train_scores) if self.train_scores else 0.0
            ),
            "train_mean_last_100": mean_tail(self.train_scores, 100),
            "best_eval_mean_score": best.get("mean_score"),
            "best_eval_median_score": best.get("median_score"),
            "best_eval_std_score": best.get("std_score"),
            "best_eval_p10_score": best.get("p10_score"),
            "best_eval_p90_score": best.get("p90_score"),
            "best_eval_record": best.get("record"),
            "best_eval_win_rate": best.get("win_rate"),
            "best_eval_truncation_rate": best.get("truncation_rate"),
            "best_eval_mean_steps": best.get("mean_steps"),
            "eval_seeds": self.config.eval_seeds(),
            "environment": environment_info(),
        }
        with open(
            os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return summary


def mean_tail(values, n):
    """Moyenne des n dernières valeurs. 0.0 si la liste est vide."""
    if not values:
        return 0.0
    return statistics.fmean(values[-n:])


# ----------------------------------------------------------------------
# Interface en ligne de commande
# ----------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description="Entraînement DQN pour Snake")
    parser.add_argument("--config", help="fichier JSON de configuration")
    parser.add_argument("--run-id")
    parser.add_argument("--algorithm", choices=["dqn", "ddqn", "dueling_ddqn",
                                                "dueling_ddqn_per"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--window",
        dest="show_window",
        action="store_true",
        default=None,
        help="ouvre Pygame après chaque bloc d'évaluation",
    )
    parser.add_argument(
        "--no-window",
        dest="show_window",
        action="store_false",
        help="n'ouvre aucune fenêtre (défaut, obligatoire en grid search)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args):
    """Fusionne le fichier de configuration et les surcharges de la ligne de commande."""
    config = Config.load(args.config) if args.config else Config()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if value is not None
        and key not in {"config", "show_window", "quiet"}
    }
    return config.replace(**overrides) if overrides else config


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    trainer = Trainer(config, show_window=args.show_window)
    summary = trainer.train(verbose=not args.quiet)

    if not args.quiet:
        print(json.dumps(
            {k: v for k, v in summary.items() if k not in {"config", "environment",
                                                           "eval_seeds"}},
            indent=2, ensure_ascii=False,
        ))
    return summary


if __name__ == "__main__":
    main()
