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
import tempfile
import time

from .agent import Agent
from .config import Config
from .diagnostics import EpisodeDiagnostics, aggregate_episode_metrics
from .evaluate import evaluate, is_better
from .game import COURSE_REWARDS, RewardProfile, SnakeGame
from .metrics import MetricsLogger, environment_info, save_replay
from .rules import RULESET
from .seed import restore_rng_state, set_global_seed
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

    def __init__(self, config, run_dir=None, show_window=None, resume=None):
        self.config = config
        self.run_dir = run_dir or os.path.join(config.output_dir, config.run_id)
        if os.path.isdir(self.run_dir) and os.listdir(self.run_dir):
            raise ValueError("run existant (potentiellement legacy) : choisir un nouvel identifiant")
        os.makedirs(self.run_dir, exist_ok=True)

        self.show_window = (
            config.show_replay_window if show_window is None else show_window
        )
        self.reward_profile = REWARD_PROFILES[config.reward_profile]
        self.max_steps_without_food = config.max_steps_without_food or None

        set_global_seed(config.seed)
        self.agent = Agent(config)

        self.train_scores = []
        self.episode_stats = []
        self.record_train = 0
        self.best_eval = None
        self.best_record_eval = 0
        self.started_at = None
        self.training_episode_seconds = 0.0
        self.elapsed_before_resume = 0.0

        if resume is not None:
            self.restore_checkpoint(resume)

        config.save(os.path.join(self.run_dir, "config.json"))

    # ------------------------------------------------------------------

    def restore_checkpoint(self, path):
        """Reprend à la frontière entre épisodes, dans un nouveau répertoire.

        Les poids/optimizer/schedule et RNG sont restaurés. La mémoire de
        replay reste vide, comme pour les checkpoints historiques.
        """
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        original = Config.from_dict(payload["config"])
        # Seuls les réglages de budget, évaluation et présentation peuvent
        # changer sans faire passer une nouvelle expérience pour une reprise.
        mutable = {"run_id", "notes", "device", "episodes", "eval_interval",
                   "replay_best_after_eval",
                   "show_replay_window", "replay_fps", "replay_speed_multiplier",
                   "long_without_food_threshold", "output_dir", "checkpoint_every"}
        changed = [key for key, value in original.to_dict().items()
                   if key not in mutable and self.config.to_dict()[key] != value]
        if changed:
            raise ValueError(f"configuration incompatible avec la reprise : {changed}")
        completed = payload.get("episode", payload.get("episodes_done", 0))
        if completed != payload.get("episodes_done", completed):
            raise ValueError("checkpoint incohérent : episode et episodes_done diffèrent")
        if self.config.episodes <= completed:
            raise ValueError("--episodes doit dépasser l'épisode du checkpoint (budget total)")
        self.agent.load_state_dict(payload)
        self.agent.episodes_done = completed
        if "agent_rng" in payload:
            self.agent._rng.bit_generator.state = payload["agent_rng"]
        if "rng" in payload:
            restore_rng_state(payload["rng"])
        state = payload.get("trainer_state", {})
        self.train_scores = list(state.get("train_scores", []))
        self.episode_stats = list(state.get("episode_stats", []))
        self.record_train = state.get("record_train", max(self.train_scores, default=0))
        self.best_eval = state.get("best_eval")
        self.best_record_eval = state.get("best_record_eval", 0)
        self.training_episode_seconds = state.get("training_episode_seconds", 0.0)
        self.elapsed_before_resume = state.get("elapsed_seconds", 0.0)
        print(f"RESUME | checkpoint {path} | next episode {completed + 1} "
              "| replay buffer empty (not bit-exact)", flush=True)

    def save_milestone(self, episode):
        """Checkpoint périodique supplémentaire, publication atomique."""
        path = os.path.join(self.run_dir, f"milestone_{episode:04d}.pt")
        trainer_state = {
            "train_scores": list(self.train_scores),
            "episode_stats": list(self.episode_stats),
            "record_train": self.record_train,
            "best_eval": None if self.best_eval is None else {
                key: value for key, value in self.best_eval.items()
                if key not in {"best_replay", "stagnation_replay"}
            },
            "best_record_eval": self.best_record_eval,
            "training_episode_seconds": self.training_episode_seconds,
            "elapsed_seconds": self.elapsed_before_resume + time.perf_counter() - self.started_at,
        }
        # Une interruption pendant l'écriture ne publie pas un milestone
        # partiel et ne touche pas aux fichiers best_mean/latest.
        with tempfile.NamedTemporaryFile(dir=self.run_dir, prefix=".milestone_",
                                         suffix=".pt", delete=False) as temporary:
            temporary_path = temporary.name
        try:
            self.agent.save(temporary_path, episode=episode, seed=self.config.seed,
                            agent_rng=self.agent._rng.bit_generator.state,
                            trainer_state=trainer_state, replay_buffer_saved=False)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        print(f"MILESTONE SAVED | episode {episode} | {path}", flush=True)
        return path

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

        diagnostics = EpisodeDiagnostics(game, self.config.long_without_food_threshold)
        episode_epsilon = self.agent.epsilon

        while not game.done:
            state = build_state(game)
            mask = game.legal_action_mask()
            legal_q = self.agent.q_values(state, mask)[mask]
            action = self.agent.act(state, mask=mask)

            result = game.step(action)
            next_state = build_state(game)
            next_mask = game.legal_action_mask()

            # Une troncature n'est PAS un état terminal : l'épisode est coupé
            # par une limite de temps, pas par une règle du jeu. On garde donc
            # le bootstrap sur l'état suivant, sinon on apprendrait au réseau
            # que ces situations ne valent rien.
            self.agent.remember(
                state, action, result.reward, next_state, result.terminated, next_mask
            )

            learned = self.agent.learn()
            diagnostics.observe(game, result, state=next_state, q_values=legal_q,
                                loss=learned[0] if learned is not None else None)

        self.agent.episodes_done += 1
        stats = diagnostics.summary()
        stats["epsilon"] = episode_epsilon
        return stats

    # ------------------------------------------------------------------

    def run_evaluation(self, episode_index, logger):
        """Bloc d'évaluation : modèle figé, epsilon nul, aucun apprentissage."""
        block = evaluate(
            self.agent,
            self.config.eval_seeds(),
            record_best_frames=self.config.replay_best_after_eval or self.show_window,
            max_steps_without_food=self.max_steps_without_food,
            long_without_food_threshold=self.config.long_without_food_threshold,
        )

        eval_dir = os.path.join(self.run_dir, "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        summary = {key: value for key, value in block.items()
                   if key not in {"best_replay", "stagnation_replay"}}
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

        if block["stagnation_replay"] is not None:
            save_replay(
                os.path.join(self.run_dir, "replays", f"stagnation_eval_{episode_index:04d}.json"),
                block["stagnation_replay"], self.config, self.config.algorithm,
                block=episode_index, record=self.best_record_eval,
            )

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
            **{f"eval_{key}": value for key, value in block["behavior_metrics"].items()
               if key not in {"mean_score", "median_score", "std_score", "p10_score", "p90_score",
                              "record", "mean_steps", "win_rate", "truncation_rate"}},
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
            termination_counts=block["termination_counts"],
            max_steps_without_food=self.max_steps_without_food,
            epsilon=0.0,
            loss_kind="frozen_one_step_td_diagnostic",
            wall_time_seconds=self.elapsed_before_resume + time.perf_counter() - self.started_at,
        )

        if self.show_window and replay_path:
            # Import tardif : pygame ne doit jamais être chargé pendant un
            # entraînement headless ou une grid search.
            from .render import replay_file

            best = block["best_replay"]
            print(f"REPLAY BEST EVAL | episode {episode_index} | score {best['score']} | seed {best['seed']}",
                  flush=True)
            replay_file(replay_path, fps=self.config.replay_fps,
                        close_when_done=True, linger_seconds=0)

        return block

    # ------------------------------------------------------------------

    def train(self, verbose=True):
        """Lance l'entraînement complet. Retourne le résumé du run."""
        cfg = self.config
        self.started_at = time.perf_counter()

        with MetricsLogger(self.run_dir) as logger:
            first_episode = self.agent.episodes_done + 1
            for episode in range(first_episode, cfg.episodes + 1):
                episode_started = time.perf_counter()
                stats = self.run_training_episode(episode)
                self.training_episode_seconds += time.perf_counter() - episode_started
                self.episode_stats.append(stats)
                self.train_scores.append(stats["score"])
                self.record_train = max(self.record_train, stats["score"])

                logger.log(
                    **{key: value for key, value in stats.items()
                       if key not in {"score", "reward", "steps", "cause", "truncated", "loss_mean", "q_mean", "epsilon"}},
                    run_id=cfg.run_id,
                    episode=episode,
                    phase="train",
                    train_score=stats["score"],
                    train_reward=round(stats["reward"], 3),
                    episode_steps=stats["steps"],
                    cause=stats["cause"],
                    truncated=stats["truncated"],
                    max_steps_without_food=self.max_steps_without_food,
                    epsilon=round(stats["epsilon"], 4),
                    loss_mean=stats["loss_mean"],
                    q_mean=stats["q_mean"],
                    loss_kind="training_replay_batch",
                    truncations_so_far=sum(row["truncated"] for row in self.episode_stats),
                    truncation_rate_so_far=sum(row["truncated"] for row in self.episode_stats) / len(self.episode_stats),
                    replay_size=len(self.agent.memory),
                    learning_rate=cfg.learning_rate,
                    record_train=self.record_train,
                    wall_time_seconds=round(
                        self.elapsed_before_resume + time.perf_counter() - self.started_at, 3
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
                            f"| {episode / (self.elapsed_before_resume + time.perf_counter() - self.started_at):.1f} ep/s",
                            flush=True,
                        )

                if cfg.checkpoint_every > 0 and episode % cfg.checkpoint_every == 0:
                    self.save_milestone(episode)

            self.agent.save(os.path.join(self.run_dir, "latest.pt"), episode=cfg.episodes)

        # Import tardif : matplotlib ne doit pas être chargé si on ne trace rien.
        from .plotting import plot_run

        plot_run(self.run_dir, title=f"{cfg.run_id} — {cfg.algorithm}")
        return self.write_summary()

    # ------------------------------------------------------------------

    def write_summary(self):
        """Résumé du run, suffisant pour le comparer à un autre."""
        duration = self.elapsed_before_resume + time.perf_counter() - self.started_at
        best = self.best_eval or {}
        summary = {
            "ruleset": RULESET,
            "run_id": self.config.run_id,
            "algorithm": self.config.algorithm,
            "config": self.config.to_dict(),
            "episodes": self.config.episodes,
            "training_seconds": round(duration, 2),
            "training_episode_seconds": self.training_episode_seconds,
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
            "train_behavior_metrics": aggregate_episode_metrics(self.episode_stats),
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
    parser.add_argument("--resume", help="checkpoint de reprise (utiliser un nouveau run-id)")
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
    parser.add_argument("--checkpoint-interval", type=int, dest="checkpoint_every",
                        help="milestone tous les N épisodes ; 0 désactive (défaut config : 500)")
    parser.add_argument("--replay-fps", type=int, help="cadence visuelle du replay (défaut : 60 FPS)")
    windows = parser.add_mutually_exclusive_group()
    windows.add_argument(
        "--window",
        dest="show_window",
        action="store_true",
        default=True,
        help="ouvre Pygame après chaque bloc d'évaluation",
    )
    windows.add_argument(
        "--no-window",
        dest="show_window",
        action="store_false",
        help="désactive les fenêtres de replay (obligatoire en grid search CLI)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args):
    """Fusionne le fichier de configuration et les surcharges de la ligne de commande."""
    if args.config:
        config = Config.load(args.config)
    elif args.resume:
        import torch
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = Config.from_dict(payload["config"])
    else:
        config = Config()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if value is not None
        and key not in {"config", "show_window", "quiet", "resume"}
    }
    # --no-window est l'interrupteur CLI, même pour un ancien JSON headless.
    return config.replace(**overrides, show_replay_window=args.show_window)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    options = {"show_window": args.show_window}
    if args.resume:
        options["resume"] = args.resume
    trainer = Trainer(config, **options)
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
