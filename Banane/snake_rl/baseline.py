"""Observation DQN multi-seeds à configuration fixe, sans grid search."""

import argparse
import json
import os
import time

from .config import Config
from .diagnostics import aggregate_episode_metrics
from .evaluate import evaluate
from .metrics import environment_info
from .rules import RULESET
from .train import Trainer


def run_baseline(config, seeds=(0, 1, 2), report_path=None):
    if config.algorithm != "dqn" or config.reward_profile != "course":
        raise ValueError("cette observation utilise uniquement DQN et le reward du cours")
    if config.max_steps_without_food < 1:
        raise ValueError("déclarer une limite expérimentale pour cette observation bornée")
    if config.eval_interval < 1 or config.episodes < 1 or config.episodes % config.eval_interval:
        raise ValueError("budget positif divisible par l'intervalle d'évaluation")
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("fournir des seeds distinctes")
    campaign_dir = os.path.join(config.output_dir, config.run_id)
    if os.path.isdir(campaign_dir) and os.listdir(campaign_dir):
        raise ValueError("campagne existante : utiliser un nouveau run_id")
    os.makedirs(campaign_dir, exist_ok=True)
    started = time.perf_counter()
    results, initial_episodes, final_episodes, training_episodes = [], [], [], []
    for seed in seeds:
        run_config = config.replace(run_id=f"dqn_seed{seed}", seed=seed, show_replay_window=False)
        run_dir = os.path.join(campaign_dir, run_config.run_id)
        trainer = Trainer(run_config, run_dir=run_dir, show_window=False)
        initial = evaluate(trainer.agent, config.eval_seeds(), record_best_frames=False,
                           max_steps_without_food=config.max_steps_without_food,
                           long_without_food_threshold=config.long_without_food_threshold)
        with open(os.path.join(run_dir, "initial_evaluation.json"), "w", encoding="utf-8") as handle:
            json.dump(initial, handle, indent=2)
        print(f"SEED {seed} initial score={initial['mean_score']:.3f} "
              f"truncation={initial['truncation_rate']:.0%}", flush=True)
        summary = trainer.train(verbose=True)
        with open(os.path.join(run_dir, "evaluations", f"evaluation_{config.episodes:04d}.json"),
                  encoding="utf-8") as handle:
            final = json.load(handle)
        # Dernier bloc, pas meilleur checkpoint choisi a posteriori.
        initial_episodes.extend(initial["episode_metrics"])
        final_episodes.extend(final["episode_metrics"])
        training_episodes.extend(trainer.episode_stats)
        replay_dir = os.path.join(run_dir, "replays")
        replay_files = sorted(os.listdir(replay_dir)) if os.path.isdir(replay_dir) else []
        results.append({"seed": seed, "run_dir": run_dir,
                        "training_seconds": summary["training_seconds"],
                        "training_episode_seconds": summary["training_episode_seconds"],
                        "initial": initial["behavior_metrics"],
                        "final": final["behavior_metrics"],
                        "train": summary["train_behavior_metrics"],
                        "evaluation_blocks": [os.path.join(run_dir, "evaluations", filename)
                                              for filename in sorted(os.listdir(os.path.join(run_dir, "evaluations")))],
                        "stagnation_replays": [os.path.join(replay_dir, filename)
                                               for filename in replay_files
                                               if filename.startswith("stagnation")]})
    report = {
        "ruleset": RULESET, "algorithm": "dqn", "training_seeds": seeds,
        "config": config.to_dict(), "environment": environment_info(),
        "campaign_seconds": time.perf_counter() - started,
        "training_seconds_including_periodic_evaluation_and_plot": sum(row["training_seconds"] for row in results),
        "training_episode_seconds": sum(row["training_episode_seconds"] for row in results),
        "initial": aggregate_episode_metrics(initial_episodes),
        "final": aggregate_episode_metrics(final_episodes),
        "train": aggregate_episode_metrics(training_episodes),
        "runs": results,
        "selection": "final policy of every seed; no best-seed or best-block selection",
    }
    for path in [os.path.join(campaign_dir, "baseline_report.json"), report_path]:
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("initial", "final", "train", "campaign_seconds")}, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dqn_torus_short.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    return run_baseline(Config.load(args.config), args.seeds, args.report)


if __name__ == "__main__":
    main()
