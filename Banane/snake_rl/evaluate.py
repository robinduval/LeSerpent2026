"""Protocole d'évaluation, strictement séparé de l'entraînement.

Pendant une évaluation :

    epsilon = 0          aucune exploration
    torch.no_grad()      aucun gradient
    aucune écriture      rien n'entre dans le replay buffer
    seeds fixées         la même liste de seeds pour toutes les configurations

C'est la seule mesure qui a le droit de désigner un champion. Un record obtenu
pendant l'entraînement, avec exploration, ne prouve rien.
"""

import statistics
import time
from collections import Counter

from .diagnostics import EpisodeDiagnostics, aggregate_episode_metrics
from .game import SnakeGame
from .rules import RULESET
from .state import build_state


def play_episode(agent, seed, record_frames=False, max_steps_without_food=None,
                 long_without_food_threshold=100):
    """Joue une partie complète en mode greedy. Retourne un dictionnaire.

    Args:
        agent: l'agent à évaluer. Sa politique n'est pas modifiée.
        seed: graine de la partie.
        record_frames: si True, enregistre la trajectoire complète pour le
            replay. On ne le fait pas systématiquement : c'est du mémoire
            gaspillée pour les parties qu'on ne rejouera jamais.
        max_steps_without_food: garde-fou expérimental, normalement None.
    """
    game = SnakeGame(seed=seed, max_steps_without_food=max_steps_without_food)
    frames = []
    decision_time = 0.0
    decisions = 0
    diagnostics = EpisodeDiagnostics(game, long_without_food_threshold)

    while not game.done:
        state = build_state(game)
        mask = game.legal_action_mask()
        legal_q = agent.q_values(state, mask)[mask] if hasattr(agent, "q_values") else None

        started = time.perf_counter()
        action = agent.act(state, mask=mask, greedy=True)
        decision_time += time.perf_counter() - started
        decisions += 1

        if record_frames:
            frames.append(game.snapshot(action=action))

        result = game.step(action)
        next_state = build_state(game)
        loss = agent.diagnostic_td_loss(state, action, result, next_state, game.legal_action_mask()) \
            if hasattr(agent, "diagnostic_td_loss") else None
        diagnostics.observe(game, result, state=next_state, q_values=legal_q, loss=loss)

    if record_frames:
        # Image finale : elle montre la position où la partie s'est terminée.
        frames.append(game.snapshot(action=None, reward=result.reward))

    return {
        **diagnostics.summary(),
        "ruleset": RULESET,
        "seed": seed,
        "max_steps_without_food": max_steps_without_food,
        "score": game.score,
        "steps": game.steps,
        "won": game.won,
        "truncated": result.truncated,
        "terminated": result.terminated,
        "cause": result.info.get("cause"),
        "mean_decision_seconds": decision_time / max(decisions, 1),
        "frames": frames,
        "epsilon": 0.0,
        "loss_kind": "frozen_one_step_td_diagnostic",
    }


def evaluate(agent, seeds, record_best_frames=True, max_steps_without_food=None,
             long_without_food_threshold=100):
    """Évalue l'agent sur une liste de seeds et agrège les métriques.

    Les frames sont capturées pendant la partie réellement mesurée. Seuls le
    meilleur replay et l'exemple de stagnation courant sont conservés ; aucune
    politique n'est réexécutée pour reconstruire un replay.
    """
    seeds = list(seeds)
    episodes = []
    best_replay = None
    stagnation_replay = None
    for seed in seeds:
        episode = play_episode(agent, seed, record_frames=record_best_frames,
                               max_steps_without_food=max_steps_without_food,
                               long_without_food_threshold=long_without_food_threshold)
        if record_best_frames:
            if best_replay is None or (episode["score"], -episode["steps"]) > (
                best_replay["score"], -best_replay["steps"]
            ):
                best_replay = episode
            if episode["probable_cycle"] or episode["long_without_food"]:
                if stagnation_replay is None or (
                    episode["probable_cycle"], episode["longest_without_food"]
                ) > (stagnation_replay["probable_cycle"], stagnation_replay["longest_without_food"]):
                    stagnation_replay = episode
        # Les métriques n'ont pas besoin de retenir toutes les frames.
        episodes.append({key: value for key, value in episode.items() if key != "frames"})
    scores = [episode["score"] for episode in episodes]
    steps = [episode["steps"] for episode in episodes]
    best = max(episodes, key=lambda episode: (episode["score"], -episode["steps"]))

    return {
        "behavior_metrics": aggregate_episode_metrics(episodes),
        "episode_metrics": [{key: value for key, value in episode.items() if key != "frames"}
                            for episode in episodes],
        "ruleset": RULESET,
        "max_steps_without_food": max_steps_without_food,
        "termination_counts": dict(Counter(episode["cause"] for episode in episodes
                                          if episode["terminated"])),
        "truncation_count": sum(episode["truncated"] for episode in episodes),
        "episodes": len(episodes),
        "seeds": list(seeds),
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "p10_score": percentile(scores, 10),
        "p90_score": percentile(scores, 90),
        "record": max(scores),
        "min_score": min(scores),
        "win_rate": sum(episode["won"] for episode in episodes) / len(episodes),
        # Part des parties coupées par le garde-fou anti-boucle. À surveiller :
        # une valeur élevée signale une politique qui survit sans progresser.
        "truncation_rate": sum(episode["truncated"] for episode in episodes)
        / len(episodes),
        "mean_steps": statistics.fmean(steps),
        "median_steps": statistics.median(steps),
        "epsilon": 0.0,
        "mean_decision_seconds": statistics.fmean(
            episode["mean_decision_seconds"] for episode in episodes
        ),
        "scores": scores,
        "best_seed": best["seed"],
        "best_replay": best_replay,
        "stagnation_replay": stagnation_replay,
    }


def percentile(values, q):
    """Percentile par interpolation linéaire, sans dépendre de NumPy."""
    if not values:
        raise ValueError("liste vide")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (q / 100) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def is_better(candidate, incumbent):
    """Départage deux blocs d'évaluation selon le critère du cadrage (§11.3).

    Ordre : score moyen, puis médiane, puis percentile 10, puis taux de
    victoire, puis variance plus faible. Le record isolé n'intervient jamais.
    """
    if candidate.get("ruleset") != RULESET:
        return False
    if incumbent is None or incumbent.get("ruleset") != RULESET:
        return True
    key = lambda block: (  # noqa: E731
        block["mean_score"],
        block["median_score"],
        block["p10_score"],
        block["win_rate"],
        -block["std_score"],
    )
    return key(candidate) > key(incumbent)
