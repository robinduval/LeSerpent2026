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

from .game import COURSE_REWARDS, SnakeGame
from .state import build_state


def play_episode(agent, seed, record_frames=False, max_steps_without_food=None):
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

    while not game.done:
        state = build_state(game)
        mask = game.legal_action_mask()

        started = time.perf_counter()
        action = agent.act(state, mask=mask, greedy=True)
        decision_time += time.perf_counter() - started
        decisions += 1

        if record_frames:
            frames.append(game.snapshot(action=action))

        result = game.step(action)

    if record_frames:
        # Image finale : elle montre la position où la partie s'est terminée.
        frames.append(game.snapshot(action=None, reward=result.reward))

    return {
        "seed": seed,
        "score": game.score,
        "steps": game.steps,
        "won": game.won,
        "truncated": result.truncated,
        "cause": result.info.get("cause"),
        "mean_decision_seconds": decision_time / max(decisions, 1),
        "frames": frames,
    }


def evaluate(agent, seeds, record_best_frames=True, max_steps_without_food=None):
    """Évalue l'agent sur une liste de seeds et agrège les métriques.

    Deux passes quand on veut le replay du meilleur épisode : une première
    passe sans enregistrement, puis on rejoue la meilleure seed en
    enregistrant. C'est déterministe (même seed, politique figée, epsilon nul),
    donc la trajectoire obtenue est exactement celle qui a produit le score
    mesuré, et on évite de garder en mémoire toutes les trajectoires du bloc.
    """
    episodes = [
        play_episode(agent, seed, max_steps_without_food=max_steps_without_food)
        for seed in seeds
    ]
    scores = [episode["score"] for episode in episodes]
    steps = [episode["steps"] for episode in episodes]

    best = max(episodes, key=lambda episode: (episode["score"], -episode["steps"]))
    best_replay = None
    if record_best_frames:
        best_replay = play_episode(
            agent,
            best["seed"],
            record_frames=True,
            max_steps_without_food=max_steps_without_food,
        )
        assert best_replay["score"] == best["score"], (
            "la rejouée doit reproduire le score mesuré ; sinon la politique "
            "ou le jeu n'est pas déterministe"
        )

    return {
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
        "mean_decision_seconds": statistics.fmean(
            episode["mean_decision_seconds"] for episode in episodes
        ),
        "scores": scores,
        "best_seed": best["seed"],
        "best_replay": best_replay,
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
    if incumbent is None:
        return True
    key = lambda block: (  # noqa: E731
        block["mean_score"],
        block["median_score"],
        block["p10_score"],
        block["win_rate"],
        -block["std_score"],
    )
    return key(candidate) > key(incumbent)
