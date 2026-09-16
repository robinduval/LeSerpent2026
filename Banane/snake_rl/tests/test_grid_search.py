"""Tests de l'outil de recherche d'hyperparamètres."""

import os

import pytest

from snake_rl.grid_search import (
    aggregate_groups,
    build_trials,
    expand_grid,
    ranking_key,
    run_search,
)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# ----------------------------------------------------------------------
# Expansion de la grille
# ----------------------------------------------------------------------


def test_expand_grid_produces_the_cartesian_product():
    combinations = expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combinations) == 6
    assert {"a": 1, "b": "x"} in combinations
    assert {"a": 2, "b": "z"} in combinations


def test_expand_empty_grid_gives_one_default_trial():
    assert expand_grid({}) == [{}]


def test_trials_cross_every_combination_with_every_seed():
    trials = build_trials(
        {
            "base": {"episodes": 10},
            "grid": {"learning_rate": [0.001, 0.0003]},
            "seeds": [0, 1, 2],
        }
    )
    assert len(trials) == 6
    assert len({trial["trial_id"] for trial in trials}) == 6
    assert {trial["config"].seed for trial in trials} == {0, 1, 2}


def test_every_trial_shares_the_same_budget_and_eval_seeds():
    """Comparaison équitable : même nombre d'épisodes, mêmes seeds d'évaluation."""
    trials = build_trials(
        {
            "base": {"episodes": 250, "eval_episodes": 20},
            "grid": {"algorithm": ["dqn", "ddqn"], "gamma": [0.9, 0.99]},
            "seeds": [0, 1],
        }
    )
    assert len({trial["config"].episodes for trial in trials}) == 1
    assert len({tuple(trial["config"].eval_seeds()) for trial in trials}) == 1


def test_trials_never_open_a_window():
    trials = build_trials(
        {"base": {"show_replay_window": True}, "grid": {"gamma": [0.9]}}
    )
    assert all(trial["config"].show_replay_window is False for trial in trials)


def test_overrides_are_applied_to_the_config():
    trials = build_trials(
        {"base": {"episodes": 10}, "grid": {"gamma": [0.87]}, "seeds": [0]}
    )
    assert trials[0]["config"].gamma == 0.87
    assert trials[0]["overrides"] == {"gamma": 0.87}


def test_an_invalid_grid_value_is_refused_at_build_time():
    """Mieux vaut échouer tout de suite qu'après une heure de calcul."""
    with pytest.raises(ValueError):
        build_trials({"grid": {"algorithm": ["rainbow"]}, "seeds": [0]})


# ----------------------------------------------------------------------
# Classement
# ----------------------------------------------------------------------


def summary(mean, p10=0.0, median=0.0, std=0.0, status="ok"):
    return {
        "status": status,
        "best_eval_mean_score": mean,
        "best_eval_p10_score": p10,
        "best_eval_median_score": median,
        "best_eval_std_score": std,
    }


def test_ranking_prefers_the_higher_mean():
    assert ranking_key(summary(10.0)) > ranking_key(summary(9.0))


def test_ranking_falls_back_to_p10_then_median():
    assert ranking_key(summary(10.0, p10=5.0)) > ranking_key(summary(10.0, p10=1.0))
    assert ranking_key(summary(10.0, 5.0, median=9.0)) > ranking_key(
        summary(10.0, 5.0, median=4.0)
    )


def test_ranking_prefers_lower_variance_on_a_full_tie():
    assert ranking_key(summary(10.0, 5.0, 9.0, std=1.0)) > ranking_key(
        summary(10.0, 5.0, 9.0, std=6.0)
    )


def test_failed_trials_rank_last():
    assert ranking_key(summary(0.0)) > ranking_key(summary(99.0, status="failed"))


# ----------------------------------------------------------------------
# Agrégation par configuration
# ----------------------------------------------------------------------


def test_groups_average_across_seeds_not_best_seed():
    """Une configuration se juge sur sa moyenne, pas sur sa seed chanceuse."""
    summaries = [
        {"group": "a", "overrides": {}, "best_eval_mean_score": 20.0,
         "best_eval_p10_score": 0.0, "best_eval_record": 30, "seconds": 1.0},
        {"group": "a", "overrides": {}, "best_eval_mean_score": 0.0,
         "best_eval_p10_score": 0.0, "best_eval_record": 1, "seconds": 1.0},
        {"group": "b", "overrides": {}, "best_eval_mean_score": 11.0,
         "best_eval_p10_score": 0.0, "best_eval_record": 12, "seconds": 1.0},
        {"group": "b", "overrides": {}, "best_eval_mean_score": 9.0,
         "best_eval_p10_score": 0.0, "best_eval_record": 10, "seconds": 1.0},
    ]
    groups = aggregate_groups(summaries)
    assert groups[0]["group"] == "b", "b est plus faible au pic mais plus régulier"
    assert groups[0]["mean_of_eval_means"] == 10.0
    assert groups[1]["spread"] == 20.0


# ----------------------------------------------------------------------
# Campagne complète
# ----------------------------------------------------------------------


def test_a_small_search_produces_all_its_artefacts(tmp_path):
    result = run_search(
        {
            "name": "mini",
            "base": {
                "episodes": 6, "eval_interval": 6, "eval_episodes": 3,
                "hidden_size": 16, "batch_size": 8, "learning_starts": 8,
                "device": "cpu", "checkpoint_every": 0,
            },
            "grid": {"gamma": [0.9, 0.95]},
            "seeds": [0],
        },
        output_dir=str(tmp_path),
        workers=1,
    )

    search_dir = tmp_path / "mini"
    assert (search_dir / "search_summary.csv").exists()
    assert (search_dir / "search_report.md").exists()
    assert (search_dir / "search_space.json").exists()

    assert len(result["summaries"]) == 2
    assert all(s["status"] == "ok" for s in result["summaries"])
    # Un répertoire par trial, avec sa configuration et ses métriques.
    for trial in result["summaries"]:
        trial_dir = search_dir / trial["trial_id"]
        assert (trial_dir / "config.json").exists()
        assert (trial_dir / "metrics.csv").exists()
        assert (trial_dir / "summary.json").exists()

    report = (search_dir / "search_report.md").read_text()
    assert "Décision pour le round suivant" in report
