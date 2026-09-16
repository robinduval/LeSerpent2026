"""Les expériences à murs mortels ne peuvent plus sélectionner un champion."""

import json

import pytest
import torch

from snake_rl.agent import Agent
from snake_rl.config import Config
from snake_rl.evaluate import evaluate, is_better
from snake_rl.grid_search import aggregate_groups, ranking_key, run_search, write_reports
from snake_rl.metrics import MetricsLogger
from snake_rl.rules import RULESET
from snake_rl.train import Trainer


@pytest.mark.parametrize("tag", [None, "snake-deadly-walls-v1"])
def test_legacy_checkpoint_is_refused(tmp_path, tag):
    payload = Agent(Config(hidden_size=16, replay_capacity=10, device="cpu")).state_dict()
    if tag is None:
        payload.pop("ruleset")
    else:
        payload["ruleset"] = tag
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="legacy"):
        Agent.load(path)


def test_legacy_evaluation_cannot_be_champion():
    assert not is_better({"mean_score": 999}, None)


@pytest.mark.parametrize("tag", [None, "snake-deadly-walls-v1"])
def test_legacy_search_results_are_excluded_from_ranking_and_report(tmp_path, tag):
    legacy = {"ruleset": tag, "status": "ok", "best_eval_mean_score": 999,
              "group": "legacy", "trial_id": "old"}
    current = {"ruleset": RULESET, "status": "ok", "best_eval_mean_score": 0,
               "best_eval_median_score": 0, "best_eval_p10_score": 0,
               "best_eval_record": 0, "group": "torus", "overrides": {},
               "seconds": 1, "trial_id": "new"}
    assert ranking_key(current) > ranking_key(legacy)
    assert aggregate_groups([legacy]) == []
    assert aggregate_groups([legacy, current])[0]["group"] == "torus"
    write_reports(tmp_path, [legacy, current], [], 1, {})
    report = (tmp_path / "search_report.md").read_text()
    assert "`old`" not in report
    assert "Meilleure configuration : `torus`" in report


def test_existing_run_is_preserved_and_refused(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"legacy": true}')
    with pytest.raises(ValueError, match="legacy"):
        Trainer(Config(), run_dir=str(tmp_path))
    assert json.loads(path.read_text()) == {"legacy": True}


def test_existing_search_is_preserved_and_refused(tmp_path):
    campaign = tmp_path / "legacy"
    campaign.mkdir()
    path = campaign / "search_space.json"
    path.write_text('{"legacy": true}')
    with pytest.raises(ValueError, match="legacy"):
        run_search({"name": "legacy"}, output_dir=str(tmp_path), workers=1)
    assert json.loads(path.read_text()) == {"legacy": True}


def test_metrics_cannot_append_to_legacy_rows(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"cause": "wall"}\n')
    with pytest.raises(ValueError, match="existantes"):
        MetricsLogger(str(tmp_path))
    assert path.read_text() == '{"cause": "wall"}\n'


def test_evaluation_records_only_toric_causes_and_explicit_truncation():
    class StraightAgent:
        def act(self, state, mask=None, greedy=False):
            return 3

    block = evaluate(StraightAgent(), [1000, 1001], record_best_frames=False,
                     max_steps_without_food=60)
    assert block["ruleset"] == RULESET
    assert block["termination_counts"] == {}
    assert block["truncation_count"] == 2
    assert block["max_steps_without_food"] == 60
    assert block["truncation_rate"] == 1
