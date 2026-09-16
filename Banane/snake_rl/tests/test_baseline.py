"""Contrôles de la campagne DQN fixe et du diagnostic TD sans apprentissage."""

import numpy as np
import pytest
import torch

from snake_rl.agent import Agent
from snake_rl.baseline import run_baseline
from snake_rl.config import Config
from snake_rl.game import StepResult


@pytest.mark.parametrize("algorithm", ["ddqn", "dueling_ddqn", "dueling_ddqn_per"])
def test_observation_refuses_other_algorithms(tmp_path, algorithm):
    with pytest.raises(ValueError, match="uniquement DQN"):
        run_baseline(Config(algorithm=algorithm, output_dir=str(tmp_path)))
    assert list(tmp_path.iterdir()) == []


def test_observation_refuses_a_changed_reward(tmp_path):
    with pytest.raises(ValueError, match="reward du cours"):
        run_baseline(Config(reward_profile="experimental", output_dir=str(tmp_path)))


def test_td_diagnostic_keeps_bootstrap_on_truncation_without_learning():
    agent = Agent(Config(hidden_size=16, replay_capacity=10, device="cpu", gamma=0.95))
    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()
        for parameter in agent.target_net.parameters():
            parameter.zero_()
        agent.target_net.net[-1].bias.fill_(2)
    state = np.zeros(11, dtype=np.float32)
    before = {name: value.clone() for name, value in agent.policy_net.state_dict().items()}
    cutoff = StepResult(reward=0.1, score=0, truncated=True)
    death = StepResult(reward=-10, score=0, terminated=True)
    # Huber(0 - (0.1 + .95*2)) = 1.5, Huber(0 - -10) = 9.5.
    assert agent.diagnostic_td_loss(state, 3, cutoff, state, [True] * 4) == pytest.approx(1.5)
    assert agent.diagnostic_td_loss(state, 3, death, state, [True] * 4) == pytest.approx(9.5)
    assert len(agent.memory) == agent.learn_steps == 0
    assert all(torch.equal(before[name], value) for name, value in agent.policy_net.state_dict().items())
    assert all(parameter.grad is None for parameter in agent.policy_net.parameters())


@pytest.mark.parametrize("record_replays", [False, True])
def test_tiny_dqn_observation_records_initial_final_and_cycle_replay(tmp_path, record_replays):
    config = Config(run_id="observation", output_dir=str(tmp_path), device="cpu",
                    hidden_size=16, batch_size=8, learning_starts=8,
                    replay_capacity=100, episodes=2, eval_interval=2, eval_episodes=2,
                    max_steps_without_food=30, long_without_food_threshold=10,
                    replay_best_after_eval=record_replays)
    report = run_baseline(config, seeds=[0])
    assert report["algorithm"] == "dqn"
    assert report["train"]["episodes"] == 2
    assert report["initial"]["episodes"] == report["final"]["episodes"] == 2
    assert report["training_episode_seconds"] > 0
    assert (tmp_path / "observation" / "baseline_report.json").exists()
    assert (tmp_path / "observation" / "dqn_seed0" / "initial_evaluation.json").exists()
    assert report["runs"][0]["evaluation_blocks"]
    if not record_replays:
        assert report["runs"][0]["stagnation_replays"] == []
