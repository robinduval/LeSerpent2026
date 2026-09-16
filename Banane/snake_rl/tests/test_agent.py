"""Tests de l'agent : exploration, masque, apprentissage, checkpoints."""

import numpy as np
import pytest
import torch

from snake_rl.agent import MASKED_Q, Agent
from snake_rl.config import Config
from snake_rl.rules import N_ACTIONS
from snake_rl.state import STATE_SIZE


def tiny_config(**overrides):
    """Config minuscule : les tests doivent rester rapides et sur CPU."""
    base = dict(
        hidden_size=16,
        batch_size=8,
        learning_starts=8,
        replay_capacity=200,
        device="cpu",
        seed=0,
    )
    base.update(overrides)
    return Config(**base)


def fill_memory(agent, n=64, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        agent.remember(
            rng.random(STATE_SIZE).astype(np.float32),
            int(rng.integers(N_ACTIONS)),
            float(rng.normal()),
            rng.random(STATE_SIZE).astype(np.float32),
            bool(rng.random() < 0.1),
            np.ones(N_ACTIONS, dtype=bool),
        )


# ----------------------------------------------------------------------
# Epsilon
# ----------------------------------------------------------------------


def test_epsilon_decays_linearly_then_plateaus():
    agent = Agent(tiny_config(epsilon_decay_episodes=100))
    agent.episodes_done = 0
    assert agent.epsilon == pytest.approx(1.0)
    agent.episodes_done = 50
    assert agent.epsilon == pytest.approx(0.51)
    agent.episodes_done = 100
    assert agent.epsilon == pytest.approx(0.02)
    agent.episodes_done = 10_000
    assert agent.epsilon == pytest.approx(0.02), "epsilon ne descend plus après"


def test_greedy_mode_ignores_epsilon_entirely():
    """En évaluation, epsilon vaut 0 même si le schedule dit 1.0."""
    agent = Agent(tiny_config())
    agent.episodes_done = 0
    assert agent.epsilon == 1.0
    state = np.zeros(STATE_SIZE, dtype=np.float32)
    expected = int(agent.q_values(state).argmax())
    assert all(agent.act(state, greedy=True) == expected for _ in range(50))


# ----------------------------------------------------------------------
# Masque d'actions
# ----------------------------------------------------------------------


def test_greedy_choice_never_picks_a_masked_action():
    agent = Agent(tiny_config())
    state = np.zeros(STATE_SIZE, dtype=np.float32)
    for forbidden in range(N_ACTIONS):
        mask = np.ones(N_ACTIONS, dtype=bool)
        mask[forbidden] = False
        for _ in range(20):
            assert agent.act(state, mask=mask, greedy=True) != forbidden


def test_exploration_never_picks_a_masked_action():
    # epsilon_decay_episodes=0 fige epsilon à epsilon_end : ici 1.0, donc
    # l'agent explore à chaque décision.
    agent = Agent(tiny_config(epsilon_end=1.0, epsilon_decay_episodes=0))
    assert agent.epsilon == 1.0
    state = np.zeros(STATE_SIZE, dtype=np.float32)
    mask = np.array([True, False, False, True])
    chosen = {agent.act(state, mask=mask) for _ in range(200)}
    assert chosen <= {0, 3}
    assert len(chosen) == 2, "les deux actions légales doivent être explorées"


def test_q_values_are_damped_on_masked_actions():
    agent = Agent(tiny_config())
    mask = np.array([True, False, True, True])
    q = agent.q_values(np.zeros(STATE_SIZE, dtype=np.float32), mask=mask)
    assert q[1] == MASKED_Q
    assert all(value > MASKED_Q for i, value in enumerate(q) if i != 1)


# ----------------------------------------------------------------------
# Apprentissage
# ----------------------------------------------------------------------


def test_learn_waits_for_learning_starts():
    agent = Agent(tiny_config(learning_starts=50))
    fill_memory(agent, n=10)
    assert agent.learn() is None
    fill_memory(agent, n=60)
    assert agent.learn() is not None


def test_learn_returns_a_finite_loss_and_moves_the_weights():
    agent = Agent(tiny_config())
    fill_memory(agent, n=100)
    before = agent.policy_net.net[0].weight.clone()

    loss, q_mean = agent.learn()

    assert np.isfinite(loss) and np.isfinite(q_mean)
    assert not torch.allclose(before, agent.policy_net.net[0].weight)


def test_repeated_learning_stays_numerically_stable():
    agent = Agent(tiny_config())
    fill_memory(agent, n=200)
    losses = [agent.learn()[0] for _ in range(100)]
    assert all(np.isfinite(loss) for loss in losses)


@pytest.mark.parametrize("algorithm", ["dqn", "ddqn", "dueling_ddqn"])
def test_every_algorithm_can_learn(algorithm):
    agent = Agent(tiny_config(algorithm=algorithm))
    fill_memory(agent, n=100)
    loss, _ = agent.learn()
    assert np.isfinite(loss)


def test_target_network_never_accumulates_gradient():
    agent = Agent(tiny_config())
    fill_memory(agent, n=100)
    agent.learn()
    assert all(p.grad is None for p in agent.target_net.parameters())


# ----------------------------------------------------------------------
# Target network
# ----------------------------------------------------------------------


def test_hard_update_copies_the_policy_weights_on_schedule():
    agent = Agent(tiny_config(target_update_interval=5))
    fill_memory(agent, n=200)
    for _ in range(4):
        agent.learn()
    policy_weight = agent.policy_net.net[0].weight
    assert not torch.allclose(policy_weight, agent.target_net.net[0].weight)

    agent.learn()  # 5e mise à jour : synchronisation
    assert torch.allclose(policy_weight, agent.target_net.net[0].weight)


def test_soft_update_moves_the_target_gradually():
    agent = Agent(tiny_config(target_update_interval=0, tau=0.5))
    fill_memory(agent, n=200)
    target_before = agent.target_net.net[0].weight.clone()
    agent.learn()
    target_after = agent.target_net.net[0].weight
    assert not torch.allclose(target_before, target_after)
    assert not torch.allclose(agent.policy_net.net[0].weight, target_after)


def test_hard_and_soft_update_are_mutually_exclusive():
    with pytest.raises(ValueError):
        Config(tau=0.01, target_update_interval=500)


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def test_checkpoint_roundtrip_preserves_q_values(tmp_path):
    """Critère §23.7 : mêmes entrées, mêmes Q values après rechargement."""
    agent = Agent(tiny_config())
    fill_memory(agent, n=200)
    for _ in range(20):
        agent.learn()
    agent.episodes_done = 42

    probe = np.random.default_rng(1).random(STATE_SIZE).astype(np.float32)
    expected = agent.q_values(probe)

    path = tmp_path / "checkpoint.pt"
    agent.save(path, eval_mean_score=3.5)

    restored = Agent.load(path, device=torch.device("cpu"))
    assert np.allclose(restored.q_values(probe), expected)
    assert restored.episodes_done == 42


def test_checkpoint_embeds_the_full_configuration(tmp_path):
    agent = Agent(tiny_config(run_id="essai", algorithm="ddqn", gamma=0.97))
    path = tmp_path / "checkpoint.pt"
    agent.save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["config"]["run_id"] == "essai"
    assert payload["config"]["algorithm"] == "ddqn"
    assert payload["config"]["gamma"] == 0.97
    assert "rng" in payload, "les états RNG permettent de reprendre un run"


def test_checkpoint_carries_extra_metrics(tmp_path):
    agent = Agent(tiny_config())
    path = tmp_path / "checkpoint.pt"
    agent.save(path, eval_mean_score=12.5, episode=700)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["eval_mean_score"] == 12.5
    assert payload["episode"] == 700
