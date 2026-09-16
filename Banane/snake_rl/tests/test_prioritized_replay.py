"""Tests du Prioritized Experience Replay.

Un PER faux est silencieux : il n'échoue jamais, il apprend simplement moins
bien qu'un buffer uniforme. Ces tests vérifient donc les propriétés qu'on ne
verrait pas passer sur une courbe.
"""

import numpy as np
import pytest

from snake_rl.config import Config
from snake_rl.prioritized_replay import (
    PrioritizedReplayBuffer,
    SumTree,
    build_replay_buffer,
)
from snake_rl.replay_buffer import ReplayBuffer
from snake_rl.rules import N_ACTIONS
from snake_rl.state import STATE_SIZE


# ----------------------------------------------------------------------
# Arbre de sommes
# ----------------------------------------------------------------------


def test_total_is_the_sum_of_the_leaves():
    tree = SumTree(8)
    for index, priority in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.update(index, priority)
    assert tree.total == pytest.approx(10.0)


def test_find_returns_the_leaf_owning_the_interval():
    tree = SumTree(4)
    for index, priority in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.update(index, priority)
    # Intervalles cumulés : [0,1) -> 0, [1,3) -> 1, [3,6) -> 2, [6,10) -> 3.
    assert tree.find(0.5) == 0
    assert tree.find(2.0) == 1
    assert tree.find(4.0) == 2
    assert tree.find(9.9) == 3


def test_updating_a_leaf_propagates_to_the_root():
    tree = SumTree(4)
    tree.update(0, 5.0)
    assert tree.total == pytest.approx(5.0)
    tree.update(0, 1.0)
    assert tree.total == pytest.approx(1.0)


def test_sampling_follows_the_priority_distribution():
    """Une priorité neuf fois plus grande doit sortir environ neuf fois plus."""
    tree = SumTree(4)
    tree.update(0, 9.0)
    tree.update(1, 1.0)
    rng = np.random.default_rng(0)
    draws = [tree.find(rng.uniform(0, tree.total)) for _ in range(4000)]
    share = draws.count(0) / len(draws)
    assert 0.85 < share < 0.95


# ----------------------------------------------------------------------
# Buffer prioritaire
# ----------------------------------------------------------------------


def transition(value=0.0, action=0, reward=1.0, done=False):
    state = np.full(STATE_SIZE, value, dtype=np.float32)
    return (state, action, reward, state, done, np.ones(N_ACTIONS, dtype=bool))


def filled_buffer(n=32, **kwargs):
    buffer = PrioritizedReplayBuffer(capacity=64, seed=0, **kwargs)
    for i in range(n):
        buffer.push(*transition(value=float(i), action=i % N_ACTIONS))
    return buffer


def test_length_and_capacity_behave_like_the_uniform_buffer():
    buffer = filled_buffer(10)
    assert len(buffer) == 10
    assert not buffer.is_full


def test_the_buffer_overwrites_the_oldest_transitions():
    buffer = PrioritizedReplayBuffer(capacity=4, seed=0)
    for i in range(6):
        buffer.push(*transition(value=float(i)))
    assert len(buffer) == 4
    assert buffer.is_full
    # Les deux premières transitions ont été écrasées par les deux dernières.
    assert set(buffer.states[:, 0]) == {4.0, 5.0, 2.0, 3.0}


def test_a_new_transition_gets_the_maximal_priority():
    """Une transition jamais rejouée doit avoir sa chance au moins une fois."""
    buffer = filled_buffer(8)
    buffer.update_priorities(np.arange(8), np.full(8, 0.001))
    buffer.push(*transition(value=99.0))
    leaves = buffer._tree.tree[buffer.capacity:]
    assert leaves[8] == pytest.approx(leaves.max())
    assert leaves[8] > leaves[0]


def test_sample_returns_indices_and_importance_weights():
    buffer = filled_buffer(32)
    batch = buffer.sample(8)
    assert len(batch) == 8  # 6 tenseurs + indices + poids
    states, actions, rewards, next_states, dones, masks, indices, weights = batch
    assert states.shape == (8, STATE_SIZE)
    assert masks.shape == (8, N_ACTIONS)
    assert indices.shape == (8,)
    assert weights.shape == (8,)
    assert indices.min() >= 0 and indices.max() < len(buffer)


def test_importance_weights_are_normalised_to_one():
    """Poids dans (0, 1] : ils réduisent le pas, ils ne l'amplifient jamais."""
    buffer = filled_buffer(32)
    buffer.update_priorities(np.arange(32), np.linspace(0.1, 10.0, 32))
    weights = buffer.sample(16)[-1].numpy()
    assert weights.max() == pytest.approx(1.0)
    assert weights.min() > 0.0


def test_the_most_surprising_transition_is_replayed_most_often():
    """Propriété centrale du PER, mesurée et pas seulement supposée."""
    buffer = filled_buffer(32, alpha=0.6)
    errors = np.full(32, 0.01)
    errors[7] = 100.0  # une transition très mal prédite
    buffer.update_priorities(np.arange(32), errors)

    counts = np.zeros(32, dtype=int)
    for _ in range(50):
        indices = buffer.sample(16)[-2]
        counts += np.bincount(indices, minlength=32)
    assert counts.argmax() == 7
    assert counts[7] > 4 * np.median(counts)


def test_a_zero_td_error_never_freezes_a_transition_forever():
    """priority_epsilon garantit une probabilité non nulle."""
    buffer = filled_buffer(4)
    buffer.update_priorities(np.arange(4), np.zeros(4))
    assert buffer._tree.total > 0.0
    indices = buffer.sample(64)[-2]
    assert set(np.unique(indices)) == {0, 1, 2, 3}


def test_alpha_zero_degenerates_into_uniform_sampling():
    """alpha = 0 doit annuler la priorisation : garde-fou sur l'exposant."""
    buffer = filled_buffer(16, alpha=0.0)
    buffer.update_priorities(np.arange(16), np.linspace(0.1, 50.0, 16))
    leaves = buffer._tree.tree[buffer.capacity: buffer.capacity + 16]
    assert leaves.std() == pytest.approx(0.0, abs=1e-9)


def test_beta_climbs_from_its_start_to_its_end():
    buffer = PrioritizedReplayBuffer(
        capacity=16, seed=0, beta_start=0.4, beta_end=1.0, beta_steps=10
    )
    for i in range(4):
        buffer.push(*transition(value=float(i)))
    assert buffer.beta == pytest.approx(0.4)
    for _ in range(10):
        buffer.sample(2)
    assert buffer.beta == pytest.approx(1.0)
    buffer.sample(2)
    assert buffer.beta == pytest.approx(1.0), "beta ne doit pas dépasser beta_end"


def test_sampling_an_empty_buffer_is_an_error_not_a_silent_batch():
    with pytest.raises(ValueError):
        PrioritizedReplayBuffer(capacity=8).sample(4)


# ----------------------------------------------------------------------
# Sélection depuis la configuration
# ----------------------------------------------------------------------


def test_the_algorithm_decides_which_memory_is_built():
    assert isinstance(build_replay_buffer(Config(algorithm="ddqn")), ReplayBuffer)
    assert isinstance(
        build_replay_buffer(Config(algorithm="dueling_ddqn_per")),
        PrioritizedReplayBuffer,
    )


def test_per_hyperparameters_reach_the_buffer():
    buffer = build_replay_buffer(
        Config(algorithm="dueling_ddqn_per", per_alpha=0.3, per_beta_start=0.7)
    )
    assert buffer.alpha == 0.3
    assert buffer.beta == pytest.approx(0.7)


# ----------------------------------------------------------------------
# Intégration avec l'agent
# ----------------------------------------------------------------------


def per_config(**overrides):
    base = dict(
        algorithm="dueling_ddqn_per",
        hidden_size=16,
        batch_size=8,
        learning_starts=8,
        replay_capacity=64,
        device="cpu",
        seed=0,
    )
    base.update(overrides)
    return Config(**base)


def test_the_agent_learns_through_the_prioritized_memory():
    from snake_rl.agent import Agent

    agent = Agent(per_config())
    assert isinstance(agent.memory, PrioritizedReplayBuffer)
    for i in range(32):
        agent.remember(*transition(value=float(i), action=i % N_ACTIONS))

    result = agent.learn()
    assert result is not None
    loss, q_mean = result
    assert np.isfinite(loss) and np.isfinite(q_mean)


def test_learning_actually_rewrites_the_priorities():
    from snake_rl.agent import Agent

    agent = Agent(per_config())
    for i in range(32):
        agent.remember(*transition(value=float(i), action=i % N_ACTIONS))

    before = agent.memory._tree.tree[agent.memory.capacity:].copy()
    agent.learn()
    after = agent.memory._tree.tree[agent.memory.capacity:]
    assert not np.allclose(before, after), "les priorités doivent bouger"


def test_a_uniform_agent_keeps_a_uniform_buffer():
    from snake_rl.agent import Agent

    agent = Agent(per_config(algorithm="dueling_ddqn"))
    assert isinstance(agent.memory, ReplayBuffer)
    for i in range(32):
        agent.remember(*transition(value=float(i), action=i % N_ACTIONS))
    assert agent.learn() is not None
