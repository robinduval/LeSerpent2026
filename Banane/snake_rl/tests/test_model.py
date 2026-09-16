"""Tests des architectures et du buffer de replay."""

import numpy as np
import pytest
import torch

from snake_rl.config import Config
from snake_rl.model import DQN, DuelingDQN, build_model
from snake_rl.replay_buffer import ReplayBuffer
from snake_rl.rules import N_ACTIONS
from snake_rl.state import STATE_SIZE


# ----------------------------------------------------------------------
# Modèles
# ----------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", [DQN, DuelingDQN])
def test_shapes_are_eleven_in_four_out(model_cls):
    model = model_cls(hidden_size=32)
    out = model(torch.zeros(1, STATE_SIZE))
    assert out.shape == (1, N_ACTIONS) == (1, 4)


@pytest.mark.parametrize("model_cls", [DQN, DuelingDQN])
def test_forward_handles_batches_and_stays_finite(model_cls):
    model = model_cls(hidden_size=32)
    out = model(torch.randn(64, STATE_SIZE))
    assert out.shape == (64, N_ACTIONS)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_cls", [DQN, DuelingDQN])
def test_batch_produces_a_finite_loss_and_gradients(model_cls):
    model = model_cls(hidden_size=32)
    loss = torch.nn.SmoothL1Loss()(
        model(torch.randn(32, STATE_SIZE)), torch.randn(32, N_ACTIONS)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


def test_dueling_advantage_is_centred():
    """Q - V doit être de moyenne nulle : c'est la contrainte d'identifiabilité."""
    model = DuelingDQN(hidden_size=32)
    x = torch.randn(16, STATE_SIZE)
    with torch.no_grad():
        features = model.trunk(x)
        value = model.value_head(features)
        q = model(x)
    assert torch.allclose(
        (q - value).mean(dim=1), torch.zeros(16), atol=1e-6
    )


def test_build_model_follows_the_configuration():
    assert isinstance(build_model(Config(algorithm="dqn")), DQN)
    assert isinstance(build_model(Config(algorithm="ddqn")), DQN)
    assert isinstance(build_model(Config(algorithm="dueling_ddqn")), DuelingDQN)
    assert isinstance(
        build_model(Config(algorithm="dueling_ddqn_per")), DuelingDQN
    )


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------


def push_dummy(buffer, value, action=0):
    buffer.push(
        np.full(STATE_SIZE, value, dtype=np.float32),
        action,
        float(value),
        np.full(STATE_SIZE, value + 1, dtype=np.float32),
        False,
    )


def test_buffer_respects_its_capacity():
    buffer = ReplayBuffer(capacity=10, seed=0)
    for i in range(25):
        push_dummy(buffer, i)
    assert len(buffer) == 10
    assert buffer.is_full


def test_buffer_overwrites_the_oldest_transitions():
    buffer = ReplayBuffer(capacity=3, seed=0)
    for i in range(5):
        push_dummy(buffer, i)
    # Après 5 insertions dans 3 cases, il reste les valeurs 2, 3 et 4.
    assert sorted(buffer.rewards.tolist()) == [2.0, 3.0, 4.0]


def test_sample_returns_the_expected_tensor_shapes():
    buffer = ReplayBuffer(capacity=100, seed=0)
    for i in range(50):
        push_dummy(buffer, i, action=i % N_ACTIONS)

    states, actions, rewards, next_states, dones, masks = buffer.sample(16)
    assert states.shape == (16, STATE_SIZE)
    assert actions.shape == (16,) and actions.dtype == torch.int64
    assert rewards.shape == (16,) and rewards.dtype == torch.float32
    assert next_states.shape == (16, STATE_SIZE)
    assert dones.shape == (16,)
    assert masks.shape == (16, N_ACTIONS) and masks.dtype == torch.bool


def test_sample_can_exceed_the_stored_count():
    """Tirage avec remise : un batch plus grand que la mémoire reste valide."""
    buffer = ReplayBuffer(capacity=100, seed=0)
    push_dummy(buffer, 1)
    states, *_ = buffer.sample(8)
    assert states.shape == (8, STATE_SIZE)


def test_sample_on_empty_buffer_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10, seed=0).sample(4)


def test_buffer_sampling_is_reproducible():
    def draw():
        buffer = ReplayBuffer(capacity=100, seed=7)
        for i in range(40):
            push_dummy(buffer, i)
        return buffer.sample(16)[2].tolist()

    assert draw() == draw()


def test_next_mask_defaults_to_all_legal():
    buffer = ReplayBuffer(capacity=4, seed=0)
    push_dummy(buffer, 1)
    assert buffer.next_masks[0].all()


def test_next_mask_is_stored_as_given():
    buffer = ReplayBuffer(capacity=4, seed=0)
    mask = np.array([True, False, True, True])
    buffer.push(
        np.zeros(STATE_SIZE, dtype=np.float32),
        0,
        0.0,
        np.zeros(STATE_SIZE, dtype=np.float32),
        False,
        next_mask=mask,
    )
    assert buffer.next_masks[0].tolist() == mask.tolist()
