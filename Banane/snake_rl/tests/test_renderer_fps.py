"""La cadence du lecteur concerne uniquement les frames enregistrées."""

import copy

import pytest

from snake_rl import rules
from snake_rl.game import SnakeGame


@pytest.mark.parametrize("fps", [60, 120])
def test_renderer_ticks_visual_fps_closes_and_preserves_frames(monkeypatch, fps):
    from snake_rl import render

    class RecordingClock:
        def __init__(self):
            self.rates = []

        def tick(self, rate):
            self.rates.append(rate)

    clock = RecordingClock()
    monkeypatch.setattr(render.pygame.time, "Clock", lambda: clock)
    game = SnakeGame(seed=1)
    frames = [game.snapshot(), game.snapshot()]
    original = copy.deepcopy(frames)
    replay = {"run_id": "fps_test", "frames": frames}
    played = render.replay_frames(replay, fps=fps, linger_seconds=0)
    assert played == len(frames)
    assert clock.rates == [fps, fps]
    assert frames == original
    assert rules.GAME_SPEED == 5
    assert render.pygame.display.get_surface() is None


def test_renderer_default_is_60_fps(monkeypatch):
    from snake_rl import render

    rates = []
    class Clock:
        def tick(self, rate):
            rates.append(rate)

    monkeypatch.setattr(render.pygame.time, "Clock", Clock)
    render.replay_frames({"frames": [SnakeGame(seed=1).snapshot()]}, linger_seconds=0)
    assert rates == [60]
