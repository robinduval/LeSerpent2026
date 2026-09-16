"""Read-only episode diagnostics; none of these observations change game rules.

Apple intervals count moves from reset to the first apple, then moves between
successive apples. ``mean_inter_apple_steps`` excludes the reset-to-first gap;
it is undefined with fewer than two apples. ``steps_per_apple`` includes the
final unfinished gap and is undefined without apples. Neither is a geometric
distance. Spatial distance is recorded separately as toroidal Manhattan
distance between successive eaten apple coordinates (not from the initial
head). Long foodless sequences include the move on which food is eaten and
are counted once per food segment when they reach the declared threshold.

Repetition histories reset at each apple. A repeated *physical* state (ordered
body, direction, food, pending growth) is evidence of a probable deterministic
cycle, not proof that a stochastic training policy cannot escape it. Repeated
11-value observations and head/direction pairs are reported independently and
never establish a cycle. All averages with no observations are JSON null.
"""

from statistics import mean, median

from . import rules


def _average(values):
    return mean(values) if values else None


class EpisodeDiagnostics:
    """Call ``observe`` once after each move; initial game state is recorded."""

    def __init__(self, game, long_without_food_threshold=100):
        if long_without_food_threshold <= 0:
            raise ValueError("long_without_food_threshold must be positive")
        self.long_without_food_threshold = long_without_food_threshold
        self._initial_steps = game.steps
        self._last_food_step = game.steps
        self._previous_apple_step = None
        self._previous_apple_position = None
        self._steps = 0
        self._score = game.score
        self._reward = 0.0
        self._apples = 0
        self._steps_since_food = 0
        self._longest_without_food = 0
        self._long_sequences = 0
        self._segment_is_long = False
        self._apple_intervals = []
        self._inter_apple_distances = []
        self._inter_apple_toric_distances = []
        self._head_repetitions = 0
        self._compressed_repetitions = 0
        self._cycle_repetitions = 0
        self._max_head_count = 1
        self._first_cycle_step = None
        self._first_cycle_period = None
        self._q_sum = 0.0
        self._q_count = 0
        self._q_max = None
        self._loss_sum = 0.0
        self._loss_count = 0
        self._terminated = False
        self._truncated = False
        self._won = bool(game.won)
        self._cause = None
        self._reset_histories(game)

    @staticmethod
    def _physical_state(game):
        return (
            tuple(tuple(position) for position in game.body),
            tuple(game.direction),
            tuple(game.food) if game.food is not None else None,
            bool(game._grow_pending),
        )

    def _reset_histories(self, game, state=None):
        self._heads = {(tuple(game.head), tuple(game.direction)): 1}
        self._physical = {self._physical_state(game): game.steps}
        self._compressed = {} if state is None else {tuple(state): 1}

    def observe(self, game, result, state=None, q_values=None, loss=None):
        """Observe a post-move state and optional pre-move legal-action Qs."""
        self._steps = game.steps - self._initial_steps
        self._score = game.score
        self._reward += float(result.reward)
        self._truncated = bool(result.truncated)
        self._won = bool(result.won)
        self._terminated = bool(
            getattr(result, "terminated", result.done and not result.truncated)
        )
        self._cause = result.info.get("cause")
        gap = game.steps - self._last_food_step
        self._longest_without_food = max(self._longest_without_food, gap)
        if gap >= self.long_without_food_threshold and not self._segment_is_long:
            self._long_sequences += 1
            self._segment_is_long = True

        if result.ate:
            self._apples += 1
            self._apple_intervals.append(gap)
            if self._previous_apple_step is not None:
                self._inter_apple_distances.append(game.steps - self._previous_apple_step)
                dx = abs(game.head[0] - self._previous_apple_position[0])
                dy = abs(game.head[1] - self._previous_apple_position[1])
                self._inter_apple_toric_distances.append(
                    min(dx, rules.GRID_SIZE - dx) + min(dy, rules.GRID_SIZE - dy)
                )
            self._previous_apple_step = game.steps
            self._previous_apple_position = tuple(game.head)
            self._last_food_step = game.steps
            self._steps_since_food = 0
            self._segment_is_long = False
            self._reset_histories(game, state)
        else:
            self._steps_since_food = gap
            head = (tuple(game.head), tuple(game.direction))
            count = self._heads.get(head, 0) + 1
            self._heads[head] = count
            self._head_repetitions += count > 1
            self._max_head_count = max(self._max_head_count, count)
            physical = self._physical_state(game)
            if physical in self._physical:
                self._cycle_repetitions += 1
                if self._first_cycle_step is None:
                    self._first_cycle_step = self._steps
                    self._first_cycle_period = game.steps - self._physical[physical]
            else:
                self._physical[physical] = game.steps
            if state is not None:
                compressed = tuple(state)
                count = self._compressed.get(compressed, 0) + 1
                self._compressed[compressed] = count
                self._compressed_repetitions += count > 1

        if q_values is not None:
            values = [float(value) for value in q_values]
            if values:
                self._q_sum += mean(values)
                self._q_count += 1
                maximum = max(values)
                self._q_max = maximum if self._q_max is None else max(self._q_max, maximum)
        if loss is not None:
            self._loss_sum += float(loss)
            self._loss_count += 1

    def summary(self):
        """Return JSON-safe metrics; only actual self collisions count as death."""
        return {
            "score": self._score,
            "apples": self._apples,
            "steps": self._steps,
            "reward": self._reward,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "won": self._won,
            "truncation_count": int(self._truncated),
            "death_count": int(self._terminated and self._cause == "self"),
            "cause": self._cause,
            "steps_since_food": self._steps_since_food,
            "longest_without_food": self._longest_without_food,
            "long_without_food_threshold": self.long_without_food_threshold,
            "long_without_food_sequences": self._long_sequences,
            "long_without_food": self._long_sequences > 0,
            "apple_step_intervals": list(self._apple_intervals),
            "inter_apple_distances": list(self._inter_apple_distances),
            "mean_inter_apple_steps": _average(self._inter_apple_distances),
            "inter_apple_toric_distances": list(self._inter_apple_toric_distances),
            "mean_inter_apple_toric_distance": _average(self._inter_apple_toric_distances),
            "steps_per_apple": self._steps / self._apples if self._apples else None,
            "head_direction_repetitions": self._head_repetitions,
            "compressed_state_repetitions": self._compressed_repetitions,
            "cycle_repetitions": self._cycle_repetitions,
            "probable_cycle": self._cycle_repetitions > 0,
            "first_cycle_step": self._first_cycle_step,
            "first_cycle_period": self._first_cycle_period,
            "max_repeated_head_direction_count": self._max_head_count,
            "q_mean": self._q_sum / self._q_count if self._q_count else None,
            "q_max": self._q_max,
            "q_count": self._q_count,
            "loss_mean": self._loss_sum / self._loss_count if self._loss_count else None,
            "loss_count": self._loss_count,
        }


def aggregate_episode_metrics(episodes):
    """Pool intervals/Q decisions/loss updates; macro-average per-apple ratios.

    ``steps_per_apple`` is the campaign's total steps / total apples, including
    zero-apple episodes; ``mean_steps_per_apple`` averages only defined episode
    ratios. Frequencies are fractions of episodes, not fractions of moves.
    """
    episodes = list(episodes)
    count = len(episodes)
    apples = sum(episode["apples"] for episode in episodes)
    steps = sum(episode["steps"] for episode in episodes)
    truncations = sum(bool(episode["truncated"]) for episode in episodes)
    intervals = [gap for episode in episodes for gap in episode["inter_apple_distances"]]
    spatial_distances = [
        distance for episode in episodes
        for distance in episode["inter_apple_toric_distances"]
    ]
    q_count = sum(episode.get("q_count", 0) for episode in episodes)
    loss_count = sum(episode.get("loss_count", 0) for episode in episodes)
    scores = [episode["score"] for episode in episodes]
    return {
        "episodes": count,
        "mean_score": _average(scores),
        "median_score": median(scores) if scores else None,
        "record": max(scores) if scores else None,
        "mean_reward": _average([episode["reward"] for episode in episodes]),
        "mean_steps": steps / count if count else None,
        "steps_total": steps,
        "apples_total": apples,
        "truncation_count": truncations,
        "truncation_rate": truncations / count if count else None,
        "terminated_count": sum(bool(episode["terminated"]) for episode in episodes),
        "death_count": sum(episode["death_count"] for episode in episodes),
        "mean_steps_per_apple": _average([
            episode["steps_per_apple"] for episode in episodes
            if episode["steps_per_apple"] is not None
        ]),
        "steps_per_apple": steps / apples if apples else None,
        "mean_inter_apple_steps": _average(intervals),
        "inter_apple_interval_count": len(intervals),
        "mean_inter_apple_toric_distance": _average(spatial_distances),
        "inter_apple_toric_distance_count": len(spatial_distances),
        "longest_without_food": max(
            (episode["longest_without_food"] for episode in episodes), default=0
        ),
        "long_without_food_sequences": sum(
            episode["long_without_food_sequences"] for episode in episodes
        ),
        "long_without_food_rate": sum(
            episode["long_without_food"] for episode in episodes
        ) / count if count else None,
        "probable_cycle_rate": sum(
            episode["probable_cycle"] for episode in episodes
        ) / count if count else None,
        "q_mean": sum(
            episode["q_mean"] * episode["q_count"] for episode in episodes
            if episode.get("q_count", 0)
        ) / q_count if q_count else None,
        "q_max": max(
            (episode["q_max"] for episode in episodes if episode["q_max"] is not None),
            default=None,
        ),
        "loss_mean": sum(
            episode["loss_mean"] * episode["loss_count"] for episode in episodes
            if episode.get("loss_count", 0)
        ) / loss_count if loss_count else None,
    }
