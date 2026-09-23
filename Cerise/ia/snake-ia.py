"""Agent Double DQN léger pour le Snake du groupe Cerise.

Commande principale :
    python snake-ia.py

Le réseau neuronal est volontairement écrit en Python standard afin que son
fonctionnement soit visible et qu'aucune dépendance à Torch ne soit nécessaire.
Pygame n'est utilisé que pour la démonstration graphique.
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import math
from pathlib import Path
import random
import time


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "cerise-double-dqn.json"


def load_game_module():
    """Charge serpent-algo.py malgré le tiret présent dans son nom."""
    source = HERE / "serpent-algo.py"
    spec = importlib.util.spec_from_file_location("cerise_serpent_algo", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


game_code = load_game_module()


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


class TinyQNetwork:
    """Réseau 11 -> 64 -> 3 avec ReLU, entraîné par descente de gradient."""

    def __init__(self, input_size=11, hidden_size=64, output_size=3, seed=2026):
        generator = random.Random(seed)
        scale1 = math.sqrt(2.0 / input_size)
        scale2 = math.sqrt(2.0 / hidden_size)
        self.w1 = [
            [generator.gauss(0.0, scale1) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        self.b1 = [0.0] * hidden_size
        self.w2 = [
            [generator.gauss(0.0, scale2) for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
        self.b2 = [0.0] * output_size

    def predict(self, state):
        hidden = []
        for hidden_index, bias in enumerate(self.b1):
            value = bias
            for input_index, state_value in enumerate(state):
                value += state_value * self.w1[input_index][hidden_index]
            hidden.append(max(0.0, value))

        output = []
        for output_index, bias in enumerate(self.b2):
            value = bias
            for hidden_index, hidden_value in enumerate(hidden):
                value += hidden_value * self.w2[hidden_index][output_index]
            output.append(value)
        return output

    def train_one(self, state, targets, learning_rate):
        pre_activation = []
        hidden = []
        for hidden_index, bias in enumerate(self.b1):
            value = bias + sum(
                state[input_index] * self.w1[input_index][hidden_index]
                for input_index in range(len(state))
            )
            pre_activation.append(value)
            hidden.append(max(0.0, value))

        output = []
        for output_index, bias in enumerate(self.b2):
            value = bias + sum(
                hidden[hidden_index] * self.w2[hidden_index][output_index]
                for hidden_index in range(len(hidden))
            )
            output.append(value)

        # Dérivée de la perte de Huber : elle limite les gradients trop grands.
        output_gradient = []
        for predicted, target in zip(output, targets):
            difference = predicted - target
            output_gradient.append(max(-1.0, min(1.0, difference)))

        hidden_gradient = []
        for hidden_index, pre_value in enumerate(pre_activation):
            gradient = sum(
                output_gradient[output_index] * self.w2[hidden_index][output_index]
                for output_index in range(len(output))
            )
            hidden_gradient.append(gradient if pre_value > 0.0 else 0.0)

        for hidden_index in range(len(hidden)):
            for output_index in range(len(output)):
                self.w2[hidden_index][output_index] -= (
                    learning_rate * output_gradient[output_index] * hidden[hidden_index]
                )
        for output_index in range(len(output)):
            self.b2[output_index] -= learning_rate * output_gradient[output_index]

        for input_index, state_value in enumerate(state):
            for hidden_index in range(len(hidden)):
                self.w1[input_index][hidden_index] -= (
                    learning_rate * hidden_gradient[hidden_index] * state_value
                )
        for hidden_index in range(len(hidden)):
            self.b1[hidden_index] -= learning_rate * hidden_gradient[hidden_index]

    def copy_from(self, other):
        self.w1 = [row[:] for row in other.w1]
        self.b1 = other.b1[:]
        self.w2 = [row[:] for row in other.w2]
        self.b2 = other.b2[:]

    def to_dict(self):
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def load_dict(self, values):
        self.w1 = values["w1"]
        self.b1 = values["b1"]
        self.w2 = values["w2"]
        self.b2 = values["b2"]


class DoubleDQNAgent:
    GAMMA = 0.90
    LEARNING_RATE = 0.001
    MEMORY_SIZE = 20_000
    REPLAY_SIZE = 64
    TARGET_UPDATE_FREQUENCY = 250
    TRAIN_EVERY = 4
    DISTANCE_GUIDANCE = 1.0

    def __init__(self, seed=2026):
        # Apple utilise le générateur global : le fixer rend les tests reproductibles.
        random.seed(seed)
        self.random = random.Random(seed)
        self.online = TinyQNetwork(seed=seed)
        self.target = TinyQNetwork(seed=seed + 1)
        self.target.copy_from(self.online)
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.training_steps = 0

    @staticmethod
    def _toroidal_delta(start, target):
        delta = (target - start) % game_code.GRID_SIZE
        if delta > game_code.GRID_SIZE // 2:
            delta -= game_code.GRID_SIZE
        return delta

    def get_state(self, game):
        snake = game.snake
        apple = game.apple.position
        apple_dx = self._toroidal_delta(snake.head_pos[0], apple[0])
        apple_dy = self._toroidal_delta(snake.head_pos[1], apple[1])
        direction = snake.direction

        return [
            float(game.would_collide(0)),
            float(game.would_collide(1)),
            float(game.would_collide(2)),
            float(direction == game_code.LEFT),
            float(direction == game_code.RIGHT),
            float(direction == game_code.UP),
            float(direction == game_code.DOWN),
            float(apple_dx < 0),
            float(apple_dx > 0),
            float(apple_dy < 0),
            float(apple_dy > 0),
        ]

    @staticmethod
    def epsilon(episode):
        return max(0.05, 1.0 - episode / 180.0)

    def choose_action(self, state, safe_actions, episode, training=True, game=None):
        if training and self.random.random() < self.epsilon(episode):
            return self.random.choice(safe_actions)
        q_values = self.online.predict(state)
        if game is None:
            return max(safe_actions, key=lambda action: q_values[action])

        current_distance = game.distance_to_apple()
        return max(
            safe_actions,
            key=lambda action: q_values[action]
            + self.DISTANCE_GUIDANCE
            * (current_distance - self._distance_after_action(game, action)),
        )

    def guided_exploration(self, game, safe_actions):
        """Explore en rencontrant plus souvent la récompense associée aux pommes.

        Ce guidage sert à rencontrer plus souvent les récompenses pendant
        l'entraînement. En évaluation, les valeurs Q restent combinées au progrès
        vers la pomme par ``choose_action``.
        """
        if self.random.random() >= 0.80:
            return self.random.choice(safe_actions)

        distances = {
            action: self._distance_after_action(game, action) for action in safe_actions
        }
        shortest = min(distances.values())
        best_actions = [
            action for action in safe_actions if distances[action] == shortest
        ]
        return self.random.choice(best_actions)

    @staticmethod
    def _distance_after_action(game, action):
        position = game.next_position(action)
        apple = game.apple.position
        size = game_code.GRID_SIZE
        dx = abs(position[0] - apple[0])
        dy = abs(position[1] - apple[1])
        return min(dx, size - dx) + min(dy, size - dy)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _targets_for(self, state, action, reward, next_state, done):
        targets = self.online.predict(state)
        if done:
            target_value = reward
        else:
            # Double DQN : le réseau online choisit, le réseau target évalue.
            best_next_action = argmax(self.online.predict(next_state))
            future_value = self.target.predict(next_state)[best_next_action]
            target_value = reward + self.GAMMA * future_value
        targets[action] = target_value
        return targets

    def train_transition(self, transition):
        state, action, reward, next_state, done = transition
        targets = self._targets_for(state, action, reward, next_state, done)
        self.online.train_one(state, targets, self.LEARNING_RATE)
        self.training_steps += 1
        if self.training_steps % self.TARGET_UPDATE_FREQUENCY == 0:
            self.target.copy_from(self.online)

    def replay(self):
        if not self.memory:
            return
        sample_size = min(self.REPLAY_SIZE, len(self.memory))
        for transition in self.random.sample(list(self.memory), sample_size):
            self.train_transition(transition)

    def save(self, path=MODEL_PATH):
        values = {
            "algorithm": "Double DQN",
            "state_size": 11,
            "hidden_size": 64,
            "action_size": 3,
            "network": self.online.to_dict(),
        }
        path.write_text(json.dumps(values))

    def load(self, path=MODEL_PATH):
        values = json.loads(path.read_text())
        self.online.load_dict(values["network"])
        self.target.copy_from(self.online)


def train(agent, episodes, max_seconds=None):
    best_score = -1
    scores = []
    started_at = time.time()
    deadline = started_at + max_seconds if max_seconds is not None else None

    # Le modèle déjà livré participe au concours : un entraînement continu ne
    # peut le remplacer que par une politique réellement meilleure.
    initial_metrics = evaluate_metrics(agent, games=5, seed_offset=10_000)
    best_key = (
        initial_metrics["mean_score"],
        -initial_metrics["mean_steps_to_target"],
    )
    best_metrics = initial_metrics
    best_network = json.loads(json.dumps(agent.online.to_dict()))

    episode = 0
    while episode < episodes:
        if deadline is not None and time.time() >= deadline:
            break

        game = game_code.SnakeGameAI(training=True)
        done = False
        episode_steps = 0
        time_limit_reached = False

        while not done:
            if deadline is not None and time.time() >= deadline:
                time_limit_reached = True
                break
            state = agent.get_state(game)
            # Pendant l'apprentissage, le masque immédiat laisse l'agent
            # rencontrer des impasses et apprendre grâce à la pénalité -10.
            # Le flood-fill complet reste actif pour l'évaluation et le jeu.
            safe_actions = game.safe_actions(advanced=False)
            if agent.random.random() < agent.epsilon(episode):
                action = agent.guided_exploration(game, safe_actions)
            else:
                action = agent.choose_action(
                    state, safe_actions, episode, training=False, game=game
                )
            reward, done, score, _ = game.play_step(action)
            next_state = agent.get_state(game)
            transition = (state, action, reward, next_state, done)
            agent.remember(*transition)
            episode_steps += 1
            if episode_steps % agent.TRAIN_EVERY == 0 or done:
                agent.train_transition(transition)

        if time_limit_reached:
            break

        agent.replay()
        scores.append(score)
        best_score = max(best_score, score)

        if (episode + 1) % 100 == 0 or episode == 0:
            recent = scores[-20:]
            average = sum(recent) / len(recent)
            metrics = evaluate_metrics(agent, games=5, seed_offset=10_000)
            candidate_key = (
                metrics["mean_score"],
                -metrics["mean_steps_to_target"],
            )
            if candidate_key > best_key:
                best_key = candidate_key
                best_metrics = metrics
                # Passage par JSON pour obtenir une copie indépendante des listes.
                best_network = json.loads(json.dumps(agent.online.to_dict()))
                agent.save()
            elapsed = time.time() - started_at
            print(
                f"Batch {episode + 1:4d} | temps={elapsed:.3f}s "
                f"| score={score:3d} "
                f"| meilleur={best_score:2d} | moyenne(20)={average:.2f} "
                f"| évaluation={metrics['mean_score']:.2f} "
                f"| pas vers 11={metrics['mean_steps_to_target']:.1f} "
                f"| epsilon={agent.epsilon(episode):.2f}"
            )
        episode += 1

    agent.online.load_dict(best_network)
    agent.target.copy_from(agent.online)
    agent.save()
    elapsed = time.time() - started_at
    print(
        f"Entraînement terminé en {elapsed:.1f}s. "
        f"Meilleure évaluation : score moyen={best_metrics['mean_score']:.2f}, "
        f"pas moyens vers 11={best_metrics['mean_steps_to_target']:.1f}. "
        f"Modèle sauvegardé dans {MODEL_PATH.name}."
    )
    return scores


def evaluate_metrics(agent, games=5, seed_offset=0, target_score=11):
    """Mesure les points puis le temps, dans l'ordre du classement."""
    random_state = random.getstate()
    scores = []
    target_steps = []
    try:
        for game_index in range(games):
            random.seed(seed_offset + game_index)
            game = game_code.SnakeGameAI(training=True)
            done = False
            steps = 0
            reached_at = None
            while not done:
                state = agent.get_state(game)
                action = agent.choose_action(
                    state, game.safe_actions(), episode=10_000, training=False, game=game
                )
                _, done, score, _ = game.play_step(action)
                steps += 1
                if score >= target_score and reached_at is None:
                    reached_at = steps
            scores.append(score)
            target_steps.append(reached_at if reached_at is not None else float("inf"))
    finally:
        random.setstate(random_state)
    return {
        "mean_score": sum(scores) / len(scores),
        "mean_steps_to_target": sum(target_steps) / len(target_steps),
    }


def print_score(score, elapsed, steps):
    """Affiche immédiatement une progression exploitable depuis un terminal."""
    print(
        f"Score {score:3d} | temps={elapsed:.3f}s | déplacements={steps}",
        flush=True,
    )


def play_in_terminal(agent, stop_at_score=None):
    """Joue sans fenêtre, à la vitesse officielle, jusqu'à la fin de la partie."""
    game = game_code.SnakeGameAI(training=False)
    started_at = time.perf_counter()
    next_tick = started_at
    steps = 0
    previous_score = 0
    done = False
    victory = False

    print(
        f"Partie terminal démarrée à {game_code.GAME_SPEED} déplacements/s.",
        flush=True,
    )
    while not done:
        state = agent.get_state(game)
        action = agent.choose_action(
            state, game.safe_actions(), episode=10_000, training=False, game=game
        )
        _, done, score, victory = game.play_step(action)
        steps += 1

        elapsed = time.perf_counter() - started_at
        if score != previous_score:
            print_score(score, elapsed, steps)
            previous_score = score

        if stop_at_score is not None and score >= stop_at_score:
            return score, elapsed, steps, False

        next_tick += 1.0 / game_code.GAME_SPEED
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    elapsed = time.perf_counter() - started_at
    if victory:
        print(
            f"VICTOIRE : toutes les pommes ont été mangées | score={score} "
            f"| temps={elapsed:.3f}s | déplacements={steps}",
            flush=True,
        )
    else:
        print(
            f"DÉFAITE : collision | score={score} | temps={elapsed:.3f}s "
            f"| déplacements={steps}",
            flush=True,
        )
    return score, elapsed, steps, victory


def demonstrate(agent):
    pygame = game_code.pygame
    if pygame is None:
        print("Démonstration ignorée : installez Pygame avec 'python -m pip install pygame'.")
        return

    pygame.init()
    screen = pygame.display.set_mode((game_code.SCREEN_WIDTH, game_code.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake IA - Double DQN - Groupe Cerise")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 40)
    game = game_code.SnakeGameAI(training=False)
    start_time = time.time()
    running = True
    done = False
    steps = 0
    previous_score = 0

    print(
        f"Partie graphique démarrée à {game_code.GAME_SPEED} déplacements/s.",
        flush=True,
    )

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not done:
            state = agent.get_state(game)
            action = agent.choose_action(
                state, game.safe_actions(), episode=10_000, training=False, game=game
            )
            _, done, score, victory = game.play_step(action)
            steps += 1
            elapsed = time.time() - start_time
            if score != previous_score:
                print_score(score, elapsed, steps)
                previous_score = score
            if done:
                result = "VICTOIRE : toutes les pommes ont été mangées" if victory else "DÉFAITE : collision"
                print(
                    f"{result} | score={score} | temps={elapsed:.3f}s "
                    f"| déplacements={steps}",
                    flush=True,
                )

        screen.fill(game_code.GRIS_FOND)
        area = pygame.Rect(
            0,
            game_code.SCORE_PANEL_HEIGHT,
            game_code.SCREEN_WIDTH,
            game_code.SCREEN_WIDTH,
        )
        pygame.draw.rect(screen, game_code.NOIR, area)
        game_code.draw_grid(screen)
        game.apple.draw(screen)
        game.snake.draw(screen)
        game_code.display_info(screen, font, game.snake, start_time)
        if done:
            message = "VICTOIRE !" if game.victory else "FIN DE PARTIE"
            color = game_code.VERT if game.victory else game_code.ROUGE
            game_code.display_message(screen, font, message, color)

        pygame.display.flip()
        clock.tick(game_code.GAME_SPEED)

    pygame.quit()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Double DQN pour le Snake du groupe Cerise")
    parser.add_argument("--episodes", type=int, default=200, help="nombre d'épisodes d'entraînement")
    parser.add_argument(
        "--train-minutes",
        type=float,
        help="continue l'entraînement pendant cette durée réelle en minutes",
    )
    parser.add_argument("--retrain", action="store_true", help="ignore le modèle sauvegardé")
    parser.add_argument("--no-demo", action="store_true", help="n'ouvre pas la fenêtre Pygame")
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="joue sans fenêtre et affiche score et temps dans le terminal",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.train_minutes is not None and args.train_minutes <= 0:
        raise SystemExit("--train-minutes doit être strictement positif")
    agent = DoubleDQNAgent()

    if MODEL_PATH.exists() and not args.retrain:
        agent.load()
        print(f"Modèle chargé depuis {MODEL_PATH.name}.")

    if args.train_minutes is not None:
        train(agent, episodes=1_000_000, max_seconds=args.train_minutes * 60.0)
    elif not MODEL_PATH.exists() or args.retrain:
        train(agent, args.episodes)

    if args.terminal:
        play_in_terminal(agent)
    elif not args.no_demo:
        demonstrate(agent)


if __name__ == "__main__":
    main()
