"""Snake pilote par un agent Deep Q-Learning (PyTorch).

Reutilise telles quelles les classes Snake/Apple et les regles de jeu de
serpent-algo.py (charge via importlib, cf. Claude.md) : l'agent se contente
de remplacer les touches clavier par un choix d'action relative.

Modes :
    --train           entrainement headless (pas d'affichage ni d'horloge)
    --play            demonstration visuelle avec le modele sauvegarde
    --eval N          evalue N parties avec le modele sauvegarde (epsilon=0)
    --eval N --fast   idem mais sans horloge (tri rapide, mesure NON officielle)
"""

import argparse
import csv
import importlib.util
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reseau minuscule (11->256->3) : le multithreading coute plus en synchronisation
# qu'il ne rapporte, et serialise mal les milliers de petits batchs.
torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Une partie n'est valable que si score > SCORE_THRESHOLD (donc >= 11 pommes).
SCORE_THRESHOLD = 10

# Penalite par pas neutre (essai 2). A ajuster : trop faible, aucun effet sur le
# ratio ; trop forte, l'agent prend des risques mortels pour gagner du temps.
STEP_PENALTY = 0.02

# Coefficient du shaping de potentiel vers la pomme (0 = desactive).
SHAPING = 0.0

GAMMA = 0.9

# Suffixe des fichiers de sortie, pour comparer plusieurs essais sans les ecraser.
TAG = ""


def _load_base_game():
    """Charge serpent-algo.py comme module (sans executer son main())."""
    path = os.path.join(BASE_DIR, "serpent-algo.py")
    spec = importlib.util.spec_from_file_location("serpent_algo_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_game()
pygame = base.pygame

GRID_SIZE = base.GRID_SIZE
CELL_SIZE = base.CELL_SIZE
GAME_SPEED = base.GAME_SPEED
UP, DOWN, LEFT, RIGHT = base.UP, base.DOWN, base.LEFT, base.RIGHT
# Ordre horaire (coordonnees ecran : y vers le bas) pour tourner droite/gauche.
CLOCKWISE = [RIGHT, DOWN, LEFT, UP]

assert GRID_SIZE == 15, "GRID_SIZE doit rester 15 (reference serpent-algo.py)"
assert CELL_SIZE == 30, "CELL_SIZE doit rester 30 (reference serpent-algo.py)"
assert GAME_SPEED == 5, "GAME_SPEED doit rester 5 (reference serpent-algo.py)"
print(f"[Config] GRID_SIZE={GRID_SIZE} CELL_SIZE={CELL_SIZE} GAME_SPEED={GAME_SPEED}")


# --------------------------------------------------------------------------
# Environnement de jeu pour l'agent
# --------------------------------------------------------------------------
class SnakeGameAI:
    """Enveloppe autour de Snake/Apple : reset() et play_step(action)."""

    def __init__(self, render=False, training=False):
        self.render = render
        # Garde-fou anti-boucle autorise uniquement pendant l'entrainement.
        self.training = training
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((base.SCREEN_WIDTH, base.SCREEN_HEIGHT))
            pygame.display.set_caption("Snake IA")
            self.clock = pygame.time.Clock()
            self.font_main = pygame.font.Font(None, 40)
        self.reset()

    def reset(self):
        self.snake = base.Snake()
        self.apple = base.Apple(self.snake.body)
        self.total_frames = 0
        # Pas depuis la derniere pomme (remis a zero a chaque repas) : sert
        # au garde-fou anti-boucle, independamment de la duree totale deja
        # jouee, pour ne pas couper une partie qui progresse encore.
        self.frames_since_food = 0
        self.start_time = time.time()
        self.victory = False

    def _apply_action(self, action):
        """action = [tout_droit, droite, gauche] en one-hot."""
        idx = CLOCKWISE.index(self.snake.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = CLOCKWISE[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = CLOCKWISE[(idx + 1) % 4]
        else:
            new_dir = CLOCKWISE[(idx - 1) % 4]
        self.snake.set_direction(new_dir)

    def play_step(self, action):
        self.total_frames += 1
        self.frames_since_food += 1

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        dist_before = torus_distance(self.snake.head_pos, self.apple.position)
        self._apply_action(action)
        self.snake.move()

        # Penalite a chaque pas neutre : cumulee entre deux pommes, elle revient
        # a penaliser le temps mis pour atteindre la pomme suivante (ratio
        # score/temps). Ne s'applique pas aux pas qui mangent ou qui tuent.
        reward = -STEP_PENALTY
        game_over = False

        if self.snake.check_self_collision():
            game_over = True
            reward = -10
        elif self.training and self.frames_since_food > 100 * len(self.snake.body):
            # Anti-boucle infinie : entrainement seulement, ne change pas les
            # regles du jeu de reference. Mesure le temps depuis la derniere
            # pomme, pas la duree totale de la partie.
            game_over = True
            reward = -10
        elif self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.frames_since_food = 0
            reward = 10
            if not self.apple.relocate(self.snake.body):
                self.victory = True
                game_over = True
        elif SHAPING:
            # Shaping de potentiel (Ng, Harada & Russell 1999) : F = g*phi(s') - phi(s)
            # avec phi = -distance torique a la pomme. Ne change pas la politique
            # optimale, mais oriente chaque pas vers le chemin court.
            dist_after = torus_distance(self.snake.head_pos, self.apple.position)
            reward += SHAPING * (dist_before - GAMMA * dist_after)

        if self.render:
            self._draw()
            self.clock.tick(GAME_SPEED)

        return reward, game_over, self.snake.score

    def _draw(self):
        self.screen.fill(base.GRIS_FOND)
        game_area_rect = pygame.Rect(0, base.SCORE_PANEL_HEIGHT, base.SCREEN_WIDTH, base.SCREEN_WIDTH)
        pygame.draw.rect(self.screen, base.NOIR, game_area_rect)
        base.draw_grid(self.screen)
        self.apple.draw(self.screen)
        self.snake.draw(self.screen)
        base.display_info(self.screen, self.font_main, self.snake, self.start_time)
        pygame.display.flip()

    def elapsed_time(self):
        return time.time() - self.start_time


# --------------------------------------------------------------------------
# Etat (11 booleens, adapte au monde torique)
# --------------------------------------------------------------------------
def _torus_delta(a, b):
    """Ecart le plus court de a vers b sur un axe de taille GRID_SIZE (torique)."""
    d = b - a
    if d > GRID_SIZE / 2:
        d -= GRID_SIZE
    elif d < -GRID_SIZE / 2:
        d += GRID_SIZE
    return d


def torus_distance(a, b):
    """Distance de Manhattan torique entre deux cases (les bords se traversent)."""
    return abs(_torus_delta(a[0], b[0])) + abs(_torus_delta(a[1], b[1]))


def get_state(game):
    snake = game.snake
    head = snake.head_pos
    direction = snake.direction
    idx = CLOCKWISE.index(direction)
    dir_straight = CLOCKWISE[idx]
    dir_right = CLOCKWISE[(idx + 1) % 4]
    dir_left = CLOCKWISE[(idx - 1) % 4]

    # move() fait pop() de la queue AVANT le test de collision : entrer sur la
    # case de la queue est donc legal, sauf si le serpent est en croissance
    # (grow_pending), auquel cas la queue ne se libere pas.
    occupied = set(map(tuple, snake.body if snake.grow_pending else snake.body[:-1]))

    def danger(d):
        # Pas de mur en monde torique : seul le corps peut tuer.
        nx = (head[0] + d[0]) % GRID_SIZE
        ny = (head[1] + d[1]) % GRID_SIZE
        return (nx, ny) in occupied

    apple = game.apple.position
    dx = _torus_delta(head[0], apple[0]) if apple else 0
    dy = _torus_delta(head[1], apple[1]) if apple else 0

    state = [
        danger(dir_straight),
        danger(dir_right),
        danger(dir_left),
        direction == LEFT,
        direction == RIGHT,
        direction == UP,
        direction == DOWN,
        dx < 0,
        dx > 0,
        dy < 0,
        dy > 0,
    ]
    return np.array(state, dtype=int)


# --------------------------------------------------------------------------
# Modele (Linear_QNet) + entrainement (QTrainer, Bellman)
# --------------------------------------------------------------------------
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name="model.pth"):
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(MODEL_DIR, file_name))

    def load(self, file_name="model.pth"):
        path = os.path.join(MODEL_DIR, file_name)
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.detach().clone()

        # Bellman vectorise : un seul forward pass sur tout le batch au lieu
        # d'un par transition. Q_new = r + gamma * max(Q(s')), et r seul si done.
        not_done = torch.tensor([not d for d in done], dtype=torch.float)
        with torch.no_grad():
            next_q = self.model(next_state).max(dim=1).values
        q_new = reward + self.gamma * next_q * not_done
        target[torch.arange(target.shape[0]), action.argmax(dim=1)] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, training=True):
        # Epsilon-greedy decroissant : exploration forte au debut, nulle en eval.
        self.epsilon = max(0, 80 - self.n_games) if training else 0
        final_move = [0, 0, 0]
        if training and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item())
        final_move[move] = 1
        return final_move


# --------------------------------------------------------------------------
# Entrainement
# --------------------------------------------------------------------------
def _plot_scores(scores, mean_scores):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponible : graphique ignore.")
        return

    plt.figure()
    plt.title("Entrainement Snake IA")
    plt.xlabel("Partie")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Moyenne mobile")
    plt.ylim(ymin=0)
    plt.legend()
    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, f"training_scores{TAG}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Graphique sauvegarde : {out_path}")


def train(n_games_target=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    agent = Agent()
    game = SnakeGameAI(render=False, training=True)

    scores = []
    mean_scores = []
    total_score = 0
    best_ratio = 0.0
    best_score = 0
    training_start = time.time()

    csv_path = os.path.join(LOG_DIR, f"training_log{TAG}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["game", "score", "steps", "steps_per_apple", "record", "best_ratio", "cumulative_time_s", "valid"]
    )

    try:
        while True:
            state_old = get_state(game)
            final_move = agent.get_action(state_old, training=True)
            reward, done, score = game.play_step(final_move)
            state_new = get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                steps = game.total_frames
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                valid = score > SCORE_THRESHOLD
                if score > best_score:
                    best_score = score

                if valid:
                    # Temps simule (pas d'horloge en headless) coherent avec
                    # la definition --fast : nb_pas / GAME_SPEED.
                    simulated_time = steps / GAME_SPEED
                    ratio = score / simulated_time if simulated_time > 0 else 0.0
                    if ratio > best_ratio:
                        best_ratio = ratio
                        agent.model.save(f"model{TAG}.pth")

                total_score += score
                mean_score = total_score / agent.n_games
                scores.append(score)
                mean_scores.append(mean_score)

                steps_per_apple = steps / score if score > 0 else steps
                cumulative_time = time.time() - training_start
                csv_writer.writerow(
                    [agent.n_games, score, steps, f"{steps_per_apple:.2f}", best_score,
                     f"{best_ratio:.3f}", f"{cumulative_time:.1f}", valid]
                )
                csv_file.flush()

                print(
                    f"Partie {agent.n_games} Score {score} Record {best_score} "
                    f"MeilleurRatio {best_ratio:.3f} ScoreMoyen {mean_score:.2f}"
                )

                if n_games_target and agent.n_games >= n_games_target:
                    break
    except KeyboardInterrupt:
        print("Entrainement interrompu par l'utilisateur.")
    finally:
        csv_file.close()
        # Le modele "best ratio" est choisi sur UNE partie, donc en partie sur
        # de la chance ; on garde aussi le modele final pour pouvoir comparer
        # les deux sur une vraie evaluation a epsilon = 0.
        agent.model.save(f"model{TAG}_final.pth")
        _plot_scores(scores, mean_scores)

    return agent


# --------------------------------------------------------------------------
# Demonstration visuelle (--play)
# --------------------------------------------------------------------------
def play():
    agent = Agent()
    agent.model.load(f"model{TAG}.pth")
    game = SnakeGameAI(render=True, training=False)

    while True:
        state = get_state(game)
        action = agent.get_action(state, training=False)
        reward, done, score = game.play_step(action)
        if done:
            elapsed = game.elapsed_time()
            print(f"Partie terminee. Score={score} Temps={elapsed:.1f}s Victoire={game.victory}")
            time.sleep(1.5)
            game.reset()


# --------------------------------------------------------------------------
# Evaluation officielle (--eval N [--fast])
# --------------------------------------------------------------------------
def evaluate(n, fast=False):
    agent = Agent()
    agent.model.load(f"model{TAG}.pth")
    game = SnakeGameAI(render=not fast, training=False)

    results = []  # (score, temps, ratio, valide)
    victory_flag = False

    for i in range(1, n + 1):
        game.reset()
        done = False
        while not done:
            state = get_state(game)
            action = agent.get_action(state, training=False)
            reward, done, score = game.play_step(action)

        steps = game.total_frames
        elapsed = (steps / GAME_SPEED) if fast else game.elapsed_time()

        valid = score > SCORE_THRESHOLD
        ratio = (score / elapsed) if elapsed > 0 else 0.0
        results.append((score, elapsed, ratio, valid))
        if game.victory:
            victory_flag = True

        tag = "VALIDE" if valid else f"INVALIDE (score <= {SCORE_THRESHOLD})"
        label = " [--fast, non officiel]" if fast else ""
        print(f"Partie {i}: score={score} temps={elapsed:.1f}s ratio={ratio:.3f}{label} -> {tag}")

    valid_results = [r for r in results if r[3]]
    print("\n--- Resume evaluation ---")
    print(f"Parties valables : {len(valid_results)}/{n} ({100 * len(valid_results) / n:.1f}%)")
    if valid_results:
        best_ratio = max(r[2] for r in valid_results)
        avg_ratio = sum(r[2] for r in valid_results) / len(valid_results)
        print(f"Meilleur ratio (parties valables) : {best_ratio:.3f}")
        print(f"Ratio moyen (parties valables) : {avg_ratio:.3f}")
    else:
        print("Aucune partie valable.")
    best_score = max(r[0] for r in results)
    print(f"Meilleur score (toutes parties) : {best_score}")
    print(f"Victoire obtenue : {'oui' if victory_flag else 'non'}")
    if fast:
        print("[Rappel] Mesure --fast : temps = nb_pas / GAME_SPEED -> NON officielle.")


# --------------------------------------------------------------------------
def main():
    global STEP_PENALTY, SHAPING, TAG

    parser = argparse.ArgumentParser(description="Snake IA - Deep Q-Learning (ratio score/temps)")
    parser.add_argument("--train", action="store_true", help="Entrainement headless")
    parser.add_argument("--play", action="store_true", help="Demonstration visuelle (modele sauvegarde)")
    parser.add_argument("--eval", type=int, metavar="N", help="Evalue N parties (epsilon=0)")
    parser.add_argument("--fast", action="store_true", help="Avec --eval : sans horloge, mesure non officielle")
    parser.add_argument("--games", type=int, default=None, help="Nombre de parties d'entrainement (defaut illimite, Ctrl+C pour arreter)")
    parser.add_argument("--step-penalty", type=float, default=STEP_PENALTY, help="Penalite par pas neutre")
    parser.add_argument("--shaping", type=float, default=SHAPING, help="Coefficient du shaping vers la pomme (0 = desactive)")
    parser.add_argument("--seed", type=int, default=None, help="Graine aleatoire (reproductibilite des essais)")
    parser.add_argument("--tag", type=str, default="", help="Suffixe des fichiers de sortie (comparaison d'essais)")
    args = parser.parse_args()

    STEP_PENALTY = args.step_penalty
    SHAPING = args.shaping
    TAG = args.tag

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.train:
        train(n_games_target=args.games)
    elif args.play:
        play()
    elif args.eval is not None:
        evaluate(args.eval, fast=args.fast)
    else:
        # Sans argument : le correcteur lance simplement "python snake-ia.py"
        # et doit voir l'agent entraine jouer, au rythme du jeu de reference.
        play()


if __name__ == "__main__":
    main()
