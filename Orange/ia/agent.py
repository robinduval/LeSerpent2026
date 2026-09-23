"""Bloc 'Agent' : construit l'etat, decide de l'action (exploration/exploitation),
memorise les transitions et pilote la boucle d'entrainement (cf. PDF slide 4).
"""

import os
import json
import random
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from game import SnakeGameAI, GRID_SIZE, UP, DOWN, LEFT, RIGHT, format_time, flood_fill_size
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
TARGET_UPDATE_EVERY = 10  # resynchronise le reseau cible toutes les 10 parties

# Meme logique que MODEL_DIR dans model.py : chemin ancre au fichier, pas au
# repertoire courant du terminal (sinon la sauvegarde/lecture ne pointe pas
# toujours vers le meme endroit d'un lancement a l'autre).
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
STATS_PATH = os.path.join(MODEL_DIR, "stats.json")


def load_stats():
    """Nombre de parties deja jouees + meilleur score, d'une session a l'autre."""
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            data = json.load(f)
        return data.get("n_games", 0), data.get("record", 0)
    return 0, 0


def save_stats(n_games, record):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump({"n_games": n_games, "record": record}, f)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # taux d'exploration courant (voir get_action)
        self.gamma = 0.9  # facteur d'actualisation des recompenses futures
        self.memory = deque(maxlen=MAX_MEMORY)  # replay buffer (FIFO)

        # 14 entrees d'etat -> 256 neurones caches -> 4 actions (HAUT, BAS, GAUCHE, DROITE)
        self.model = Linear_QNet(14, 256, 4)
        if self.model.load():
            print("Modele precedent recharge (model/model.pth) : l'entrainement reprend.")
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """Construit le vecteur d'etat : les 11 booleens du PDF (slide 6) --
        3 dangers (relatifs a la direction courante) + 4 direction + 4 pomme --
        plus 3 valeurs continues ajoutees pour eviter que le serpent se
        bloque lui-meme : la quantite d'espace libre atteignable (flood fill,
        normalisee entre 0 et 1) dans chacune des 3 directions candidates.
        Un danger ne dit que "la case juste a cote est bloquee" ; l'espace
        libre dit en plus "et si elle ne l'est pas, est-ce que j'ai la place
        de manoeuvrer derriere, ou est-ce un cul-de-sac ?".
        """
        snake = game.snake
        head = snake.head_pos
        direction = snake.direction

        # Rotation horaire des directions, pour deduire "tout droit / droite / gauche"
        # a partir de la direction absolue courante du serpent.
        clock_wise = [RIGHT, DOWN, LEFT, UP]
        idx = clock_wise.index(direction)
        dir_straight = clock_wise[idx]
        dir_right = clock_wise[(idx + 1) % 4]
        dir_left = clock_wise[(idx - 1) % 4]

        def next_point(d):
            return [(head[0] + d[0]) % GRID_SIZE, (head[1] + d[1]) % GRID_SIZE]

        # Danger = la case atteinte mordrait le corps du serpent.
        # (Le mur n'est jamais un danger : la grille boucle sur elle-meme.)
        danger_straight = snake.check_self_collision(next_point(dir_straight))
        danger_right = snake.check_self_collision(next_point(dir_right))
        danger_left = snake.check_self_collision(next_point(dir_left))

        food = game.apple.position

        # Espace libre atteignable dans chaque direction candidate, normalise
        # par la taille de la grille pour rester dans [0, 1] comme les autres
        # entrees. 0 si la case est deja bloquee (coherent avec danger=True).
        max_area = GRID_SIZE * GRID_SIZE
        space_straight = flood_fill_size(next_point(dir_straight), snake.body) / max_area
        space_right = flood_fill_size(next_point(dir_right), snake.body) / max_area
        space_left = flood_fill_size(next_point(dir_left), snake.body) / max_area

        state = [
            danger_straight,
            danger_right,
            danger_left,

            direction == LEFT,
            direction == RIGHT,
            direction == UP,
            direction == DOWN,

            food[0] < head[0],  # pomme a gauche
            food[0] > head[0],  # pomme a droite
            food[1] < head[1],  # pomme en haut
            food[1] > head[1],  # pomme en bas

            space_straight,
            space_right,
            space_left,
        ]
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Rejoue un batch aleatoire d'experiences passees (experience replay).

        Objectif : casser la correlation entre transitions consecutives (qui
        biaiserait l'apprentissage) et reutiliser les experiences rares (ex:
        manger une pomme) plus d'une fois.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Apprend immediatement de la derniere transition (avant meme qu'elle
        ne soit rejouee plus tard depuis la memoire longue)."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Politique epsilon-greedy : au debut, on explore beaucoup (actions
        aleatoires) pour decouvrir le jeu ; puis, au fil des parties, on fait
        de plus en plus confiance au reseau (exploitation).
        """
        self.epsilon = max(5, 80 - self.n_games)
        action = [0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()

        action[move] = 1
        return action


def plot(scores, mean_scores):
    """Courbe demandee slide 12 ('Qui a fait le meilleur score ? La courbe ?')."""
    plt.clf()
    plt.title("Entrainement de l'agent")
    plt.xlabel("Nombre de parties")
    plt.ylabel("Score")
    plt.plot(scores, label="score")
    plt.plot(mean_scores, label="score moyen")
    plt.ylim(ymin=0)
    plt.legend()
    plt.pause(0.001)


def train(render=False, n_games_target=None, speed=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    agent = Agent()
    game = SnakeGameAI(render=render, speed=speed)

    # Reprise : on recupere ou en etait la session precedente (le modele lui-
    # meme est deja recharge dans Agent.__init__). Sans ca, epsilon repartirait
    # de "80 - 0" a chaque lancement et l'agent se remettrait a explorer
    # beaucoup, meme avec un reseau deja entraine.
    agent.n_games, record = load_stats()
    if agent.n_games:
        print(f"Reprise a la partie {agent.n_games}, record precedent : {record}.")

    plt.ion()
    target = None if n_games_target is None else agent.n_games + n_games_target

    while target is None or agent.n_games < target:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            elapsed = game.get_elapsed_time()
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games % TARGET_UPDATE_EVERY == 0:
                agent.trainer.update_target()

            if score > record:
                record = score

            # Sauvegarde systematique (pas seulement sur un record) : si on
            # interrompt la session en cours de route, rien n'est perdu.
            agent.model.save()
            save_stats(agent.n_games, record)

            print(f"Partie {agent.n_games} | Score {score} | Record {record} | Temps {format_time(elapsed)}")

            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    import sys
    # Par defaut : entrainement rapide, sans fenetre (headless).
    # `python agent.py --render` ouvre la fenetre pygame et respecte GAME_SPEED
    # (l'horloge du jeu de base n'est jamais modifiee), utile pour un temps de
    # partie comparable a une partie humaine.
    # `python agent.py --render --speed=30` ouvre la fenetre mais rejoue plus
    # vite, uniquement pour observer l'agent rapidement pendant les tests.
    render = "--render" in sys.argv
    speed = None
    for arg in sys.argv:
        if arg.startswith("--speed="):
            speed = int(arg.split("=", 1)[1])
    train(render=render, speed=speed)
