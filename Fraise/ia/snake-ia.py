import argparse
import csv
import importlib.util
import os
import random
import time
from collections import deque

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CHARGEMENT DU JEU DE BASE (SANS LE MODIFIER) ---
# On importe serpent-algo.py tel quel : clock, grille et scoring restent ceux du jeu d'origine.
# (import classique impossible à cause du tiret dans le nom du fichier)
_chemin_jeu = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serpent-algo.py")
_spec = importlib.util.spec_from_file_location("serpent", _chemin_jeu)
serpent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serpent)

# --- RÉCOMPENSES (propres au RL, n'affectent pas le score du jeu) ---
RECOMPENSE_POMME = 10
RECOMPENSE_DEFAITE = -10
RECOMPENSE_VICTOIRE = 100
RECOMPENSE_DEPLACEMENT = 0

# Anti-boucle infinie pendant l'entraînement : fin de partie si aucune pomme
# n'est mangée après LIMITE_ETAPES_PAR_SEGMENT * longueur du serpent étapes.
LIMITE_ETAPES_PAR_SEGMENT = 100

# --- ACTIONS ---
# L'agent choisit une action RELATIVE à sa direction actuelle :
#   [1, 0, 0] = tout droit, [0, 1, 0] = tourner à droite, [0, 0, 1] = tourner à gauche
# Elle est convertie en direction ABSOLUE (UP/DOWN/LEFT/RIGHT) du jeu.
# Avantage : le demi-tour (interdit par le jeu) n'est jamais proposé.
SENS_HORAIRE = [serpent.RIGHT, serpent.DOWN, serpent.LEFT, serpent.UP]


class SnakeGameAI:
    """Bloc GAME : enveloppe le jeu de base pour qu'un agent puisse y jouer."""

    def __init__(self, affichage=True, limite_etapes=True, facteur_vitesse=1):
        self.affichage = affichage
        self.limite_etapes = limite_etapes
        # Mode rapide : la partie tourne facteur_vitesse fois plus vite que GAME_SPEED,
        # et le temps compté est multiplié d'autant (x10 en vitesse = x10 sur le temps)
        self.facteur_vitesse = facteur_vitesse
        if self.affichage:
            pygame.init()
            self.screen = pygame.display.set_mode((serpent.SCREEN_WIDTH, serpent.SCREEN_HEIGHT))
            titre = "Snake IA - Deep Q-Learning (Fraise)"
            if facteur_vitesse != 1:
                titre += f" - MODE RAPIDE x{facteur_vitesse} (temps affiché = temps réel x{facteur_vitesse})"
            pygame.display.set_caption(titre)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 40)
        self.reset()

    def reset(self):
        """Démarre une nouvelle partie avec les objets du jeu de base."""
        self.snake = serpent.Snake()
        self.apple = serpent.Apple(self.snake.body)
        self.etapes_sans_pomme = 0
        self.victoire = False
        self.start_time = time.time()

    def temps_ecoule(self):
        """Temps de la partie en secondes, multiplié par le facteur de vitesse en mode rapide."""
        return (time.time() - self.start_time) * self.facteur_vitesse

    def play_step(self, action):
        """Joue une action. Retourne (reward, game_over, score)."""
        if self.affichage:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        self.snake.set_direction(self._direction_depuis_action(action))
        self.snake.move()
        self.etapes_sans_pomme += 1
        reward = RECOMPENSE_DEPLACEMENT

        # Mêmes vérifications que la boucle principale du jeu de base
        if self.snake.is_game_over():
            return RECOMPENSE_DEFAITE, True, self.snake.score

        if self.limite_etapes and self.etapes_sans_pomme > LIMITE_ETAPES_PAR_SEGMENT * len(self.snake.body):
            return RECOMPENSE_DEFAITE, True, self.snake.score

        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.etapes_sans_pomme = 0
            reward = RECOMPENSE_POMME
            if not self.apple.relocate(self.snake.body):
                self.victoire = True
                return RECOMPENSE_VICTOIRE, True, self.snake.score

        if self.affichage:
            self._dessiner()
            if self.facteur_vitesse == 1:
                self.clock.tick(serpent.GAME_SPEED)  # clock d'origine, comme le jeu de base
            else:
                # Mode rapide : même fréquence, mais attente précise. Avec tick() classique, ~3 ms
                # d'imprécision par image, multipliées par le facteur, gonfleraient le temps compté.
                self.clock.tick_busy_loop(serpent.GAME_SPEED * self.facteur_vitesse)

        return reward, False, self.snake.score

    def _direction_depuis_action(self, action):
        idx = SENS_HORAIRE.index(self.snake.direction)
        if action[1] == 1:  # droite
            idx = (idx + 1) % 4
        elif action[2] == 1:  # gauche
            idx = (idx - 1) % 4
        return SENS_HORAIRE[idx]

    def _dessiner(self):
        """Réutilise les fonctions d'affichage du jeu de base."""
        self.screen.fill(serpent.GRIS_FOND)
        zone_jeu = pygame.Rect(0, serpent.SCORE_PANEL_HEIGHT, serpent.SCREEN_WIDTH, serpent.SCREEN_WIDTH)
        pygame.draw.rect(self.screen, serpent.NOIR, zone_jeu)
        serpent.draw_grid(self.screen)
        self.apple.draw(self.screen)
        self.snake.draw(self.screen)
        # display_info affiche (maintenant - début) : on lui passe un début qui donne le temps compté
        serpent.display_info(self.screen, self.font, self.snake, time.time() - self.temps_ecoule())
        pygame.display.flip()


# --- BLOC MODEL (TORCH) ---

DOSSIER_MODELE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


class Linear_QNet(nn.Module):
    """Réseau DQN : état (14 valeurs) -> Q-valeur de chaque action (3 valeurs)."""

    def __init__(self, input_size=14, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name="model.pth", dossier=DOSSIER_MODELE):
        os.makedirs(dossier, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(dossier, file_name))

    def load(self, file_name="model.pth", dossier=DOSSIER_MODELE):
        self.load_state_dict(torch.load(os.path.join(dossier, file_name)))
        self.eval()


class QTrainer:
    """Entraîne le réseau avec l'équation de Bellman : Q_cible = r + gamma * max Q(s')."""

    def __init__(self, model, lr=0.001, gamma=0.9):
        self.model = model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Accepte une seule transition (mémoire courte) ou un lot (mémoire longue)."""
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        if state.dim() == 1:  # une seule transition -> lot de taille 1
            state, next_state, action = state.unsqueeze(0), next_state.unsqueeze(0), action.unsqueeze(0)
            reward, done = reward.unsqueeze(0), done.unsqueeze(0)

        # Cible de Bellman, calculée sans gradient : si la partie est finie, pas de futur
        with torch.no_grad():
            q_futur = self.model(next_state).max(dim=1).values
            q_cible = reward + self.gamma * q_futur * (~done)

        # Q prédit pour l'action réellement jouée (action one-hot -> indice)
        q_predit = self.model(state).gather(1, action.argmax(dim=1, keepdim=True)).squeeze(1)

        loss = self.criterion(q_predit, q_cible)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# --- BLOC AGENT ---

MAX_MEMOIRE = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
TAILLE_ETAT = 14
# Exploration : probabilité de jouer au hasard, de EPSILON_DEPART à 0 en NB_PARTIES_EXPLORATION parties
EPSILON_DEPART = 0.4
NB_PARTIES_EXPLORATION = 80


class Agent:
    """Bloc AGENT : observe le jeu, choisit les actions et fait apprendre le modèle."""

    def __init__(self):
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMOIRE)  # les plus anciens souvenirs sont oubliés
        self.model = Linear_QNet(TAILLE_ETAT, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)

    @staticmethod
    def get_state(game):
        """État à 14 valeurs : 3 dangers, 4 directions, 4 positions de pomme, 3 espaces libres."""
        snake = game.snake
        hx, hy = snake.head_pos
        direction = snake.direction
        g = serpent.GRID_SIZE

        # Coup à partir duquel chaque case du corps est libre : le segment n°i (tête = 0)
        # quitte sa case après (longueur - i) coups, un coup de plus si le serpent vient de manger.
        longueur = len(snake.body)
        retard = 1 if snake.grow_pending else 0
        libre_au_coup = {tuple(segment): longueur - i + retard for i, segment in enumerate(snake.body)}

        # Cases occupées APRÈS le prochain déplacement (coup 1) : la queue libère sa case,
        # sauf si le serpent vient de manger.
        occupees = {case for case, coup in libre_au_coup.items() if coup > 1}

        def danger(d):
            # Les murs se traversent (modulo) : seul le corps est dangereux
            return ((hx + d[0]) % g, (hy + d[1]) % g) in occupees

        idx = SENS_HORAIRE.index(direction)
        dir_droite = SENS_HORAIRE[(idx + 1) % 4]
        dir_gauche = SENS_HORAIRE[(idx - 1) % 4]

        # Position de la pomme par le chemin le plus court (en traversant les bords si besoin)
        ax, ay = game.apple.position
        dx = (ax - hx) % g  # 0 = même colonne, 1..g//2 = à droite, sinon à gauche
        dy = (ay - hy) % g  # 0 = même ligne,   1..g//2 = en bas,    sinon en haut

        def espace_libre(d):
            """Part des cases libres atteignables si la tête va dans la direction d, en tenant compte
            de la queue qui libère des cases en avançant. 0 = mort immédiate, 1 = rien n'est bloqué."""
            depart = ((hx + d[0]) % g, (hy + d[1]) % g)
            if depart in occupees:
                return 0.0
            vus = {depart}
            frontiere = [depart]  # cases atteintes au coup n°coup
            coup = 1
            while frontiere:  # propagation couche par couche (1 couche = 1 coup), bords traversants
                coup += 1
                suivante = []
                for x, y in frontiere:
                    for voisin in ((x + 1) % g, y), ((x - 1) % g, y), (x, (y + 1) % g), (x, (y - 1) % g):
                        # une case du corps est franchissable si son segment l'a quittée d'ici là
                        if voisin not in vus and libre_au_coup.get(voisin, 0) <= coup:
                            vus.add(voisin)
                            suivante.append(voisin)
                frontiere = suivante
            cases_libres = g * g - len(occupees)
            return min(1.0, (len(vus) - 1) / max(1, cases_libres - 1))

        state = [
            danger(direction),   # danger en face
            danger(dir_droite),  # danger à droite
            danger(dir_gauche),  # danger à gauche

            direction == serpent.LEFT,
            direction == serpent.RIGHT,
            direction == serpent.UP,
            direction == serpent.DOWN,

            dx > g // 2,         # pomme à gauche
            0 < dx <= g // 2,    # pomme à droite
            dy > g // 2,         # pomme en haut
            0 < dy <= g // 2,    # pomme en bas

            espace_libre(direction),   # espace libre tout droit
            espace_libre(dir_droite),  # espace libre à droite
            espace_libre(dir_gauche),  # espace libre à gauche
        ]
        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """Apprend immédiatement du coup qui vient d'être joué."""
        return self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """Rejoue un lot aléatoire de souvenirs (experience replay) en fin de partie."""
        if len(self.memory) > BATCH_SIZE:
            lot = random.sample(self.memory, BATCH_SIZE)
        else:
            lot = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*lot)
        return self.trainer.train_step(np.array(states), np.array(actions), np.array(rewards),
                                       np.array(next_states), np.array(dones))

    def get_action(self, state, exploration=True):
        """Epsilon-greedy : hasard au début (exploration), puis le modèle (exploitation)."""
        epsilon = EPSILON_DEPART * max(0, NB_PARTIES_EXPLORATION - self.n_games) / NB_PARTIES_EXPLORATION
        action = [0, 0, 0]
        if exploration and random.random() < epsilon:
            action[random.randrange(3)] = 1
        else:
            with torch.no_grad():
                q_valeurs = self.model(torch.tensor(state, dtype=torch.float))
            action[int(torch.argmax(q_valeurs))] = 1
        return action


# --- ENTRAÎNEMENT ET DÉMO ---

def train(nb_parties=300, dossier=DOSSIER_MODELE, seed=None):
    """Entraînement rapide sans affichage. Sauvegarde le meilleur modèle et l'historique en CSV.
    Retourne True si toutes les parties ont été jouées, False si interrompu (Ctrl+C)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    agent = Agent()
    game = SnakeGameAI(affichage=False)
    record, total_score, etapes = 0, 0, 0
    complet = True
    debut = time.time()

    os.makedirs(dossier, exist_ok=True)
    chemin_csv = os.path.join(dossier, "scores.csv")
    with open(chemin_csv, "w", newline="") as fichier:
        historique = csv.writer(fichier)
        historique.writerow(["partie", "score", "record", "moyenne", "etapes", "temps_ecoule_s", "victoire"])

        try:
            state_old = agent.get_state(game)
            while agent.n_games < nb_parties:
                action = agent.get_action(state_old)
                reward, done, score = game.play_step(action)
                state_new = agent.get_state(game)
                etapes += 1

                agent.train_short_memory(state_old, action, reward, state_new, done)
                agent.remember(state_old, action, reward, state_new, done)
                state_old = state_new

                if done:
                    victoire = game.victoire
                    game.reset()
                    state_old = agent.get_state(game)
                    agent.n_games += 1
                    agent.train_long_memory()

                    if score > record:
                        record = score
                        agent.model.save(dossier=dossier)

                    total_score += score
                    moyenne = total_score / agent.n_games
                    ecoule = time.time() - debut
                    historique.writerow([agent.n_games, score, record, f"{moyenne:.2f}", etapes, f"{ecoule:.1f}", victoire])
                    print(f"Partie {agent.n_games:4d} | Score {score:3d} | Record {record:3d} | "
                          f"Moyenne {moyenne:6.2f} | {ecoule:6.1f}s" + (" | VICTOIRE !" if victoire else ""))
                    etapes = 0
        except KeyboardInterrupt:
            print("\nEntraînement interrompu.")
            complet = False

    print(f"Terminé : record {record}, modèle dans {dossier}, historique dans {chemin_csv}")
    return complet


ALERTE_BOUCLE_DEMO = 500  # coups sans pomme avant d'afficher une alerte


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_temps(secondes):
    return f"{int(secondes // 60):02d}:{secondes % 60:04.1f}"


def demo(dossier=DOSSIER_MODELE, facteur_vitesse=1):
    """Partie réelle : modèle entraîné, affichage et clock d'origine (GAME_SPEED), logs dans le terminal.
    facteur_vitesse > 1 : mode rapide, le temps compté est le temps réel multiplié par ce facteur."""
    agent = Agent()
    agent.model.load(dossier=dossier)
    game = SnakeGameAI(affichage=True, limite_etapes=False, facteur_vitesse=facteur_vitesse)
    font_fin = pygame.font.Font(None, 80)

    print("=" * 78)
    print("  SNAKE IA - Deep Q-Learning (groupe Fraise)")
    print(f"  Modèle : {os.path.join(dossier, 'model.pth')} ({TAILLE_ETAT} entrées)")
    print(f"  Grille {serpent.GRID_SIZE}x{serpent.GRID_SIZE} | Clock {serpent.GAME_SPEED} FPS | "
          "ESPACE = rejouer après une partie | Échap = quitter")
    if facteur_vitesse != 1:
        print(f"  MODE RAPIDE x{facteur_vitesse} : jeu à {serpent.GAME_SPEED * facteur_vitesse} FPS, "
              f"tous les temps affichés = temps réel x{facteur_vitesse}")
    print("=" * 78, flush=True)

    parties = []  # (numéro, score, temps, victoire)
    record = None
    numero = 0
    try:
        while True:
            numero += 1
            game.reset()
            score_precedent, done = 0, False
            log(f"Partie {numero} | Début")

            while not done:
                state = agent.get_state(game)
                _, done, score = game.play_step(agent.get_action(state, exploration=False))
                temps = game.temps_ecoule()

                if score > score_precedent:
                    log(f"Partie {numero} | Score {score:3d} | Temps {format_temps(temps)}")
                    score_precedent = score
                elif not done and game.etapes_sans_pomme % ALERTE_BOUCLE_DEMO == 0:
                    log(f"Partie {numero} | ATTENTION : aucune pomme depuis {game.etapes_sans_pomme} coups (boucle ?)")

            temps = game.temps_ecoule()
            fin_partie = "VICTOIRE (grille remplie)" if game.victoire else "GAME OVER (collision avec le corps)"
            parties.append((numero, score, temps, game.victoire))
            # Classement : meilleur score, puis temps le plus court
            if record is None or (score, -temps) > (record[1], -record[2]):
                record = parties[-1]
            log(f"Partie {numero} | {fin_partie} | Score {score} | Temps {format_temps(temps)}")
            log(f"Record de la session : {record[1]} en {format_temps(record[2])} (partie {record[0]})"
                " | ESPACE = rejouer, Échap = quitter")

            # Écran de fin comme le jeu de base : score et chrono figés, ESPACE pour rejouer
            game._dessiner()
            if game.victoire:
                serpent.display_message(game.screen, font_fin, "VICTOIRE !", serpent.VERT)
            else:
                serpent.display_message(game.screen, font_fin, "GAME OVER", serpent.ROUGE)
            serpent.display_message(game.screen, game.font, "ESPACE pour rejouer.", serpent.BLANC, y_offset=100)
            pygame.display.flip()

            attente = True
            while attente:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        attente = False
                game.clock.tick(serpent.GAME_SPEED)
    except (SystemExit, KeyboardInterrupt):
        if not parties or parties[-1][0] != numero:
            log(f"Partie {numero} | Interrompue (non comptée)")
    finally:
        pygame.quit()

    print("=" * 78)
    print(f"  BILAN : {len(parties)} partie(s) terminée(s)")
    for n, s, t, v in parties:
        print(f"    Partie {n} : score {s} en {format_temps(t)}"
              + (" VICTOIRE" if v else ""))
    if record:
        print(f"  RECORD : {record[1]} en {format_temps(record[2])} (partie {record[0]})")
    print("=" * 78, flush=True)


# Au-delà de ce nombre de coups sans pomme, la partie réelle tournerait en boucle sans fin
LIMITE_BOUCLE_EVAL = 2000


def evaluate(nb_parties=100, dossier=DOSSIER_MODELE, seed=None):
    """Évalue le modèle comme en partie réelle (sans hasard ni limite), mais sans affichage.
    Temps de jeu = coups / GAME_SPEED, c'est-à-dire le chrono qu'afficherait la démo."""
    if seed is not None:
        random.seed(seed)
    agent = Agent()
    agent.model.load(dossier=dossier)
    game = SnakeGameAI(affichage=False, limite_etapes=False)
    resultats = []  # (score, temps de jeu en s, fin de partie)

    for _ in range(nb_parties):
        game.reset()
        coups, done = 0, False
        while not done:
            state = agent.get_state(game)
            _, done, score = game.play_step(agent.get_action(state, exploration=False))
            coups += 1
            if game.etapes_sans_pomme > LIMITE_BOUCLE_EVAL:
                break
        fin = "victoire" if game.victoire else ("boucle" if not done else "mort")
        resultats.append((score, coups / serpent.GAME_SPEED, fin))

    scores = [r[0] for r in resultats]
    total_temps = sum(r[1] for r in resultats)
    meilleur = max(resultats, key=lambda r: (r[0], -r[1]))  # meilleur score, puis le plus rapide
    print(f"Évaluation sur {nb_parties} parties :")
    print(f"  Record          : {meilleur[0]} en {meilleur[1]:.1f}s de jeu ({meilleur[0] / meilleur[1]:.3f} pomme/s)")
    print(f"  Score moyen     : {sum(scores) / nb_parties:.2f} (médiane {sorted(scores)[nb_parties // 2]})")
    print(f"  Pommes/s de jeu : {sum(scores) / total_temps:.3f} (global)")
    print(f"  Fins de partie  : " + ", ".join(f"{f} {sum(r[2] == f for r in resultats)}" for f in ("mort", "boucle", "victoire")))
    return resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake - Deep Q-Learning (groupe Fraise)")
    # Sans argument : partie réelle avec le modèle pré-entraîné (commande du professeur)
    sous = parser.add_subparsers(dest="mode")
    p_train = sous.add_parser("train", help="entraîner le modèle sans affichage")
    p_train.add_argument("--parties", type=int, default=300)
    p_train.add_argument("--seed", type=int, default=None)
    sous.add_parser("demo", help="(par défaut) partie réelle avec le modèle entraîné et logs")
    p_rapide = sous.add_parser("rapide", help="partie accélérée, temps compté = temps réel x facteur")
    p_rapide.add_argument("facteur", type=int, nargs="?", default=10, help="accélération (défaut : 10)")
    p_eval = sous.add_parser("eval", help="mesurer le modèle sur N parties réelles sans affichage")
    p_eval.add_argument("--parties", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train(nb_parties=args.parties, seed=args.seed)
    elif args.mode == "eval":
        evaluate(nb_parties=args.parties, seed=args.seed)
    else:
        if not os.path.exists(os.path.join(DOSSIER_MODELE, "model.pth")):
            log("Aucun modèle pré-entraîné trouvé : entraînement de 300 parties avant la partie réelle...")
            if not train(nb_parties=300, seed=2):
                os.remove(os.path.join(DOSSIER_MODELE, "model.pth"))  # modèle incomplet : on ne le garde pas
                raise SystemExit
        demo(facteur_vitesse=args.facteur if args.mode == "rapide" else 1)
