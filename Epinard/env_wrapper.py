"""
Bloc "Game" du slide 4 : expose play_step(action) -> reward, game_over, score.

Réutilise Snake et Apple du jeu original, sans modifier serpent-algo.py.

RÈGLES RETENUES
---------------
Reward (slide 5, signaux d'apprentissage du RL) :
    pomme        : +10
    mort         : -10
    déplacement  : +0.1
    grille pleine: +100

Score (jeu, code original ligne 74) : +1 par pomme. Échelle distincte du reward.

Fin de partie : auto-morsure, ou grille pleine.
    Les murs ne tuent PAS : move() lignes 57-58 applique `% GRID_SIZE`,
    le plateau est un tore. check_wall_collision() ne peut donc jamais
    renvoyer True (commenté "ne fonctionne pas volontairement" dans l'original).
    Aucun seuil de points ne termine la partie.

Garde-fou anti-boucle : plafond de pas sans pomme. C'est de l'hygiène
d'entraînement (éviter les parties infinies), pas une règle de jeu.
"""
import time

from game_core import GRID_SIZE, Apple, Snake, turn

# Rewards du slide 5
REWARD_POMME = 10.0
REWARD_MORT = -10.0
REWARD_DEPLACEMENT = 0.1
REWARD_VICTOIRE = 100.0


class SnakeEnv:
    """Environnement RL construit par-dessus le jeu original."""

    def __init__(self, patience_factor=60, patience_min=150, shaping=1.0):
        # Plafond de pas sans pomme : proportionnel à la longueur du serpent,
        # car plus il est long, plus rejoindre une pomme demande de détours.
        self.patience_factor = patience_factor
        self.patience_min = patience_min
        # Reward shaping : petit bonus quand on se rapproche de la pomme,
        # petite pénalité quand on s'en éloigne. Somme nulle sur un
        # aller-retour, donc ça ne crée pas d'exploit ; ça donne juste un
        # signal DENSE vers la pomme. Sans lui, le +0.1 par déplacement
        # domine et l'agent apprend à survivre sans jamais manger.
        self.shaping = shaping
        self.reset()

    def _apple_dist(self):
        """Distance torique (Manhattan sur le tore) tête <-> pomme."""
        hx, hy = self.snake.head_pos
        ax, ay = self.apple.position
        dx = abs(hx - ax)
        dy = abs(hy - ay)
        return min(dx, GRID_SIZE - dx) + min(dy, GRID_SIZE - dy)

    def reset(self):
        self.snake = Snake()
        self.apple = Apple(self.snake.body)
        self.steps = 0
        self.steps_since_apple = 0
        self.start_time = time.time()
        self.game_over = False
        self.victory = False
        self.end_reason = None
        self.prev_dist = self._apple_dist()
        return self

    @property
    def score(self):
        """Score du jeu : +1 par pomme (échelle du code original)."""
        return self.snake.score

    @property
    def elapsed(self):
        return time.time() - self.start_time

    def _patience(self):
        return max(self.patience_min, self.patience_factor * len(self.snake.body))

    def play_step(self, action_index):
        """
        Joue un pas. action_index : 0 = tout droit, 1 = droite, 2 = gauche.
        Retourne (reward, game_over, score).
        """
        if self.game_over:
            return 0.0, True, self.score

        self.steps += 1
        self.steps_since_apple += 1

        # 1. Appliquer l'action via l'API du jeu original
        new_dir = turn(self.snake.direction, action_index)
        self.snake.set_direction(new_dir)
        self.snake.move()

        # 2. Mort ? Seule l'auto-morsure tue (plateau torique).
        if self.snake.check_self_collision():
            self.game_over = True
            self.end_reason = "collision"
            return REWARD_MORT, True, self.score

        reward = REWARD_DEPLACEMENT

        # Reward shaping : + si on se rapproche, - si on s'éloigne.
        new_dist = self._apple_dist()
        reward += self.shaping * (1.0 if new_dist < self.prev_dist else -1.0)
        self.prev_dist = new_dist

        # 3. Pomme mangée ?
        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.steps_since_apple = 0
            reward = REWARD_POMME

            # relocate() renvoie False quand il n'y a plus de case libre :
            # c'est la condition de victoire du jeu original (ligne 259).
            if not self.apple.relocate(self.snake.body):
                self.game_over = True
                self.victory = True
                self.end_reason = "victoire"
                return REWARD_VICTOIRE, True, self.score

            # Nouvelle pomme placée : on recalcule la distance de référence.
            self.prev_dist = self._apple_dist()

        # 4. Garde-fou anti-boucle
        if self.steps_since_apple > self._patience():
            self.game_over = True
            self.end_reason = "timeout"
            return reward, True, self.score

        return reward, False, self.score

    def stats(self):
        """Métriques d'un essai : temps, score, ratio score/temps."""
        t = self.elapsed
        return {
            "score": self.score,
            "temps": round(t, 2),
            "ratio_score_temps": round(self.score / t, 3) if t > 0 else 0.0,
            "steps": self.steps,
            "longueur": len(self.snake.body),
            "remplissage_pct": round(100 * len(self.snake.body) / (GRID_SIZE ** 2), 1),
            "fin": self.end_reason,
            "victoire": self.victory,
        }
