"""Moteur Snake pur, sans dépendance à Pygame.

Différences assumées avec `serpent-algo.py`, et pourquoi :

1. Plateau torique, comme le modulo du socle : traverser un bord ne tue pas.
2. Aucun import Pygame : l'entraînement headless ne doit rien afficher.
   Le rendu vit dans `render.py` et consomme les snapshots produits ici.
3. RNG locale (`random.Random(seed)`) au lieu du `random` global, pour que
   deux parties de même seed soient strictement identiques.
4. Positions en tuples immuables plutôt qu'un mélange listes / tuples.

Tout le reste (dimension de grille, longueur initiale, position de départ,
direction initiale, +1 par pomme, victoire quand la grille est pleine,
interdiction du demi-tour) est repris à l'identique du socle.
"""

import random
from dataclasses import dataclass, field

from . import rules


@dataclass(frozen=True)
class RewardProfile:
    """Barème de récompense. Le profil par défaut est celui des slides.

    Attention : la récompense est le signal d'apprentissage, elle n'a rien à
    voir avec le score officiel. Le score officiel reste le nombre de pommes
    mangées et n'est jamais influencé par ce barème.
    """

    name: str = "course"
    apple: float = 10.0
    death: float = -10.0
    step: float = 0.1
    victory: float = 100.0


COURSE_REWARDS = RewardProfile()


@dataclass
class StepResult:
    """Résultat d'un pas de simulation."""

    reward: float
    score: int
    terminated: bool = False
    won: bool = False
    ate: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.terminated and self.truncated:
            raise ValueError("terminated et truncated sont mutuellement exclusifs")

    @property
    def done(self):
        """Compatibilité : fin de collecte, pas masque de bootstrap RL."""
        return self.terminated or self.truncated


class SnakeGame:
    """Moteur de jeu headless, déterministe à seed fixée.

    Usage :
        game = SnakeGame(seed=42)
        game.reset()
        result = game.step(action)  # action dans [0, 3]
    """

    def __init__(
        self,
        seed=None,
        reward_profile=COURSE_REWARDS,
        max_steps_without_food=None,
    ):
        """
        Args:
            seed: graine de placement des pommes. `None` = non déterministe.
            reward_profile: barème de récompense (voir `RewardProfile`).
            max_steps_without_food: garde-fou EXPÉRIMENTAL contre les boucles
                infinies. `None` par défaut : la règle officielle ne connaît
                que deux fins de partie (corps, victoire). Si une valeur
                est fournie, l'épisode est marqué `truncated` et non `won` ;
                cette troncature doit apparaître dans la configuration et dans
                le rapport d'expérience.
        """
        self.seed = seed
        self.reward_profile = reward_profile
        if max_steps_without_food is not None and (
            type(max_steps_without_food) is not int or max_steps_without_food < 0
        ):
            raise ValueError("max_steps_without_food doit être un entier positif, 0 ou None")
        self.max_steps_without_food = max_steps_without_food or None
        self._rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """Réinitialise une partie. Repositionne le serpent comme le socle."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)

        head_x = rules.GRID_SIZE // 4
        head_y = rules.GRID_SIZE // 2
        # Corps du socle : tête, puis deux segments vers la gauche.
        self.body = [(head_x - i, head_y) for i in range(rules.INITIAL_LENGTH)]
        self.direction = rules.RIGHT
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.terminated = False
        self.truncated = False
        self.won = False
        self._grow_pending = False
        self.food = self._place_food()
        return self

    @property
    def head(self):
        return self.body[0]

    @property
    def done(self):
        """Vraie fin du jeu OU fin expérimentale de collecte."""
        return self.terminated or self.truncated

    # ------------------------------------------------------------------
    # Pomme
    # ------------------------------------------------------------------

    def _place_food(self):
        """Tire une case libre. Retourne None si la grille est pleine.

        On itère sur une liste ordonnée de façon déterministe pour qu'une seed
        donnée produise toujours exactement la même suite de pommes.
        """
        occupied = set(self.body)
        free = [
            (x, y)
            for x in range(rules.GRID_SIZE)
            for y in range(rules.GRID_SIZE)
            if (x, y) not in occupied
        ]
        if not free:
            return None
        return self._rng.choice(free)

    # ------------------------------------------------------------------
    # Collisions
    # ------------------------------------------------------------------

    def hits_body(self, point, body=None):
        """Vrai si `point` touche le corps (hors tête, comme dans le socle)."""
        body = self.body if body is None else body
        return point in body[1:]

    def is_collision(self, point=None):
        """Collision actuelle, ou prédiction corporelle du prochain pas.

        Un point proposé est ramené sur le tore. La queue est libérée avant
        le test de collision, sauf si une croissance est déjà en attente.
        """
        if point is None:
            return self.hits_body(self.head)
        point = (point[0] % rules.GRID_SIZE, point[1] % rules.GRID_SIZE)
        occupied = self.body if self._grow_pending else self.body[:-1]
        return point in occupied

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def legal_action_mask(self):
        """Masque des actions valides : le demi-tour immédiat est interdit.

        Le socle ignore silencieusement la touche opposée. On expose plutôt
        l'information pour que le réseau ne gaspille pas une sortie sur une
        action qui serait de toute façon ignorée.
        """
        banned = rules.opposite(self.direction)
        return [direction != banned for direction in rules.ACTIONS]

    def step(self, action):
        """Exécute une action et fait avancer le jeu d'une case.

        Args:
            action: entier dans [0, 3] (0 haut, 1 bas, 2 gauche, 3 droite).

        Returns:
            StepResult.
        """
        if self.done:
            raise RuntimeError("step() appelé sur une partie terminée ; reset() d'abord")
        if not 0 <= action < rules.N_ACTIONS:
            raise ValueError(f"action invalide : {action!r}")

        # 1. Direction. Une action opposée est ignorée (comportement du socle).
        new_direction = rules.ACTIONS[action]
        if new_direction != rules.opposite(self.direction):
            self.direction = new_direction

        # 2. Nouvelle tête sur le plateau torique officiel.
        new_head = (
            (self.head[0] + self.direction[0]) % rules.GRID_SIZE,
            (self.head[1] + self.direction[1]) % rules.GRID_SIZE,
        )

        self.steps += 1
        self.steps_since_food += 1

        # 3. Avance le corps. On applique l'ordre du socle : insertion de la
        #    tête, puis retrait de la queue si le serpent ne grandit pas. La
        #    case libérée par la queue est donc jouable, ce qui est correct.
        self.body.insert(0, new_head)
        if self._grow_pending:
            self._grow_pending = False
        else:
            self.body.pop()

        # 4. Auto-morsure.
        if self.hits_body(new_head):
            self.terminated = True
            return StepResult(
                reward=self.reward_profile.death,
                terminated=True,
                score=self.score,
                info={"cause": "self"},
            )

        # 5. Pomme.
        if new_head == self.food:
            self.score += 1
            self._grow_pending = True
            self.steps_since_food = 0
            reward = self.reward_profile.apple

            # Victoire : la grille est pleine. On compte la croissance en
            # attente, car elle est déjà acquise — la queue ne sera pas
            # retirée au prochain pas.
            #
            # Sans ce `+ 1`, il resterait exactement une case libre (l'ancienne
            # position de la queue), on y placerait une dernière pomme, et
            # cette pomme serait en pratique inatteignable : elle est adjacente
            # à la queue, pas à la tête. La partie ne pourrait jamais être
            # gagnée. Ce n'est pas un assouplissement des règles : le score
            # officiel est identique, seule la détection de fin est corrigée.
            grid_is_full = len(self.body) + 1 >= rules.TOTAL_CELLS

            self.food = None if grid_is_full else self._place_food()
            if self.food is None:
                self.terminated = True
                self.won = True
                reward += self.reward_profile.victory
                return StepResult(
                    reward=reward,
                    terminated=True,
                    score=self.score,
                    won=True,
                    ate=True,
                    info={"cause": "victory"},
                )
            return StepResult(reward=reward, score=self.score, ate=True)

        # 6. Déplacement normal.
        if (
            self.max_steps_without_food is not None
            and self.steps_since_food >= self.max_steps_without_food
        ):
            # Troncature expérimentale, explicitement distincte d'une défaite.
            self.truncated = True
            return StepResult(
                reward=self.reward_profile.step,
                score=self.score,
                truncated=True,
                info={"cause": "truncated"},
            )

        return StepResult(
            reward=self.reward_profile.step, score=self.score
        )

    # ------------------------------------------------------------------
    # Sérialisation pour les replays
    # ------------------------------------------------------------------

    def snapshot(self, action=None, reward=None):
        """Photo de l'état courant, suffisante pour rejouer sans PyTorch."""
        return {
            "head": list(self.head),
            "body": [list(p) for p in self.body],
            "food": list(self.food) if self.food else None,
            "direction": list(self.direction),
            "action": action,
            "reward": reward,
            "score": self.score,
            "steps": self.steps,
            "done": self.done,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "cause": "truncated" if self.truncated else (
                "victory" if self.won else "self" if self.terminated else None
            ),
        }
