"""Configuration centralisée d'une expérience.

Règle du projet : aucun hyperparamètre ne doit être dispersé dans le code.
Une expérience est entièrement décrite par un objet `Config` sérialisable, et
doit pouvoir être relancée sans éditer une seule constante.

    python train.py --config configs/baseline.json
"""

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    """Tous les réglages d'un entraînement.

    Les valeurs par défaut correspondent à la baseline indicative du cadrage
    (§9). Elles ne sont pas déclarées optimales : c'est un point de départ que
    la recherche d'hyperparamètres devra remettre en question.
    """

    # --- Identité de l'expérience ---
    run_id: str = "baseline"
    algorithm: str = "dqn"  # dqn | ddqn | dueling_ddqn | dueling_ddqn_per
    notes: str = ""

    # --- Reproductibilité ---
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda

    # --- Réseau ---
    hidden_size: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.95
    gradient_clip: float = 10.0
    loss: str = "huber"  # huber | mse

    # --- Replay ---
    replay_capacity: int = 100_000
    batch_size: int = 256
    learning_starts: int = 1_000

    # --- Exploration ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 500

    # --- Target network ---
    # Hard update (intervalle) OU soft update (tau), jamais les deux.
    target_update_interval: int = 500
    tau: float = 0.0  # 0 = désactivé, on utilise le hard update

    # --- Entraînement ---
    episodes: int = 1_000

    # --- Évaluation ---
    eval_interval: int = 100
    eval_episodes: int = 30
    eval_seed_start: int = 1_000

    # --- Replay visuel ---
    replay_best_after_eval: bool = True
    show_replay_window: bool = True  # à couper pendant les grid searches
    replay_fps: int = 60  # cadence graphique uniquement, indépendante du moteur
    replay_speed_multiplier: int = 8

    # --- Récompense ---
    reward_profile: str = "course"  # course | experimental

    # --- Garde-fou expérimental (HORS règles officielles) ---
    #
    # Nécessité technique, pas confort d'apprentissage : avec epsilon à 0, une
    # politique déterministe encore imparfaite tourne en rond indéfiniment et
    # l'évaluation ne se termine jamais. Mesuré en pratique dès le premier
    # smoke test.
    #
    # 500 pas sans pomme est très généreux : la grille ne compte que 225 cases,
    # mais cela ne garantit pas l'absence de troncature sur un tore. Elle est
    # comptabilisée séparément (`truncation_rate`) et n'est ni une défaite ni
    # une victoire. La valeur apparaît dans la configuration et dans tous les
    # rapports. 0 désactive complètement le garde-fou.
    max_steps_without_food: int = 500
    # Seuil ANALYTIQUE, n'arrête jamais le jeu.
    long_without_food_threshold: int = 100

    # --- Prioritized Experience Replay ---
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 100_000
    priority_epsilon: float = 1e-6

    # --- Sorties ---
    output_dir: str = "runs"
    checkpoint_every: int = 500

    def __post_init__(self):
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 0:
            raise ValueError("checkpoint_every doit être un entier >= 0")
        if type(self.replay_fps) is not int or self.replay_fps < 1:
            raise ValueError("replay_fps doit être un entier >= 1")
        if type(self.max_steps_without_food) is not int or self.max_steps_without_food < 0:
            raise ValueError("max_steps_without_food doit être un entier >= 0")
        if type(self.long_without_food_threshold) is not int or self.long_without_food_threshold < 1:
            raise ValueError("long_without_food_threshold doit être un entier >= 1")
        if self.tau > 0 and self.target_update_interval > 0:
            raise ValueError(
                "hard update et soft update sont exclusifs : mettez "
                "`target_update_interval` à 0 pour utiliser `tau`"
            )
        if self.algorithm not in {
            "dqn",
            "ddqn",
            "dueling_ddqn",
            "dueling_ddqn_per",
        }:
            raise ValueError(f"algorithme inconnu : {self.algorithm!r}")

    # ------------------------------------------------------------------

    @property
    def uses_per(self):
        return self.algorithm.endswith("_per")

    @property
    def uses_double(self):
        return self.algorithm in {"ddqn", "dueling_ddqn", "dueling_ddqn_per"}

    @property
    def uses_dueling(self):
        return self.algorithm.startswith("dueling")

    def eval_seeds(self):
        """Liste déterministe des seeds d'évaluation de ce run."""
        return list(
            range(self.eval_seed_start, self.eval_seed_start + self.eval_episodes)
        )

    # ------------------------------------------------------------------

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data):
        """Construit une config, en refusant les clés inconnues.

        Une faute de frappe dans un JSON de configuration doit lever une
        erreur, pas être silencieusement ignorée : sinon on croit tester un
        hyperparamètre alors qu'on relance la valeur par défaut.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"clés de configuration inconnues : {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def replace(self, **overrides):
        """Copie modifiée, pour décliner une grille d'hyperparamètres."""
        data = self.to_dict()
        data.update(overrides)
        return Config.from_dict(data)
