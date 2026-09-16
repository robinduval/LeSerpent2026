"""Gestion centralisée des graines aléatoires.

Une seule fonction configure tous les générateurs du projet, pour qu'aucune
expérience ne dépende d'un état aléatoire oublié quelque part.

Convention de séparation des seeds (§24 du cadrage) :

    entraînement   0     à 9
    validation     1000  à 1049
    test final     10000 à 10099

Les seeds de test final ne doivent JAMAIS servir à régler des
hyperparamètres : elles mesurent la généralisation sur de l'inconnu.
"""

import random

import numpy as np
import torch

TRAIN_SEEDS = tuple(range(0, 10))
VALIDATION_SEEDS = tuple(range(1000, 1050))
TEST_SEEDS = tuple(range(10000, 10100))


def set_global_seed(seed, deterministic_torch=False):
    """Fixe Python, NumPy, PyTorch CPU et CUDA.

    Args:
        seed: la graine.
        deterministic_torch: force les noyaux CUDA déterministes. Plus lent,
            réservé aux runs de reproduction exacte.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def capture_rng_state():
    """Photographie l'état des générateurs, pour reprendre un entraînement."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restaure un état capturé par `capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
