"""Chargement du moteur fourni par l'enseignant, sans le modifier.

Le fichier s'appelle `serpent-algo.py` (tiret) : il faut passer par importlib.
"""
import importlib.util
from pathlib import Path

ENGINE_PATH = Path(__file__).resolve().parent.parent / "serpent-algo.py"

_engine = None


def load_engine():
    global _engine
    if _engine is None:
        spec = importlib.util.spec_from_file_location("serpent_engine", ENGINE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _engine = module
    return _engine
