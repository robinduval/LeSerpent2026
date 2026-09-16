"""Journalisation des métriques et écriture des replays.

Deux formats, pour deux usages :

    metrics.jsonl  une ligne par épisode, écrite au fil de l'eau. Robuste : un
                   run interrompu garde tout ce qui a été mesuré avant.
    metrics.csv    export final, plus commode pour un tableur ou pandas.
"""

import csv
import json
import os
import subprocess
import time

from .rules import RULESET


class MetricsLogger:
    """Écrit les métriques d'un run en JSONL, puis exporte un CSV."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        if os.path.exists(self.jsonl_path) and os.path.getsize(self.jsonl_path):
            raise ValueError("métriques existantes : utiliser un nouveau répertoire de run")
        self.rows = []
        self._handle = open(self.jsonl_path, "a", encoding="utf-8")

    def log(self, **row):
        """Enregistre une ligne et la vide immédiatement sur le disque."""
        row["ruleset"] = RULESET
        self.rows.append(row)
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()
        return row

    def export_csv(self, path=None):
        """Exporte toutes les lignes, en unifiant les colonnes rencontrées."""
        if not self.rows:
            return None
        path = path or os.path.join(self.run_dir, "metrics.csv")
        columns = []
        for row in self.rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.export_csv()
        self.close()
        return False


def git_commit():
    """Commit courant, pour tracer l'expérience. None hors dépôt Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def environment_info():
    """Contexte machine, à joindre à tout rapport final."""
    import platform

    import torch

    return {
        "ruleset": RULESET,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_replay(path, episode, config, algorithm, block=None, record=None):
    """Sérialise une trajectoire en JSON, rejouable sans PyTorch.

    Args:
        episode: le dictionnaire produit par `evaluate.play_episode` avec
            `record_frames=True`.
    """
    from . import rules

    payload = {
        "ruleset": episode["ruleset"],
        "run_id": config.run_id,
        "algorithm": algorithm,
        "seed": episode["seed"],
        "score": episode["score"],
        "steps": episode["steps"],
        "won": episode["won"],
        "cause": episode["cause"],
        "truncated": episode["truncated"],
        "terminated": episode["terminated"],
        "metrics": {key: value for key, value in episode.items() if key != "frames"},
        "max_steps_without_food": episode["max_steps_without_food"],
        "episode": block,
        "record": record,
        "grid_size": rules.GRID_SIZE,
        "frames": episode["frames"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


def load_replay(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
