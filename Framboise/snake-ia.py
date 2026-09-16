"""
Snake piloté par apprentissage par renforcement (Q-learning tabulaire).

Usage :
    python snake-ia.py                       # joue avec l'agent embarqué dans ce fichier (pygame seul requis)
    python snake-ia.py train                 # enchaîne toutes les phases de PHASES (numpy requis)
    python snake-ia.py train --phase p2      # lance une seule phase (après ajustement à la main)
    python snake-ia.py train --no-watch      # sans visualisation en fin de phase
    python snake-ia.py watch --phase p1      # revoit le top 3 d'une phase déjà entraînée
    python snake-ia.py watch --phase p1 --top 1
    python snake-ia.py export --phase p2     # embarque le meilleur agent de p2 dans ce fichier (--rank N sinon)

L'entraînement est stocké dans runs/ (local, non versionné). Seul l'agent exporté voyage avec le script.

Jeu : HAUT/BAS vitesse, ESPACE rejouer, ECHAP fermer.
Visualisation : ESPACE pause, HAUT/BAS vitesse, N nouvelle partie, ECHAP fermer.
"""

import argparse
import base64
import itertools
import json
import multiprocessing as mp
import os
import pprint
import random
import re
import sys
import time
import zlib
from collections import deque
from pathlib import Path

# --- CONSTANTES DE JEU (identiques au jeu de base) ---
GRID_SIZE = 15
CELL_SIZE = 30
GAME_SPEED = 5              # images par seconde en mode jeu (1 pas par image)
SCORE_PANEL_HEIGHT = 80
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

BLANC = (255, 255, 255)
NOIR = (0, 0, 0)
ORANGE = (255, 165, 0)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS_FOND = (50, 50, 50)
GRIS_GRILLE = (80, 80, 80)

# --- RESSOURCES ---
RAM_BUDGET_GB = 24          # 50 % des 48 Go
MAIN_RESERVE_MB = 1024      # processus principal
WORKER_EST_MB = 150         # estimation initiale, remplacée par le pic mesuré après chaque phase
RUNS_DIR = Path(__file__).resolve().parent / "runs"

# --- ÉVALUATION / CLASSEMENT ---
EVAL_GAMES = 300            # parties gloutonnes (epsilon = 0) par agent : ±1 pomme d'erreur type
EVAL_SEED = 1_000_000       # mêmes parties pour tous les agents : comparaison équitable
BUDGET_STEPS = 1000         # classement : pommes mangées dans les 1000 premiers pas (200 s à 5 fps)
TOP_VIEW = 3

# --- HYPERPARAMÈTRES ---
DEFAULTS = dict(
    episodes=3000,
    alpha=0.4,
    gamma=0.8,
    eps_start=1.0,
    eps_min=0.01,
    eps_decay=0.99,
    r_apple=10.0,
    r_death=-10.0,
    r_closer=0.1,
    r_farther=-0.15,
    r_step=-0.01,
    timeout_factor=100,     # fin de partie après timeout_factor * longueur pas sans pomme
    init_from=None,         # nom d'une phase : démarre depuis sa meilleure table Q
)

# Chaque phase = produit cartésien de "grid" x "seeds", appliqué sur DEFAULTS + "base".
# Après visualisation : ajuster la phase suivante ici puis `train --phase <nom>`.
PHASES = [
    {"name": "v2_hyper", "seeds": 5, "grid": {
        "alpha": [0.2, 0.4, 0.6],
        "gamma": [0.7, 0.8, 0.9],
    }},
    {"name": "v2_reward", "seeds": 3, "base": {"init_from": "v2_hyper", "eps_start": 0.1, "episodes": 2000},
     "grid": {
        "r_step": [0.0, -0.01, -0.05],
        "r_farther": [-0.1, -0.15, -0.3],
    }},
]


# --- JEU (logique du jeu de base, sans pygame) ---

class Snake:
    """Représente le serpent, sa position, sa direction et son corps."""
    def __init__(self):
        self.head_pos = [GRID_SIZE // 4, GRID_SIZE // 2]
        self.body = [self.head_pos,
                     [self.head_pos[0] - 1, self.head_pos[1]],
                     [self.head_pos[0] - 2, self.head_pos[1]]]
        self.direction = RIGHT
        self.grow_pending = False
        self.score = 0

    def set_direction(self, new_dir):
        """Change la direction, empêchant le mouvement inverse immédiat."""
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

    def move(self):
        """Déplace le serpent d'une case (la grille est torique)."""
        new_head_pos = [(self.head_pos[0] + self.direction[0]) % GRID_SIZE,
                        (self.head_pos[1] + self.direction[1]) % GRID_SIZE]
        self.body.insert(0, new_head_pos)
        self.head_pos = new_head_pos
        if not self.grow_pending:
            self.body.pop()
        else:
            self.grow_pending = False

    def grow(self):
        """Prépare le serpent à grandir au prochain mouvement."""
        self.grow_pending = True
        self.score += 1

    def is_game_over(self):
        """Seule l'auto-morsure tue : les bords bouclent."""
        return self.head_pos in self.body[1:]


class Apple:
    """Représente la pomme ; tirage via un générateur dédié pour rejouer une partie à l'identique."""
    def __init__(self, snake_body, rng):
        self.rng = rng
        self.position = self.random_position(snake_body)

    def random_position(self, occupied_positions):
        available = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                     if [x, y] not in occupied_positions]
        return self.rng.choice(available) if available else None

    def relocate(self, snake_body):
        new_pos = self.random_position(snake_body)
        if new_pos:
            self.position = new_pos
            return True
        return False


# --- ENVIRONNEMENT RL ---

STATE_BITS = 12
STATE_SIZE = 2 ** STATE_BITS


class SnakeEnv:
    """Actions relatives : 0 tout droit, 1 droite, 2 gauche.

    État (12 bits), tout relatif au cap du serpent :
      - 3 bits : mort immédiate pour chaque action
      - 4 bits : pomme devant / derrière / à droite / à gauche
      - 3 bits : piège pour chaque action (espace atteignable < longueur et queue inatteignable)
      - 2 bits : action menant au plus grand espace (0, 1 ou 2)
    """
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.rng = random.Random(seed)

    def reset(self):
        self.snake = Snake()
        self.apple = Apple(self.snake.body, self.rng)
        self.steps = 0
        self.since_food = 0
        self.won = False
        return self.state()

    def dirs(self):
        dx, dy = self.snake.direction
        return [(dx, dy), (-dy, dx), (dy, -dx)]

    def free_times(self):
        """Case du corps -> nombre de pas avant qu'elle se libère (la queue part en premier)."""
        n = len(self.snake.body)
        delay = 1 if self.snake.grow_pending else 0
        return {tuple(seg): n - i + delay for i, seg in enumerate(self.snake.body)}

    def food_delta(self):
        """Écart signé tête -> pomme par le plus court chemin torique."""
        def delta(a, h):
            v = (a - h) % GRID_SIZE
            return v - GRID_SIZE if v > GRID_SIZE // 2 else v
        return (delta(self.apple.position[0], self.snake.head_pos[0]),
                delta(self.apple.position[1], self.snake.head_pos[1]))

    def distance(self):
        fx, fy = self.food_delta()
        return abs(fx) + abs(fy)

    @staticmethod
    def escape_space(start, free_at, limit):
        """Espace atteignable en entrant dans start au pas 1, plafonné à limit.

        Une case du corps est traversable si on y arrive après sa libération : l'atteindre
        revient à suivre sa queue, ce qui laisse toujours une sortie -> retourne limit.
        """
        if free_at.get(start, 0) > 1:
            return 0
        if start in free_at:
            return limit
        seen = {start}
        queue = deque([(start, 1)])
        while queue:
            (x, y), t = queue.popleft()
            for dx, dy in (UP, DOWN, LEFT, RIGHT):
                n = ((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
                if n in seen:
                    continue
                if n in free_at:
                    if t + 1 >= free_at[n]:
                        return limit
                    continue
                seen.add(n)
                if len(seen) >= limit:
                    return limit
                queue.append((n, t + 1))
        return len(seen)

    def state(self):
        hx, hy = self.snake.head_pos
        dx, dy = self.snake.direction
        n = len(self.snake.body)
        free_at = self.free_times()
        cells = [((hx + ax) % GRID_SIZE, (hy + ay) % GRID_SIZE) for ax, ay in self.dirs()]
        fx, fy = self.food_delta()
        ahead, right = fx * dx + fy * dy, fx * -dy + fy * dx
        space = [self.escape_space(c, free_at, n) for c in cells]
        best = max(range(3), key=lambda a: space[a])

        bits = [free_at.get(c, 0) > 1 for c in cells]
        bits += [ahead > 0, ahead < 0, right > 0, right < 0]
        bits += [s < n for s in space]
        bits += [best & 1, best >> 1]
        return sum(int(b) << i for i, b in enumerate(bits))

    def step(self, action):
        """Retourne (état, récompense, terminé). L'état terminal vaut 0 (jamais utilisé)."""
        c = self.cfg
        self.snake.set_direction(self.dirs()[action])
        old_dist = self.distance()
        self.snake.move()
        self.steps += 1
        self.since_food += 1

        if self.snake.is_game_over() or self.since_food > c["timeout_factor"] * len(self.snake.body):
            return 0, c["r_death"], True

        if self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.since_food = 0
            if not self.apple.relocate(self.snake.body):
                self.won = True
                return 0, c["r_apple"], True
            return self.state(), c["r_apple"], False

        reward = c["r_closer"] if self.distance() < old_dist else c["r_farther"]
        return self.state(), reward + c["r_step"], False


# --- ENTRAÎNEMENT (processus workers) ---

def peak_rss_mb():
    try:
        import resource
    except ImportError:         # Windows : pas de mesure, le garde-fou RAM est inactif
        return 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10   # octets sur macOS, Ko sur Linux


def q_path(phase_name, run_id):
    return RUNS_DIR / phase_name / "q" / f"{run_id}.npy"


def load_ranked(phase_name):
    path = RUNS_DIR / phase_name / "results.json"
    if not path.exists():
        sys.exit(f"Phase {phase_name} introuvable dans {RUNS_DIR} (lancer `train --phase {phase_name}`)")
    return json.loads(path.read_text(encoding="utf-8"))["ranked"]


def best_q(phase_name):
    import numpy as np
    ranked = load_ranked(phase_name)
    if not ranked:
        raise RuntimeError(f"Aucun agent classé dans la phase {phase_name}")
    return np.load(q_path(phase_name, ranked[0]["run_id"]))


def evaluate(choose, cfg):
    """Joue EVAL_GAMES parties avec choose(état) -> action."""
    scores, budget, steps, timeouts = [], [], 0, 0
    for g in range(EVAL_GAMES):
        env = SnakeEnv(cfg, EVAL_SEED + g)
        s, done = env.reset(), False
        in_budget = None
        while not done:
            s, _, done = env.step(choose(s))
            if env.steps == BUDGET_STEPS:
                in_budget = env.snake.score
        scores.append(env.snake.score)
        budget.append(env.snake.score if in_budget is None else in_budget)
        steps += env.steps
        timeouts += not env.won and env.since_food > cfg["timeout_factor"] * len(env.snake.body)
    return {
        "budget_apples": sum(budget) / EVAL_GAMES,   # critère de classement : score dans un temps fixé
        "mean_score": sum(scores) / EVAL_GAMES,
        "max_score": int(max(scores)),
        "mean_steps": steps / EVAL_GAMES,
        "ratio": sum(scores) / max(steps, 1),       # pommes par pas sur la partie entière
        "timeouts": timeouts,
    }


def train_run(job):
    """Entraîne puis évalue un agent. Sauvegarde sa table Q sur disque (pas de transfert via pipe)."""
    import numpy as np
    cfg, seed, run_id = job["cfg"], job["seed"], job["run_id"]
    base = {"run_id": run_id, "label": job["label"], "seed": seed, "cfg": cfg}
    t0 = time.time()
    rng = random.Random(seed)
    env = SnakeEnv(cfg, seed)

    Q = np.zeros((STATE_SIZE, 3))
    if cfg["init_from"]:
        init = best_q(cfg["init_from"])
        if init.shape != Q.shape:
            return {**base, "error": f"table Q de {cfg['init_from']} incompatible (ancien format d'état)"}
        Q = init.copy()

    eps = cfg["eps_start"]
    for ep in range(cfg["episodes"]):
        s, done = env.reset(), False
        while not done:
            a = rng.randrange(3) if rng.random() < eps else int(Q[s].argmax())
            s2, r, done = env.step(a)
            target = r if done else r + cfg["gamma"] * Q[s2].max()
            Q[s, a] += cfg["alpha"] * (target - Q[s, a])
            s = s2
        eps = max(cfg["eps_min"], eps * cfg["eps_decay"])
        if ep % 100 == 0 and peak_rss_mb() > job["rss_cap_mb"]:
            return {**base, "error": f"RAM worker > {job['rss_cap_mb']:.0f} Mo"}

    np.save(q_path(job["phase"], run_id), Q)
    return {**base, **evaluate(lambda st: int(Q[st].argmax()), cfg), "train_s": time.time() - t0, "rss_mb": peak_rss_mb()}


# --- ORCHESTRATION DES PHASES ---

def expand_phase(phase):
    base = {**DEFAULTS, **phase.get("base", {})}
    grid = phase.get("grid", {})
    jobs = []
    for i, values in enumerate(itertools.product(*grid.values())):
        combo = dict(zip(grid, values))
        cfg = {**base, **combo}
        label = " ".join(f"{k}={v}" for k, v in combo.items())
        for seed in range(phase.get("seeds", 1)):
            jobs.append({"cfg": cfg, "seed": seed, "run_id": f"c{i:03d}_s{seed}",
                         "label": label, "phase": phase["name"]})
    return jobs


def plan_workers(n_jobs):
    """Nombre de workers limité par les cœurs, le nombre de runs et RAM_BUDGET_GB."""
    est_file = RUNS_DIR / "worker_mb.json"
    worker_mb = json.loads(est_file.read_text())["mb"] if est_file.exists() else WORKER_EST_MB
    worker_mb = max(worker_mb, 1.0)     # mesure à 0 sous Windows
    budget_mb = RAM_BUDGET_GB * 1024 - MAIN_RESERVE_MB
    workers = max(1, min(os.cpu_count() or 1, n_jobs, int(budget_mb // worker_mb)))
    return workers, budget_mb / workers, worker_mb


def efficiency(r):
    """Pommes dans les BUDGET_STEPS premiers pas : mourir tôt et faire des détours coûtent tous les deux."""
    return r["budget_apples"]


def rank(results):
    return sorted((r for r in results if "error" not in r), key=efficiency, reverse=True)


def print_table(ranked, n=10):
    print(f"\n{'#':>2}  {'run':<9} {'@' + str(BUDGET_STEPS):>6} {'score':>6} {'max':>4} {'ratio':>7}"
          f" {'pas':>6} {'bloq':>4} {'s':>5}  paramètres")
    for i, r in enumerate(ranked[:n], 1):
        print(f"{i:>2}  {r['run_id']:<9} {r['budget_apples']:>6.1f} {r['mean_score']:>6.1f} {r['max_score']:>4}"
              f" {r['ratio']:>7.4f} {r['mean_steps']:>6.0f} {r['timeouts']:>4} {r['train_s']:>5.0f}  {r['label']}")


def run_phase(phase):
    name = phase["name"]
    (RUNS_DIR / name / "q").mkdir(parents=True, exist_ok=True)
    jobs = expand_phase(phase)
    workers, cap_mb, worker_mb = plan_workers(len(jobs))
    for job in jobs:
        job["rss_cap_mb"] = cap_mb
    print(f"Phase {name} : {len(jobs)} runs, {workers} workers "
          f"(estimation {worker_mb:.0f} Mo/worker, plafond {cap_mb:.0f} Mo, budget {RAM_BUDGET_GB} Go)")

    results, t0 = [], time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(train_run, jobs), 1):
            results.append(res)
            info = res["error"] if "error" in res else f"@{BUDGET_STEPS} {res['budget_apples']:.1f}  score {res['mean_score']:.1f}"
            print(f"[{k}/{len(jobs)}] {time.time() - t0:6.0f}s  {res['run_id']}  {info}")

    ranked = rank(results)      # toutes les tables Q sont conservées (~100 Ko chacune)

    (RUNS_DIR / name / "results.json").write_text(json.dumps(
        {"phase": phase, "ranked": ranked, "all": results}, indent=2), encoding="utf-8")
    measured = [r["rss_mb"] for r in results if r.get("rss_mb")]
    if measured:
        (RUNS_DIR / "worker_mb.json").write_text(json.dumps({"mb": max(measured) * 1.5}))

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n{len(errors)} runs en erreur, ex. : {errors[0]['run_id']} : {errors[0]['error']}")
    print_table(ranked)
    return ranked


# --- AGENT EMBARQUÉ (voyage avec le script : seule l'action gloutonne par état est conservée) ---

POLICY_BLOCK = re.compile(r"^# --- POLITIQUE EMBARQUÉE.*?^# --- FIN POLITIQUE EMBARQUÉE ---$", re.S | re.M)


def encode_policy(Q):
    return base64.b64encode(zlib.compress(bytes(int(a) for a in Q.argmax(axis=1)), 9)).decode("ascii")


def decode_policy(data):
    return list(zlib.decompress(base64.b64decode(data)))


def export(phase_name, rank_index):
    """Réécrit le bloc POLITIQUE EMBARQUÉE de ce fichier avec l'agent choisi."""
    import numpy as np
    ranked = load_ranked(phase_name)
    if not 1 <= rank_index <= len(ranked):
        sys.exit(f"--rank doit être entre 1 et {len(ranked)}")
    r = ranked[rank_index - 1]
    data = encode_policy(np.load(q_path(phase_name, r["run_id"])))

    # Vérification : la politique décodée rejoue exactement l'évaluation de l'agent
    policy = decode_policy(data)
    check = evaluate(lambda st: policy[st], r["cfg"])
    if any(abs(check[k] - r[k]) > 1e-9 for k in check):
        sys.exit(f"Export refusé : politique décodée {check} != évaluation {r}")

    info = {"phase": phase_name, "run_id": r["run_id"], "label": r["label"], "cfg": r["cfg"],
            **check, "exported": time.strftime("%Y-%m-%d %H:%M")}
    chunks = "".join(f'    "{data[i:i + 88]}"\n' for i in range(0, len(data), 88))
    block = ("# --- POLITIQUE EMBARQUÉE (générée par `python snake-ia.py export`, ne pas modifier à la main) ---\n"
             f"POLICY_INFO = {pprint.pformat(info, width=100, sort_dicts=False)}\n"
             f"POLICY_DATA = (\n{chunks})\n"
             "# --- FIN POLITIQUE EMBARQUÉE ---")
    path = Path(__file__).resolve()
    src, n = POLICY_BLOCK.subn(lambda _: block, path.read_text(encoding="utf-8"))
    if n != 1:
        sys.exit("Bloc POLITIQUE EMBARQUÉE introuvable dans le script")
    path.write_text(src, encoding="utf-8")
    print(f"Agent {phase_name}/{r['run_id']} embarqué ({len(data)} caractères) : "
          f"{check['budget_apples']:.1f} pommes en {BUDGET_STEPS} pas, score {check['mean_score']:.1f}."
          " Commit + push du script suffisent.")


# --- AFFICHAGE (pygame importé à l'usage : les workers n'en ont pas besoin) ---

def draw_board(pygame, screen, env, x0, y0, cell):
    board = GRID_SIZE * cell
    pygame.draw.rect(screen, NOIR, (x0, y0, board, board))
    for k in range(GRID_SIZE + 1):
        pygame.draw.line(screen, GRIS_GRILLE, (x0 + k * cell, y0), (x0 + k * cell, y0 + board))
        pygame.draw.line(screen, GRIS_GRILLE, (x0, y0 + k * cell), (x0 + board, y0 + k * cell))
    if env.apple.position:
        ax, ay = env.apple.position
        pygame.draw.rect(screen, ROUGE, (x0 + ax * cell, y0 + ay * cell, cell, cell), border_radius=5)
    for j, (sx, sy) in reversed(list(enumerate(env.snake.body))):   # tête dessinée en dernier
        rect = (x0 + sx * cell, y0 + sy * cell, cell, cell)
        pygame.draw.rect(screen, ORANGE if j == 0 else VERT, rect)
        pygame.draw.rect(screen, NOIR, rect, 2 if j == 0 else 1)


def display_message(pygame, screen, font, message, color=BLANC, y_offset=0):
    text = font.render(message, True, color)
    rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + y_offset))
    bg = rect.inflate(40, 40)
    pygame.draw.rect(screen, NOIR, bg, border_radius=10)
    pygame.draw.rect(screen, BLANC, bg, 2, border_radius=10)
    screen.blit(text, rect)


def play():
    """Mode par défaut : l'agent embarqué joue au jeu de base."""
    import pygame

    if POLICY_INFO is None:
        sys.exit("Aucun agent embarqué : `python snake-ia.py train` puis `python snake-ia.py export --phase <nom>`.")
    policy = decode_policy(POLICY_DATA)
    cfg = POLICY_INFO["cfg"]
    if len(policy) != STATE_SIZE:
        sys.exit("Agent embarqué corrompu (taille incohérente) : refaire l'export.")

    width = GRID_SIZE * CELL_SIZE
    pygame.init()
    screen = pygame.display.set_mode((width, width + SCORE_PANEL_HEIGHT))
    pygame.display.set_caption(f"Snake IA - agent {POLICY_INFO['phase']}/{POLICY_INFO['run_id']}")
    font_main, font_small, font_big = pygame.font.Font(None, 34), pygame.font.Font(None, 22), pygame.font.Font(None, 80)
    clock = pygame.time.Clock()
    fps = GAME_SPEED

    def new_game():
        env = SnakeEnv(cfg, None)       # pommes aléatoires à chaque partie
        return env, env.reset(), time.time(), None

    env, s, start, end = new_game()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    fps = min(fps * 2, 640)
                elif event.key == pygame.K_DOWN:
                    fps = max(fps // 2, 1)
                elif event.key == pygame.K_SPACE and end:
                    env, s, start, end = new_game()

        if not end:
            s, _, done = env.step(policy[s])
            if done:
                end = time.time()

        elapsed = (end or time.time()) - start
        screen.fill(GRIS_FOND)
        draw_board(pygame, screen, env, 0, SCORE_PANEL_HEIGHT, CELL_SIZE)
        pygame.draw.line(screen, BLANC, (0, SCORE_PANEL_HEIGHT - 2), (width, SCORE_PANEL_HEIGHT - 2), 2)
        fill_rate = len(env.snake.body) / GRID_SIZE ** 2 * 100
        screen.blit(font_main.render(f"Score: {env.snake.score}", True, BLANC), (10, 12))
        time_text = font_main.render(f"Temps: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}", True, BLANC)
        screen.blit(time_text, time_text.get_rect(topright=(width - 10, 12)))
        info = (f"Remplissage {fill_rate:.1f}%  pommes/s {env.snake.score / max(elapsed, 1e-9):.2f}"
                f"  pommes/pas {env.snake.score / max(env.steps, 1):.3f}  {fps} fps")
        screen.blit(font_small.render(info, True, BLANC), (10, 50))

        if end:
            if env.won:
                display_message(pygame, screen, font_big, "VICTOIRE !", VERT)
            elif env.since_food > cfg["timeout_factor"] * len(env.snake.body):
                display_message(pygame, screen, font_big, "BLOQUÉ", ROUGE)
            else:
                display_message(pygame, screen, font_big, "GAME OVER", ROUGE)
            display_message(pygame, screen, font_main, "ESPACE pour rejouer.", BLANC, y_offset=100)

        pygame.display.flip()
        clock.tick(fps)


def watch(phase_name, top=TOP_VIEW):
    import numpy as np
    import pygame

    ranked = load_ranked(phase_name)[:top]
    if not ranked:
        print(f"Aucun agent à afficher pour la phase {phase_name}")
        return
    agents = [(r, np.load(q_path(phase_name, r["run_id"]))) for r in ranked]

    cell, header, pad = 24, 95, 10
    board = GRID_SIZE * cell
    width = len(agents) * (board + pad) + pad
    height = header + board + pad

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    fps, paused, game_seed = 20, False, EVAL_SEED

    def new_games(seed):
        games = []
        for r, Q in agents:
            env = SnakeEnv(r["cfg"], seed)
            games.append({"env": env, "Q": Q, "s": env.reset(), "done": False})
        return games

    games = new_games(game_seed)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    fps = min(fps * 2, 640)
                elif event.key == pygame.K_DOWN:
                    fps = max(fps // 2, 1)
                elif event.key == pygame.K_n:
                    game_seed += 1
                    games = new_games(game_seed)

        if not paused:
            for g in games:
                if not g["done"]:
                    g["s"], _, g["done"] = g["env"].step(int(g["Q"][g["s"]].argmax()))

        all_done = all(g["done"] for g in games)
        status = "terminé - N : nouvelle partie" if all_done else ("pause" if paused else f"{fps} fps")
        pygame.display.set_caption(f"Top {len(agents)} - phase {phase_name} - partie {game_seed - EVAL_SEED} - {status}")
        screen.fill(GRIS_FOND)

        for i, ((r, _), g) in enumerate(zip(agents, games)):
            x0 = pad + i * (board + pad)
            env = g["env"]
            ratio = env.snake.score / max(env.steps, 1)
            lines = [
                f"#{i + 1}  {r['run_id']}" + ("  MORT" if g["done"] and not env.won else "  VICTOIRE" if env.won else ""),
                r["label"],
                f"partie : score {env.snake.score}  pas {env.steps}  ratio {ratio:.3f}",
                f"éval : @{BUDGET_STEPS} {r['budget_apples']:.1f}  score {r['mean_score']:.1f}  ratio {r['ratio']:.3f}",
            ]
            screen.set_clip((x0, 0, board, header))
            for j, line in enumerate(lines):
                screen.blit(font.render(line, True, BLANC), (x0, 8 + j * 21))
            screen.set_clip(None)
            draw_board(pygame, screen, env, x0, header, cell)

        pygame.display.flip()
        clock.tick(fps if not all_done else 20)


# --- POLITIQUE EMBARQUÉE (générée par `python snake-ia.py export`, ne pas modifier à la main) ---
POLICY_INFO = {'phase': 'v2_hyper',
 'run_id': 'c002_s0',
 'label': 'alpha=0.2 gamma=0.9',
 'cfg': {'episodes': 3000,
         'alpha': 0.2,
         'gamma': 0.9,
         'eps_start': 1.0,
         'eps_min': 0.01,
         'eps_decay': 0.99,
         'r_apple': 10.0,
         'r_death': -10.0,
         'r_closer': 0.1,
         'r_farther': -0.15,
         'r_step': -0.01,
         'timeout_factor': 100,
         'init_from': None},
 'budget_apples': 85.65666666666667,
 'mean_score': 144.30333333333334,
 'max_score': 208,
 'mean_steps': 2923.2433333333333,
 'ratio': 0.04936411953389671,
 'timeouts': 1,
 'exported': '2026-09-16 21:22'}
POLICY_DATA = (
    "eNrtllEKwCAMQ03uf+ixWR1Wq7i/uj6QYsM+GoMupTkwaoFGPQeqHaf6aQ5gcOarjATf7UbOD19veS+UFro8Pj2q"
    "/LkLIVgN0IYMa6ezrYG74DfL6u/q7i9b6dXZqOYTnTxg/h9D5ju8VKu/qwd+AjD8wZQHH6ID7fmC8uJZ3wdBEBhc"
    "59IA6w=="
)
# --- FIN POLITIQUE EMBARQUÉE ---


# --- POINT D'ENTRÉE ---

def main():
    parser = argparse.ArgumentParser(description="Snake - apprentissage par renforcement")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("play", help="(défaut) l'agent embarqué joue")
    train = sub.add_parser("train", help="entraîne les phases de PHASES")
    train.add_argument("--phase", help="nom d'une seule phase à lancer")
    train.add_argument("--no-watch", action="store_true", help="pas de visualisation en fin de phase")
    view = sub.add_parser("watch", help="visualise les meilleurs agents d'une phase")
    view.add_argument("--phase", required=True)
    view.add_argument("--top", type=int, default=TOP_VIEW)
    exp = sub.add_parser("export", help="embarque un agent entraîné dans ce fichier")
    exp.add_argument("--phase", required=True)
    exp.add_argument("--rank", type=int, default=1, help="rang dans le classement de la phase (1 = meilleur)")
    args = parser.parse_args()

    if args.cmd in (None, "play"):
        play()
        return
    if args.cmd == "watch":
        watch(args.phase, args.top)
        return
    if args.cmd == "export":
        export(args.phase, args.rank)
        return

    phases = [p for p in PHASES if args.phase in (None, p["name"])]
    if not phases:
        sys.exit(f"Phase inconnue : {args.phase}")
    for i, phase in enumerate(phases):
        run_phase(phase)
        if not args.no_watch:
            watch(phase["name"])
        if i < len(phases) - 1:
            answer = input(f"\nEntrée : phase {phases[i + 1]['name']}  |  q : arrêter pour ajuster PHASES > ")
            if answer.strip().lower() == "q":
                print(f"Relancer ensuite avec : python snake-ia.py train --phase {phases[i + 1]['name']}")
                break


if __name__ == "__main__":
    main()
