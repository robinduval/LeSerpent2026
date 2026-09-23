"""Snake RL : entraînement accélérable, évaluation à 5 Hz, durées réelles.

python snake-ia.py
python3 snake-ia.py --mode train --resume runs/<session>/latest.pt --episodes 100
python3 snake-ia.py --mode eval --resume runs/<session>/latest.pt --episodes 10
"""
import argparse
import csv
import importlib.util
import json
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import pygame
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("snake_base", ROOT / "serpent-algo.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
DIRECTIONS = [base.RIGHT, base.DOWN, base.LEFT, base.UP]
STATE_SIZE = 13
STATE_SIZES = {'basic': 13, 'spatial': 22}


def free_regions(occupied):
    """Tailles des composantes libres sur la grille torique (corps figé)."""
    n = base.GRID_SIZE
    unseen = {(x, y) for x in range(n) for y in range(n)} - occupied
    sizes = {}
    while unseen:
        start = unseen.pop()
        component = [start]
        for x, y in component:
            for dx, dy in DIRECTIONS:
                neighbour = ((x+dx) % n, (y+dy) % n)
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.append(neighbour)
        for pos in component:
            sizes[pos] = len(component)
    return sizes


# --- GAME (PyGame) : règles, état et avancement d'une partie ---
class Game:
    """Bloc Game des slides, réutilisant les objets et le rendu PyGame du socle."""
    def __init__(self, seed, features='basic'):
        self.features = features
        self._cached_key = None
        self._cached_state = None
        self.rng = random.Random(seed)
        self.snake = base.Snake()
        self.apple = base.Apple.__new__(base.Apple)
        self.apple.position = self.place_apple()
        self.done = False
        self.reason = "running"
        self.steps = 0
        self.without_food = 0

    def place_apple(self):
        # Même liste de cases libres et même tirage uniforme que le socle.
        free = [(x, y) for x in range(base.GRID_SIZE)
                for y in range(base.GRID_SIZE) if [x, y] not in self.snake.body]
        return self.rng.choice(free) if free else None

    def direction(self, action):
        return DIRECTIONS[(DIRECTIONS.index(self.snake.direction) + (0, 1, -1)[action]) % 4]

    def state(self):
        s = self.snake
        key = (tuple(map(tuple, s.body)), s.direction, s.grow_pending, self.apple.position)
        if key == self._cached_key:
            return list(self._cached_state)
        danger = []
        for action in range(3):
            dx, dy = self.direction(action)
            pos = [(s.head_pos[0] + dx) % base.GRID_SIZE,
                   (s.head_pos[1] + dy) % base.GRID_SIZE]
            occupied = s.body if s.grow_pending else s.body[:-1]
            danger.append(float(pos in occupied))
        apple = self.apple.position or tuple(s.head_pos)
        # Déplacement signé le plus court sur la grille torique, taille impaire.
        half = base.GRID_SIZE // 2
        dx = (apple[0] - s.head_pos[0] + half) % base.GRID_SIZE - half
        dy = (apple[1] - s.head_pos[1] + half) % base.GRID_SIZE - half
        result = danger + [float(s.direction == d) for d in DIRECTIONS] + [
            float(dx < 0), float(dx > 0), float(dy < 0), float(dy > 0),
            len(s.body) / base.GRID_SIZE ** 2, float(s.grow_pending)]
        if self.features == 'spatial':
            # Après un mouvement, les obstacles sont l'ancien corps, moins la
            # queue uniquement si elle bouge. La nouvelle tête reste accessible.
            occupied = set(map(tuple, s.body if s.grow_pending else s.body[:-1]))
            regions = free_regions(occupied)
            spaces, distances = [], []
            for action in range(3):
                vx, vy = self.direction(action)
                pos = ((s.head_pos[0]+vx) % base.GRID_SIZE,
                       (s.head_pos[1]+vy) % base.GRID_SIZE)
                spaces.append(regions.get(pos, 0) / base.GRID_SIZE ** 2)
                distance = base.GRID_SIZE
                for step in range(1, base.GRID_SIZE):
                    ahead = ((s.head_pos[0]+step*vx) % base.GRID_SIZE,
                             (s.head_pos[1]+step*vy) % base.GRID_SIZE)
                    if ahead in occupied:
                        distance = step
                        break
                distances.append(distance / base.GRID_SIZE)
            result += spaces + distances + [dx/half, dy/half, (abs(dx)+abs(dy))/(2*half)]
        self._cached_key, self._cached_state = key, result
        return list(result)

    def play_step(self, action):
        """Contrat des slides : action → (reward, game_over, score)."""
        if self.done:
            raise RuntimeError("Créer un nouvel environnement après la fin de partie")
        self.snake.set_direction(self.direction(action))
        self.snake.move()
        self.steps += 1
        self.without_food += 1
        reward = 0.1
        if self.snake.is_game_over():
            self.done, self.reason, reward = True, "body_collision", -10.0
        elif self.snake.head_pos == list(self.apple.position):
            self.snake.grow()
            self.without_food = 0
            reward = 10.0
            pos = self.place_apple()
            if pos is None:
                self.done, self.reason, reward = True, "victory", 100.0
            else:
                self.apple.position = pos
        return reward, self.done, self.snake.score

    def step(self, action):
        """Compatibilité avec les tests/outils de la première version."""
        reward, done, _ = self.play_step(action)
        return self.state(), reward, done


# Ancien nom gardé pour les scripts d'analyse existants.
Environment = Game


# --- MODEL (Torch) : estimation des valeurs Q des trois actions ---
class Linear_QNet(nn.Sequential):
    """DQN : entrées → couche cachée ReLU → valeurs Q.

    Les clés numériques 0.weight/2.weight sont conservées pour les checkpoints.
    """
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super().__init__(nn.Linear(input_size, hidden_size), nn.ReLU(),
                         nn.Linear(hidden_size, output_size))

    def predict(self, state):
        """Prédit l'action sans exploration ni mise à jour des poids."""
        with torch.no_grad():
            return int(self(torch.as_tensor(state, dtype=torch.float32)).argmax())


# --- AGENT : observation, exploration, mémoire et entraînement Double DQN ---
class Agent:
    def __init__(self, seed, features='basic'):
        self.features = features
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.model = Linear_QNet(STATE_SIZES[features])
        self.target = Linear_QNet(STATE_SIZES[features])
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=3e-4)
        self.memory = deque(maxlen=20_000)
        self.steps = 0
        self.updates = 0
        self.loss = None
        self.epsilon_floor = 0.05

    @property
    def online(self):
        """Nom Double DQN du modèle principal (agent.model dans les slides)."""
        return self.model

    def get_state(self, game):
        return game.state()

    @property
    def epsilon(self):
        return max(self.epsilon_floor, 1.0 - (1.0-self.epsilon_floor) * self.steps / 8000)

    def get_move(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(3)
        return self.model.predict(state)

    def act(self, state, training):
        """Ancien nom conservé pour les outils existants."""
        return self.get_move(state, training)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, transition):
        self.remember(*transition)
        self.steps += 1
        if len(self.memory) < 64:
            return
        states, actions, rewards, next_states, dones = zip(*self.rng.sample(list(self.memory), 64))
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        with torch.no_grad():
            choices = self.online(next_states).argmax(1, keepdim=True)
            future = self.target(next_states).gather(1, choices).squeeze(1)
            target = torch.tensor(rewards) + 0.99 * (~torch.tensor(dones)).float() * future
        predicted = self.online(states).gather(1, torch.tensor(actions).unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()
        self.loss = float(loss.detach())
        self.updates += 1
        if self.steps % 500 == 0:
            self.target.load_state_dict(self.online.state_dict())

    def act_batch(self, states):
        with torch.no_grad():
            actions = self.online(torch.tensor(states, dtype=torch.float32)).argmax(1).tolist()
        return [self.rng.randrange(3) if self.rng.random() < self.epsilon else action
                for action in actions]

    def save(self, path):
        data = dict(online=self.online.state_dict(), target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(), steps=self.steps, updates=self.updates,
                    memory=list(self.memory), rng=self.rng.getstate(), torch_rng=torch.get_rng_state(),
                    features=self.features, epsilon_floor=self.epsilon_floor)
        temporary = path.with_suffix(".tmp")
        torch.save(data, temporary)
        temporary.replace(path)

    def load(self, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get('features', 'basic') != self.features:
            raise ValueError('État incompatible : utiliser --warm-start pour basic → spatial')
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data.get("target", data["online"]))
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon_floor = data.get('epsilon_floor', 0.05)
        self.steps, self.updates = data.get("steps", 0), data.get("updates", 0)
        self.memory.extend(data.get("memory", []))
        if 'rng' in data:
            self.rng.setstate(data["rng"])
        if 'torch_rng' in data:
            torch.set_rng_state(data["torch_rng"])

    def warm_start(self, path):
        """Transfert exact du comportement basic, nouvelles entrées à poids zéro."""
        data = torch.load(path, map_location='cpu', weights_only=True)
        if self.features != 'spatial' or data.get('features', 'basic') != 'basic':
            raise ValueError('Le transfert nécessite un checkpoint basic et --features spatial')
        for name in ('online', 'target'):
            weights = dict(data[name])
            enlarged = torch.zeros((128, STATE_SIZES['spatial']))
            enlarged[:, :STATE_SIZE] = weights['0.weight']
            weights['0.weight'] = enlarged
            getattr(self, name).load_state_dict(weights)
        # Replay basic inexploitable sans les plateaux d'origine. Nouvel optimiseur.
        # Compteur conservé : même epsilon, même calendrier de synchronisation.
        self.steps = data['steps']
        self.updates = data['updates']


def game_timing(row):
    """Équivalent à 5 Hz pour le training ; temps mesuré pour l'évaluation."""
    duration = row['duration_s'] if row['mode'] == 'eval' else row['steps'] / base.GAME_SPEED
    return duration, row['score'] / duration if duration > 0 else 0.0


def publish(directory, rows, live, budget_start):
    elapsed = max(0, time.time() - budget_start)
    completed = [r for r in rows if r["completed"]]
    scores = [r["score"] for r in completed]
    summary = dict(live=live, episodes=rows, completed=len(completed),
                   session_status='finished' if live.get('session_finished') else 'running',
                   best_score=max(scores, default=0), mean_score=sum(scores) / max(1, len(scores)),
                   success_rate=sum(s >= 10 for s in scores) / max(1, len(scores)),
                   budget_elapsed_minutes=elapsed / 60, budget_remaining_minutes=max(0, 120-elapsed/60),
                   budget_start=datetime.fromtimestamp(budget_start).astimezone().isoformat())
    temp = directory / "metrics.tmp"
    temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temp.replace(directory / "metrics.json")
    points = " ".join(f"{30 + i * 680 / max(1,len(scores)-1):.1f},{170 - s * 140 / max(10,max(scores,default=0)):.1f}" for i,s in enumerate(scores))
    table = "".join(f'<tr><td>{r["episode"]}</td><td>{r["mode"]}</td><td>{r["score"]}</td><td>{game_timing(r)[0]:.1f}s</td><td>{r["steps"]}</td><td>{game_timing(r)[1]:.3f}</td><td>{r["duration_s"]:.2f}s</td><td>{r["reason"]}</td></tr>' for r in reversed(rows))
    game_duration, game_ratio = game_timing(live)
    refresh = '' if live.get('session_finished') else '<meta http-equiv="refresh" content="2">'
    html = f'''<!doctype html><html lang="fr"><meta charset="utf-8">{refresh}
<title>Citron · Suivi Snake RL</title><style>body{{background:#111827;color:#eef2ff;font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}}h1{{color:#bef264}}.cards{{display:flex;gap:14px;flex-wrap:wrap}}.card{{background:#1f2937;padding:20px;border-radius:12px;min-width:160px}}strong{{display:block;font-size:30px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #374151}}small,p{{color:#cbd5e1}}svg{{background:#1f2937;border-radius:12px;width:100%}}</style>
<h1>Citron · Snake RL</h1><p>Clock : {live.get('clock_hz', 5) or 'illimitée (entraînement)'} · serpents : {live.get('num_envs', 1)} · grille 15 × 15 · score : pommes · seuil : 10 points</p>
<p><b>Session {'terminée — modèle sauvegardé' if live.get('session_finished') and live['mode']=='train' else 'terminée' if live.get('session_finished') else 'en cours'}</b></p>
<div class="cards"><div class="card">Meilleur score terminé<strong>{summary['best_score']} / 10</strong></div><div class="card">Parties terminées<strong>{len(completed)}</strong></div><div class="card">Parties ≥ 10<strong>{summary['success_rate']:.0%}</strong></div><div class="card">Budget restant<strong id="remaining">{summary['budget_remaining_minutes']:.1f} min</strong></div></div>
<p>Début du budget : {summary['budget_start']} · session : {directory.name} · les scores d'entraînement ne constituent pas une évaluation du modèle figé.</p>
<p>Cadence de référence du jeu : 5 déplacements/s. En entraînement, durée équivalente = déplacements / 5 et score/s à 5 Hz = score / durée équivalente. Il s'agit d'une estimation pour cette trajectoire, pas d'une évaluation du modèle figé. En évaluation, la durée réellement mesurée est utilisée.</p>
<h2>Partie {live['episode']} · {live['mode']} · {live['reason']}</h2><p>Score : {live['score']} · durée {'mesurée' if live['mode']=='eval' else 'équivalente à 5 Hz'} : {game_duration:.1f}s · score/s à 5 Hz : {game_ratio:.3f} · déplacements : {live['steps']}</p>
<p>Déplacements sans pomme : {live['steps_without_food']} · epsilon : {live['epsilon']:.3f} · mises à jour : {live['updates']} · loss : {live['loss']}</p>
<details><summary>Performance de calcul — temps réellement écoulé</summary><p>Session : {live.get('session_elapsed_s', live['duration_s']):.1f}s · partie : {live['duration_s']:.2f}s · débit global : {live.get('throughput_steps_s', live['steps']/max(.001,live['duration_s'])):.1f} transitions/s. Ces mesures servent au budget d'entraînement, pas au ratio de jeu. Les durées des serpents simultanés se chevauchent.</p></details>
<h2>Score par partie terminée</h2><svg viewBox="0 0 750 200"><text x="10" y="20" fill="white">Score (max échelle : {max(10,max(scores,default=0))})</text><line x1="30" y1="170" x2="720" y2="170" stroke="#64748b"/><polyline points="{points}" fill="none" stroke="#bef264" stroke-width="3"/>{''.join(f'<circle cx="{30+i*680/max(1,len(scores)-1):.1f}" cy="{170-s*140/max(10,max(scores,default=0)):.1f}" r="5" fill="#bef264"/>' for i,s in enumerate(scores))}</svg>
<p>Un premier résultat valide le fonctionnement. Il ne permet pas encore de prédire l'atteinte de 10 points avant l'échéance.</p>
<table><tr><th>Partie</th><th>Mode</th><th>Score</th><th>Durée à 5 Hz*</th><th>Pas</th><th>Score/s à 5 Hz*</th><th>Temps écoulé</th><th>Fin</th></tr>{table}</table>
<p>* En entraînement : équivalent théorique à 5 Hz. En évaluation : temps chronométré. Le budget de deux heures utilise toujours le temps réellement écoulé.</p>
<script>function budget(){{document.getElementById('remaining').textContent=Math.max(0,({budget_start + 7200}-Date.now()/1000)/60).toFixed(1)+' min';}}budget();setInterval(budget,1000);</script></html>'''
    temp = directory / "dashboard.tmp"
    temp.write_text(html, encoding="utf-8")
    temp.replace(directory / "dashboard.html")


def run_parallel(args, agent, directory, budget_start, clock):
    """Environnements entrelacés sur CPU et une inférence groupée par tour."""
    session_start = time.monotonic()
    initial_steps = agent.steps
    rows = []
    slots = []
    next_episode = 1
    last_publish = session_start
    last_save = session_start

    def new_slot(index):
        return dict(env=Game(args.seed+index-1, agent.features), episode=index,
                    started=time.monotonic(), reward=0., first_ten=None)

    def record(slot):
        now = time.monotonic()
        env = slot['env']
        duration = now-slot['started']
        elapsed = now-session_start
        return dict(episode=slot['episode'], mode='train', seed=args.seed+slot['episode']-1,
                    score=env.snake.score, duration_s=duration, steps=env.steps,
                    score_per_second=env.snake.score/max(duration,.001), epsilon=agent.epsilon,
                    updates=agent.updates, training_steps=agent.steps, loss=agent.loss,
                    steps_without_food=env.without_food, time_to_10_s=slot['first_ten'],
                    reward=slot['reward'], reason=env.reason, completed=env.done,
                    clock_hz=args.train_fps, num_envs=args.num_envs,
                    session_elapsed_s=elapsed, session_steps=agent.steps-initial_steps,
                    throughput_steps_s=(agent.steps-initial_steps)/max(elapsed,.001))

    def finish(slot):
        row = record(slot)
        rows.append(row)
        with (directory/'episodes.csv').open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if len(rows)==1:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row),flush=True)

    for _ in range(min(args.num_envs,args.episodes)):
        slots.append(new_slot(next_episode))
        next_episode += 1
    while slots:
        clock.tick(args.train_fps)
        reason = None
        if any(e.type==pygame.QUIT for e in pygame.event.get()):
            reason='interrupted'
        elif time.time() >= budget_start+7200:
            reason='budget_exhausted'
        elif args.max_seconds and time.monotonic()-session_start >= args.max_seconds:
            reason='session_limit'
        if reason:
            for slot in slots:
                slot['env'].reason=reason
                finish(slot)
            break
        states=[agent.get_state(slot['env']) for slot in slots]
        actions=agent.act_batch(states)
        remaining=[]
        for slot,state,action in zip(slots,states,actions):
            env=slot['env']
            reward,done,score=env.play_step(action)
            new_state=agent.get_state(env)
            slot['reward']+=reward
            agent.learn((state,action,reward,new_state,done))
            if env.snake.score>=10 and slot['first_ten'] is None:
                slot['first_ten']=time.monotonic()-slot['started']
            if done:
                finish(slot)
                if next_episode<=args.episodes:
                    remaining.append(new_slot(next_episode))
                    next_episode+=1
            else:
                remaining.append(slot)
        slots=remaining
        if time.monotonic()-last_publish>=1:
            publish(directory,rows,record(slots[0]) if slots else rows[-1],budget_start)
            last_publish=time.monotonic()
        if time.monotonic()-last_save>=30:
            agent.save(directory/'latest.pt')
            agent.save(directory/f'checkpoint-{agent.steps}.pt')
            last_save=time.monotonic()
    agent.save(directory/'latest.pt')
    publish(directory,rows,dict(rows[-1],session_finished=True),budget_start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--features", choices=['basic', 'spatial'], default=None)
    parser.add_argument("--warm-start", type=Path, help="Transférer un modèle basic vers spatial, sans son replay")
    parser.add_argument("--epsilon-floor", type=float, default=None, help="Minimum d'exploration en entraînement")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate pour une nouvelle expérience")
    parser.add_argument("--headless", action="store_true", help="Pas de fenêtre")
    parser.add_argument("--train-fps", type=int, default=0, help="Clock entraînement : 0 = illimitée")
    parser.add_argument("--num-envs", type=int, default=1, help="Serpents simultanés, entraînement uniquement")
    parser.add_argument("--max-seconds", type=float, default=0, help="Limite réelle de session ; interruption signalée")
    parser.add_argument("--budget-start", default=None, help="Début du budget de deux heures ; aucune limite horaire en démo par défaut")
    args = parser.parse_args()
    enforce_budget = args.mode == 'train' or args.budget_start is not None
    if args.budget_start is None:
        start = datetime.now().astimezone()
        if args.mode == 'train':
            start = start.replace(hour=20,minute=0,second=0,microsecond=0)
        args.budget_start = start.isoformat()
    if args.mode == "eval" and not args.resume:
        args.resume = ROOT / 'model' / 'model.pt'
    if args.resume and not args.resume.is_file():
        parser.error(f'Modèle introuvable : {args.resume}. Récupérer aussi le dossier model du dépôt.')
    if args.epsilon_floor is not None and not 0<=args.epsilon_floor<=1:
        parser.error('--epsilon-floor doit être entre 0 et 1')
    if args.learning_rate is not None and not 0<args.learning_rate<1:
        parser.error('--learning-rate doit être entre 0 et 1, exclus')
    if args.mode=='eval' and (args.epsilon_floor is not None or args.learning_rate is not None):
        parser.error('Les réglages d’apprentissage ne s’appliquent pas en évaluation')
    if args.resume and args.warm_start:
        parser.error('--resume et --warm-start sont exclusifs')
    if args.warm_start and args.features != 'spatial':
        parser.error('--warm-start nécessite --features spatial')
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if args.mode=='train' and saved.get('inference_only'):
            parser.error("Pour reprendre l'entraînement, utiliser un checkpoint complet de runs/ avec --resume")
        checkpoint_features = saved.get('features', 'basic')
        if args.features is not None and args.features != checkpoint_features:
            parser.error('Le checkpoint et --features sont incompatibles')
        args.features = checkpoint_features
    args.features = args.features or 'basic'
    if args.episodes < 1:
        parser.error("--episodes doit être positif")
    if args.num_envs<1 or args.train_fps<0 or args.max_seconds<0:
        parser.error("Nombre d'environnements positif ; clock et durée non négatives")
    if args.mode=='eval' and args.num_envs!=1:
        parser.error("Évaluation : un seul serpent à 5 Hz")
    budget_start = datetime.fromisoformat(args.budget_start).timestamp()
    directory = ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.mode)
    directory.mkdir(parents=True)
    actual_fps=base.GAME_SPEED if args.mode=='eval' else args.train_fps
    (directory / "config.json").write_text(json.dumps(dict(mode=args.mode, seed=args.seed, episodes=args.episodes, grid=base.GRID_SIZE, clock=actual_fps, num_envs=args.num_envs, max_seconds=args.max_seconds, features=args.features, state_size=STATE_SIZES[args.features], warm_start=str(args.warm_start), timing='real_monotonic_wall_time', budget_start=args.budget_start, resume=str(args.resume), algorithm="Double DQN", rewards=dict(move=.1, apple=10, death=-10, victory=100)), indent=2))
    torch.set_num_threads(1)
    agent = Agent(args.seed, args.features)
    if args.resume:
        agent.load(args.resume)
    elif args.warm_start:
        agent.warm_start(args.warm_start)
    if args.epsilon_floor is not None:
        agent.epsilon_floor=args.epsilon_floor
    if args.learning_rate is not None:
        for group in agent.optimizer.param_groups:
            group['lr']=args.learning_rate
    config_path=directory/'config.json'
    config=json.loads(config_path.read_text())
    config.update(epsilon_floor=agent.epsilon_floor, learning_rate=agent.optimizer.param_groups[0]['lr'], selection_objective='maximum_score')
    config_path.write_text(json.dumps(config,indent=2))
    pygame.init()
    screen = None if args.headless or args.num_envs>1 else pygame.display.set_mode((850, base.SCREEN_HEIGHT))
    if screen:
        pygame.display.set_caption(f"Citron | Snake RL | {actual_fps or 'illimite'} Hz | objectif 10")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    rows = []
    stop = False
    print(f"DASHBOARD={directory / 'dashboard.html'}", flush=True)
    session_started=time.monotonic()
    initial_steps=agent.steps
    try:
        if args.num_envs>1:
            run_parallel(args,agent,directory,budget_start,clock)
            return
        for episode in range(1, args.episodes + 1):
            env = Game(args.seed + episode - 1, agent.features)
            started = time.monotonic()
            first_ten = None
            total_reward = 0.0
            while not env.done:
                clock.tick(actual_fps)
                if any(e.type == pygame.QUIT for e in pygame.event.get()):
                    stop = True
                    env.reason = "interrupted"
                    break
                if enforce_budget and time.time() >= budget_start + 7200:
                    stop = True
                    env.reason = "budget_exhausted"
                    break
                if args.max_seconds and time.monotonic()-session_started>=args.max_seconds:
                    stop=True
                    env.reason='session_limit'
                    break
                state = agent.get_state(env)
                action = agent.get_move(state, args.mode == "train")
                reward, done, score = env.play_step(action)
                next_state = agent.get_state(env)
                total_reward += reward
                if args.mode == "train":
                    agent.learn((state, action, reward, next_state, done))
                duration = time.monotonic() - started
                if env.snake.score >= 10 and first_ten is None:
                    first_ten = duration
                live = dict(episode=episode, mode=args.mode, seed=args.seed+episode-1,
                            score=env.snake.score, duration_s=duration, steps=env.steps,
                            score_per_second=env.snake.score/max(duration,.001),
                            epsilon=agent.epsilon if args.mode == "train" else 0,
                            updates=agent.updates, training_steps=agent.steps, loss=agent.loss,
                            steps_without_food=env.without_food, time_to_10_s=first_ten,
                            reward=total_reward, reason=env.reason, completed=env.done,
                            clock_hz=actual_fps, num_envs=1, session_elapsed_s=time.monotonic()-session_started,
                            throughput_steps_s=(agent.steps-initial_steps if args.mode=='train' else sum(r['steps'] for r in rows)+env.steps)/max(.001,time.monotonic()-session_started))
                if env.steps % 5 == 0 or env.done:
                    publish(directory, rows, live, budget_start)
                if screen:
                    screen.fill(base.GRIS_FOND)
                    pygame.draw.rect(screen, base.NOIR, (0,base.SCORE_PANEL_HEIGHT,base.SCREEN_WIDTH,base.SCREEN_WIDTH))
                    base.draw_grid(screen)
                    env.apple.draw(screen)
                    env.snake.draw(screen)
                    lines = [f"Partie {episode} | {args.mode}", f"Score {env.snake.score} / objectif 10", f"Duree reelle {duration:.1f} s | pas {env.steps}", f"Epsilon {live['epsilon']:.3f}", f"Updates {agent.updates}", f"Sans pomme : {env.without_food} pas", f"Budget restant : {max(0,(budget_start+7200-time.time())/60):.1f} min", f"Clock : {actual_fps or 'illimitee'}", "Dashboard : runs/.../dashboard.html"]
                    for i,line in enumerate(lines):
                        screen.blit(font.render(line,True,base.BLANC),(460,30+35*i))
                    screen.blit(font.render(f"Score : {env.snake.score} | {duration:.1f}s",True,base.BLANC),(15,25))
                    pygame.display.flip()
            duration = time.monotonic() - started
            row = dict(episode=episode, mode=args.mode, seed=args.seed+episode-1,
                       score=env.snake.score, duration_s=duration, steps=env.steps,
                       score_per_second=env.snake.score/max(duration,.001),
                       epsilon=agent.epsilon if args.mode=="train" else 0,
                       updates=agent.updates, training_steps=agent.steps, loss=agent.loss,
                       steps_without_food=env.without_food, time_to_10_s=first_ten,
                       reward=total_reward, reason=env.reason, completed=env.done,
                       clock_hz=actual_fps, num_envs=1, session_elapsed_s=time.monotonic()-session_started,
                       throughput_steps_s=(agent.steps-initial_steps if args.mode=='train' else sum(r['steps'] for r in rows)+env.steps)/max(.001,time.monotonic()-session_started))
            rows.append(row)
            with (directory / "episodes.csv").open("a", newline="") as f:
                writer = csv.DictWriter(f,fieldnames=list(row))
                if episode == 1:
                    writer.writeheader()
                writer.writerow(row)
            if args.mode == "train":
                agent.save(directory / "latest.pt")
            publish(directory, rows, dict(row, session_finished=stop or episode==args.episodes), budget_start)
            print(json.dumps(row), flush=True)
            if stop:
                break
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
