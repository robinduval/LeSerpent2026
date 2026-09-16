"""15 x 15 pygame presentation, one learned action per original 5 Hz tick."""
from __future__ import annotations
import json
import time
import statistics
import pygame

from .env import Env, GAME_SPEED, GRID_SIZE
from .agent import DQNAgent
from .evaluation import action_for, checkpoint_id
from .training import append_json
from .policy_io import load_policy, policy_kind

CELL = 30
PANEL = 80
WIDTH = GRID_SIZE*CELL


class Display:
    def __init__(self, label, hybrid=False):
        pygame.init()
        self.screen=pygame.display.set_mode((WIDTH,WIDTH+PANEL))
        pygame.display.set_caption('Snake RL - Double DQN - Chou')
        self.font=pygame.font.Font(None,27)
        self.big=pygame.font.Font(None,54)
        self.label=label
        self.hybrid=hybrid

    def draw(self,env,start_time,ended=False):
        screen=self.screen
        screen.fill((50,50,50))
        pygame.draw.rect(screen,(0,0,0),(0,PANEL,WIDTH,WIDTH))
        for x in range(0,WIDTH,CELL):
            pygame.draw.line(screen,(80,80,80),(x,PANEL),(x,WIDTH+PANEL))
        for y in range(PANEL,WIDTH+PANEL,CELL):
            pygame.draw.line(screen,(80,80,80),(0,y),(WIDTH,y))
        for i,(x,y) in enumerate(env.body):
            rect=pygame.Rect(x*CELL,y*CELL+PANEL,CELL,CELL)
            pygame.draw.rect(screen,(255,165,0) if i==0 else (0,200,0),rect)
            pygame.draw.rect(screen,(0,0,0),rect,2 if i==0 else 1)
        if env.apple and not env.completed:
            x,y=env.apple
            rect=pygame.Rect(x*CELL,y*CELL+PANEL,CELL,CELL)
            pygame.draw.rect(screen,(200,0,0),rect,border_radius=5)
            pygame.draw.circle(screen,(255,255,255),(rect.x+21,rect.y+9),3)
        elapsed=time.time()-start_time
        labels=[(f'Score : {env.score}',(10,12)),
                (f'Temps : {int(elapsed//60):02d}:{int(elapsed%60):02d}',(290,12)),
                (f'{len(env.body)/225:.1%}  |  {"RL + filtre" if self.hybrid else "RL"}  |  5 Hz',(10,47))]
        for label,pos in labels:
            screen.blit(self.font.render(label,True,(255,255,255)),pos)
        if ended:
            rect=pygame.Rect(35,220,380,140)
            pygame.draw.rect(screen,(0,0,0),rect,border_radius=10)
            pygame.draw.rect(screen,(255,255,255),rect,2,border_radius=10)
            label='VICTOIRE !' if env.completed else 'GAME OVER'
            text=self.big.render(label,True,(0,200,0) if env.completed else (220,60,60))
            screen.blit(text,text.get_rect(center=(225,265)))
            text=self.font.render('ESPACE : rejouer   |   ECHAP : quitter',True,(255,255,255))
            screen.blit(text,text.get_rect(center=(225,320)))
        pygame.display.flip()


def play(checkpoint,base,seed=None,smoke_steps=0):
    import torch
    torch.set_num_threads(1)
    agent=load_policy(checkpoint)
    label=checkpoint_id(checkpoint)
    display=Display(label, hybrid=policy_kind(agent)=='rl_with_safety_filter')
    clock=pygame.time.Clock()
    env=Env(seed=seed,move_reward=agent.config.move_reward)
    start_time=time.time()
    total_return=0.0
    total_steps=0
    latencies=[]
    ended=False
    running=True
    def record(reason):
        metadata=agent.metadata
        row=dict(model_id=label,configuration_id=metadata.get('run_id',label),seed=seed,
                 official_score=env.score,official_time_seconds=time.time()-start_time,
                 rl_return=total_return,apples=env.score,steps=env.steps,completed=env.completed,
                 termination_reason=reason,final_length=len(env.body),clock_hz=GAME_SPEED,
                 decision_latency=statistics.mean(latencies) if latencies else 0.0,
                 training_transitions=agent.env_steps,training_updates=agent.updates,
                 training_wall_time=metadata.get('training_wall_time'),
                 demonstration_cost=metadata.get('demonstration_cost',0),simulation_cost=metadata.get('simulation_cost',0),
                 policy_kind=policy_kind(agent), safety_filter=metadata.get('safety_filter'),
                 filter_overrides=getattr(agent, 'overrides', 0),
                 learned_choice_states=getattr(agent, 'choice_states', 0),
                 time_convention='wall time at terminal event or user quit; display continues afterward')
        append_json(base/'runs'/'interactive.jsonl',row)
        print(json.dumps(row),flush=True)
    print(f'Loaded {label}: {agent.env_steps} training transitions; CPU, evaluation only, 5 Hz; SDL {pygame.display.get_driver()}.',flush=True)
    try:
        while running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                    running=False
                if event.type==pygame.KEYDOWN and event.key==pygame.K_SPACE and ended:
                    seed=seed+1 if seed is not None else None
                    env.reset(seed=seed)
                    if hasattr(agent, 'reset_episode'):
                        agent.reset_episode()
                    start_time=time.time()
                    total_return=0.0
                    latencies=[]
                    ended=False
            if not running:
                if not ended:
                    record('user_quit')
                break
            if not ended:
                began=time.perf_counter()
                action=action_for(agent,env)
                latencies.append(time.perf_counter()-began)
                reward,ended,_=env.step(action)
                total_return+=reward
                total_steps+=1
                if ended:
                    record(env.termination_reason)
            display.draw(env,start_time,ended)
            if smoke_steps and (total_steps>=smoke_steps or ended):
                pygame.image.save(display.screen,str(base/'runs'/'smoke.png'))
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            clock.tick(GAME_SPEED)
    finally:
        pygame.quit()
    print('Window closed cleanly; checkpoint unchanged.',flush=True)
