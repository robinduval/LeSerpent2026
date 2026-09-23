"""Tests des règles et de l'apprentissage, sans lancer une partie accélérée."""
import importlib.util
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

spec = importlib.util.spec_from_file_location("snake_ia", Path(__file__).with_name("snake-ia.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SnakeTests(unittest.TestCase):
    def test_slides_api_and_legacy_model_weights(self):
        game=m.Game(42)
        game.apple.position=(4,7)
        agent=m.Agent(42)
        self.assertIsInstance(agent.model,m.Linear_QNet)
        state=agent.get_state(game)
        self.assertEqual(agent.get_move(state,False),agent.model.predict(state))
        reward,done,score=game.play_step(0)
        self.assertEqual((reward,done,score),(10.,False,1))
        legacy=m.nn.Sequential(m.nn.Linear(13,128),m.nn.ReLU(),m.nn.Linear(128,3))
        legacy.load_state_dict(agent.model.state_dict(),strict=True)
        tensor=m.torch.tensor(state)
        self.assertTrue(m.torch.equal(legacy(tensor),agent.model(tensor)))

    def test_finetuning_settings_survive_resume(self):
        agent=m.Agent(1,'spatial')
        agent.steps=10000
        agent.epsilon_floor=.01
        agent.optimizer.param_groups[0]['lr']=1e-4
        self.assertEqual(agent.epsilon,.01)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';agent.save(path)
            other=m.Agent(2,'spatial');other.load(path)
            self.assertEqual(other.epsilon,.01)
            self.assertEqual(other.optimizer.param_groups[0]['lr'],1e-4)

    def test_spatial_regions_wrap_and_detect_pockets(self):
        self.assertEqual(m.free_regions(set())[(0,0)],225)
        enclosed={(1,0),(14,0),(0,1),(0,14)}
        self.assertEqual(m.free_regions(enclosed)[(0,0)],1)
        self.assertEqual(m.free_regions(enclosed)[(7,7)],220)

    def test_spatial_features_do_not_change_game(self):
        basic=m.Environment(42)
        spatial=m.Environment(42,'spatial')
        rng=m.random.Random(3)
        for _ in range(1000):
            state=spatial.state()
            self.assertEqual(len(state),22)
            self.assertEqual(state[:13],basic.state())
            self.assertTrue(all(-1<=x<=1 for x in state))
            action=rng.randrange(3)
            _,reward,done=basic.step(action)
            _,other_reward,other_done=spatial.step(action)
            self.assertEqual((reward,done),(other_reward,other_done))
            self.assertEqual(basic.snake.body,spatial.snake.body)
            self.assertEqual(basic.snake.score,spatial.snake.score)
            self.assertEqual(basic.apple.position,spatial.apple.position)
            if done:
                break

    def test_spatial_transfer_preserves_initial_predictions(self):
        basic=m.Agent(42)
        spatial=m.Agent(43,'spatial')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'basic.pt';basic.save(path);spatial.warm_start(path)
            state=m.Environment(42,'spatial').state()
            self.assertTrue(m.torch.allclose(basic.online(m.torch.tensor(state[:13])),spatial.online(m.torch.tensor(state)),atol=1e-6))
            self.assertEqual(len(spatial.memory),0)
            path=Path(folder)/'spatial.pt';spatial.save(path)
            loaded=m.Agent(2,'spatial');loaded.load(path)
            self.assertEqual(loaded.act(state,False),spatial.act(state,False))
            with self.assertRaises(ValueError):
                basic.load(path)

    def test_display_uses_five_hz_equivalent_only_for_training(self):
        row=dict(mode='train',steps=505,score=50,duration_s=1.8167)
        duration,ratio=m.game_timing(row)
        self.assertEqual(duration,101)
        self.assertAlmostEqual(ratio,50/101)
        row.update(mode='eval',duration_s=102.5)
        self.assertEqual(m.game_timing(row),(102.5,50/102.5))
        self.assertEqual(m.game_timing(dict(mode='train',steps=0,score=0,duration_s=.1)),(0,0))

    def test_parallel_real_time_and_interruptions(self):
        agent=m.Agent(42)
        args=SimpleNamespace(seed=42,episodes=100,num_envs=3,train_fps=0,max_seconds=.1)
        m.pygame.init()
        try:
            with tempfile.TemporaryDirectory() as folder:
                start=m.time.monotonic()
                m.run_parallel(args,agent,Path(folder),m.time.time(),m.pygame.time.Clock())
                elapsed=m.time.monotonic()-start
                data=m.json.loads((Path(folder)/'metrics.json').read_text())
                live=data['live']
                self.assertGreaterEqual(live['session_elapsed_s'],.1)
                self.assertLessEqual(live['session_elapsed_s'],elapsed)
                self.assertEqual(live['num_envs'],3)
                self.assertEqual(live['clock_hz'],0)
                self.assertEqual(live['reason'],'session_limit')
                self.assertFalse(live['completed'])
                self.assertAlmostEqual(live['throughput_steps_s'],live['session_steps']/live['session_elapsed_s'])
                self.assertTrue((Path(folder)/'latest.pt').exists())
        finally:
            m.pygame.quit()

    def test_border_wrap_and_reward(self):
        env = m.Environment(42)
        env.snake.head_pos = [14, 7]
        env.snake.body = [[14, 7], [13, 7], [12, 7]]
        env.apple.position = (5, 5)
        _, reward, done = env.step(0)
        self.assertEqual(env.snake.head_pos, [0, 7])
        self.assertEqual(reward, .1)
        self.assertFalse(done)

    def test_growth_delayed_and_score_preserved(self):
        env = m.Environment(42)
        env.apple.position = (4, 7)
        _, reward, done = env.step(0)
        self.assertEqual((reward, env.snake.score, len(env.snake.body)), (10, 1, 3))
        self.assertTrue(env.snake.grow_pending)
        env.step(0)
        self.assertEqual(len(env.snake.body), 4)

    def test_danger_accounts_for_vacating_tail(self):
        env = m.Environment(42)
        env.snake.head_pos = [3, 3]
        env.snake.body = [[3, 3], [3, 4], [4, 4], [4, 3]]
        self.assertEqual(env.state()[0], 0)
        env.snake.grow_pending = True
        self.assertEqual(env.state()[0], 1)
        _, reward, done = env.step(0)
        self.assertEqual(reward, -10)
        self.assertTrue(done)

    def test_optimizer_and_checkpoint_resume(self):
        agent = m.Agent(42)
        state = m.Environment(42).state()
        before = [p.detach().clone() for p in agent.online.parameters()]
        for _ in range(64):
            agent.learn((state, 0, 10., state, True))
        self.assertEqual(agent.updates, 1)
        self.assertTrue(any(not m.torch.equal(a,b) for a,b in zip(before,agent.online.parameters())))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            agent.save(path)
            loaded = m.Agent(0)
            loaded.load(path)
            self.assertEqual(loaded.steps, 64)
            self.assertEqual(len(loaded.memory),64)
            self.assertEqual(loaded.act(state,False),agent.act(state,False))


if __name__ == "__main__":
    unittest.main()
