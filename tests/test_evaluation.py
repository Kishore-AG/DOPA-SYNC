import tempfile
import unittest
import numpy as np
from src.agent import QLearningAgent
from src.evaluate import evaluate
from src.environment import EnvConfig,IntersectionEnv


class EvaluationTests(unittest.TestCase):
    def config(self):
        return {"seed":11,"environment":{"episode_steps":20,"step_seconds":1.0,"arrival_rate_ns_per_step":.3,
            "arrival_rate_ew_per_step":.25,"discharge_per_step":1,"minimum_green_steps":2,
            "queue_bin_edges":[0,2,4],"switching_cost":0.0},"agent":{"episodes":2,"learning_rate":.1,
            "discount_factor":.9,"epsilon_start":1,"epsilon_end":.1,"epsilon_decay_episodes":1},
            "baseline":{"phase_duration_steps":4},"evaluation":{"scenarios":3,"seed_start":900}}

    def test_paired_seed_rows_and_no_agent_update(self):
        cfg=self.config();env=IntersectionEnv(EnvConfig.from_dict(cfg["environment"]));agent=QLearningAgent(env.state_count)
        q=agent.q.copy();
        with tempfile.TemporaryDirectory() as d: df,summary=evaluate(cfg,agent,d)
        self.assertEqual(agent.td_errors,[]);np.testing.assert_array_equal(q,agent.q)
        for seed in range(900,903):
            self.assertEqual(set(df[df.seed==seed].controller),{"rl","fixed"})
        self.assertEqual(len(df),6);self.assertIn("std",summary["results"]["rl"]["switches"])

    def test_same_random_arrivals_for_identical_seed(self):
        cfg=EnvConfig(episode_steps=20,minimum_green_steps=2)
        a=IntersectionEnv(cfg,seed=432);b=IntersectionEnv(cfg,seed=432)
        for _ in range(20):
            aa=a.step(0);bb=b.step(0);self.assertEqual(aa[1:3],bb[1:3]);self.assertEqual(aa[3]["arrivals"],bb[3]["arrivals"])


if __name__ == "__main__": unittest.main()
