import tempfile
import unittest
from pathlib import Path
from src.agent import QLearningAgent
from src.environment import EnvConfig,IntersectionEnv
from src.export_policy import export_policy,validate_policy


class PolicyTests(unittest.TestCase):
    def test_round_trip_and_checksum(self):
        cfg={"seed":4,"environment":{"episode_steps":8,"step_seconds":1.0,"arrival_rate_ns_per_step":.2,
            "arrival_rate_ew_per_step":.2,"discharge_per_step":1,"minimum_green_steps":2,
            "queue_bin_edges":[0,2,4],"switching_cost":0.0},"agent":{"episodes":1,"learning_rate":.2,
            "discount_factor":.9,"epsilon_start":1,"epsilon_end":.1,"epsilon_decay_episodes":1},"baseline":{"phase_duration_steps":3}}
        env=IntersectionEnv(EnvConfig.from_dict(cfg["environment"])); agent=QLearningAgent(env.state_count)
        agent.q[0,0]=1.25
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"policy.json"; export_policy(agent,cfg,path)
            self.assertAlmostEqual(validate_policy(path,agent)["q_values"][0][0],1.25)
            path.write_text(path.read_text().replace("1.25","1.5"))
            with self.assertRaises(ValueError): validate_policy(path)

    def test_missing_and_malformed(self):
        with self.assertRaises(FileNotFoundError): validate_policy("does-not-exist-policy.json")
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.json";p.write_text("{}")
            with self.assertRaises(ValueError):validate_policy(p)


if __name__ == "__main__": unittest.main()
