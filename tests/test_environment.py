import unittest
from src.environment import EnvConfig, IntersectionEnv


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig(episode_steps=9, arrival_rate_ns_per_step=0, arrival_rate_ew_per_step=0,
                             minimum_green_steps=2, queue_bin_edges=(0, 2, 4))
        self.env = IntersectionEnv(self.cfg, seed=5)

    def test_state_round_trip_and_bounds(self):
        for phase in (0,1):
            for ns in range(self.env.queue_bin_count):
                for ew in range(self.env.queue_bin_count):
                    for age in range(self.cfg.minimum_green_steps+1):
                        code = self.env.encode_components(phase,ns,ew,age)
                        self.assertEqual(self.env.decode_state(code),(phase,ns,ew,age))
        with self.assertRaises(ValueError): self.env.decode_state(self.env.state_count)

    def test_phase_minimum_green_and_service(self):
        state, _ = self.env.reset(seed=10)
        self.assertEqual(self.env.legal_actions(),(0,))
        state, _, _, _ = self.env.step(0, arrivals=(2,1))
        self.assertEqual(self.env.metrics()["completed"],1)
        self.assertEqual(self.env.metrics()["queue_ns"],1)
        self.assertEqual(self.env.metrics()["queue_ew"],1)
        with self.assertRaises(ValueError): self.env.step(1, arrivals=(0,0))
        self.env.step(0,arrivals=(0,0)); self.assertIn(1,self.env.legal_actions())
        self.env.step(1,arrivals=(0,0)); self.assertEqual(self.env.phase,1)

    def test_waiting_accounting_and_terminal(self):
        self.env.reset(seed=2)
        for i in range(9):
            _,_,done,info=self.env.step(0,arrivals=(1,0))
            if i < 8: self.assertFalse(done)
        self.assertTrue(done); self.assertEqual(info["steps"],9)
        self.assertGreater(info["waiting_seconds"],0)
        self.assertEqual(info["completed"],9)

    def test_reproducibility(self):
        a=IntersectionEnv(self.cfg,seed=44); b=IntersectionEnv(self.cfg,seed=44)
        trace_a=[a.step(0)[1:3] for _ in range(2)]
        trace_b=[b.step(0)[1:3] for _ in range(2)]
        self.assertEqual(trace_a,trace_b)


if __name__ == "__main__": unittest.main()
