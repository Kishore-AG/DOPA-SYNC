import tempfile
import unittest
import numpy as np
from src.agent import QLearningAgent


class AgentTests(unittest.TestCase):
    def test_update_nonterminal_and_terminal(self):
        a=QLearningAgent(3,learning_rate=.5,discount_factor=.9,seed=1)
        a.q[1,0]=4
        delta=a.update(0,1,2,1,False)
        self.assertAlmostEqual(delta,5.6); self.assertAlmostEqual(a.q[0,1],2.8)
        delta=a.update(0,1,-1,1,True)
        self.assertAlmostEqual(delta,-3.8); self.assertAlmostEqual(a.q[0,1],.9)

    def test_actions_and_table_io(self):
        a=QLearningAgent(4,seed=2); a.q[1]=[0,3]
        self.assertEqual(a.greedy_action(1),1); self.assertEqual(a.greedy_action(2,(0,)),0)
        self.assertIn(a.act(0,(0,1),1.0),(0,1))
        before=a.q.copy()
        with tempfile.TemporaryDirectory() as d:
            path=f"{d}/q.npy"; a.save(path); b=QLearningAgent(4); b.load(path)
        np.testing.assert_array_equal(before,b.q)
        with self.assertRaises(ValueError): a.greedy_action(4)


if __name__ == "__main__": unittest.main()
