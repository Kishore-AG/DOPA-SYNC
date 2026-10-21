import json
import shutil
import subprocess
import unittest
from pathlib import Path
from src.agent import QLearningAgent
from src.environment import EnvConfig, IntersectionEnv
from src.export_policy import export_policy


@unittest.skipUnless(shutil.which("node"), "Node.js is required for JavaScript parity checks")
class BrowserParityTests(unittest.TestCase):
    def test_state_transition_action_and_td_error_match(self):
        cfg={"seed":9,"environment":{"episode_steps":8,"step_seconds":1.0,"arrival_rate_ns_per_step":.2,
            "arrival_rate_ew_per_step":.2,"discharge_per_step":1,"minimum_green_steps":2,
            "queue_bin_edges":[0,2,4],"switching_cost":0.0},"agent":{"episodes":1,"learning_rate":.2,
            "discount_factor":.9,"epsilon_start":1,"epsilon_end":.1,"epsilon_decay_episodes":1},
            "baseline":{"phase_duration_steps":3}}
        env=IntersectionEnv(EnvConfig.from_dict(cfg["environment"])); env.phase=0; env.green_age=2; env.queues=[[2.,0.],[0.]]
        state=env.encode_state(); next_state=env.encode_components(1,1,0,1)
        agent=QLearningAgent(env.state_count,discount_factor=.9); agent.q[state]=[1,2.5]; agent.q[next_state]=[4,100]
        action=agent.greedy_action(state,env.legal_actions()); py_next,py_reward,terminal,info=env.step(action,arrivals=(1,0))
        py_td=py_reward+.9*agent.q[py_next,0]-agent.q[state,action]
        with self.subTest("python setup"): self.assertEqual((state,next_state,action,py_next),(env.encode_components(0,1,0,2),next_state,1,next_state))
        with __import__("tempfile").TemporaryDirectory() as d:
            path=Path(d)/"policy.json"; export_policy(agent,cfg,path)
            js=r'''import fs from "node:fs";
import {makeModel,transition,legalActions} from "./simulation/environment.js";
import {validatePolicy,chooseAction,predictionError} from "./simulation/policy.js";
const p=await validatePolicy(JSON.parse(fs.readFileSync(process.argv[1],"utf8")));
const m=makeModel();m.phase=0;m.greenAge=2;m.queues=[[2,0],[0]];
const choice=chooseAction(p,m,[2,1]);const n=transition(m,choice.action,[1,0],p.environment);
const pred=predictionError(p,choice.state,choice.action,n.reward,n.state,n.terminal,legalActions(m.greenAge,p.environment));
console.log(JSON.stringify({state:choice.state,action:choice.action,next:n.state,reward:n.reward,delta:pred.delta,queues:m.queues.map(q=>q.length)}));'''
            run=subprocess.run(["node","--input-type=module","-e",js,str(path)],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
        got=json.loads(run.stdout)
        self.assertEqual((got["state"],got["action"],got["next"],got["reward"]),(state,action,py_next,py_reward))
        self.assertEqual(got["queues"],[len(env.queues[0]),len(env.queues[1])])
        self.assertAlmostEqual(got["delta"],py_td,places=10)


if __name__ == "__main__": unittest.main()
