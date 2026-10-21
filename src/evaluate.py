"""Paired held-out evaluation against a fixed-duration controller."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .agent import QLearningAgent
from .environment import EnvConfig, IntersectionEnv


def run_episode(env: IntersectionEnv, controller: str, agent: QLearningAgent | None = None,
                phase_duration: int = 12, seed: int = 0) -> dict:
    state, _ = env.reset(seed=seed)
    phase_age = 0
    while True:
        if controller == "rl":
            assert agent is not None
            action = agent.greedy_action(state, env.legal_actions())
        elif controller == "fixed":
            action = int(phase_age >= phase_duration and env.legal_actions() == (0, 1))
        else:
            raise ValueError("controller must be 'rl' or 'fixed'")
        state, _, done, info = env.step(action)
        phase_age = 0 if action == 1 else phase_age + 1
        if done:
            return {"controller": controller, "seed": seed, "mean_wait_seconds_per_completed_vehicle": info["mean_completed_wait_seconds"],
                    "mean_queue_vehicles": info["waiting_seconds"] / (2 * env.config.episode_steps * env.config.step_seconds),
                    "completed_vehicles": info["completed"], "episode_return": info["episode_return"],
                    "switches": info["switches"], "waiting_vehicle_seconds": info["waiting_seconds"]}


def evaluate(config: dict, agent: QLearningAgent, results: str | Path = "results") -> tuple[pd.DataFrame, dict]:
    results = Path(results); results.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig.from_dict(config["environment"])
    n = config["evaluation"]["scenarios"]
    seeds = range(config["evaluation"]["seed_start"], config["evaluation"]["seed_start"] + n)
    rows = []
    for seed in seeds:
        for controller in ("rl", "fixed"):
            env = IntersectionEnv(cfg, seed=seed)
            rows.append(run_episode(env, controller, agent, config["baseline"]["phase_duration_steps"], seed))
    df = pd.DataFrame(rows)
    df.to_csv(results / "evaluation_scenarios.csv", index=False)
    summary = {"configuration": config, "training_seed": config["seed"],
               "evaluation_seeds": list(seeds), "scenario_count": n,
               "metric_definitions": {
                   "mean_wait_seconds_per_completed_vehicle": "Mean simulated seconds waited by completed vehicles; zero when none complete.",
                   "mean_queue_vehicles": "Mean of aggregate queue count across both approaches, time-averaged; reported as vehicle count.",
                   "completed_vehicles": "Vehicles discharged from the intersection during the episode.",
                   "episode_return": "Sum of negative post-service queue counts and configured switching costs.",
                   "switches": "Number of phase changes during the episode."}, "results": {}}
    for controller, group in df.groupby("controller"):
        summary["results"][controller] = {}
        for metric in ("mean_wait_seconds_per_completed_vehicle", "mean_queue_vehicles", "completed_vehicles", "episode_return", "switches"):
            summary["results"][controller][metric] = {"mean": float(group[metric].mean()), "std": float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0}
    (results / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df, summary
