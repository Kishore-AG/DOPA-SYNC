"""Training loop and experiment artifact generation."""
from __future__ import annotations
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from .agent import QLearningAgent
from .environment import EnvConfig, IntersectionEnv


def load_config(path: str | Path = "config.json") -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    EnvConfig.from_dict(cfg["environment"]).validate()
    if cfg["agent"]["episodes"] <= 0:
        raise ValueError("agent.episodes must be positive")
    return cfg


def epsilon_at(episode: int, cfg: dict) -> float:
    a = cfg["agent"]
    fraction = min(1.0, episode / max(1, a["epsilon_decay_episodes"]))
    return float(a["epsilon_start"] + fraction * (a["epsilon_end"] - a["epsilon_start"]))


def train(config: dict, artifacts: str | Path = "artifacts", results: str | Path = "results") -> tuple[QLearningAgent, pd.DataFrame]:
    artifacts, results = Path(artifacts), Path(results)
    artifacts.mkdir(parents=True, exist_ok=True); results.mkdir(parents=True, exist_ok=True)
    env_cfg = EnvConfig.from_dict(config["environment"])
    env = IntersectionEnv(env_cfg, seed=config["seed"])
    ac = config["agent"]
    agent = QLearningAgent(env.state_count, learning_rate=ac["learning_rate"],
                           discount_factor=ac["discount_factor"], seed=config["seed"] + 1)
    rows = []
    for episode in range(ac["episodes"]):
        state, _ = env.reset(seed=config["seed"] + episode)
        epsilon = epsilon_at(episode, config)
        ret = 0.0
        while True:
            action = agent.act(state, env.legal_actions(), epsilon)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done, env.legal_actions())
            ret += reward; state = next_state
            if done:
                break
        rows.append({"episode": episode + 1, "return": ret,
                     "mean_wait_seconds_per_completed_vehicle": info["mean_completed_wait_seconds"],
                     "throughput_vehicles": info["completed"], "mean_queue_vehicles": info["waiting_seconds"] / (2 * env_cfg.episode_steps * env_cfg.step_seconds),
                     "epsilon": epsilon, "td_error_mean": float(np.mean(agent.td_errors[-env_cfg.episode_steps:])),
                     "td_error_std": float(np.std(agent.td_errors[-env_cfg.episode_steps:]))})
    df = pd.DataFrame(rows)
    df.to_csv(results / "training_episodes.csv", index=False)
    agent.save(artifacts / "q_table.npy")
    metadata = {"seed": config["seed"], "episodes": ac["episodes"], "state_count": env.state_count,
                "action_count": 2, "learning_rate": ac["learning_rate"], "discount_factor": ac["discount_factor"],
                "epsilon_start": ac["epsilon_start"], "epsilon_end": ac["epsilon_end"],
                "python": platform.python_version(), "q_table_changed": bool(np.any(agent.q != 0)),
                "td_error_count": len(agent.td_errors)}
    (artifacts / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return agent, df
