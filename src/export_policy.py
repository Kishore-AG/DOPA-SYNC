"""Export and validate the Python Q-table for browser inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .agent import QLearningAgent
from .environment import EnvConfig, IntersectionEnv

SCHEMA_VERSION = 1


def _canonical(payload: dict) -> bytes:
    def normalize(value):
        if isinstance(value, dict): return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list): return [normalize(v) for v in value]
        if isinstance(value, float) and value.is_integer(): return int(value)
        return value
    return json.dumps(normalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def export_policy(agent: QLearningAgent, config: dict, path: str | Path = "artifacts/policy.json") -> dict:
    env_cfg = EnvConfig.from_dict(config["environment"])
    env = IntersectionEnv(env_cfg)
    payload = {"schema_version": SCHEMA_VERSION, "algorithm": "tabular_q_learning",
        "trained": True, "state_count": env.state_count, "action_count": 2,
        "actions": {"0": "KEEP", "1": "SWITCH"},
        "state_encoding": {"order": ["phase", "ns_queue_bin", "ew_queue_bin", "green_age_capped"],
            "formula": "(((phase * queue_bin_count + ns_bin) * queue_bin_count + ew_bin) * (minimum_green_steps + 1) + green_age_capped)",
            "phase": {"0": "NS_GREEN", "1": "EW_GREEN"}, "queue_bin_edges": list(env_cfg.queue_bin_edges),
            "queue_bin_count": env.queue_bin_count, "green_age_cap": env_cfg.minimum_green_steps},
        "environment": config["environment"], "baseline": config["baseline"],
        "training": {"seed": config["seed"], **config["agent"]},
        "q_values": agent.to_list()}
    payload["sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    dest = Path(path); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    validate_policy(dest, agent)
    return payload


def validate_policy(path: str | Path, agent: QLearningAgent | None = None) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    checksum = data.pop("sha256", None)
    if data.get("schema_version") != SCHEMA_VERSION or data.get("trained") is not True:
        raise ValueError("unsupported or untrained policy")
    if hashlib.sha256(_canonical(data)).hexdigest() != checksum:
        raise ValueError("policy checksum mismatch")
    q = data.get("q_values")
    if len(q) != data["state_count"] or any(len(row) != 2 for row in q):
        raise ValueError("invalid Q-table dimensions")
    if agent is not None:
        import numpy as np
        if not np.allclose(np.asarray(q), agent.q, rtol=0, atol=1e-12):
            raise ValueError("exported Q-values differ from trained agent")
    data["sha256"] = checksum
    return data


if __name__ == "__main__":
    from .train import load_config
    from .agent import QLearningAgent
    cfg = load_config()
    env = IntersectionEnv(EnvConfig.from_dict(cfg["environment"]))
    a = QLearningAgent(env.state_count)
    a.load("artifacts/q_table.npy")
    export_policy(a, cfg)
