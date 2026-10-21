"""Synthetic two-phase intersection with shared, discrete-time semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    episode_steps: int = 300
    step_seconds: float = 1.0
    arrival_rate_ns_per_step: float = 0.32
    arrival_rate_ew_per_step: float = 0.28
    discharge_per_step: int = 1
    minimum_green_steps: int = 6
    queue_bin_edges: tuple[int, ...] = (0, 3, 7, 12)
    switching_cost: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvConfig":
        values = dict(data)
        values["queue_bin_edges"] = tuple(values.get("queue_bin_edges", (0, 3, 7, 12)))
        return cls(**values)

    def validate(self) -> None:
        if self.episode_steps <= 0 or self.step_seconds <= 0:
            raise ValueError("episode_steps and step_seconds must be positive")
        if min(self.arrival_rate_ns_per_step, self.arrival_rate_ew_per_step) < 0:
            raise ValueError("arrival rates cannot be negative")
        if self.discharge_per_step < 1 or self.minimum_green_steps < 1:
            raise ValueError("discharge capacity and minimum green must be positive")
        if tuple(sorted(set(self.queue_bin_edges))) != self.queue_bin_edges or self.queue_bin_edges[0] != 0:
            raise ValueError("queue_bin_edges must be unique, increasing, and start at zero")
        if self.switching_cost < 0:
            raise ValueError("switching_cost cannot be negative")


class IntersectionEnv:
    """Actions are KEEP=0 and SWITCH=1; one action advances one fixed time step.

    Queues store individual waiting ages. Arrivals enter before service; each queued
    vehicle accrues one step of wait, then the active green serves up to capacity.
    Reward is negative post-service aggregate queue length minus switching cost.
    Thus one reward unit is one vehicle remaining for one decision interval.
    """
    KEEP, SWITCH = 0, 1

    def __init__(self, config: EnvConfig | dict[str, Any], seed: int | None = None):
        self.config = config if isinstance(config, EnvConfig) else EnvConfig.from_dict(config)
        self.config.validate()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.reset()

    @property
    def queue_bin_count(self) -> int:
        return len(self.config.queue_bin_edges)

    @property
    def state_count(self) -> int:
        return 2 * self.queue_bin_count ** 2 * (self.config.minimum_green_steps + 1)

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed
        self.phase = 0  # 0=NS green; 1=EW green
        self.green_age = 0
        self.queues: list[list[float]] = [[], []]
        self.steps = 0
        self.waiting_seconds = 0.0
        self.completed = 0
        self.completed_wait_seconds = 0.0
        self.switches = 0
        self.return_total = 0.0
        return self.encode_state(), self.metrics()

    def queue_bins(self, queues: tuple[int, int] | None = None) -> tuple[int, int]:
        lengths = queues if queues is not None else (len(self.queues[0]), len(self.queues[1]))
        return tuple(int(np.digitize(length, self.config.queue_bin_edges[1:], right=False)) for length in lengths)

    def encode_components(self, phase: int, ns_bin: int, ew_bin: int, green_age: int) -> int:
        if phase not in (0, 1) or not (0 <= ns_bin < self.queue_bin_count) or not (0 <= ew_bin < self.queue_bin_count):
            raise ValueError("state component out of bounds")
        if not (0 <= green_age <= self.config.minimum_green_steps):
            raise ValueError("green_age out of bounds")
        return (((phase * self.queue_bin_count + ns_bin) * self.queue_bin_count + ew_bin)
                * (self.config.minimum_green_steps + 1) + green_age)

    def decode_state(self, state: int) -> tuple[int, int, int, int]:
        if not isinstance(state, (int, np.integer)) or not 0 <= int(state) < self.state_count:
            raise ValueError("state out of bounds")
        n = self.config.minimum_green_steps + 1
        age = state % n
        ew = (state // n) % self.queue_bin_count
        ns = (state // (n * self.queue_bin_count)) % self.queue_bin_count
        phase = state // (n * self.queue_bin_count ** 2)
        return int(phase), int(ns), int(ew), int(age)

    def encode_state(self) -> int:
        ns, ew = self.queue_bins()
        return self.encode_components(self.phase, ns, ew, min(self.green_age, self.config.minimum_green_steps))

    def legal_actions(self) -> tuple[int, int]:
        return (self.KEEP, self.SWITCH) if self.green_age >= self.config.minimum_green_steps else (self.KEEP,)

    def step(self, action: int, arrivals: tuple[int, int] | None = None) -> tuple[int, float, bool, dict[str, Any]]:
        if action not in (self.KEEP, self.SWITCH):
            raise ValueError("action must be KEEP=0 or SWITCH=1")
        if action == self.SWITCH and self.green_age < self.config.minimum_green_steps:
            raise ValueError("switch is not permitted before minimum green")
        switched = action == self.SWITCH
        if switched:
            self.phase = 1 - self.phase
            self.green_age = 0
            self.switches += 1
        if arrivals is None:
            arrivals = (int(self.rng.poisson(self.config.arrival_rate_ns_per_step)),
                        int(self.rng.poisson(self.config.arrival_rate_ew_per_step)))
        if len(arrivals) != 2 or min(arrivals) < 0:
            raise ValueError("arrivals must contain two nonnegative integers")
        for p in (0, 1):
            self.queues[p].extend([0.0] * int(arrivals[p]))
        self.waiting_seconds += (len(self.queues[0]) + len(self.queues[1])) * self.config.step_seconds
        for p in (0, 1):
            self.queues[p] = [age + self.config.step_seconds for age in self.queues[p]]
        served = min(self.config.discharge_per_step, len(self.queues[self.phase]))
        for _ in range(served):
            age = self.queues[self.phase].pop(0)
            self.completed += 1
            self.completed_wait_seconds += age
        self.steps += 1
        self.green_age = min(self.config.minimum_green_steps, self.green_age + 1)
        reward = -(len(self.queues[0]) + len(self.queues[1])) - (self.config.switching_cost if switched else 0.0)
        self.return_total += reward
        done = self.steps >= self.config.episode_steps
        info = {"arrivals": tuple(map(int, arrivals)), "served": served, "switched": switched,
                "queues": (len(self.queues[0]), len(self.queues[1])), "reward": reward,
                "waiting_seconds": self.waiting_seconds, **self.metrics()}
        return self.encode_state(), float(reward), done, info

    def metrics(self) -> dict[str, float | int]:
        return {"queue_ns": len(self.queues[0]), "queue_ew": len(self.queues[1]),
                "mean_queue": (len(self.queues[0]) + len(self.queues[1])) / 2.0,
                "waiting_seconds": self.waiting_seconds, "completed": self.completed,
                "mean_completed_wait_seconds": self.completed_wait_seconds / self.completed if self.completed else 0.0,
                "switches": self.switches, "steps": self.steps, "episode_return": self.return_total}
