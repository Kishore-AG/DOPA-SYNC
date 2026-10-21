"""Transparent tabular Q-learning implementation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class QLearningAgent:
    def __init__(self, state_count: int, action_count: int = 2, learning_rate: float = .18,
                 discount_factor: float = .96, seed: int | None = None):
        if state_count <= 0 or action_count <= 0:
            raise ValueError("state_count and action_count must be positive")
        self.q = np.zeros((state_count, action_count), dtype=np.float64)
        self.alpha, self.gamma = learning_rate, discount_factor
        self.rng = np.random.default_rng(seed)
        self.td_errors: list[float] = []

    def act(self, state: int, legal_actions: tuple[int, ...] = (0, 1), epsilon: float = 0.0) -> int:
        self._validate(state, legal_actions)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(legal_actions))
        vals = self.q[state, list(legal_actions)]
        return int(legal_actions[int(np.argmax(vals))])

    def greedy_action(self, state: int, legal_actions: tuple[int, ...] = (0, 1)) -> int:
        return self.act(state, legal_actions, epsilon=0.0)

    def update(self, state: int, action: int, reward: float, next_state: int, terminal: bool,
               next_legal_actions: tuple[int, ...] = (0, 1)) -> float:
        self._validate(state, (action,))
        if not 0 <= next_state < self.q.shape[0]:
            raise ValueError("next_state out of bounds")
        self._validate(next_state, next_legal_actions)
        future = 0.0 if terminal else self.gamma * float(np.max(self.q[next_state, list(next_legal_actions)]))
        delta = float(reward + future - self.q[state, action])
        self.q[state, action] += self.alpha * delta
        self.td_errors.append(delta)
        return delta

    def _validate(self, state: int, actions: tuple[int, ...]) -> None:
        if not 0 <= state < self.q.shape[0]:
            raise ValueError("state out of bounds")
        if not actions or any(a < 0 or a >= self.q.shape[1] for a in actions):
            raise ValueError("invalid action set")

    def save(self, path: str | Path) -> None:
        np.save(path, self.q)

    def load(self, path: str | Path) -> None:
        q = np.load(path)
        if q.shape != self.q.shape:
            raise ValueError("Q-table shape mismatch")
        self.q[:] = q

    def to_list(self) -> list[list[float]]:
        return self.q.tolist()
