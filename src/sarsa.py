"""
SARSA implementation for ant agents.
On-policy TD control using epsilon-greedy behavior/target.
"""

import pickle
import random
from typing import Dict, Tuple, Any, Optional


class SARSAAgent:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.9,
        epsilon_decay: float = 1,
        min_epsilon: float = 0.01,
        n_actions: int = 5,
        seed: Optional[int] = None,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_actions = n_actions
        self.q_table: Dict[Tuple[Any, int], float] = {}
        self.rng = random.Random(seed)

    def _select_action(self, state: Tuple) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(range(self.n_actions))
        q_values = [self.q_table.get((state, a), 0.0) for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return self.rng.choice(best_actions)

    def step(self, state: Tuple, env_step_func) -> Tuple[int, int, Tuple]:
        # On-policy SARSA update using the next epsilon-greedy action with terminal handling.
        action = self._select_action(state)
        reward, next_state, terminated, truncated = env_step_func(action)

        # Only bootstrap if the episode continues.
        if terminated or truncated:
            td_target = reward
        else:
            next_action = self._select_action(next_state)
            next_q = self.q_table.get((next_state, next_action), 0.0)
            td_target = reward + self.discount_factor * next_q

        current_q = self.q_table.get((state, action), 0.0)
        td_error = td_target - current_q
        self.q_table[(state, action)] = current_q + self.learning_rate * td_error

        return action, reward, next_state

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename: str):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print(f"Q-table file {filename} not found. Starting with empty Q-table.")
