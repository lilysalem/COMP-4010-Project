"""
Q-learning implementation for ant agents.
Handles Q-table, action selection, and learning updates.
"""

import pickle
from random import random, choice
from typing import Dict, Tuple, Any, Optional


class QLearningAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.9, epsilon_decay: float = 1, min_epsilon: float = 0.01,
                 n_actions: int = 5, seed: Optional[int] = None):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table: Dict[Tuple[Any, int], float] = {}
        self.n_actions = n_actions

    def step(self, state: Tuple, env_step_func) -> Tuple[int, int, Tuple]:
        # Perform complete Q-learning step: action selection, environment interaction, Q-update

        # Args - state: Current state tuple, env_step_func: Function that takes action and returns (reward, next_state, terminated, truncated)

        # Returns - (action, reward, next_state)

        # Epsilon-greedy action selection
        if random() < self.epsilon:
            action = choice(range(self.n_actions))
        else:
            q_values = [self.q_table.get((state, a), 0.0) for a in range(self.n_actions)]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            action = choice(best_actions)

        # Environment step
        reward, next_state, terminated, truncated = env_step_func(action)

        # Q-value update
        current_q = self.q_table.get((state, action), 0.0)
        # If episode terminates, next state has no future value
        if terminated or truncated:
            td_target = reward
        else:
            max_next_q = max([self.q_table.get((next_state, a), 0.0) for a in range(self.n_actions)])
            td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error
        self.q_table[(state, action)] = new_q

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