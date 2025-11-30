"""
Dyna-Q agent for ant simulation.
Builds on Q-learning by adding model-based planning updates.
"""

from random import choice, random
from typing import Dict, Tuple, Any

from q_learning import QLearningAgent


class DynaQAgent(QLearningAgent):
    """
    Dyna-Q agent that augments Q-learning with model-based planning updates.
    Stores a simple model of (state, action) -> (reward, next_state) and replays it.
    """

    def __init__(self, planning_steps: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.planning_steps = planning_steps
        self.model: Dict[Tuple[Any, int], Tuple[float, Tuple]] = {}

    def _select_action(self, state: Tuple) -> int:
        if random() < self.epsilon:
            return choice(range(5))
        q_values = [self.q_table.get((state, a), 0.0) for a in range(5)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return choice(best_actions)

    def _update_q(self, state: Tuple, action: int, reward: float, next_state: Tuple) -> None:
        current_q = self.q_table.get((state, action), 0.0)
        max_next_q = max([self.q_table.get((next_state, a), 0.0) for a in range(5)])
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        self.q_table[(state, action)] = current_q + self.learning_rate * td_error

    def step(self, state: Tuple, env_step_func) -> Tuple[int, int, Tuple]:
        # Standard Q-learning update using real experience
        action = self._select_action(state)
        reward, next_state, terminated, truncated = env_step_func(action)

        # Update model and Q-values from real experience
        self.model[(state, action)] = (reward, next_state)
        self._update_q(state, action, reward, next_state)

        # Planning: replay from stored model
        for _ in range(self.planning_steps):
            if not self.model:
                break
            sampled_state_action = choice(list(self.model.keys()))
            sim_reward, sim_next_state = self.model[sampled_state_action]
            sim_state, sim_action = sampled_state_action
            self._update_q(sim_state, sim_action, sim_reward, sim_next_state)

        return action, reward, next_state
