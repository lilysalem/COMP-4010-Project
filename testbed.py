"""
Q-learning training testbed for ant simulation.
Runs 500 episodes with fixed hyperparameters and produces training graphs.
"""

import sys
import os
from typing import List, Tuple

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

from hex_grid_world import HexGridWorld
from q_learning import QLearningAgent
from dyna_q import DynaQAgent


def train_agent(animate: bool = False, agent_cls=QLearningAgent, agent_kwargs: dict | None = None) -> Tuple[List[int], List[int]]:
    episodes = 500
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.9
    epsilon_decay = 0.99
    min_epsilon = epsilon
    agent_kwargs = agent_kwargs or {}
    filtered_agent_kwargs = dict(agent_kwargs)
    filtered_agent_kwargs.pop("epsilon_decay", None)
    filtered_agent_kwargs.pop("min_epsilon", None)

    world = HexGridWorld(train=True, worldType=1, animate=animate)

    # Create Q-learning agent with fixed hyperparameters (no decay)
    if len(world.colony) > 1:
        q_agent = QLearningAgent(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=agent_kwargs.get("epsilon_decay", epsilon_decay),  # No decay
            min_epsilon=agent_kwargs.get("min_epsilon", min_epsilon),  # Keep epsilon constant
            **filtered_agent_kwargs,
        )
        world.colony[1].q_agent = q_agent

    episode_rewards = []
    episode_lengths = []

    print(f"Training with fixed hyperparameters: lr={learning_rate}, γ={discount_factor}, ε={epsilon}")

    for episode in range(episodes):
        world.reset()

        # Ensure Q-agent exists after reset with fixed hyperparameters
        if len(world.colony) > 1 and world.colony[1].q_agent is None:
            world.colony[1].q_agent = agent_cls(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=agent_kwargs.get("epsilon_decay", epsilon_decay),
                min_epsilon=agent_kwargs.get("min_epsilon", min_epsilon),
                **filtered_agent_kwargs,
            )

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < 1000:
            _, reward, terminated, truncated, _ = world.step(None)

            if reward is not None:
                total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if episode % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            avg_length = sum(episode_lengths[-100:]) / min(100, len(episode_lengths))
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}")

    return episode_rewards, episode_lengths


def dyna_q_smoke_test() -> None:
    """
    Lightweight Dyna-Q smoke test to ensure the model and Q-table update paths work.
    Uses a fake environment step to avoid touching the full simulator.
    """
    agent = DynaQAgent(
        planning_steps=2,
        learning_rate=1.0,
        discount_factor=0.9,
        epsilon=0.0,
        min_epsilon=0.0,
    )

    # Make one action strictly greedy to remove randomness
    state = (False, "E", "E", "E")
    next_state = (False, "E", "E", "E")
    agent.q_table[(state, 1)] = 1.0

    def fake_env_step(action: int):
        # Should pick the greedy action (1)
        assert action == 1, f"Expected greedy action 1, got {action}"
        reward = 2.0
        terminated = False
        truncated = False
        return reward, next_state, terminated, truncated

    action, reward, returned_next_state = agent.step(state, fake_env_step)

    # Basic expectations: model populated, Q-value updated upward, next_state passed through
    assert (state, action) in agent.model, "Dyna-Q model did not record the real transition"
    assert agent.q_table[(state, action)] >= reward, "Q-value not updated by real + planning steps"
    assert returned_next_state == next_state, "Returned next_state mismatch in Dyna-Q step"

    print("Dyna-Q smoke test passed.")


def plot_training_results(episode_rewards: List[int], episode_lengths: List[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(episode_rewards)
    ax1.set_title('Training Rewards Over Episodes (lr=0.1, ε=0.9, γ=0.9)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)

    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths Over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


def compare_q_vs_dyna(planning_steps: int = 20) -> None:
    """
    Train both Q-learning and Dyna-Q (with the same hyperparameters) and plot a side-by-side comparison.
    """
    q_rewards, q_lengths = train_agent(animate=False, agent_cls=QLearningAgent)
    dyna_rewards, dyna_lengths = train_agent(
        animate=False,
        agent_cls=DynaQAgent,
        agent_kwargs={"planning_steps": planning_steps, "epsilon_decay": 0.99},
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(q_rewards, label="Q-learning")
    ax1.plot(dyna_rewards, label=f"Dyna-Q (planning={planning_steps})")
    ax1.set_title('Training Rewards Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(q_lengths, label="Q-learning")
    ax2.plot(dyna_lengths, label=f"Dyna-Q (planning={planning_steps})")
    ax2.set_title('Episode Lengths Comparison')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_results_comparison.png')
    plt.show()


def main() -> None:
    print("Starting Q-learning training: 500 episodes with fixed hyperparameters")
    rewards, lengths = train_agent(animate=True)
    plot_training_results(rewards, lengths)
    print("Training completed!")


if __name__ == "__main__":
    main()
