"""
Q-learning training testbed for ant simulation.
Runs 500 episodes with fixed hyperparameters and produces training graphs.
"""

import sys
import os
from typing import List, Tuple

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib.pyplot as plt
from hex_grid_world import HexGridWorld
from q_learning import QLearningAgent


def train_agent(animate: bool = False) -> Tuple[List[int], List[int]]:
    episodes = 500
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.9

    world = HexGridWorld(train=True, worldType=1, animate=animate)

    # Create Q-learning agent with fixed hyperparameters (no decay)
    if len(world.colony) > 1:
        q_agent = QLearningAgent(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=0.99,  # No decay
            min_epsilon=epsilon  # Keep epsilon constant
        )
        world.colony[1].q_agent = q_agent

    episode_rewards = []
    episode_lengths = []

    print(f"Training with fixed hyperparameters: lr={learning_rate}, γ={discount_factor}, ε={epsilon}")

    for episode in range(episodes):
        world.reset()

        # Ensure Q-agent exists after reset with fixed hyperparameters
        if len(world.colony) > 1 and world.colony[1].q_agent is None:
            world.colony[1].q_agent = QLearningAgent(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=1.0,
                min_epsilon=epsilon
            )

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < 1000:
            next_state, reward, terminated, truncated, _ = world.step(None)

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


def plot_training_results(episode_rewards: List[int], episode_lengths: List[int]) -> None:
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


def main() -> None:
    print("Starting Q-learning training: 500 episodes with fixed hyperparameters")
    rewards, lengths = train_agent( animate=True)
    plot_training_results(rewards, lengths)
    print("Training completed!")


if __name__ == "__main__":
    main()