"""
Q-learning training testbed for ant simulation.
Runs 500 episodes with fixed hyperparameters and produces training graphs.
"""

import sys
import os
from typing import List, Tuple, Optional
import numpy as np
import random

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

from hex_grid_world import HexGridWorld
from q_learning import QLearningAgent
from dyna_q import DynaQAgent


def train_agent(
    animate: bool = False,
    agent_cls=QLearningAgent,
    agent_kwargs: dict | None = None,
    episodes: int = 500,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    epsilon: float = 0.9,
    epsilon_decay: float = 0.99,
    min_epsilon: float = 0.01,
    max_steps: int = 1000,
    collect_deliveries: bool = False,
    return_agent: bool = False,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    agent_kwargs = agent_kwargs or {}
    filtered_agent_kwargs = dict(agent_kwargs)
    filtered_agent_kwargs.pop("epsilon_decay", None)
    filtered_agent_kwargs.pop("min_epsilon", None)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    world = HexGridWorld(train=True, worldType=1, animate=animate)

    # Instantiate the requested agent type once and reuse across episodes
    q_agent = agent_cls(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=agent_kwargs.get("epsilon_decay", epsilon_decay),
        min_epsilon=agent_kwargs.get("min_epsilon", min_epsilon),
        **filtered_agent_kwargs,
    )
    if len(world.colony) > 1:
        world.colony[1].q_agent = q_agent

    episode_rewards = []
    episode_lengths = []
    episode_deliveries = []

    print(f"Training with fixed hyperparameters: lr={learning_rate}, γ={discount_factor}, ε={epsilon}")

    for episode in range(episodes):
        world.reset()
        start_food = getattr(world.colony[0], "food", 0) if len(world.colony) > 0 else 0

        # Reattach the persistent agent after reset (new worker object each reset)
        if len(world.colony) > 1:
            world.colony[1].q_agent = q_agent

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_steps:
            _, reward, terminated, truncated, _ = world.step(None)

            if reward is not None:
                total_reward += reward
            steps += 1

        end_food = getattr(world.colony[0], "food", 0) if len(world.colony) > 0 else 0
        deliveries = max(0, end_food - start_food)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if collect_deliveries:
            episode_deliveries.append(deliveries)

        # Decay epsilon each episode for agents that support it
        if hasattr(q_agent, "decay_epsilon"):
            q_agent.decay_epsilon()

        if episode % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            avg_length = sum(episode_lengths[-100:]) / min(100, len(episode_lengths))
            current_eps = getattr(q_agent, "epsilon", epsilon)
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}, ε={current_eps:.3f}")

    if collect_deliveries and return_agent:
        return episode_rewards, episode_lengths, episode_deliveries, q_agent
    if collect_deliveries:
        return episode_rewards, episode_lengths, episode_deliveries
    if return_agent:
        return episode_rewards, episode_lengths, q_agent
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


def compare_q_vs_dyna(
    planning_steps: int = 20,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    episodes: int = 500,
    max_steps: int = 600,
    smooth_window: int = 50,
    seed: Optional[int] = 42,
    output_path: Optional[str] = None,
    timestamped: bool = False,
) -> None:
    """
    Train both Q-learning and Dyna-Q (with the same hyperparameters) and plot a side-by-side comparison.
    Adds rolling mean smoothing and logs deliveries/eval summaries.
    """
    # Common hyperparams and seed for reproducibility
    q_rewards, q_lengths, q_deliveries, q_agent = train_agent(
        animate=False,
        agent_cls=QLearningAgent,
        episodes=episodes,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        max_steps=max_steps,
        collect_deliveries=True,
        return_agent=True,
        seed=seed,
    )

    dyna_rewards, dyna_lengths, dyna_deliveries, dyna_agent = train_agent(
        animate=False,
        agent_cls=DynaQAgent,
        agent_kwargs={"planning_steps": planning_steps},
        episodes=episodes,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        max_steps=max_steps,
        collect_deliveries=True,
        return_agent=True,
        seed=seed,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    def rolling_mean(data):
        if len(data) < smooth_window:
            return []
        weights = np.ones(smooth_window) / smooth_window
        return np.convolve(data, weights, mode="valid")

    # Rewards with smoothing
    axes[0].plot(q_rewards, alpha=0.3, label="Q-learning (raw)")
    axes[0].plot(dyna_rewards, alpha=0.3, label=f"Dyna-Q {planning_steps} (raw)")
    q_smooth = rolling_mean(q_rewards)
    d_smooth = rolling_mean(dyna_rewards)
    if len(q_smooth) > 0:
        axes[0].plot(range(smooth_window - 1, smooth_window - 1 + len(q_smooth)), q_smooth, label="Q-learning (smooth)")
    if len(d_smooth) > 0:
        axes[0].plot(range(smooth_window - 1, smooth_window - 1 + len(d_smooth)), d_smooth, label=f"Dyna-Q {planning_steps} (smooth)")
    axes[0].set_title('Training Rewards Comparison')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True)
    axes[0].legend()

    # Lengths with smoothing
    axes[1].plot(q_lengths, alpha=0.3, label="Q-learning (raw)")
    axes[1].plot(dyna_lengths, alpha=0.3, label=f"Dyna-Q {planning_steps} (raw)")
    ql_smooth = rolling_mean(q_lengths)
    dl_smooth = rolling_mean(dyna_lengths)
    if len(ql_smooth) > 0:
        axes[1].plot(range(smooth_window - 1, smooth_window - 1 + len(ql_smooth)), ql_smooth, label="Q-learning (smooth)")
    if len(dl_smooth) > 0:
        axes[1].plot(range(smooth_window - 1, smooth_window - 1 + len(dl_smooth)), dl_smooth, label=f"Dyna-Q {planning_steps} (smooth)")
    axes[1].set_title('Episode Lengths Comparison')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True)
    axes[1].legend()

    # Deliveries
    axes[2].plot(q_deliveries, alpha=0.3, label="Q-learning deliveries")
    axes[2].plot(dyna_deliveries, alpha=0.3, label=f"Dyna-Q {planning_steps} deliveries")
    axes[2].set_title('Food Deliveries per Episode')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Deliveries')
    axes[2].grid(True)
    axes[2].legend()

    base_dir = os.path.join("results", "comparisons")
    if timestamped:
        run_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        run_dir = base_dir
    os.makedirs(run_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(run_dir, "training_results_comparison.png")

    plt.tight_layout()
    plt.savefig(output_path)

    # Greedy eval summary
    def greedy_eval(agent_cls, q_table):
        eval_agent = agent_cls(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=0.0,
            epsilon_decay=1.0,
            min_epsilon=0.0,
            **({} if agent_cls is QLearningAgent else {"planning_steps": planning_steps}),
        )
        eval_agent.q_table = dict(q_table)
        eval_world = HexGridWorld(train=False, worldType=1, animate=False)
        if len(eval_world.colony) > 1:
            eval_world.colony[1].q_agent = eval_agent
        rewards = []
        lengths = []
        deliveries = []
        eval_episodes = 50
        for _ in range(eval_episodes):
            eval_world.reset()
            if len(eval_world.colony) > 1:
                eval_world.colony[1].q_agent = eval_agent
            start_food = getattr(eval_world.colony[0], "food", 0) if len(eval_world.colony) > 0 else 0
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated) and steps < max_steps:
                _, r, terminated, truncated, _ = eval_world.step(None)
                if r is not None:
                    total_reward += r
                steps += 1
            end_food = getattr(eval_world.colony[0], "food", 0) if len(eval_world.colony) > 0 else 0
            rewards.append(total_reward)
            lengths.append(steps)
            deliveries.append(max(0, end_food - start_food))
        return np.mean(rewards), np.mean(lengths), np.mean(deliveries)

    # Reuse the same lr/gamma from train_agent defaults
    learning_rate = 0.1
    discount_factor = 0.9
    q_eval = greedy_eval(QLearningAgent, q_agent.q_table)
    d_eval = greedy_eval(DynaQAgent, dyna_agent.q_table)
    print("Greedy eval (50 eps):")
    print(f"  Q-learning:    avg_reward={q_eval[0]:.2f}, avg_steps={q_eval[1]:.1f}, avg_deliveries={q_eval[2]:.2f}")
    print(f"  Dyna-Q({planning_steps}): avg_reward={d_eval[0]:.2f}, avg_steps={d_eval[1]:.1f}, avg_deliveries={d_eval[2]:.2f}")


def compare_q_vs_dyna_suite(
    planning_steps_list=(3, 5, 20),
    epsilon_list=(0.3, 0.5),
    epsilon_decay_list=(0.99, 0.995),
    min_epsilon: float = 0.05,
    episodes: int = 500,
    max_steps: int = 600,
    smooth_window: int = 50,
    seed: Optional[int] = 42,
    timestamped: bool = False,
) -> None:
    """
    Run multiple compare_q_vs_dyna configurations (planning_depth x epsilon schedule) and save separate plots.
    """
    base_dir = os.path.join("results", "comparisons")
    if timestamped:
        base_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(base_dir, exist_ok=True)
    for ps in planning_steps_list:
        for eps in epsilon_list:
            for ed in epsilon_decay_list:
                outfile = os.path.join(
                    base_dir,
                    f"training_results_comparison_p{ps}_eps{eps}_decay{ed}.png".replace('.', 'p'),
                )
                print(f"\n=== Comparing Q vs Dyna-Q: planning={ps}, epsilon={eps}, decay={ed} ===")
                compare_q_vs_dyna(
                    planning_steps=ps,
                    epsilon=eps,
                    epsilon_decay=ed,
                    min_epsilon=min_epsilon,
                    episodes=episodes,
                    max_steps=max_steps,
                    smooth_window=smooth_window,
                    seed=seed,
                    output_path=outfile,
                    timestamped=False,
                )


def main() -> None:
    print("Starting Q-learning training: 500 episodes with fixed hyperparameters")
    rewards, lengths = train_agent(animate=True)
    plot_training_results(rewards, lengths)
    print("Training completed!")


if __name__ == "__main__":
    main()
