"""
Hyperparameter testing for Q-learning agents.

Runs combinations of learning_rate, gamma, epsilon, and epsilon_decay.
Saves results to CSV and generates training graphs.

"""

import sys
import os
import itertools
import csv
from datetime import datetime
from typing import List, Tuple

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib.pyplot as plt
from hex_grid_world import HexGridWorld
from q_learning import QLearningAgent
from dyna_q import DynaQAgent
from sarsa import SARSAAgent


def train_agent(
    agent_cls,
    agent_kwargs: dict,
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
    epsilon_decay: float,
    episodes: int = 200,
    animate: bool = False
) -> Tuple[List[int], List[int], List[int], object, object]:
    min_epsilon = 0.01
    algo_label = agent_cls.__name__

    world = HexGridWorld(train=True, worldType=1, animate=True)

    # Create Q-learning agent with hyperparameters
    q_agent = None
    if len(world.colony) > 1:
        q_agent = agent_cls(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            **agent_kwargs,
        )
        world.colony[1].q_agent = q_agent

    episode_rewards = []
    episode_lengths = []
    episode_deliveries = []

    print(f"Training with {algo_label}: lr={learning_rate}, γ={discount_factor}, ε={epsilon}, ε_decay={epsilon_decay}")

    for episode in range(episodes):
        world.reset()
        start_food = 0
        if len(world.colony) > 0:
            start_food = getattr(world.colony[0], 'food', 0)

        # Reassign Q-agent after reset (Q-table persists across episodes)
        if len(world.colony) > 1:
            if q_agent is None:
                q_agent = agent_cls(
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    min_epsilon=min_epsilon,
                    **agent_kwargs,
                )
            world.colony[1].q_agent = q_agent

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < 1000:
            # Q-learning agent selects action internally with Worker.act() so we pass None as action since Worker.act() will use the Q-learning agent to determine the action
            _, reward, terminated, truncated, _ = world.step(None)

            if reward is not None:
                total_reward += reward
            steps += 1

        # Decay epsilon after each episode
        if q_agent is not None:
            q_agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        end_food = 0
        if len(world.colony) > 0:
            end_food = getattr(world.colony[0], 'food', 0)
        episode_deliveries.append(max(0, end_food - start_food))

        if episode % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            avg_length = sum(episode_lengths[-100:]) / min(100, len(episode_lengths))
            current_epsilon = q_agent.epsilon if q_agent else epsilon
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}, ε = {current_epsilon:.3f}")

    return episode_rewards, episode_lengths, episode_deliveries, q_agent, world


def plot_training_results(
    episode_rewards: List[int],
    episode_lengths: List[int],
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    epsilon: float = 0.9,
    epsilon_decay: float = 0.995,
    save_path: str = None
) -> None:
    """
    Plot training results with specified hyperparameters in title.
    """
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(episode_rewards)
    ax1.set_title(f'Training Rewards Over Episodes (lr={learning_rate}, ε={epsilon}, γ={discount_factor}, ε_decay={epsilon_decay})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)

    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths Over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"  -> Graph saved to {save_path}")
    else:
        plt.savefig('training_results.png')
        plt.show()


def main(timestamped: bool = False) -> None:
    lrs = [0.001, 0.01]
    gammas = [0.9, 0.99]
    epsilons = [0.3, 0.5]
    eps_decays = [0.99, 0.995]
    episodes = 300

    algos = [
        ("q_learning", QLearningAgent, {}),
        ("sarsa", SARSAAgent, {}),
        ("dyna_q_p3", DynaQAgent, {"planning_steps": 3}),
        ("dyna_q_p5", DynaQAgent, {"planning_steps": 5}),
    ]

    combos = list(itertools.product(algos, lrs, gammas, epsilons, eps_decays))
    print(f"Starting hyperparameter search: {len(combos)} combinations, {episodes} episodes each\n")

    # Directory to store results; timestamped if requested
    if timestamped:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", "hyperparameter_sweeps", f"hp_sweep_{run_id}")
    else:
        run_dir = os.path.join("results", "hyperparameter_sweeps")
    os.makedirs(run_dir, exist_ok=True)
    results = []
    all_training_data = []

    # Run all experiments and collect full training data
    for i, combo in enumerate(combos, 1):
        (algo_name, agent_cls, agent_kwargs), lr, gamma, eps, eps_decay = combo
        print(f"[{i}/{len(combos)}] Testing algo={algo_name}, lr={lr}, gamma={gamma}, eps={eps}, eps_decay={eps_decay}")

        episode_rewards, episode_lengths, episode_deliveries, q_agent, world = train_agent(
            agent_cls=agent_cls,
            agent_kwargs=agent_kwargs,
            learning_rate=lr,
            discount_factor=gamma,
            epsilon=eps,
            epsilon_decay=eps_decay,
            episodes=episodes,
            animate=True
        )
        
        # avg_last50: Average reward over the last 50 training episodes to determine final performance
        avg_last50 = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
        avg_steps = sum(episode_lengths) / len(episode_lengths)
        avg_deliveries = sum(episode_deliveries) / len(episode_deliveries)
        
        # Save per-combo training plot
        combo_name = f"{algo_name}_lr{lr}_g{gamma}_e{eps}_d{eps_decay}".replace('.', 'p')
        train_plot_path = os.path.join(run_dir, f"train_{combo_name}.png")
        plot_training_results(episode_rewards, episode_lengths, learning_rate=lr, discount_factor=gamma, epsilon=eps, epsilon_decay=eps_decay, save_path=train_plot_path)

        # Evaluate policy in full environment (train=False)
        eval_episodes = 50
        # Create a fresh copy of the Q-agent for evaluation and set epsilon to 0 for deterministic greedy policy
        eval_agent = agent_cls(
            learning_rate=lr,
            discount_factor=gamma,
            epsilon=0.0,
            epsilon_decay=eps_decay,
            min_epsilon=0.01,
            **agent_kwargs,
        )
        # Copy trained Q-table to eval agent
        eval_agent.q_table = dict(q_agent.q_table)
        world.train = False
        world.colony[1].q_agent = eval_agent
        eval_rewards = []
        eval_lengths = []
        eval_deliveries = []
        for e in range(eval_episodes):
            world.reset()
            # Reset recreates the colony, reattach the frozen eval agent to worker[1]
            if len(world.colony) > 1:
                world.colony[1].q_agent = eval_agent
            start_food = getattr(world.colony[0], 'food', 0) if len(world.colony) > 0 else 0
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated) and steps < 1000:
                s_, r, terminated, truncated, _ = world.step(None)
                if r is not None:
                    total_reward += r
                steps += 1
            end_food = getattr(world.colony[0], 'food', 0) if len(world.colony) > 0 else 0
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
            eval_deliveries.append(max(0, end_food - start_food))
        # Restore world to training mode
        world.train = True
        world.colony[1].q_agent = q_agent

        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        avg_eval_steps = sum(eval_lengths) / len(eval_lengths)
        avg_eval_deliveries = sum(eval_deliveries) / len(eval_deliveries)

        results.append((algo_name, lr, gamma, eps, eps_decay, avg_last50, avg_steps, avg_deliveries, avg_eval_reward, avg_eval_steps, avg_eval_deliveries))
        all_training_data.append((algo_name, lr, gamma, eps, eps_decay, episode_rewards, episode_lengths, episode_deliveries, eval_rewards, eval_lengths, eval_deliveries))

        print(f"  -> train_avg_reward_last50={avg_last50:.2f}, train_avg_steps={avg_steps:.2f}, train_avg_deliveries={avg_deliveries:.2f}, eval_avg_reward={avg_eval_reward:.2f}, eval_avg_steps={avg_eval_steps:.1f}, eval_avg_deliveries={avg_eval_deliveries:.2f}\n")
    
    # Find best configuration
    # Choose best configuration from the results of train_avg_last50
    best_idx = max(range(len(results)), key=lambda i: results[i][5])
    best = results[best_idx]
    best_algo, best_lr, best_gamma, best_eps, best_eps_decay = best[0], best[1], best[2], best[3], best[4]
    
    print(f"\nBest configuration:")
    print(f"  algo={best_algo}, lr={best_lr}, gamma={best_gamma}, eps={best_eps}, eps_decay={best_eps_decay}")
    print(f"  train_avg_reward_last50={best[5]:.2f}, train_avg_steps={best[6]:.2f}, train_avg_deliveries={best[7]:.2f}")
    print(f"  eval_avg_reward={best[8]:.2f}, eval_avg_steps={best[9]:.2f}, eval_avg_deliveries={best[10]:.2f}")
    
    # Write results to a CSV file
    csv_path = os.path.join(run_dir, 'hyperparameter_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algo', 'lr', 'gamma', 'epsilon', 'eps_decay', 'train_avg_last50', 'train_avg_steps', 'train_avg_deliveries', 'eval_avg_reward', 'eval_avg_steps', 'eval_avg_deliveries'])
        writer.writerows(results)
    print(f"CSV saved: {csv_path}")

    # Generate graph for best configuration
    print(f"\nGenerating training graph for best configuration...")
    best_episode_rewards, best_episode_lengths = all_training_data[best_idx][5], all_training_data[best_idx][6]

    graph_path = os.path.join(run_dir, f'{best_algo}_training_hyperparameters_results.png')
    plot_training_results(
        best_episode_rewards,
        best_episode_lengths,
        learning_rate=best_lr,
        discount_factor=best_gamma,
        epsilon=best_eps,
        epsilon_decay=best_eps_decay,
        save_path=graph_path
    )
    best_eval_rewards = all_training_data[best_idx][8]
    best_eval_lengths = all_training_data[best_idx][9]
    eval_graph_path = os.path.join(run_dir, f'{best_algo}_training_hyperparameter_evaluation_results.png')
    plot_training_results(
        best_eval_rewards,
        best_eval_lengths,
        learning_rate=best_lr,
        discount_factor=best_gamma,
        epsilon=best_eps,
        epsilon_decay=best_eps_decay,
        save_path=eval_graph_path
    )
    
    print(f'\nHyperparameter testing completed!')
    
    # Print sorted table (by train_avg_last50)
    print('\nTop configs (by train_avg_last50):')
    print('algo\tlr\tgamma\teps\tdecay\ttrain_avg_last50\ttrain_avg_steps\ttrain_avg_deliveries\teval_avg_reward\teval_avg_steps\teval_avg_deliveries')
    sorted_results = sorted(results, key=lambda r: r[5], reverse=True)
    for r in sorted_results[:10]:
        print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]:.2f}\t{r[6]:.1f}\t{r[7]:.2f}\t{r[8]:.2f}\t{r[9]:.1f}\t{r[10]:.2f}")


if __name__ == '__main__':
    main()
