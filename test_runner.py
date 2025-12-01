"""
Test runner helpers and a quick manual for running comparisons/sweeps.

How to use:
1) Pick one or more test blocks below and uncomment the call(s) in main().
2) Run: python test_runner.py

Available tests:
- compare_q_vs_dyna_single(): Q-learning vs Dyna-Q (planning_steps configurable).
- compare_q_vs_dyna_suite(): small grid over planning/epsilon/decay; saves plots.
- compare_q_dyna_sarsa(): Q-learning vs Dyna-Q vs SARSA; overwrites unless timestamped.
- sarsa_smoothed_plot(): single SARSA run with ε-decay and a rolling-mean plot.
- hyperparameter_sweep(): Q-learning vs Dyna-Q (planning 3/5) over a small lr/gamma/eps/decay grid; saves CSV and plots.

Results paths:
- Comparisons (Q vs Dyna): results/comparisons/ (timestamped if you pass timestamped=True).
- Comparisons (with SARSA): results/comparisons_with_sarsa/ (timestamped optional).
- SARSA smoothed plot: defaults to training_results_sarsa_smoothed.png (set output_path to override).
- Hyperparameter sweeps: results/hyperparameter_sweeps/ (timestamped optional).
"""

from testbed import compare_q_vs_dyna, compare_q_vs_dyna_suite
from testbed_hyperparameters import main as hyperparameter_sweep
from testbed import train_agent
from sarsa import SARSAAgent


def compare_q_vs_dyna_single():
    """
    Single comparison with custom knobs.
    Adjust planning_steps/epsilon/epsilon_decay as needed.
    """
    compare_q_vs_dyna(
        planning_steps=5,
        epsilon=0.3,
        epsilon_decay=0.99,
        min_epsilon=0.05,
        episodes=500,
        max_steps=600,
        smooth_window=50,
        seed=42,
        timestamped=False,  # set True to avoid overwriting
        output_path=None,   # default path if None
    )


def compare_q_dyna_sarsa():
    """
    Single comparison that always includes SARSA alongside Q-learning and Dyna-Q.
    Uses the same knobs as compare_q_vs_dyna_single, but forces include_sarsa=True.
    """
    compare_q_vs_dyna(
        planning_steps=5,
        epsilon=0.3,
        epsilon_decay=0.99,
        min_epsilon=0.05,
        episodes=500,
        max_steps=600,
        smooth_window=50,
        seed=42,
        timestamped=False,
        output_path=None,
        include_sarsa=True,
    )


def compare_q_vs_dyna_grid():
    """
    Small grid over planning_steps x epsilon x epsilon_decay.
    Plots saved to results/comparisons/ (or timestamped subfolder if enabled).
    """
    compare_q_vs_dyna_suite(
        planning_steps_list=(3, 5, 20),
        epsilon_list=(0.3, 0.5),
        epsilon_decay_list=(0.99, 0.995),
        min_epsilon=0.05,
        episodes=500,
        max_steps=600,
        smooth_window=50,
        seed=42,
        timestamped=False,  # set True to create a dated subfolder
    )


def hyperparameter_sweep_run():
    """
    Hyperparameter sweep for Q-learning vs Dyna-Q (planning 3/5) on a small grid.
    Outputs go to results/hyperparameter_sweeps/ (or timestamped subfolder).
    """
    hyperparameter_sweep(timestamped=False)  # set True for dated run folder


def sarsa_smoothed_plot(
    episodes: int = 500,
    epsilon: float = 0.3,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    max_steps: int = 600,
    window: int = 50,
    seed: int | None = None,
    output_path: str = "training_results_sarsa_smoothed.png",
):
    """
    Run SARSA with an epsilon schedule and save a rolling-mean rewards plot.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards, _ = train_agent(
        animate=False,
        agent_cls=SARSAAgent,
        agent_kwargs={"epsilon_decay": epsilon_decay, "min_epsilon": min_epsilon},
        episodes=episodes,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        max_steps=max_steps,
        seed=seed,
    )

    weights = np.ones(window) / window
    smooth = np.convolve(rewards, weights, mode="valid") if len(rewards) >= window else []

    fig, ax = plt.subplots()
    ax.plot(rewards, alpha=0.3, label="raw")
    if len(smooth) > 0:
        ax.plot(range(window - 1, window - 1 + len(smooth)), smooth, label=f"rolling mean ({window})")
    ax.set_title(f"SARSA rewards (ε={epsilon}, decay={epsilon_decay}, min={min_epsilon}, episodes={episodes})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"SARSA smoothed plot saved to {output_path}")


def main():
    # Uncomment the test(s) you want to run:

    compare_q_vs_dyna_single()
    # compare_q_dyna_sarsa()
    # compare_q_vs_dyna_grid()
    # hyperparameter_sweep_run()
    # sarsa_smoothed_plot()

    print("No tests selected. Edit test_runner.py main() to uncomment a test.")


if __name__ == "__main__":
    main()
