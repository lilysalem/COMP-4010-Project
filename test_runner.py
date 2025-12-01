"""
Test runner helpers and a quick manual for running comparisons/sweeps.

How to use:
1) Pick one or more test blocks below and uncomment the call(s) in main().
2) Run: python test_runner.py

Available tests:
- compare_q_vs_dyna_single(): Q-learning vs Dyna-Q (planning_steps configurable).
- compare_q_vs_dyna_suite(): small grid over planning/epsilon/decay; saves plots.
- hyperparameter_sweep(): Q-learning vs Dyna-Q (planning 3/5) over a small lr/gamma/eps/decay grid; saves CSV and plots.

Results paths:
- Comparisons: results/comparisons/ (timestamped if you pass timestamped=True).
- Hyperparameter sweeps: results/hyperparameter_sweeps/ (timestamped optional).
"""

from testbed import compare_q_vs_dyna, compare_q_vs_dyna_suite
from testbed_hyperparameters import main as hyperparameter_sweep


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


def main():
    # Uncomment the test(s) you want to run:

    # compare_q_vs_dyna_single()
    # compare_q_vs_dyna_grid()
    # hyperparameter_sweep_run()

    print("No tests selected. Edit test_runner.py main() to uncomment a test.")


if __name__ == "__main__":
    main()
