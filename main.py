"""
E-NSGA-II + X-MARL: Main Experiment Runner

This script runs the complete E-NSGA-II + X-MARL portfolio optimization system.

Usage:
    python main.py                    # Full run with default settings
    python main.py --test             # Quick test mode
    python main.py --pop_size 30 --generations 20  # Custom parameters
    python main.py --delta 0.05 --ppo_updates 5    # Custom NSGA-II params

Author: E-NSGA-II + X-MARL Research Team
"""

import os
import pandas as pd
import numpy as np
import torch
import argparse
from datetime import datetime

from env.portfolio_env import PortfolioEnv
from models.agents import ReturnAgent, RiskAgent, ExplainAgent, MultiAgentEnsemble
from algorithms.nsga_ii import NSGA2
from utils.metrics import calculate_metrics
from baselines.traditional import TraditionalBaselines


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='E-NSGA-II + X-MARL Portfolio Optimization System'
    )

    # Data parameters
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (minimal data/compute)')

    # NSGA-II parameters
    parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size for NSGA-II')
    parser.add_argument('--generations', type=int, default=10,
                        help='Number of evolutionary generations')
    parser.add_argument('--delta', type=float, default=0.05,
                        help='Explainability dominance threshold (δ)')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                        help='Mutation rate for genetic operators')

    # PPO parameters
    parser.add_argument('--ppo_updates', type=int, default=5,
                        help='PPO updates per generation (0 to disable)')

    # Agent parameters
    parser.add_argument('--hidden_sizes', type=str, default='256,128,64',
                        help='Hidden layer sizes (comma-separated)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for agents')

    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (None = auto)')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(test_mode: bool = False):
    """
    Load preprocessed data from CSV files.

    Args:
        test_mode: If True, use minimal data for quick testing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_path = "data/train.csv"
    val_path = "data/val.csv"
    test_path = "data/test.csv"

    # Check for test data files first
    if test_mode and os.path.exists("data/test_train.csv"):
        print("Loading test data files...")
        train_df = pd.read_csv("data/test_train.csv", index_col=[0, 1], parse_dates=True)
        val_df = pd.read_csv("data/test_val.csv", index_col=[0, 1], parse_dates=True)
        test_df = pd.read_csv("data/test_test.csv", index_col=[0, 1], parse_dates=True)
        return train_df, val_df, test_df

    # Check for main data files
    if not os.path.exists(train_path):
        print("=" * 60)
        print("ERROR: Data not found!")
        print("=" * 60)
        print(f"Expected file: {os.path.abspath(train_path)}")
        print("\nPlease run the following commands to generate data:")
        print("  python data/downloader.py")
        print("  python data/preprocessor.py")
        print("=" * 60)
        return None, None, None

    print("Loading preprocessed data...")
    train_df = pd.read_csv(train_path, index_col=[0, 1], parse_dates=True)
    val_df = pd.read_csv(val_path, index_col=[0, 1], parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=[0, 1], parse_dates=True)

    # Truncate for test mode (by date, not by row, to keep all stocks)
    if test_mode:
        print("Test mode: Truncating data for speed...")

        def truncate_by_dates(df, n_days):
            """Truncate dataframe to first n_days."""
            dates = df.index.get_level_values(0).unique().sort_values()
            keep_dates = dates[:n_days]
            return df[df.index.get_level_values(0).isin(keep_dates)]

        train_df = truncate_by_dates(train_df, 100)
        val_df = truncate_by_dates(val_df, 50)
        test_df = truncate_by_dates(test_df, 50)

    # Ensure consistent stocks across all splits
    train_tickers = set(train_df.index.get_level_values(1).unique())
    val_tickers = set(val_df.index.get_level_values(1).unique())
    test_tickers = set(test_df.index.get_level_values(1).unique())
    common_tickers = train_tickers & val_tickers & test_tickers

    # Check if any split has more stocks than common
    max_tickers = max(len(train_tickers), len(val_tickers), len(test_tickers))
    if len(common_tickers) < max_tickers:
        print(f"  Filtering to {len(common_tickers)} common stocks (train={len(train_tickers)}, val={len(val_tickers)}, test={len(test_tickers)})...")
        train_df = train_df[train_df.index.get_level_values(1).isin(common_tickers)]
        val_df = val_df[val_df.index.get_level_values(1).isin(common_tickers)]
        test_df = test_df[test_df.index.get_level_values(1).isin(common_tickers)]

    return train_df, val_df, test_df


def create_evaluation_function(env):
    """
    Create evaluation function for NSGA-II fitness computation.

    Args:
        env: Portfolio environment

    Returns:
        Function that evaluates an agent and returns [return, -risk, explainability]
    """
    def evaluate(agent):
        """Evaluate agent on environment, return multi-objective fitness."""
        obs, _ = env.reset()
        terminated = False

        total_sharpe = 0
        total_risk = 0
        total_explain = 0
        steps = 0

        while not terminated:
            features = obs['features'].flatten()
            action = agent.get_action(features)

            obs, rewards, terminated, truncated, _ = env.step(action)

            # rewards is [sharpe, risk_reward, explain_reward]
            if isinstance(rewards, np.ndarray) and len(rewards) >= 3:
                total_sharpe += rewards[0]
                total_risk += rewards[1]
                total_explain += rewards[2]
            else:
                # Fallback for scalar reward
                total_sharpe += rewards if not isinstance(rewards, np.ndarray) else rewards[0]

            steps += 1
            terminated = terminated or truncated

        # Return average rewards over episode
        if steps > 0:
            return [total_sharpe / steps, total_risk / steps, total_explain / steps]
        return [0.0, 0.0, 0.0]

    return evaluate


def run_baselines(env_test, results_dir):
    """
    Run traditional baseline methods for comparison.

    Args:
        env_test: Test environment
        results_dir: Directory to save results
    """
    print("\n" + "=" * 60)
    print("Running Baseline Methods")
    print("=" * 60)

    bl = TraditionalBaselines(env_test)
    baseline_results = {}

    # Equal Weight
    print("\n1. Equal Weight Strategy...")
    env_test.reset()
    terminated = False
    while not terminated:
        target_w = bl.equal_weight()
        action = np.log(target_w + 1e-8) * 100
        _, _, terminated, truncated, _ = env_test.step(action)
        terminated = terminated or truncated or env_test.current_step >= len(env_test.dates) - 1

    if len(env_test.portfolio_history) > 0:
        ew_metrics = calculate_metrics(env_test.portfolio_history)
        baseline_results['Equal Weight'] = ew_metrics
        print(f"   Return: {ew_metrics['Annualized Return']*100:.2f}%")
        print(f"   Sharpe: {ew_metrics['Sharpe Ratio']:.2f}")

    # Min Variance (if available)
    if hasattr(bl, 'min_variance'):
        print("\n2. Minimum Variance Strategy...")
        env_test.reset()
        terminated = False
        while not terminated:
            target_w = bl.min_variance()
            action = np.log(target_w + 1e-8) * 100
            _, _, terminated, truncated, _ = env_test.step(action)
            terminated = terminated or truncated or env_test.current_step >= len(env_test.dates) - 1

        if len(env_test.portfolio_history) > 0:
            mv_metrics = calculate_metrics(env_test.portfolio_history)
            baseline_results['Min Variance'] = mv_metrics
            print(f"   Return: {mv_metrics['Annualized Return']*100:.2f}%")
            print(f"   Sharpe: {mv_metrics['Sharpe Ratio']:.2f}")

    # Save baseline results
    if baseline_results:
        baseline_df = pd.DataFrame(baseline_results).T
        baseline_df.to_csv(os.path.join(results_dir, "baseline_metrics.csv"))
        print(f"\nBaseline results saved to {results_dir}/baseline_metrics.csv")

    return baseline_results


def main():
    """Main entry point for E-NSGA-II + X-MARL experiments."""
    args = parse_args()

    # Parse hidden sizes
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]

    print("\n" + "=" * 60)
    print("E-NSGA-II + X-MARL Portfolio Optimization")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'TEST' if args.test else 'PRODUCTION'}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Set seeds
    set_seed(args.seed)

    # Load data
    train_df, val_df, test_df = load_data(args.test)
    if train_df is None:
        return

    print(f"\nData loaded:")
    print(f"  Train: {len(train_df)} samples ({len(train_df.index.get_level_values(0).unique())} days)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df.index.get_level_values(0).unique())} days)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df.index.get_level_values(0).unique())} days)")

    # Create environments
    print("\nCreating environments...")
    env_train = PortfolioEnv(train_df)
    env_val = PortfolioEnv(val_df)
    env_test = PortfolioEnv(test_df)

    print(f"  Assets: {env_train.n_assets}")
    print(f"  Features: {env_train.n_features}")

    # Calculate observation dimension
    obs_dim = env_train.n_assets * env_train.n_features
    action_dim = env_train.n_assets + 1  # +1 for cash

    print(f"  Obs dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize population
    print(f"\nInitializing population...")
    print(f"  Population size: {args.pop_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Delta (δ): {args.delta}")
    print(f"  PPO updates/gen: {args.ppo_updates}")
    print(f"  Hidden sizes: {hidden_sizes}")

    population = [
        ReturnAgent(obs_dim, action_dim, hidden_sizes=hidden_sizes, lr=args.lr)
        for _ in range(args.pop_size)
    ]

    # Count parameters
    total_params = sum(p.numel() for p in population[0].parameters())
    print(f"  Parameters per agent: {total_params:,}")

    # Create NSGA-II optimizer
    nsga2 = NSGA2(
        population_size=args.pop_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        delta=args.delta,
        ppo_updates_per_gen=args.ppo_updates,
        n_workers=args.n_workers
    )

    # Create evaluation function
    evaluate_fn = create_evaluation_function(env_train)

    # Run evolution
    print("\n" + "=" * 60)
    print("Starting Evolution")
    print("=" * 60)

    final_pop, final_fitness = nsga2.evolve(
        population,
        evaluate_fn,
        env_train=env_train if args.ppo_updates > 0 else None,
        verbose=True
    )

    # Save Pareto front
    print("\nSaving Pareto front...")
    fitness_data = []
    for fit in final_fitness:
        fitness_data.append({
            'Return': fit[0],
            'Risk': -fit[1],  # Negate back to positive
            'Explainability': fit[2]
        })
    pareto_df = pd.DataFrame(fitness_data)
    pareto_df.to_csv(os.path.join(results_dir, "pareto_front.csv"), index=False)

    # Generate plots
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # 3D Pareto front
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            pareto_df['Risk'],
            pareto_df['Return'],
            pareto_df['Explainability'],
            c=pareto_df['Explainability'],
            cmap='viridis',
            s=50
        )
        ax.set_xlabel('Risk (Volatility)')
        ax.set_ylabel('Return')
        ax.set_zlabel('Explainability')
        ax.set_title('Pareto Front: Return vs Risk vs Explainability')
        plt.savefig(os.path.join(results_dir, "pareto_front_3d.png"), dpi=150)
        plt.close()

        print(f"  Saved: {results_dir}/pareto_front_3d.png")

    except ImportError:
        print("  Matplotlib not available, skipping plots")

    # Select best model (highest Sharpe from Pareto front)
    best_idx = np.argmax([f[0] for f in final_fitness])
    best_agent = final_pop[best_idx]
    print(f"\nBest agent (idx {best_idx}):")
    print(f"  Return: {final_fitness[best_idx][0]:.4f}")
    print(f"  Risk: {-final_fitness[best_idx][1]:.4f}")
    print(f"  Explainability: {final_fitness[best_idx][2]:.4f}")

    # Test best model
    print("\n" + "=" * 60)
    print("Testing Best Agent")
    print("=" * 60)

    # Verify dimension compatibility between train and test envs
    if env_test.n_assets != env_train.n_assets or env_test.n_features != env_train.n_features:
        print(f"  Warning: Test env dimensions differ from train!")
        print(f"  Train: {env_train.n_assets} assets × {env_train.n_features} features = {obs_dim}")
        print(f"  Test:  {env_test.n_assets} assets × {env_test.n_features} features = {env_test.n_assets * env_test.n_features}")
        print("  Skipping best agent test due to dimension mismatch.")
        test_metrics = None
    else:
        obs, _ = env_test.reset()
        terminated = False

        while not terminated:
            features = obs['features'].flatten()
            action = best_agent.get_action(features, deterministic=True)
            obs, _, terminated, truncated, _ = env_test.step(action)
            terminated = terminated or truncated
        test_metrics = "run"

    if test_metrics == "run" and len(env_test.portfolio_history) > 0:
        test_metrics = calculate_metrics(env_test.portfolio_history)
        print("\nTest Results:")
        for key, value in test_metrics.items():
            if 'Return' in key or 'Volatility' in key or 'Drawdown' in key or 'CVaR' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")

        # Save test metrics
        test_df = pd.DataFrame([test_metrics])
        test_df.to_csv(os.path.join(results_dir, "test_metrics_best_agent.csv"), index=False)
    elif test_metrics is None:
        print("\n  (Test metrics unavailable due to dimension mismatch)")

    # Run baselines
    baseline_results = run_baselines(env_test, results_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Experiment Complete")
    print("=" * 60)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {os.path.abspath(results_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
