import os
import pandas as pd
import numpy as np
import torch
from data.preprocessor import load_data, compute_features, split_and_normalize
from env.portfolio_env import PortfolioEnv
from models.agents import ReturnAgent, RiskAgent, ExplainAgent
from algorithms.nsga_ii import NSGA2
from utils.metrics import calculate_metrics
from baselines.traditional import TraditionalBaselines

import argparse

def main():
    parser = argparse.ArgumentParser(description='MARL NSGA-II System')
    parser.add_argument('--pop_size', type=int, default=20, help='Population size')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--test', action='store_true', help='Run in test mode (minimal data/compute)')
    args = parser.parse_args()

    print("Starting MARL NSGA-II System...")
    
    # 1. Data Loading
    # 1. Data Loading
    print(f"Current Working Directory: {os.getcwd()}")
    
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"Contents of '{data_dir}': {os.listdir(data_dir)}")
    else:
        print(f"Directory '{data_dir}' does not exist.")

    train_path = os.path.abspath("data/train.csv")
    val_path = os.path.abspath("data/val.csv")
    test_path = os.path.abspath("data/test.csv")
    
    print(f"Checking for training data at: {train_path}")

    if args.test and os.path.exists("data/test_train.csv"):
        print("Loading test data...")
        train_df = pd.read_csv("data/test_train.csv", index_col=[0, 1], parse_dates=True, date_format='%Y-%m-%d')
        val_df = pd.read_csv("data/test_val.csv", index_col=[0, 1], parse_dates=True, date_format='%Y-%m-%d')
        test_df = pd.read_csv("data/test_test.csv", index_col=[0, 1], parse_dates=True, date_format='%Y-%m-%d')
    elif not os.path.exists("data/train.csv"):
        print("ERROR: Data not found.")
        print(f"Expected file: {train_path}")
        print("Please run the following commands to generate the data:")
        print("  python data/downloader.py")
        print("  python data/preprocessor.py")
        return
    else:
        print("Loading data...")
        train_df = pd.read_csv("data/train.csv", index_col=[0, 1], parse_dates=True, date_format='%Y-%m-%d')
        val_df = pd.read_csv("data/val.csv", index_col=[0, 1], parse_dates=True, date_format='%Y-%m-%d')
        test_df = pd.read_csv("data/test.csv", index_col=[0, 1], parse_dates=True, date_format='%Y-%m-%d')
    
    if args.test:
        print("Test mode: Truncating data for speed...")
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]
    
    # 2. Environment Setup
    print("Setting up environment...")
    env_train = PortfolioEnv(train_df)
    env_val = PortfolioEnv(val_df)
    env_test = PortfolioEnv(test_df)
    
    # 3. Initialize Population
    print(f"Initializing population (Size: {args.pop_size}, Gens: {args.generations})...")
    pop_size = args.pop_size
    generations = args.generations
    
    # Population of Agents.
    # Each individual is a set of agents? Or one agent that outputs weights?
    # Guide says: "Your E-NSGA-II + X-MARL (full)"
    # Usually MARL means multiple agents interacting.
    # "Agents: ReturnAgent, RiskAgent, ExplainAgent"
    # Do they cooperate? Or are they objectives?
    # "Reward for ReturnAgent...", "Reward for RiskAgent..."
    # This implies we have 3 distinct agents acting in the environment?
    # But the environment expects ONE action (weights).
    # So there must be a mixing mechanism.
    # "Action: continuous vector ... softmax-normalized"
    # Maybe the agents vote? Or we have one policy network that optimizes multiple objectives?
    # Guide: "Pure NSGA-II (direct weight optimization, no RL)" vs "E-NSGA-II + X-MARL"
    # "Agents see only this data"
    # Let's assume we evolve a single Policy Network (Agent) that tries to satisfy all objectives.
    # OR we have a population of policies, and we evaluate each on 3 objectives.
    # Yes, NSGA-II evolves a population of solutions. Each solution is a policy.
    # The "Agents" (Return, Risk, Explain) might just be the *Reward Functions* or *Evaluators*.
    # Or maybe it's a modular architecture where 3 sub-agents propose actions and we aggregate?
    # Given the complexity, let's assume we evolve a single Policy Network (Agent) and evaluate it on 3 objectives.
    
    # Input dim: n_assets * n_features (flattened)
    obs_dim = env_train.observation_space['features'].shape[0] * env_train.observation_space['features'].shape[1]
    population = [ReturnAgent(obs_dim, env_train.n_assets + 1) for _ in range(pop_size)]
    
    nsga2 = NSGA2(pop_size, generations)
    
    # 4. Evaluation Function
    def evaluate(agent):
        # Run episode on training data
        obs, _ = env_train.reset()
        terminated = False
        
        total_sharpe = 0
        total_risk = 0
        total_explain = 0
        steps = 0
        
        while not terminated:
            # Get action from agent
            # Agent takes features as input
            features = obs['features']
            # We need to flatten or process features?
            # Agent expects (N_assets, N_features) or flattened?
            # Our simple Agent expects flat input?
            # Let's flatten features for the agent
            flat_features = features.flatten()
            
            action = agent.get_action(flat_features)
            
            obs, rewards, terminated, truncated, _ = env_train.step(action)
            
            # rewards is [sharpe, risk, explain]
            total_sharpe += rewards[0]
            total_risk += rewards[1]
            total_explain += rewards[2]
            steps += 1
            
        # Average rewards over episode?
        # Or use the final metrics?
        # Guide says: "Reward for ReturnAgent: daily Sharpe... rolling"
        # We can sum them up or take mean.
        return [total_sharpe/steps, total_risk/steps, total_explain/steps]

    # 5. Run Evolution
    print("Running evolution...")
    # We need to adapt evaluate to work with the population
    # Note: This is slow if sequential. Parallelization recommended.
    
    final_pop, final_fitness = nsga2.evolve(population, evaluate)
    
    # 6. Selection on Validation & Export Results
    print("Selecting best model and exporting results...")
    
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # Extract fitnesses of final population
    # fitnesses is a list of [Return, -Volatility, Explainability]
    # We want to save them as [Return, Volatility, Explainability]
    # So we negate the second component back to positive Volatility for reporting?
    # Wait, in env we return -Volatility. So fitness[1] is -Volatility.
    # To get Volatility, we take -fitness[1].
    
    final_fitness_data = []
    for fit in final_fitness:
        final_fitness_data.append([fit[0], -fit[1], fit[2]])
        
    results_df = pd.DataFrame(final_fitness_data, columns=['Return', 'Volatility', 'Explainability'])
    results_df.to_csv("results/pareto_front.csv", index=False)
    print("Pareto front data saved to results/pareto_front.csv")
    
    # Plotting
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 3D Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(results_df['Volatility'], results_df['Return'], results_df['Explainability'], c=results_df['Explainability'], cmap='viridis')
        ax.set_xlabel('Volatility (Risk)')
        ax.set_ylabel('Return')
        ax.set_zlabel('Explainability')
        ax.set_title('Pareto Front: Return vs Risk vs Explainability')
        plt.savefig("results/pareto_front_3d.png")
        plt.close()
        
        # 2D Pairwise Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Return vs Risk
        axes[0].scatter(results_df['Volatility'], results_df['Return'], c='b', alpha=0.6)
        axes[0].set_xlabel('Volatility')
        axes[0].set_ylabel('Return')
        axes[0].set_title('Return vs Risk')
        axes[0].grid(True)
        
        # Return vs Explainability
        axes[1].scatter(results_df['Explainability'], results_df['Return'], c='g', alpha=0.6)
        axes[1].set_xlabel('Explainability')
        axes[1].set_ylabel('Return')
        axes[1].set_title('Return vs Explainability')
        axes[1].grid(True)
        
        # Risk vs Explainability
        axes[2].scatter(results_df['Volatility'], results_df['Explainability'], c='r', alpha=0.6)
        axes[2].set_xlabel('Volatility')
        axes[2].set_ylabel('Explainability')
        axes[2].set_title('Explainability vs Risk')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig("results/pareto_front_2d.png")
        plt.close()
        
        print("Plots saved to results/pareto_front_3d.png and results/pareto_front_2d.png")
        
    except ImportError:
        print("Matplotlib not found. Skipping plotting.")

    best_model = final_pop[0] # Placeholder for actual selection logic (e.g. Hypervolume)
    
    # 7. Testing
    print("Testing best model...")
    obs, _ = env_test.reset()
    terminated = False
    while not terminated:
        features = obs['features'].flatten()
        action = best_model.get_action(features, deterministic=True)
        obs, _, terminated, _, _ = env_test.step(action)
        
    metrics = calculate_metrics(env_test.portfolio_history)
    print("Test Metrics:", metrics)
    
    # Save Test Metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("results/test_metrics_best_agent.csv", index=False)
    
    # 8. Baselines
    print("Running Baselines...")
    bl = TraditionalBaselines(env_test)
    
    # Equal Weight
    env_test.reset()
    terminated = False
    while not terminated:
        target_w = bl.equal_weight()
        action = np.log(target_w + 1e-8) * 100
        env_test.step(action)
        terminated = env_test.current_step >= len(env_test.dates) - 1
        
    ew_metrics = calculate_metrics(env_test.portfolio_history)
    print("Equal Weight Metrics:", ew_metrics)
    
    # Save Baseline Metrics
    ew_metrics_df = pd.DataFrame([ew_metrics])
    ew_metrics_df.to_csv("results/test_metrics_equal_weight.csv", index=False)

if __name__ == "__main__":
    main()
