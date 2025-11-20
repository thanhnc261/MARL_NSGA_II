"""
Ablation Study Variants

Implements ablation variants for analyzing components of E-NSGA-II + X-MARL:

1. Ablation 1: NSGA-II + X-MARL (no Explainability Dominance, δ=0)
   - Tests the contribution of the Explainability Dominance operator
   - Sets delta tolerance to 0 in dominance function

2. Ablation 2: E-NSGA-II + single-agent PPO
   - Uses Enhanced NSGA-II outer loop
   - But with single-agent PPO instead of multi-agent setup
   - Tests multi-agent vs single-agent contribution
"""

import numpy as np
import sys
import os
import copy

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.nsga_ii import NSGA2
from models.agents import ReturnAgent
from baselines.single_ppo import PPOAgent


class NSGA2NoDominance(NSGA2):
    """
    Ablation 1: NSGA-II without Explainability Dominance.

    Sets δ=0 in the dominance operator, effectively removing the explainability preference.
    """

    def dominates(self, p, q, delta=0.0):
        """
        Standard NSGA-II dominance without explainability preference.

        δ=0 means no tolerance for explainability, making it pure Pareto dominance.
        """
        # Call parent with delta=0
        return super().dominates(p, q, delta=0.0)


class EnhancedNSGA2SingleAgent:
    """
    Ablation 2: E-NSGA-II with single-agent PPO.

    Uses Enhanced NSGA-II (with Explainability Dominance)
    but optimizes single PPO agent instead of multi-agent system.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        reward_weights: tuple = (0.5, 0.3, 0.2)
    ):
        """
        Initialize ablation variant.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            population_size: Population size
            generations: Number of generations
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate
            reward_weights: Scalarization weights for PPO
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.reward_weights = reward_weights

        # Use standard NSGA-II with Explainability Dominance (δ > 0)
        self.nsga2 = NSGA2(population_size, generations, mutation_rate)

    def evolve(self, evaluate_fn):
        """
        Run evolution with single PPO agents.

        Args:
            evaluate_fn: Evaluation function

        Returns:
            (final_population, final_fitnesses)
        """
        # Initialize population with single-agent PPO
        population = [
            PPOAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_size=256,
                learning_rate=3e-4,
                gamma=0.95,
                reward_weights=self.reward_weights
            )
            for _ in range(self.pop_size)
        ]

        # Run NSGA-II evolution
        final_pop, final_fitness = self.nsga2.evolve(population, evaluate_fn)

        return final_pop, final_fitness


def run_ablation_1(env_train, env_val, env_test, pop_size=20, generations=10):
    """
    Run Ablation 1: No Explainability Dominance (δ=0).

    Tests whether the Explainability Dominance operator improves results.

    Args:
        env_train: Training environment
        env_val: Validation environment
        env_test: Test environment
        pop_size: Population size
        generations: Number of generations

    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("Ablation 1: NSGA-II + MARL (No Explainability Dominance, δ=0)")
    print("="*80)

    from main import main as run_main  # Import main training function

    # This is similar to main.py but uses NSGA2NoDominance
    state_dim = env_train.observation_space['features'].shape[0] * env_train.observation_space['features'].shape[1]
    action_dim = env_train.n_assets + 1

    # Create population
    population = [ReturnAgent(state_dim, action_dim) for _ in range(pop_size)]

    # Use modified NSGA-II
    nsga2_no_dom = NSGA2NoDominance(pop_size, generations)

    # Evaluation function
    def evaluate(agent):
        obs, _ = env_train.reset()
        terminated = False
        total_rewards = np.zeros(3)
        steps = 0

        while not terminated:
            features = obs['features'].flatten()
            action = agent.get_action(features)
            obs, rewards, terminated, truncated, _ = env_train.step(action)
            total_rewards += rewards
            steps += 1

        return (total_rewards / max(steps, 1)).tolist()

    # Run evolution
    print(f"\nEvolving (δ=0)...")
    final_pop, final_fitness = nsga2_no_dom.evolve(population, evaluate)

    # Test best model
    print("\nTesting...")
    best_idx = 0  # Placeholder
    best_model = final_pop[best_idx]

    obs, _ = env_test.reset()
    terminated = False

    while not terminated:
        features = obs['features'].flatten()
        action = best_model.get_action(features, deterministic=True)
        obs, _, terminated, _, _ = env_test.step(action)

    from utils.metrics import calculate_metrics
    test_metrics = calculate_metrics(env_test.portfolio_history)

    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")

    return {
        'pareto_front': final_fitness,
        'best_model': best_model,
        'test_metrics': test_metrics,
        'variant': 'no_explainability_dominance'
    }


def run_ablation_2(env_train, env_val, env_test, pop_size=20, generations=10):
    """
    Run Ablation 2: E-NSGA-II + single-agent PPO.

    Tests whether multi-agent architecture improves over single-agent.

    Args:
        env_train: Training environment
        env_val: Validation environment
        env_test: Test environment
        pop_size: Population size
        generations: Number of generations

    Returns:
        Dictionary with results
    """
    print("\n" + "="*80)
    print("Ablation 2: E-NSGA-II + Single-Agent PPO")
    print("="*80)

    state_dim = env_train.observation_space['features'].shape[0] * env_train.observation_space['features'].shape[1]
    action_dim = env_train.n_assets + 1

    # Create ablation optimizer
    optimizer = EnhancedNSGA2SingleAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        population_size=pop_size,
        generations=generations,
        reward_weights=(0.5, 0.3, 0.2)
    )

    # Evaluation function
    def evaluate(agent):
        obs, _ = env_train.reset()
        agent.clear_trajectory()
        terminated = False
        total_rewards = np.zeros(3)
        steps = 0

        while not terminated:
            features = obs['features'].flatten()
            action = agent.get_action(features)
            obs, rewards, terminated, truncated, _ = env_train.step(action)
            total_rewards += rewards
            steps += 1

        return (total_rewards / max(steps, 1)).tolist()

    # Run evolution
    print(f"\nEvolving with single-agent PPO...")
    final_pop, final_fitness = optimizer.evolve(evaluate)

    # Test best model
    print("\nTesting...")
    best_idx = 0
    best_model = final_pop[best_idx]

    obs, _ = env_test.reset()
    terminated = False

    while not terminated:
        features = obs['features'].flatten()
        action = best_model.get_action(features, deterministic=True)
        obs, _, terminated, _, _ = env_test.step(action)

    from utils.metrics import calculate_metrics
    test_metrics = calculate_metrics(env_test.portfolio_history)

    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")

    return {
        'pareto_front': final_fitness,
        'best_model': best_model,
        'test_metrics': test_metrics,
        'variant': 'single_agent_ppo'
    }


if __name__ == "__main__":
    print("Ablation Studies Module")
    print("\nAvailable ablations:")
    print("1. NSGA-II + MARL (No Explainability Dominance, δ=0)")
    print("2. E-NSGA-II + Single-Agent PPO")
