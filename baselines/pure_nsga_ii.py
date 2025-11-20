"""
Pure NSGA-II Baseline (Direct Weight Optimization, No RL)

This baseline directly evolves portfolio weight vectors using NSGA-II,
without neural networks or reinforcement learning.

Serves as a comparison to demonstrate the value of combining evolutionary
algorithms with deep reinforcement learning.

Reference:
    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and
    elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on
    Evolutionary Computation, 6(2), 182-197.
"""

import numpy as np
import sys
import os
import copy

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.nsga_ii import NSGA2


class WeightIndividual:
    """
    Individual in Pure NSGA-II population.

    Represents a portfolio as a vector of weights (instead of neural network).
    """

    def __init__(self, n_assets: int, initialize_random: bool = True):
        """
        Initialize weight individual.

        Args:
            n_assets: Number of assets + 1 (for cash)
            initialize_random: If True, randomize weights; else equal weights
        """
        self.n_assets = n_assets

        if initialize_random:
            # Random weights + softmax normalize
            raw_weights = np.random.randn(n_assets)
            exp_weights = np.exp(raw_weights)
            self.weights = exp_weights / np.sum(exp_weights)
        else:
            # Equal weights
            self.weights = np.ones(n_assets) / n_assets

    def get_action(self, state, deterministic=False):
        """
        Return fixed portfolio weights (doesn't depend on state).

        This implements a constant rebalancing strategy.

        Args:
            state: Market state (ignored for fixed weights)
            deterministic: Ignored

        Returns:
            Log-weights for softmax normalization in environment
        """
        # Return log of weights (so environment can softmax them)
        # Add small epsilon to avoid log(0)
        log_weights = np.log(self.weights + 1e-8)
        return log_weights * 100  # Scale up for numerical stability


class PureNSGAII:
    """
    Pure NSGA-II baseline without reinforcement learning.

    Directly optimizes portfolio weight vectors using genetic algorithm.
    """

    def __init__(
        self,
        n_assets: int,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        """
        Initialize Pure NSGA-II optimizer.

        Args:
            n_assets: Number of assets (+ 1 for cash)
            population_size: Population size
            generations: Number of generations
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
        """
        self.n_assets = n_assets
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Use adapted NSGA-II algorithm
        self.nsga2 = PureNSGA2Adapter(
            population_size,
            generations,
            mutation_rate,
            crossover_rate
        )

    def evolve(self, evaluate_fn):
        """
        Run evolution to find Pareto-optimal portfolio weight sets.

        Args:
            evaluate_fn: Function that evaluates an individual and returns [obj1, obj2, obj3]

        Returns:
            (final_population, final_fitnesses)
        """
        # Initialize population with weight individuals
        population = [
            WeightIndividual(self.n_assets, initialize_random=True)
            for _ in range(self.pop_size)
        ]

        # Run NSGA-II evolution
        final_pop, final_fitness = self.nsga2.evolve(population, evaluate_fn)

        return final_pop, final_fitness


class PureNSGA2Adapter(NSGA2):
    """
    Adapted NSGA-II for weight vectors instead of neural networks.
    """

    def __init__(self, population_size, generations, mutation_rate, crossover_rate):
        super().__init__(population_size, generations, mutation_rate)
        self.crossover_rate = crossover_rate

    def generate_offspring(self, population):
        """Generate offspring using crossover and mutation on weight vectors."""
        offspring = []

        while len(offspring) < self.pop_size:
            # Select parents
            p1 = np.random.choice(population)
            p2 = np.random.choice(population)

            # Crossover with probability
            if np.random.rand() < self.crossover_rate:
                child = self.crossover(p1, p2)
            else:
                child = copy.deepcopy(p1)

            # Mutation
            self.mutate(child)

            offspring.append(child)

        return offspring

    def crossover(self, p1, p2):
        """
        Crossover for weight vectors.

        Uses arithmetic crossover: blend weights from two parents.
        """
        child = WeightIndividual(p1.n_assets, initialize_random=False)

        # Arithmetic crossover: weighted average
        alpha = np.random.rand()
        child.weights = alpha * p1.weights + (1 - alpha) * p2.weights

        # Renormalize to ensure sum = 1
        child.weights = child.weights / np.sum(child.weights)

        return child

    def mutate(self, individual):
        """
        Mutation for weight vectors.

        Adds Gaussian noise to weights and renormalizes.
        """
        if np.random.rand() < self.mutation_rate:
            # Add Gaussian noise
            noise = np.random.randn(individual.n_assets) * 0.1

            # Add to log-weights (to maintain positive weights)
            log_weights = np.log(individual.weights + 1e-8)
            log_weights += noise

            # Convert back and normalize
            new_weights = np.exp(log_weights)
            individual.weights = new_weights / np.sum(new_weights)


def run_pure_nsga_ii_baseline(
    env_train,
    env_val,
    env_test,
    population_size: int = 20,
    generations: int = 10,
    episodes_per_eval: int = 150
):
    """
    Run Pure NSGA-II baseline and return results.

    Args:
        env_train: Training environment
        env_val: Validation environment
        env_test: Test environment
        population_size: Population size
        generations: Number of generations
        episodes_per_eval: Episodes to average for evaluation

    Returns:
        Dictionary with Pareto front, best model, and test metrics
    """
    print("\n" + "="*80)
    print("Running Pure NSGA-II Baseline (No RL)")
    print("="*80)

    n_assets = env_train.n_assets + 1  # + 1 for cash

    # Initialize Pure NSGA-II
    optimizer = PureNSGAII(
        n_assets=n_assets,
        population_size=population_size,
        generations=generations
    )

    # Define evaluation function
    def evaluate(individual):
        """Evaluate individual on training environment."""
        total_rewards = np.zeros(3)
        num_episodes = min(episodes_per_eval, 10)  # Limit for speed

        for _ in range(num_episodes):
            obs, _ = env_train.reset()
            terminated = False
            episode_rewards = np.zeros(3)
            steps = 0

            while not terminated:
                features = obs['features'].flatten()
                action = individual.get_action(features)
                obs, rewards, terminated, truncated, _ = env_train.step(action)

                episode_rewards += rewards
                steps += 1

            if steps > 0:
                total_rewards += episode_rewards / steps

        avg_rewards = total_rewards / num_episodes
        return avg_rewards.tolist()

    # Run evolution
    print(f"\nEvolving population (Size: {population_size}, Gens: {generations})...")
    final_pop, final_fitness = optimizer.evolve(evaluate)

    print(f"\nEvolution complete. Final population size: {len(final_pop)}")

    # Select best individual (highest Sharpe on validation set)
    print("\nEvaluating on validation set...")
    val_scores = []
    for individual in final_pop:
        obs, _ = env_val.reset()
        terminated = False
        sharpe_sum = 0
        steps = 0

        while not terminated:
            features = obs['features'].flatten()
            action = individual.get_action(features)
            obs, rewards, terminated, truncated, _ = env_val.step(action)
            sharpe_sum += rewards[0]
            steps += 1

        val_sharpe = sharpe_sum / max(steps, 1)
        val_scores.append(val_sharpe)

    best_idx = np.argmax(val_scores)
    best_individual = final_pop[best_idx]

    print(f"Best individual: #{best_idx} (Val Sharpe: {val_scores[best_idx]:.4f})")
    print(f"Weights: {best_individual.weights}")

    # Test on test set
    print("\nEvaluating on test set...")
    obs, _ = env_test.reset()
    terminated = False

    while not terminated:
        features = obs['features'].flatten()
        action = best_individual.get_action(features, deterministic=True)
        obs, _, terminated, _, _ = env_test.step(action)

    from utils.metrics import calculate_metrics
    test_metrics = calculate_metrics(env_test.portfolio_history)

    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")

    return {
        'pareto_front': final_fitness,
        'best_model': best_individual,
        'test_metrics': test_metrics
    }


if __name__ == "__main__":
    print("Pure NSGA-II Baseline Module")
    print("Optimizes portfolio weights directly without neural networks.")
