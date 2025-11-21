"""
Enhanced NSGA-II with PPO Training for Multi-Objective Portfolio Optimization

This module implements E-NSGA-II (Enhanced NSGA-II) which combines:
  - Multi-objective evolutionary optimization (NSGA-II)
  - PPO policy gradient training within each generation
  - Explainability dominance operator (δ-dominance)
  - Parallel fitness evaluation for efficiency

Key Features:
  - Configurable δ parameter for explainability dominance
  - PPO updates between generations for policy improvement
  - Multiprocessing for parallel agent evaluation
  - Elitism to preserve best solutions

Author: E-NSGA-II + X-MARL Research Team
"""

import numpy as np
import copy
import torch
import torch.nn.functional as F
from typing import List, Callable, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for agent policy updates.

    PPO provides stable policy gradient updates by:
      - Clipping probability ratios to prevent large updates
      - Using GAE for advantage estimation
      - Entropy bonus for exploration
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Initialize PPO trainer.

        Args:
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs per batch
            batch_size: Mini-batch size for PPO updates
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
        """
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            next_value: Value estimate for next state

        Returns:
            Tuple of (advantages, returns)
        """
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        agent,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray
    ) -> dict:
        """
        Perform PPO update on agent.

        Args:
            agent: Agent to update
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Log probabilities from old policy
            returns: Computed returns
            advantages: Computed advantages

        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        old_log_probs_t = torch.FloatTensor(old_log_probs)
        returns_t = torch.FloatTensor(returns)
        advantages_t = torch.FloatTensor(advantages)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_samples = len(states)
        metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        for _ in range(self.ppo_epochs):
            # Generate random mini-batches
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]

                # Evaluate current policy
                log_probs, values, entropy = agent.model.evaluate_actions(
                    batch_states, batch_actions
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Update
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), self.max_grad_norm)
                agent.optimizer.step()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.mean().item()

        # Average metrics
        n_updates = self.ppo_epochs * (n_samples // self.batch_size + 1)
        for k in metrics:
            metrics[k] /= max(n_updates, 1)

        return metrics


class NSGA2:
    """
    Enhanced NSGA-II with PPO Training and Explainability Dominance.

    This implementation extends standard NSGA-II with:
      1. PPO policy gradient updates within each generation
      2. Configurable δ-dominance for explainability prioritization
      3. Parallel fitness evaluation using multiprocessing
      4. Elitism to preserve top solutions
    """

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        delta: float = 0.05,
        ppo_updates_per_gen: int = 5,
        n_workers: int = None,
        use_elitism: bool = True
    ):
        """
        Initialize E-NSGA-II optimizer.

        Args:
            population_size: Number of individuals in population
            generations: Number of evolutionary generations
            mutation_rate: Probability of mutation per parameter
            delta: Explainability dominance threshold (δ)
            ppo_updates_per_gen: Number of PPO update episodes per generation
            n_workers: Number of parallel workers (None = auto)
            use_elitism: Whether to preserve best solutions
        """
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.delta = delta
        self.ppo_updates_per_gen = ppo_updates_per_gen
        self.use_elitism = use_elitism

        # Set number of workers
        if n_workers is None:
            self.n_workers = min(mp.cpu_count() - 1, population_size)
        else:
            self.n_workers = n_workers

        # PPO trainer
        self.ppo_trainer = PPOTrainer()

        # Tracking
        self.generation_history = []

    def evolve(
        self,
        population: List,
        evaluate_fn: Callable,
        env_train=None,
        verbose: bool = True
    ) -> Tuple[List, List]:
        """
        Main evolution loop with PPO training.

        Algorithm:
          For each generation:
            1. Evaluate population on objectives
            2. (Optional) Run PPO updates on top individuals
            3. Generate offspring via crossover/mutation
            4. Evaluate offspring
            5. Combine and select via non-dominated sorting

        Args:
            population: List of agents
            evaluate_fn: Function that evaluates an agent, returns [obj1, obj2, obj3]
            env_train: Training environment (for PPO updates)
            verbose: Whether to print progress

        Returns:
            Tuple of (final_population, final_fitnesses)
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"E-NSGA-II Evolution")
            print(f"{'=' * 60}")
            print(f"Population: {self.pop_size}, Generations: {self.generations}")
            print(f"Delta (δ): {self.delta}, Mutation Rate: {self.mutation_rate}")
            print(f"Workers: {self.n_workers}")
            print(f"{'=' * 60}\n")

        # Initial evaluation
        if verbose:
            print("Initial population evaluation...")

        fitnesses = self._parallel_evaluate(population, evaluate_fn)

        for gen in range(self.generations):
            if verbose:
                print(f"\n--- Generation {gen + 1}/{self.generations} ---")

            # PPO training on top individuals (optional)
            if env_train is not None and self.ppo_updates_per_gen > 0:
                self._ppo_training_phase(population, fitnesses, env_train, verbose)

            # Generate offspring
            offspring = self._generate_offspring(population)

            # Evaluate offspring
            offspring_fitnesses = self._parallel_evaluate(offspring, evaluate_fn)

            # Combine populations
            combined_pop = population + offspring
            combined_fitness = fitnesses + offspring_fitnesses

            # Non-dominated sorting with δ-dominance
            fronts = self._fast_non_dominated_sort(combined_fitness)

            # Select next generation
            population, fitnesses = self._select_next_generation(
                combined_pop, combined_fitness, fronts
            )

            # Track progress
            self._track_generation(gen, fitnesses, verbose)

        if verbose:
            print(f"\n{'=' * 60}")
            print("Evolution complete!")
            print(f"{'=' * 60}")

        return population, fitnesses

    def _parallel_evaluate(
        self,
        population: List,
        evaluate_fn: Callable
    ) -> List:
        """
        Evaluate population in parallel using ThreadPoolExecutor.

        Note: Using threads instead of processes because PyTorch models
        don't pickle well across processes.

        Args:
            population: List of agents to evaluate
            evaluate_fn: Evaluation function

        Returns:
            List of fitness values
        """
        if self.n_workers <= 1:
            # Sequential evaluation
            return [evaluate_fn(ind) for ind in population]

        # Parallel evaluation with threads
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            fitnesses = list(executor.map(evaluate_fn, population))

        return fitnesses

    def _ppo_training_phase(
        self,
        population: List,
        fitnesses: List,
        env,
        verbose: bool
    ):
        """
        Run PPO training on top individuals in the population.

        Args:
            population: Current population
            fitnesses: Current fitness values
            env: Training environment
            verbose: Print progress
        """
        # Select top individuals for PPO training
        n_train = min(5, len(population))

        # Rank by Pareto dominance (front 0 first)
        fronts = self._fast_non_dominated_sort(fitnesses)
        top_indices = []
        for front in fronts:
            top_indices.extend(front)
            if len(top_indices) >= n_train:
                break
        top_indices = top_indices[:n_train]

        if verbose:
            print(f"  PPO training on {n_train} top individuals...")

        for idx in top_indices:
            agent = population[idx]

            # Collect rollout data
            states, actions, rewards, values, dones, log_probs = self._collect_rollout(
                agent, env, n_steps=256
            )

            if len(states) < 64:
                continue

            # Compute advantages
            next_value = agent.get_value(states[-1])
            advantages, returns = self.ppo_trainer.compute_gae(
                rewards, values, dones, next_value
            )

            # PPO update
            self.ppo_trainer.update(
                agent, states, actions, log_probs, returns, advantages
            )

    def _collect_rollout(
        self,
        agent,
        env,
        n_steps: int = 256
    ) -> Tuple:
        """
        Collect rollout data for PPO training.

        Args:
            agent: Agent to collect data from
            env: Environment to interact with
            n_steps: Number of steps to collect

        Returns:
            Tuple of (states, actions, rewards, values, dones, log_probs)
        """
        states, actions, rewards, values, dones, log_probs = [], [], [], [], [], []

        obs, _ = env.reset()
        features = obs['features'].flatten()

        for _ in range(n_steps):
            # Get action and value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(features).unsqueeze(0)
                dist = agent.model.get_action_distribution(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                _, value = agent.model(state_tensor)

            action_np = action.cpu().numpy()[0]

            # Step environment
            next_obs, reward_vec, terminated, truncated, _ = env.step(action_np)

            # Store data (use first objective as reward)
            states.append(features)
            actions.append(action_np)
            rewards.append(reward_vec[0] if isinstance(reward_vec, np.ndarray) else reward_vec)
            values.append(value.cpu().numpy()[0, 0])
            dones.append(float(terminated or truncated))
            log_probs.append(log_prob.cpu().numpy())

            if terminated or truncated:
                obs, _ = env.reset()
                features = obs['features'].flatten()
            else:
                features = next_obs['features'].flatten()

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(values),
            np.array(dones),
            np.array(log_probs).flatten()
        )

    def _generate_offspring(self, population: List) -> List:
        """
        Generate offspring population via crossover and mutation.

        Args:
            population: Parent population

        Returns:
            Offspring population
        """
        offspring = []

        while len(offspring) < self.pop_size:
            # Tournament selection
            p1 = population[np.random.randint(len(population))]
            p2 = population[np.random.randint(len(population))]

            # Crossover
            child = self._crossover(p1, p2)

            # Mutation
            self._mutate(child)

            offspring.append(child)

        return offspring

    def _crossover(self, p1, p2):
        """
        Uniform crossover of neural network weights.

        Args:
            p1: Parent 1
            p2: Parent 2

        Returns:
            Child agent
        """
        child = copy.deepcopy(p1)

        if hasattr(p1, 'model') and hasattr(p1.model, 'parameters'):
            for p1_param, p2_param, child_param in zip(
                p1.model.parameters(),
                p2.model.parameters(),
                child.model.parameters()
            ):
                # Uniform crossover mask
                mask = torch.rand_like(p1_param) < 0.5
                child_param.data = torch.where(mask, p1_param.data, p2_param.data)

        return child

    def _mutate(self, agent):
        """
        Gaussian mutation of neural network weights.

        Args:
            agent: Agent to mutate
        """
        if hasattr(agent, 'model') and hasattr(agent.model, 'parameters'):
            for param in agent.model.parameters():
                if np.random.rand() < self.mutation_rate:
                    noise = torch.randn_like(param) * 0.1
                    param.data += noise

    def _fast_non_dominated_sort(self, fitnesses: List) -> List[List[int]]:
        """
        Fast non-dominated sorting with δ-dominance for explainability.

        δ-dominance: A dominates B if:
          1. A dominates B in standard Pareto sense, AND
          2. Explainability_A > Explainability_B - δ

        This prevents solutions with poor explainability from dominating
        solutions with good explainability, even if they have better
        return/risk trade-offs.

        Args:
            fitnesses: List of [return, -risk, explainability] values

        Returns:
            List of fronts, each front is a list of indices
        """
        N = len(fitnesses)
        S = [[] for _ in range(N)]  # Solutions dominated by i
        n = [0] * N  # Domination count
        rank = [0] * N
        fronts = [[]]

        # Negate for minimization (NSGA-II standard)
        fits = -np.array(fitnesses)

        for p in range(N):
            for q in range(N):
                if self._dominates(fits[p], fits[q]):
                    S[p].append(q)
                elif self._dominates(fits[q], fits[p]):
                    n[p] += 1

            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
            else:
                break

        return fronts

    def _dominates(self, p: np.ndarray, q: np.ndarray) -> bool:
        """
        Check if p dominates q with δ-dominance.

        Standard dominance (minimization):
          p dominates q if p <= q in all objectives and p < q in at least one

        δ-dominance addition:
          Additionally requires: Explainability_p > Explainability_q - δ
          (In negated space: p[2] < q[2] + δ)

        Args:
            p: Fitness values for solution p (negated)
            q: Fitness values for solution q (negated)

        Returns:
            True if p dominates q
        """
        # Standard Pareto dominance
        standard_dom = np.all(p <= q) and np.any(p < q)

        if not standard_dom:
            return False

        # δ-dominance for explainability
        # p[2] is -Explainability_p, q[2] is -Explainability_q
        # Condition: -p[2] > -q[2] - δ  =>  p[2] < q[2] + δ
        return p[2] < q[2] + self.delta

    def _select_next_generation(
        self,
        combined_pop: List,
        combined_fitness: List,
        fronts: List[List[int]]
    ) -> Tuple[List, List]:
        """
        Select next generation using crowding distance.

        Args:
            combined_pop: Combined parent + offspring population
            combined_fitness: Combined fitness values
            fronts: Non-dominated fronts

        Returns:
            Tuple of (selected_population, selected_fitnesses)
        """
        new_pop = []
        new_fitness = []

        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                # Add entire front
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_fitness.append(combined_fitness[idx])
            else:
                # Need to select subset using crowding distance
                num_needed = self.pop_size - len(new_pop)
                sorted_front = self._crowding_distance_sort(front, combined_fitness)

                for i in range(num_needed):
                    idx = sorted_front[i]
                    new_pop.append(combined_pop[idx])
                    new_fitness.append(combined_fitness[idx])
                break

        return new_pop, new_fitness

    def _crowding_distance_sort(
        self,
        front: List[int],
        fitnesses: List
    ) -> List[int]:
        """
        Sort front by crowding distance (descending).

        Crowding distance measures how isolated a solution is in objective space.
        Higher crowding distance = more diverse = preferred.

        Args:
            front: List of indices in this front
            fitnesses: All fitness values

        Returns:
            Sorted list of indices
        """
        if len(front) == 0:
            return []

        distances = {idx: 0.0 for idx in front}
        fits = -np.array(fitnesses)  # Back to minimization
        num_objs = fits.shape[1]

        for m in range(num_objs):
            # Sort by objective m
            sorted_front = sorted(front, key=lambda x: fits[x][m])

            # Boundary points get infinite distance
            distances[sorted_front[0]] = np.inf
            distances[sorted_front[-1]] = np.inf

            f_min = fits[sorted_front[0]][m]
            f_max = fits[sorted_front[-1]][m]

            if f_max == f_min:
                continue

            for i in range(1, len(sorted_front) - 1):
                distances[sorted_front[i]] += (
                    fits[sorted_front[i + 1]][m] - fits[sorted_front[i - 1]][m]
                ) / (f_max - f_min)

        # Sort by distance descending
        return sorted(front, key=lambda x: distances[x], reverse=True)

    def _track_generation(self, gen: int, fitnesses: List, verbose: bool):
        """
        Track generation statistics.

        Args:
            gen: Generation number
            fitnesses: Current fitness values
            verbose: Whether to print
        """
        fits = np.array(fitnesses)

        stats = {
            'generation': gen + 1,
            'mean_return': np.mean(fits[:, 0]),
            'mean_risk': np.mean(-fits[:, 1]),  # Negate back
            'mean_explain': np.mean(fits[:, 2]),
            'best_return': np.max(fits[:, 0]),
            'best_risk': np.min(-fits[:, 1]),
            'best_explain': np.max(fits[:, 2])
        }

        self.generation_history.append(stats)

        if verbose:
            print(f"  Return: {stats['mean_return']:.4f} (best: {stats['best_return']:.4f})")
            print(f"  Risk:   {stats['mean_risk']:.4f} (best: {stats['best_risk']:.4f})")
            print(f"  Explain: {stats['mean_explain']:.4f} (best: {stats['best_explain']:.4f})")


if __name__ == "__main__":
    # Test NSGA-II
    print("Testing E-NSGA-II implementation...")

    nsga = NSGA2(population_size=10, generations=3, delta=0.05)

    # Create dummy population
    class DummyAgent:
        def __init__(self):
            self.model = None

    population = [DummyAgent() for _ in range(10)]

    # Dummy evaluation function
    def evaluate(agent):
        return [np.random.randn(), np.random.randn(), np.random.rand()]

    # Test evolution (without PPO)
    final_pop, final_fit = nsga.evolve(population, evaluate, verbose=True)

    print(f"\nFinal population size: {len(final_pop)}")
    print(f"Final fitnesses shape: {len(final_fit)} x {len(final_fit[0])}")
    print("\nTest passed!")
