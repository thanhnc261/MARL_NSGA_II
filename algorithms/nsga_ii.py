import numpy as np
import copy
import torch

class NSGA2:
    def __init__(self, population_size, generations, mutation_rate=0.1):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
    def evolve(self, population, evaluate_fn):
        """
        Main evolution loop.
        population: List of Agents
        evaluate_fn: Function that takes an Agent and returns [obj1, obj2, obj3]
        """
        # Initial evaluation
        print("Initial evaluation...")
        fitnesses = [evaluate_fn(ind) for ind in population]
        
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")
            
            # Create offspring
            offspring = self.generate_offspring(population)
            
            # Evaluate offspring
            offspring_fitnesses = [evaluate_fn(ind) for ind in offspring]
            
            # Combine population
            combined_pop = population + offspring
            combined_fitness = fitnesses + offspring_fitnesses
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined_fitness)
            
            # Select next generation
            new_pop = []
            new_fitness = []
            
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    # Add entire front
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_fitness.append(combined_fitness[idx])
                else:
                    # Crowding distance sort
                    sorted_front = self.crowding_distance_sort(front, combined_fitness)
                    num_needed = self.pop_size - len(new_pop)
                    for i in range(num_needed):
                        idx = sorted_front[i]
                        new_pop.append(combined_pop[idx])
                        new_fitness.append(combined_fitness[idx])
                    break
            
            population = new_pop
            fitnesses = new_fitness
            
        return population, fitnesses

    def generate_offspring(self, population):
        offspring = []
        # Tournament selection and crossover/mutation
        while len(offspring) < self.pop_size:
            p1 = np.random.choice(population)
            p2 = np.random.choice(population)
            
            child = self.crossover(p1, p2)
            self.mutate(child)
            offspring.append(child)
        return offspring

    def crossover(self, p1, p2):
        # Deep copy parent 1
        child = copy.deepcopy(p1)

        # Check if individual has a model attribute (neural network agents)
        if hasattr(p1, 'model') and hasattr(p1.model, 'parameters'):
            # Simple weight averaging or uniform crossover
            # Iterate over parameters
            for p1_param, p2_param, child_param in zip(p1.model.parameters(), p2.model.parameters(), child.model.parameters()):
                # Uniform crossover mask
                mask = torch.rand_like(p1_param) < 0.5
                child_param.data = torch.where(mask, p1_param.data, p2_param.data)
        # Otherwise, let subclasses handle crossover (e.g., Pure NSGA-II with weight vectors)

        return child

    def mutate(self, agent):
        # Check if individual has a model attribute (neural network agents)
        if hasattr(agent, 'model') and hasattr(agent.model, 'parameters'):
            # Gaussian mutation
            for param in agent.model.parameters():
                if np.random.rand() < self.mutation_rate:
                    noise = torch.randn_like(param) * 0.1
                    param.data += noise
        # Otherwise, let subclasses handle mutation (e.g., Pure NSGA-II with weight vectors)

    def fast_non_dominated_sort(self, fitnesses):
        # fitnesses: List of [obj1, obj2, ...]
        # We assume minimization? Or maximization?
        # Guide: Sharpe (Max), -CVaR (Max), Explainability (Max)
        # So all are Maximization.
        # Standard NSGA-II is Minimization.
        # We convert to minimization by negating.
        
        N = len(fitnesses)
        S = [[] for _ in range(N)]
        n = [0] * N
        rank = [0] * N
        fronts = [[]]
        
        # Convert to numpy for easier comparison
        fits = -np.array(fitnesses) # Negate for minimization
        
        for p in range(N):
            S[p] = []
            n[p] = 0
            for q in range(N):
                if self.dominates(fits[p], fits[q], delta=0.05): # Delta from paper? "delta is a tolerance margin"
                    S[p].append(q)
                elif self.dominates(fits[q], fits[p], delta=0.05):
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
            if len(Q) > 0:
                fronts.append(Q)
            else:
                break # Avoid empty last front
                
        return fronts

    def dominates(self, p, q, delta=0.0):
        # p dominates q if p <= q for all and p < q for at least one (Minimization)
        # AND Explainability condition: E_p > E_q - delta
        # We assume fitnesses are [Return, Risk, Explainability]
        # But we negated them for minimization.
        # So p = [-Return, -Risk, -Explainability]
        # Original: Max Return, Min Volatility (or Max -Volatility), Max Explainability
        # Draft v2: Max Return (f1), Min Volatility (f2), Max Explainability (f3)
        # My code assumes all Maximization and negates them.
        # So p[0] = -Return, p[1] = -(-Volatility) = Volatility? 
        # Wait, let's check env.
        # Env returns [Sharpe, RiskReward, Explain].
        # RiskReward was -CVaR (Maximized).
        # If we switch to Volatility (Minimization), we should return -Volatility (Maximization) OR handle minimization here.
        # Let's stick to Maximization convention in Env: Return -Volatility.
        # Then NSGA-II negates everything -> Minimization.
        # So p[2] is -Explainability.
        # Condition: E_p > E_q - delta
        # => -p[2] > -q[2] - delta
        # => q[2] - p[2] < delta (if delta is positive)
        # => p[2] < q[2] + delta
        
        # Standard dominance (Minimization):
        standard_dom = np.all(p <= q) and np.any(p < q)
        
        if not standard_dom:
            return False
            
        # Explainability Dominance
        # p dominates q ONLY IF standard_dom AND Explainability condition
        # E_p > E_q - delta
        # p[2] is -E_p
        # -p[2] > -q[2] - delta
        # p[2] < q[2] + delta
        
        # If delta > 0, we allow p to dominate q even if E_p is slightly worse?
        # No, "prioritizing models with stable SHAP-based interpretability"
        # "A < B if (fA dominates fB) and (EA > EB - delta)"
        # This means A dominates B if A is better in objectives AND A's explainability is not too much worse than B's?
        # Or "EA > EB - delta" means A must be at least (EB - delta).
        # If delta is small positive, it relaxes the requirement?
        # Actually, usually dominance is strict.
        # If we want to FAVOR explainability, maybe we say A dominates B if A is better OR A has much better explainability?
        # The formula: A < B if (fA dominates fB) AND (EA > EB - delta).
        # This restricts dominance. A only dominates B if it's better AND has decent explainability.
        # This prevents a solution with great Return/Risk but terrible Explainability from dominating a balanced one?
        # No, if A dominates B in objectives, it usually eliminates B.
        # If we add "AND EA > EB - delta", we make it HARDER for A to dominate B.
        # If A has bad explainability (EA < EB - delta), A does NOT dominate B, even if A has better Return/Risk.
        # So B survives. This preserves solutions with high Explainability.
        
        # Implementation:
        # p[2] is -E_p. q[2] is -E_q.
        # Condition: -p[2] > -q[2] - delta  => p[2] < q[2] + delta
        
        return standard_dom and (p[2] < q[2] + delta)

    def crowding_distance_sort(self, front, fitnesses):
        l = len(front)
        if l == 0: return []
        
        distances = {idx: 0 for idx in front}
        fits = -np.array(fitnesses) # Negate back to minimization frame
        
        num_objs = fits.shape[1]
        
        for m in range(num_objs):
            # Sort by objective m
            sorted_front = sorted(front, key=lambda x: fits[x][m])
            
            distances[sorted_front[0]] = np.inf
            distances[sorted_front[-1]] = np.inf
            
            f_min = fits[sorted_front[0]][m]
            f_max = fits[sorted_front[-1]][m]
            
            if f_max == f_min: continue
            
            for i in range(1, l-1):
                distances[sorted_front[i]] += (fits[sorted_front[i+1]][m] - fits[sorted_front[i-1]][m]) / (f_max - f_min)
                
        # Sort by distance descending
        return sorted(front, key=lambda x: distances[x], reverse=True)
