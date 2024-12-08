import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.7  # Adaptive differential weight
        self.CR = 0.85  # Crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Adaptive mutation and crossover
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d, e = population[indices]
                mutant = np.clip(a + self.F * (b - c + d - e), self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if eval_count >= self.budget:
                    break

            # Efficient local search on top individuals
            top_indices = np.argsort(fitness)[:3]
            for idx in top_indices:
                individual = population[idx]
                individual_fitness = fitness[idx]
                
                # Perform a refined local search with Gaussian perturbation
                local_neighbors = individual + np.random.normal(0, 0.05, (5, self.dim))
                local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
                local_fitness = np.array([func(ind) for ind in local_neighbors])
                eval_count += len(local_neighbors)
                
                # Update if any local neighbor is better
                if np.min(local_fitness) < individual_fitness:
                    best_index = np.argmin(local_fitness)
                    population[idx] = local_neighbors[best_index]
                    fitness[idx] = local_fitness[best_index]

        # Return best found solution
        best_index = np.argmin(fitness)
        return population[best_index]