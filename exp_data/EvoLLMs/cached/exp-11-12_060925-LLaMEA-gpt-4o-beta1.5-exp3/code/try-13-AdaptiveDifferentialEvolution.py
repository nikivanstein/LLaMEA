import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.f = 0.5  # mutation factor
        self.cr = 0.9  # crossover probability

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation: select three random indices that are not i
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Perform mutation
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                self.evaluations += 1
                
                # Selection: replace if trial is better
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                
                # Adaptive strategy adjustment
                if trial_fitness < np.mean(fitness):
                    self.f = np.clip(self.f + 0.1 * (trial_fitness - np.mean(fitness)), 0.1, 1.0)
                    self.cr = np.clip(self.cr + 0.1 * (trial_fitness - np.mean(fitness)), 0.1, 1.0)
                
                # Early stopping if budget is exhausted
                if self.evaluations >= self.budget:
                    break

        # Return best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]