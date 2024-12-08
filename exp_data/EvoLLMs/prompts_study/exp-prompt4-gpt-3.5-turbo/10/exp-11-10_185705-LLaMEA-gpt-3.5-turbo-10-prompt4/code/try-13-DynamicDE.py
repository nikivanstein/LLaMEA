import numpy as np

class DynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_factor = 0.5
        self.mutation_factor_range = [0.1, 0.9]
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        for _ in range(self.budget - 1):
            mutant = population[np.random.choice(len(population), 3, replace=False)]
            mutated_vector = population[best_idx] + self.mutation_factor * (mutant[0] - mutant[1])
            mutated_vector = np.clip(mutated_vector, -5.0, 5.0)
            trial_vector = np.where(np.random.rand(self.dim) < self.mutation_factor, mutated_vector, population[best_idx])
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[best_idx]:
                population[best_idx] = trial_vector
                fitness[best_idx] = trial_fitness
                if trial_fitness < fitness[best_idx]:
                    best_solution = trial_vector

            if np.random.rand() < 0.1:  # Update mutation factor with small probability
                self.mutation_factor = np.clip(self.mutation_factor * np.random.uniform(0.8, 1.2), self.mutation_factor_range[0], self.mutation_factor_range[1])

        return best_solution