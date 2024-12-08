import numpy as np

class EnhancedDynamicDE(DynamicDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.crossover_rate = 0.9
        self.crossover_rate_range = [0.1, 0.9]
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        for _ in range(self.budget - 1):
            mutant = population[np.random.choice(len(population), 3, replace=False)]
            crossover_mask = np.random.rand(self.dim) < self.crossover_rate
            mutated_vector = population[best_idx].copy()
            mutated_vector[crossover_mask] = population[best_idx][crossover_mask] + self.mutation_factor * (mutant[0][crossover_mask] - mutant[1][crossover_mask])
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
            if np.random.rand() < 0.1:  # Update crossover rate with small probability
                self.crossover_rate = np.clip(self.crossover_rate * np.random.uniform(0.8, 1.2), self.crossover_rate_range[0], self.crossover_rate_range[1])

        return best_solution