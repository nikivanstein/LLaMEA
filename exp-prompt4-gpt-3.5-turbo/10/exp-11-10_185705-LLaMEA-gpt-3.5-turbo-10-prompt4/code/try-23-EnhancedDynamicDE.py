import numpy as np

class EnhancedDynamicDE(DynamicDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.crossover_rate = 0.7

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget - 1):
            mutant = population[np.random.choice(len(population), 3, replace=False)]
            crossover_mask = np.random.rand(self.dim) < self.crossover_rate
            crossover_vector = population[best_idx].copy()
            crossover_vector[crossover_mask] = mutant[0][crossover_mask]
            mutated_vector = crossover_vector + self.mutation_factor * (mutant[1] - mutant[2])
            mutated_vector = np.clip(mutated_vector, -5.0, 5.0)
            trial_vector = np.where(np.random.rand(self.dim) < self.mutation_factor, mutated_vector, population[best_idx])
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[best_idx]:
                population[best_idx] = trial_vector
                fitness[best_idx] = trial_fitness
                best_solution = trial_vector if trial_fitness < fitness[best_idx] else best_solution

            if np.random.rand() < 0.1:  # Update mutation factor with small probability
                self.mutation_factor = np.clip(self.mutation_factor * np.random.uniform(0.8, 1.2), self.mutation_factor_range[0], self.mutation_factor_range[1])

        return best_solution