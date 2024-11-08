import numpy as np

class OptimizedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        rand_indices = np.arange(self.population_size)

        while evaluations < self.budget:
            np.random.shuffle(rand_indices)
            for i in range(0, self.population_size, 3):
                if evaluations >= self.budget:
                    break

                indices = rand_indices[i:i+3]
                if len(indices) < 3:  # Ensure we have at least 3 indices
                    continue

                a, b, c = population[indices]
                
                mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, population[indices[0]])
                
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[indices[0]]:
                    population[indices[0]] = trial_vector
                    fitness[indices[0]] = trial_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]