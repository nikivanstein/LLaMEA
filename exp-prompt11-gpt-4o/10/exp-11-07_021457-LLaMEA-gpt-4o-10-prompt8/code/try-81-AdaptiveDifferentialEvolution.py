import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        rng = np.random.default_rng()
        population = rng.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = rng.choice(self.population_size - 1, 3, replace=False)
                indices[indices >= i] += 1  # Adjust indices to exclude 'i'
                a, b, c = population[indices]
                
                mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = rng.random(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial_vector, trial_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]