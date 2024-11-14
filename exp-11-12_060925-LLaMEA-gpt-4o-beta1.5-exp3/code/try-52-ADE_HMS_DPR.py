import numpy as np

class ADE_HMS_DPR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.base_pop_size = 10 * dim
        self.scale_factor = 0.8
        self.crossover_prob = 0.9

    def __call__(self, func):
        population_size = self.base_pop_size
        population = self.lower_bound + np.random.rand(population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break

                indices = np.random.permutation(population_size)
                x1, x2, x3 = population[indices[:3]]

                mutant_vector = x1 + self.scale_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial_vector = np.where(crossover, mutant_vector, population[i])

                trial_fitness = func(trial_vector)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_individual = trial_vector
                        best_fitness = trial_fitness

            if self.evaluations % (self.base_pop_size // 2) == 0:
                population_size = max(4, int(population_size * 0.9))
                population = population[np.argsort(fitness)[:population_size]]
                fitness = fitness[np.argsort(fitness)[:population_size]]

        return best_individual, best_fitness