import numpy as np

class DynamicAdaptationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.initial_scale_factor = 0.5
        self.initial_crossover_rate = 0.7

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        scale_factor = self.initial_scale_factor
        crossover_rate = self.initial_crossover_rate

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = population[a] + scale_factor * (population[b] - population[c])
                for j in range(self.dim):
                    if np.random.rand() > crossover_rate:
                        mutant[j] = population[i][j]
                mutant_fit = func(mutant)
                if mutant_fit < fitness[i]:
                    population[i] = mutant
                    fitness[i] = mutant_fit
            
            # Dynamic adaptation of scale factor and crossover rate
            sorted_indices = np.argsort(fitness)
            best_idx = sorted_indices[0]
            worst_idx = sorted_indices[-1]
            scale_factor = max(0.1, min(0.9, scale_factor + 0.1 * (fitness[best_idx] - fitness[worst_idx])))
            crossover_rate = max(0.1, min(0.9, crossover_rate + 0.1 * (fitness[best_idx] - fitness[worst_idx])))

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        return best_solution, best_fitness