import numpy as np

class EnhancedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rates = np.full(budget, 0.05)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutations = np.random.randn(self.budget, self.dim) * self.mutation_rates[:, np.newaxis]
            mutated_population = self.population + mutations
            mutated_fitness = [func(ind) for ind in mutated_population]
            for i in range(self.budget):
                if mutated_fitness[i] < fitness[i]:
                    self.population[i] = mutated_population[i]
                    self.mutation_rates[i] *= 1.1
                else:
                    self.mutation_rates[i] *= 0.9
        return self.population[np.argmin([func(ind) for ind in self.population])]