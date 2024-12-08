import numpy as np

class ImprovedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rate = np.full(budget, 0.05)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            for i in range(self.budget):
                if i != best_idx:
                    self.population[i] += np.random.randn(self.dim) * self.mutation_rate[i]
            if func(best_solution) == 0:
                self.mutation_rate[best_idx] *= 0.95
            else:
                self.mutation_rate[best_idx] *= 1.05
        return self.population[best_idx]