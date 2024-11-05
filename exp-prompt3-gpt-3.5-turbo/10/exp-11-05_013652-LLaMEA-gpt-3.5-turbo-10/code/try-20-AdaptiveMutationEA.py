import numpy as np

class AdaptiveMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rates = np.full(budget, 0.1)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            for i in range(self.budget):
                if i != best_idx:
                    self.mutation_rates[i] *= 1.01 if fitness[i] >= fitness[best_idx] else 0.99
                    mutation = np.random.randn(self.dim) * self.mutation_rates[i]
                    new_solution = self.population[i] + mutation
                    if func(new_solution) < fitness[i]:
                        self.population[i] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]