import numpy as np

class SelfAdaptiveMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rates = np.full(dim, 0.05)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutation = np.random.randn(self.dim) * self.mutation_rates
            new_solution = best_solution + mutation
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
                self.mutation_rates = self.mutation_rates * 0.9  # Decrease mutation rates if improvement
            else:
                self.mutation_rates = self.mutation_rates * 1.1  # Increase mutation rates if no improvement
        return self.population[np.argmin([func(ind) for ind in self.population])]