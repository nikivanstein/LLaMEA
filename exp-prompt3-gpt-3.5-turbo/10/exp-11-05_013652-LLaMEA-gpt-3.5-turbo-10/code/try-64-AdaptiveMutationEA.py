import numpy as np

class AdaptiveMutationEA:
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
            mutation_rate = self.mutation_rates[best_idx]
            mutation = np.random.randn(self.dim) * mutation_rate
            new_solution = best_solution + mutation
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
                self.mutation_rates[best_idx] *= 1.1  # Increase mutation rate if successful
            else:
                self.mutation_rates[best_idx] *= 0.9  # Decrease mutation rate if unsuccessful
        return self.population[np.argmin([func(ind) for ind in self.population])]