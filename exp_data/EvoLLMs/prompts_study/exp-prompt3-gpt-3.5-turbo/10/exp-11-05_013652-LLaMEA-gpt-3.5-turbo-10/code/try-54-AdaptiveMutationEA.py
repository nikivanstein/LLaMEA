import numpy as np

class AdaptiveMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutation_rates = np.clip(np.exp(np.random.randn(self.budget)), 0.01, 0.1)
            mutations = np.random.randn(self.budget, self.dim) * mutation_rates[:, np.newaxis]
            new_solutions = self.population + mutations
            new_fitness = [func(ind) for ind in new_solutions]
            improved_idxs = np.where(new_fitness < fitness)
            self.population[improved_idxs] = new_solutions[improved_idxs]
        return self.population[np.argmin([func(ind) for ind in self.population])]