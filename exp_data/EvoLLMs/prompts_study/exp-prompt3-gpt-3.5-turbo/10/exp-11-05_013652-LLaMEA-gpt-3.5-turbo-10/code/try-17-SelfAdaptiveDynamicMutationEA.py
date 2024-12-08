import numpy as np

class SelfAdaptiveDynamicMutationEA:
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
            new_solutions = self.population + mutations
            new_fitness = [func(new_sol) for new_sol in new_solutions]
            improving_mask = np.less(new_fitness, fitness)
            self.population[improving_mask] = new_solutions[improving_mask]
            self.mutation_rates[improving_mask] *= 1.2
            self.mutation_rates[~improving_mask] *= 0.8
        return self.population[np.argmin([func(ind) for ind in self.population])]