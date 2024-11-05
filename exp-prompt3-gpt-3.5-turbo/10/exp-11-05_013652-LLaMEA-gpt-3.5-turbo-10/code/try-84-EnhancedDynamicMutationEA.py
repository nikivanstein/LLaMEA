import numpy as np

class EnhancedDynamicMutationEA:
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
            mutations = np.random.randn(self.budget, self.dim) * self.mutation_rates
            new_solutions = self.population + mutations
            new_fitness = [func(ind) for ind in new_solutions]
            improvements = np.less(new_fitness, fitness)
            self.population[improvements] = new_solutions[improvements]
            self.mutation_rates *= 1.1  # Increase mutation rates for successful mutations
            self.mutation_rates *= 0.9  # Decrease mutation rates for unsuccessful mutations
        return self.population[np.argmin([func(ind) for ind in self.population])]