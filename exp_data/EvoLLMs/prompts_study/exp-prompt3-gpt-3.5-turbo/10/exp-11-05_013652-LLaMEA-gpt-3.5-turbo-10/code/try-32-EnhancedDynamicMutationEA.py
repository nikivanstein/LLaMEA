import numpy as np

class EnhancedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rates = np.full(budget, 0.01)
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutations = np.random.randn(self.budget, self.dim) * self.mutation_rates[:, np.newaxis]
            new_solutions = self.population + mutations
            new_fitness = [func(ind) for ind in new_solutions]
            improvements = np.less(new_fitness, fitness)
            self.population[improvements] = new_solutions[improvements]
            self.mutation_rates[improvements] *= 1.2  # Increase mutation rate for successful individuals
            self.mutation_rates[~improvements] *= 0.8  # Decrease mutation rate for unsuccessful individuals
        return self.population[np.argmin([func(ind) for ind in self.population])]