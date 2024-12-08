import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        func_population = [func(ind) for ind in population]  # Pre-calculate function values
        for _ in range(self.budget):
            idx_sorted = np.argsort(func_population)
            elites = population[idx_sorted[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            idx_worst = idx_sorted[-1]
            population[idx_worst] = offspring
            func_population[idx_worst] = func(offspring)  # Update function value
        return population[np.argmin(func_population)]