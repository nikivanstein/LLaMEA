import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            func_vals = np.array([func(ind) for ind in population])  # Reduced redundant function evaluations
            elites = population[np.argsort(func_vals)[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            idx = np.argmax(func_vals)
            population[idx] = offspring
        return population[np.argmin(func_vals)]