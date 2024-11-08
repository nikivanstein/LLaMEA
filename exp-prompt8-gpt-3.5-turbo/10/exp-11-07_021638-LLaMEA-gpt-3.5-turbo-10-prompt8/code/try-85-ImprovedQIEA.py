import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        func_vals = [func(ind) for ind in population]
        for _ in range(self.budget):
            sorted_indices = np.argsort(func_vals)
            elites = population[sorted_indices[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            max_idx = np.argmax(func_vals)
            func_vals[max_idx] = func(offspring)
            population[max_idx] = offspring
        return population[np.argmin(func_vals)]