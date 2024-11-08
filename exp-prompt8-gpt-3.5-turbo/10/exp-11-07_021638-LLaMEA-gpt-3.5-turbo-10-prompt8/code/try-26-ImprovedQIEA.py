import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        func_values = [func(ind) for ind in population]  # Cache function evaluations
        for _ in range(self.budget):
            best_indices = np.argsort(func_values)[:2]
            best_elites = population[best_indices]
            offspring = np.mean(best_elites, axis=0) + np.random.normal(0, 1, self.dim)
            replace_idx = np.argmax(func_values)
            population[replace_idx] = offspring
            func_values[replace_idx] = func(offspring)  # Update function value
        return population[np.argmin(func_values)]