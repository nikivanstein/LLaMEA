import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            idx = np.argsort([func(ind) for ind in population])
            elites = population[idx[:2]]
            current_best = func(population[idx[0]])
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            if func(offspring) < current_best:
                population[idx[-1]] = offspring
        return population[np.argmin([func(ind) for ind in population])]