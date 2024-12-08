import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness = np.array([func(ind) for ind in population])
            elites_idx = np.argsort(fitness)[:2]
            elites = population[elites_idx]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            population[elites_idx[0]] = offspring
        return population[np.argmin(fitness)]