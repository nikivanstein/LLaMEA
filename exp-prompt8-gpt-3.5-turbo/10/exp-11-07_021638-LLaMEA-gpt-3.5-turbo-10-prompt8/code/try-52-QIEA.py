import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            elites = population[np.argsort([func(ind) for ind in population])[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            population[np.argmax([func(ind) for ind in population])] = offspring
        return population[np.argmin([func(ind) for ind in population])]