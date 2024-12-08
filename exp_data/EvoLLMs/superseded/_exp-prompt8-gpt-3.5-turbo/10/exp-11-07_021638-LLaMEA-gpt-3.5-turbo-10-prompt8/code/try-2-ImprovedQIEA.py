import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            elite_indices = np.argsort([func(ind) for ind in population])[:2]
            elites = population[elite_indices]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            max_index = elite_indices[np.argmax([func(ind) for ind in elites])]
            population[max_index] = offspring
        return population[np.argmin([func(ind) for ind in population])]