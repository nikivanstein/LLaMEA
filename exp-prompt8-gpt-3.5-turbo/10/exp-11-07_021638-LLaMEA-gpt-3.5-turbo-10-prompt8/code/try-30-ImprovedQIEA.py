import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_scores = [func(ind) for ind in population]
            elites = population[np.argsort(fitness_scores)[:2]]
            best_idx = np.argmin(fitness_scores)
            population[best_idx] = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
        return population[best_idx]