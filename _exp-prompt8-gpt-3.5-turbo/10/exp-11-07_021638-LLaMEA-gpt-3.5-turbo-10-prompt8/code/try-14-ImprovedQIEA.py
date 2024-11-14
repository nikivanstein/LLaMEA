import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_scores = [func(ind) for ind in population]
            idx = np.argsort(fitness_scores)
            elites = population[idx[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            idx_worst = np.argmax(fitness_scores)
            population[idx_worst] = offspring
        return population[np.argmin(fitness_scores)]