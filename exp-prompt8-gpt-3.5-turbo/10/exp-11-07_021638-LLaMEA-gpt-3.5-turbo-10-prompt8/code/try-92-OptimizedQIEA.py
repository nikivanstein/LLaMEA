import numpy as np

class OptimizedQIEA:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_scores = [func(ind) for ind in population]
            elites = population[np.argsort(fitness_scores)[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            least_fit_idx = np.argmax(fitness_scores)
            population[least_fit_idx] = offspring
        return population[np.argmin(fitness_scores)]