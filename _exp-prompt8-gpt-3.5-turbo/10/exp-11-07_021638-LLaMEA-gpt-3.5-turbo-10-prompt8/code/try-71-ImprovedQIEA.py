import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness_values = [func(ind) for ind in population]
        for _ in range(self.budget):
            elites_idx = np.argsort(fitness_values)[:2]
            elites = population[elites_idx]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            max_idx = np.argmax(fitness_values)
            population[max_idx] = offspring
            fitness_values[max_idx] = func(offspring)
        return population[np.argmin(fitness_values)]