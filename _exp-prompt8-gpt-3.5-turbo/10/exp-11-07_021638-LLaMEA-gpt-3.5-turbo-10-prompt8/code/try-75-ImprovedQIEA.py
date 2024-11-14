import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness_values = [func(ind) for ind in population]
        for _ in range(self.budget):
            elite_indices = np.argsort(fitness_values)[:2]
            elites = population[elite_indices]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            max_index = np.argmax(fitness_values)
            population[max_index] = offspring
            fitness_values[max_index] = func(offspring)
        return population[np.argmin(fitness_values)]