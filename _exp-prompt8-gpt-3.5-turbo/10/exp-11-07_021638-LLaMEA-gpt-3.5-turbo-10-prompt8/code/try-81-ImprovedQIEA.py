import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        calculate_fitness = lambda p: func(p)
        for _ in range(self.budget):
            fitness_values = np.apply_along_axis(calculate_fitness, 1, population)
            elites_indices = np.argpartition(fitness_values, 2)[:2]
            elites = population[elites_indices]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            population[np.argmax(fitness_values)] = offspring
        return population[np.argmin(np.apply_along_axis(calculate_fitness, 1, population))]