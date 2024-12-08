import numpy as np

class EQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_values = [func(ind) for ind in population]  # Parallelize fitness evaluations
            elites = population[np.argsort(fitness_values)[:2]]
            mutation_strength = 1 / np.sqrt(2 * np.sqrt(self.dim))  # Dynamic mutation strategy
            offspring = np.mean(elites, axis=0) + mutation_strength * np.random.normal(0, 1, self.dim)
            population[np.argmax(fitness_values)] = offspring
        return population[np.argmin(fitness_values)]