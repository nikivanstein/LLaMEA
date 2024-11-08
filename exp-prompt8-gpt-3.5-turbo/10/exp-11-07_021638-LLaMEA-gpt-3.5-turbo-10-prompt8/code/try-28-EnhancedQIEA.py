import numpy as np

class EnhancedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.elite_count = 2  # Define elite count as a variable for flexibility
        self.mutation_scale = 1.0  # Adaptive mutation scale

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_values = np.array([func(ind) for ind in population])
            elites_idx = np.argsort(fitness_values)[:self.elite_count]
            elites = population[elites_idx]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, self.mutation_scale, self.dim)
            replace_idx = np.argmax(fitness_values)
            population[replace_idx] = offspring
        return population[np.argmin([func(ind) for ind in population])]