import numpy as np

class QuantumInspiredEnhancedExplorationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = population[np.argmin([func(x) for x in population])]

        for _ in range(self.budget):
            new_population = population + np.random.normal(0, 1, (self.budget, self.dim))
            current_fitness = [func(x) for x in population]
            new_fitness = [func(x) for x in new_population]

            population = new_population[new_fitness < current_fitness]
            best_solution = population[np.argmin([func(x) for x in population])]

        return best_solution