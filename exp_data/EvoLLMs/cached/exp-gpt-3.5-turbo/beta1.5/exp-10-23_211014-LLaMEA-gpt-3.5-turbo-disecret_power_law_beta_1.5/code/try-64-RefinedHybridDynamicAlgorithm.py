import numpy as np

class RefinedHybridDynamicAlgorithm(HybridDynamicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta = 0.1

    def _firefly_algorithm(self, population, func):
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        attractiveness = 1 / (1 + np.sqrt(np.sum((population - best_solution) ** 2, axis=1)))
        for i in range(self.population_size):
            for j in range(self.population_size):
                if fitness[i] > fitness[j]:
                    movement = attractiveness[j] * (population[j] - population[i]) + self.beta * np.random.standard_cauchy(self.dim)
                    population[i] += movement
                    population[i] = np.clip(population[i], -5.0, 5.0)
        return population