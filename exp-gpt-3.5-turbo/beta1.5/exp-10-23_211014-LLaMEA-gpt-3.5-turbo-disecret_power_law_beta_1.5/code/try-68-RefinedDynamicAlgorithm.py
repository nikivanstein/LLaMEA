import numpy as np

class RefinedDynamicAlgorithm(HybridDynamicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta = 1.0

    def _firefly_algorithm(self, population, func):
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        attractiveness = 1 / (1 + np.sqrt(np.sum((population - best_solution) ** 2, axis=1)))
        for i in range(self.population_size):
            for j in range(self.population_size):
                if fitness[i] > fitness[j]:
                    distance = np.linalg.norm(population[j] - population[i])
                    population[i] += attractiveness[j] * np.exp(-self.beta * distance ** 2) * (population[j] - population[i]) + np.random.uniform(-0.1, 0.1, self.dim)
                    population[i] = np.clip(population[i], -5.0, 5.0)
        return population

    def _levy_flight(self, population):
        levy = np.random.standard_cauchy(size=(self.population_size, self.dim))
        population += 0.01 * levy
        population = np.clip(population, -5.0, 5.0)
        return population