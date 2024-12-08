import numpy as np

class EnhancedHybridDynamicAlgorithm(HybridDynamicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._differential_evolution(population, func)
            population = self._firefly_algorithm(population, func)
            population = self._pso(population, func)
            population = self._levy_flight(population)
            for i in range(self.population_size):
                if np.random.rand() < self.beta:
                    population[i] = np.clip(population[i] + np.random.normal(0, 0.1, self.dim), -5.0, 5.0)

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution