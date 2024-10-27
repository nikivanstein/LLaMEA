# import numpy as np

class EnhancedHybridDynamicAlgorithm(HybridDynamicAlgorithm):
    def _enhanced_levy_flight(self, population):
        levy = np.random.standard_cauchy(size=(self.population_size, self.dim))
        population += 0.01 * levy
        population = np.clip(population, -5.0, 5.0)
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._differential_evolution(population, func)
            population = self._firefly_algorithm(population, func)
            population = self._pso(population, func)
            population = self._enhanced_levy_flight(population)

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution