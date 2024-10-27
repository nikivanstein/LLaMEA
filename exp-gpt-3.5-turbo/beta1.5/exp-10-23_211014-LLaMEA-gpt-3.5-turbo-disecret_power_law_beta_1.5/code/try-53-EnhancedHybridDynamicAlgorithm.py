# import numpy as np

class EnhancedHybridDynamicAlgorithm(HybridDynamicAlgorithm):
    def _levy_flight(self, population):
        levy = np.random.standard_cauchy(size=(self.population_size, self.dim))
        population += 0.01 * levy
        population = np.clip(population, -5.0, 5.0)
        return population