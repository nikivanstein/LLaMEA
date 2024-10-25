import numpy as np

class EnhancedHybridDynamicAlgorithm(HybridDynamicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta = 0.1

    def _levy_flight(self, population):
        levy = np.random.standard_cauchy(size=(self.population_size, self.dim))
        population += self.beta * levy
        population = np.clip(population, -5.0, 5.0)
        return population