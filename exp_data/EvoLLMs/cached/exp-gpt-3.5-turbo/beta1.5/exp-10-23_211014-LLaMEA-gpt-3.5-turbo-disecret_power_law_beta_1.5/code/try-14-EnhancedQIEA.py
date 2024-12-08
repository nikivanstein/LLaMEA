import numpy as np

class EnhancedQIEA(QIEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta = 1.5

    def _levy_update(self, population):
        for i in range(1, self.population_size):
            levy = np.random.standard_cauchy(size=self.dim) / (np.power(np.abs(np.random.normal(0, 1, size=self.dim)), 1/self.beta))
            population[i] += levy
            population[i] = np.clip(population[i], -5.0, 5.0)
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._quantum_update(population)
            population = self._levy_update(population)
            population = self._evolutionary_update(population, func)

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution