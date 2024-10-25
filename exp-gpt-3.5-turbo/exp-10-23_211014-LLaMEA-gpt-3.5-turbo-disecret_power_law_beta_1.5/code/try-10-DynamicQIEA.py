import numpy as np

class DynamicQIEA(QIEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.5

    def _dynamic_learning_rate_adjustment(self, population, func, iteration):
        learning_rate = self.beta / (1 + self.alpha * iteration)
        for i in range(1, self.population_size):
            population[i] += learning_rate * np.random.standard_cauchy(size=self.dim) / (np.power(np.abs(np.random.normal(0, 1, size=self.dim)), 1/self.beta))
            population[i] = np.clip(population[i], -5.0, 5.0)
        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for iteration in range(self.budget // self.population_size):
            population = self._quantum_update(population)
            population = self._dynamic_learning_rate_adjustment(population, func, iteration)
            population = self._evolutionary_update(population, func)

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution