import numpy as np

class DQIEA(QIEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta = 1.5
        self.history = []

    def _adaptive_quantum_update(self, population):
        for i in range(1, self.population_size):
            self.beta = np.mean(self.history) if len(self.history) > 0 else self.beta
            levy = np.random.standard_cauchy(size=self.dim) / (np.power(np.abs(np.random.normal(0, 1, size=self.dim)), 1/self.beta))
            population[i] += levy
            population[i] = np.clip(population[i], -5.0, 5.0)
        return population

    def _update_history(self, performance):
        self.history.append(performance)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._adaptive_quantum_update(population)
            population = self._levy_update(population)
            population = self._evolutionary_update(population, func)

            performances = [func(individual) for individual in population]
            best_solution = population[np.argmin(performances)]
            self._update_history(np.min(performances))

        return best_solution