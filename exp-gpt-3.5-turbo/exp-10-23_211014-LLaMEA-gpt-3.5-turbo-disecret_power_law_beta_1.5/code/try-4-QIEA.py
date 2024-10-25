import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.alpha = 0.9

    def _quantum_update(self, population):
        for i in range(self.population_size):
            for j in range(self.dim):
                population[i][j] = np.sign(np.random.uniform(-1, 1)) * abs(np.cos(np.pi * population[i][j])) ** self.alpha
        return population

    def _evolutionary_update(self, population, func):
        fitness_values = [func(individual) for individual in population]
        sorted_indices = np.argsort(fitness_values)
        elite = population[sorted_indices[0]]

        for i in range(1, self.population_size):
            population[i] = elite + np.random.normal(0, 0.1, size=self.dim)

        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._quantum_update(population)
            population = self._evolutionary_update(population, func)

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution