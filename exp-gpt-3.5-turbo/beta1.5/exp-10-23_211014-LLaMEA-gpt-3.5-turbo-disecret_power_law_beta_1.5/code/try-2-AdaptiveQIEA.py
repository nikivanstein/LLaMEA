import numpy as np

class AdaptiveQIEA(QIEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.1

    def _adaptive_evolutionary_update(self, population, func):
        fitness_values = [func(individual) for individual in population]
        sorted_indices = np.argsort(fitness_values)
        elite = population[sorted_indices[0]]

        for i in range(1, self.population_size):
            mutation_rate = self.mutation_rate / np.sqrt(i)  # Adaptive mutation rate
            population[i] = elite + np.random.normal(0, mutation_rate, size=self.dim)

        return population

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._quantum_update(population)
            population = self._adaptive_evolutionary_update(population, func)

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution