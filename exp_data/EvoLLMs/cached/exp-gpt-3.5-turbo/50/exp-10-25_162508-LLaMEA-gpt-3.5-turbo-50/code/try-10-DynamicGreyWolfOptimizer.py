import numpy as np

class DynamicGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 5 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _update_position(self, alpha, beta, delta, individual, a, c):
        new_position = np.clip(individual + a * (alpha - individual) + c * (beta - delta), self.lower_bound, self.upper_bound)
        return new_position

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            fitness_values = self._get_fitness(population, func)
            alpha, beta, delta = population[np.argsort(fitness_values)[:3]]
            a = 2 - 2 * evals / self.budget
            c = 2 * np.random.rand()

            for i in range(self.pop_size):
                new_position = self._update_position(alpha, beta, delta, population[i], a, c)
                evals += 1

                if evals >= self.budget:
                    break

        best_solution = population[np.argmin(self._get_fitness(population, func))]
        return best_solution