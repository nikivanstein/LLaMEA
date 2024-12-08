import numpy as np

class EnhancedMHSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size
        self.bandwidth = 0.01

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def harmony_search_move(x, best):
            r = np.random.rand(self.dim)
            x = (1 - self.bandwidth) * x + self.bandwidth * best + self.bandwidth * r
            x = np.clip(x, -5.0, 5.0)
            return x

        population = initialize_population()
        fitness_values = np.array([objective_function(ind) for ind in population])
        best_idx = np.argmin(fitness_values)
        best = population[best_idx].copy()

        for _ in range(self.max_iter):
            for idx, ind in enumerate(population):
                population[idx] = harmony_search_move(ind, best)

            new_fitness_values = np.array([objective_function(ind) for ind in population])
            best_idx = np.argmin(new_fitness_values)

            if new_fitness_values[best_idx] < fitness_values[best_idx]:
                best = population[best_idx]

            fitness_values = new_fitness_values

            # Dynamic adaptation of bandwidth
            self.bandwidth *= 0.99

        return best