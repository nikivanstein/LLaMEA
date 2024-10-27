import numpy as np

class AdaptiveBandwidthMHSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def harmony_search_move(x, best, bandwidth=0.01):
            r = np.random.rand(self.dim)
            prob = np.random.rand(self.dim)
            x = np.where(prob < 0.5, (1 - bandwidth) * x + bandwidth * best + bandwidth * r, np.random.uniform(-5.0, 5.0, size=(self.dim)))
            x = np.clip(x, -5.0, 5.0)
            return x

        population = initialize_population()
        fitness_values = np.array([objective_function(ind) for ind in population])
        best_idx = np.argmin(fitness_values)
        best = population[best_idx].copy()

        for _ in range(self.max_iter):
            for idx, ind in enumerate(population):
                bandwidth = 0.01 + 0.24 * np.random.rand()
                population[idx] = harmony_search_move(ind, best, bandwidth)

            new_fitness_values = np.array([objective_function(ind) for ind in population])
            best_idx = np.argmin(new_fitness_values)

            if new_fitness_values[best_idx] < fitness_values[best_idx]:
                best = population[best_idx]

            fitness_values = new_fitness_values

        return best