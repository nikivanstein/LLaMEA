import numpy as np

class ADELS:
    def __init__(self, budget, dim, n_individuals=50, max_local_iter=15, mutation_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.n_individuals = n_individuals
        self.max_local_iter = max_local_iter
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        def local_search(x):
            best_x = x.copy()
            best_fitness = func(x)
            for _ in range(self.max_local_iter):
                new_x = x + np.random.uniform(-0.05, 0.05, size=self.dim)
                new_fitness = func(new_x)
                if new_fitness < best_fitness:
                    best_x = new_x
                    best_fitness = new_fitness
            return best_x

        population = np.random.uniform(-5.0, 5.0, size=(self.n_individuals, self.dim))
        population_fitness = np.array([func(p) for p in population])
        best_idx = np.argmin(population_fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget // self.n_individuals):
            for i in range(self.n_individuals):
                new_position = population[i] + np.random.uniform(-0.5, 0.5, size=self.dim) * (best_solution - population[i])
                new_position = np.clip(new_position, -5.0, 5.0)

                if np.random.rand() < self.mutation_rate:
                    new_position += np.random.normal(0, 0.3, size=self.dim)

                new_position = local_search(new_position)
                new_fitness = func(new_position)

                if new_fitness < population_fitness[i]:
                    population[i] = new_position
                    population_fitness[i] = new_fitness

                    if new_fitness < func(best_solution):
                        best_solution = new_position

        return best_solution