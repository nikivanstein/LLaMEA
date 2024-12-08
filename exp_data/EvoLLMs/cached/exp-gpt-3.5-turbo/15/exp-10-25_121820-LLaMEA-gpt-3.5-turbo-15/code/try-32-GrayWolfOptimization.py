import numpy as np

class GrayWolfOptimization:
    def __init__(self, budget, dim, a=2, b=1, c=1):
        self.budget = budget
        self.dim = dim
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        population = initialize_population()
        fitness = np.array([func(individual) for individual in population])
        g_best_idx = np.argmin(fitness)
        g_best = population[g_best_idx]

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                alpha = np.random.uniform(0, 2 * self.a) - self.a
                beta = np.random.uniform(0, 2 * self.b) - self.b
                delta = np.random.uniform(0, 2 * self.c) - self.c

                X1 = g_best - alpha * np.abs(population[i])
                X2 = p_best[i] - beta * np.abs(population[i])
                X3 = np.mean(population, axis=0) - delta * np.abs(population[i])

                population[i] = (X1 + X2 + X3) / 3
                population[i] = np.clip(population[i], -5.0, 5.0)

                fitness_i = func(population[i])
                if fitness_i < fitness[i]:
                    fitness[i] = fitness_i
                    if fitness_i < fitness[g_best_idx]:
                        g_best_idx = i
                        g_best = population[i]

        return g_best